"""Discovery, manifest, TIFF loading and hygiene."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import uniform_filter
from skimage.transform import rescale

SAMPLE_RE = re.compile(
    r"^(?:Batch_(?P<pb>\d+)_)?img_(?P<sid>[a-z0-9]+)_(?P<det>BSE|ETD|SE|Inlens)\.tif$",
    re.IGNORECASE,
)
BATCH_RE = re.compile(r"batch[\s_]*(\d)", re.IGNORECASE)


def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def batch_from_path(path: Path) -> str:
    m = BATCH_RE.search(str(path.parent))
    return m.group(1) if m else "unknown"


@dataclass
class Sample:
    sample_id: str
    batch: str
    path_bse: Path | None = None
    path_etd: Path | None = None
    path_inlens: Path | None = None
    etd_label_in_filename: str = ""
    has_batch_prefix: bool = False
    prefix_batch: str = ""
    warnings: list = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.path_bse is not None and self.path_etd is not None and self.path_inlens is not None


def discover(data_dir: str | Path) -> list[Sample]:
    """Walk data_dir, group TIFFs into samples keyed on (directory, sample id)."""
    data_dir = Path(data_dir)
    groups: dict[tuple[str, str], Sample] = {}
    for path in sorted(data_dir.rglob("*.tif")) + sorted(data_dir.rglob("*.tiff")):
        m = SAMPLE_RE.match(path.name)
        if not m:
            continue
        sid, det = m.group("sid").lower(), m.group("det").upper()
        key = (str(path.parent), sid)
        if key not in groups:
            groups[key] = Sample(
                sample_id=f"img_{sid}",
                batch=batch_from_path(path),
            )
        s = groups[key]
        if m.group("pb"):
            s.has_batch_prefix = True
            s.prefix_batch = m.group("pb")
            if s.batch != "unknown" and s.prefix_batch != s.batch:
                s.warnings.append(
                    f"filename prefix Batch_{s.prefix_batch} disagrees with folder batch {s.batch}"
                )
        if det == "BSE":
            s.path_bse = path
        elif det in ("ETD", "SE"):
            s.etd_label_in_filename = det
            s.path_etd = path
        elif det == "INLENS":
            s.path_inlens = path
    samples = list(groups.values())
    for s in samples:
        if s.path_bse is None:
            s.warnings.append("missing BSE")
        if s.path_etd is None:
            s.warnings.append("missing ETD")
        if s.path_inlens is None:
            s.warnings.append("missing InLens")
        elif not s.complete:
            s.warnings.append("incomplete triple")
    return samples


def px_nm_from_tags(path: Path) -> tuple[float, bool]:
    """Decode pixel size from TIFF resolution tags; default 25 nm/px + flag."""
    try:
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            xres = page.tags.get("XResolution")
            unit = page.tags.get("ResolutionUnit")
            if xres is None:
                return 25.0, True
            res = float(xres.value[0]) / float(xres.value[1])
            u = int(unit.value) if unit is not None else 2
            # res = pixels per unit; nm per px = unit_nm / res
            if u == 3:  # centimetre
                return 1e7 / res, False
            # default inch
            return 2.54e7 / res, False
    except Exception:
        return 25.0, True


def load_grey(path: Path) -> tuple[np.ndarray, dict]:
    """Load TIFF -> uint8 greyscale; drop rows/cols where channels disagree."""
    img = tifffile.imread(path)
    info = {"dropped_rows": [], "dropped_cols": []}
    if img.ndim == 3:
        rgb_ok = (img[..., 0] == img[..., 1]) & (img[..., 1] == img[..., 2])
        col_bad = ~rgb_ok.all(axis=0)
        info["dropped_cols"] = np.where(col_bad)[0].tolist()
        img = img[..., 0]
        if col_bad.any():
            img = img[:, ~col_bad]
            rgb_ok = rgb_ok[:, ~col_bad]
        row_bad = ~rgb_ok.all(axis=1)
        info["dropped_rows"] = np.where(row_bad)[0].tolist()
        if row_bad.any():
            img = img[~row_bad, :]
    return img.astype(np.uint8), info


def resample_to(img: np.ndarray, px_nm: float, target: float) -> np.ndarray:
    if abs(px_nm - target) / target <= 0.02:
        return img
    scale = px_nm / target  # >1 means coarser pixels -> shrink
    out = rescale(img, 1.0 / scale, order=1, anti_aliasing=True, preserve_range=True)
    return out.astype(np.uint8)


def detect_edge_bands(
    inlens: np.ndarray, etd: np.ndarray, max_frac: float = 0.10
) -> tuple[int, int]:
    """Rows of un-sectioned nodular material at top/bottom.

    Band rows: InLens row median < 0.6x global median AND ETD local
    (5x5) variance > 2x global median. Contiguous from an edge, <=10% H.
    Returns (masked_rows_top, masked_rows_bottom).
    """
    h = inlens.shape[0]
    il_row_med = np.median(inlens, axis=1)
    il_glob = np.median(inlens)
    etd_var = uniform_filter(etd.astype(float) ** 2, size=5) - uniform_filter(
        etd.astype(float), size=5
    ) ** 2
    etd_row_var = np.median(etd_var, axis=1)
    etd_glob = np.median(etd_row_var)
    flagged = (il_row_med < 0.6 * il_glob) & (etd_row_var > 2 * etd_glob)

    def run_from(edge_idx, step):
        n = 0
        i = edge_idx
        while 0 <= i < h and flagged[i] and n < int(max_frac * h):
            n += 1
            i += step
        return n

    top = run_from(0, 1)
    bottom = run_from(h - 1, -1)
    return top, bottom


@dataclass
class Channels:
    bse: np.ndarray
    etd: np.ndarray | None
    inlens: np.ndarray | None
    valid: np.ndarray
    px_nm: float
    px_nm_assumed: bool
    masked_rows_top: int
    masked_rows_bottom: int
    shape: tuple


def load_sample(sample: Sample, cfg: dict) -> Channels:
    """Load a sample's three channels with hygiene + edge-band masking."""
    bse, bse_info = load_grey(sample.path_bse)
    px_nm, assumed = px_nm_from_tags(sample.path_bse)
    px_nm_target = cfg["px_nm_target"]
    resampled = abs(px_nm - px_nm_target) / px_nm_target > 0.02
    if resampled:
        bse = resample_to(bse, px_nm, px_nm_target)
        px_nm = px_nm_target

    etd = inlens = None
    if sample.path_etd is not None:
        etd, _ = load_grey(sample.path_etd)
        if resampled:
            etd = resample_to(etd, px_nm_from_tags(sample.path_etd)[0], px_nm_target)
        if etd.shape != bse.shape:
            etd = _crop_to(etd, bse.shape)
    if sample.path_inlens is not None:
        inlens, _ = load_grey(sample.path_inlens)
        if resampled:
            inlens = resample_to(inlens, px_nm_from_tags(sample.path_inlens)[0], px_nm_target)
        if inlens.shape != bse.shape:
            inlens = _crop_to(inlens, bse.shape)

    top = bottom = 0
    manual = cfg.get("manual_masks", {}).get(sample.sample_id)
    if manual:
        top, bottom = manual.get("top", 0), manual.get("bottom", 0)
    elif inlens is not None and etd is not None:
        top, bottom = detect_edge_bands(inlens, etd)

    valid = np.ones(bse.shape, dtype=bool)
    if top:
        valid[:top, :] = False
    if bottom:
        valid[bse.shape[0] - bottom :, :] = False

    return Channels(
        bse=bse, etd=etd, inlens=inlens, valid=valid,
        px_nm=px_nm, px_nm_assumed=assumed or resampled,
        masked_rows_top=top, masked_rows_bottom=bottom, shape=bse.shape,
    )


def _crop_to(img: np.ndarray, shape: tuple) -> np.ndarray:
    return img[: shape[0], : shape[1]]


def write_manifest(samples: list[Sample], out_path: Path) -> pd.DataFrame:
    rows = []
    for s in samples:
        w = h = None
        px_nm, assumed = (25.0, True)
        sha = ""
        if s.path_bse is not None:
            img = tifffile.imread(s.path_bse)
            h, w = img.shape[:2]
            px_nm, assumed = px_nm_from_tags(s.path_bse)
            sha = sha1_of(s.path_bse)
        rows.append(
            dict(
                sample_id=s.sample_id, batch=s.batch,
                path_bse=str(s.path_bse or ""), path_etd=str(s.path_etd or ""),
                path_inlens=str(s.path_inlens or ""),
                etd_label_in_filename=s.etd_label_in_filename,
                has_batch_prefix=s.has_batch_prefix,
                width=w, height=h, px_nm=px_nm, px_nm_assumed=assumed,
                sha1_bse=sha, warnings=";".join(s.warnings),
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df
