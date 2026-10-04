"""KPI 11 - defects: cracks (Sato), largest void, high-Z inclusions."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sato
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from .common import Ctx


def curtaining_angle_deg(inlens: np.ndarray | None) -> float:
    """Dominant stripe angle of ion-milling curtaining in InLens, 50-80 deg.

    Estimated as the orientation of the strongest off-axis Fourier peak.
    Returns NaN when undetectable.
    """
    if inlens is None:
        return float("nan")
    sub = inlens[::4, ::4].astype(float)
    sub = sub - sub.mean()
    win = np.outer(np.hanning(sub.shape[0]), np.hanning(sub.shape[1]))
    F = np.abs(np.fft.fftshift(np.fft.fft2(sub * win)))
    F[F.shape[0] // 2 - 3 : F.shape[0] // 2 + 4,
      F.shape[1] // 2 - 3 : F.shape[1] // 2 + 4] = 0  # remove DC
    h, w = F.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    ang = np.degrees(np.arctan2(fy, fx))
    # curtaining stripes are near-vertical in image space; the Fourier peak is
    # perpendicular to the stripes -> stripes at angle theta have peak at theta+90
    band = (np.abs(np.abs(ang) - 90) >= 50) & (np.abs(np.abs(ang) - 90) <= 80)
    band |= (np.abs(ang) >= 50) & (np.abs(ang) <= 80)
    if not band.any():
        return float("nan")
    idx = np.argmax(np.where(band, F, 0))
    iy, ix = np.unravel_index(idx, F.shape)
    peak_ang = float(np.degrees(np.arctan2(fy[iy, 0], fx[0, ix])))
    stripe_ang = peak_ang + 90 if peak_ang <= 0 else peak_ang - 90
    return float(abs(stripe_ang))


def sato_threshold_for(bse_s: np.ndarray, valid: np.ndarray, cfg: dict) -> tuple[float, np.ndarray]:
    """Sato ridge response on inverted BSE; return (p99.5 response, response map)."""
    sigmas = cfg.get("crack_sato_sigmas", [1, 2, 3])
    sub = bse_s[::2, ::2]
    inv = 255.0 - sub
    resp = sato(inv, sigmas=sigmas, black_ridges=False)
    v = valid[::2, ::2]
    thr = float(np.percentile(resp[v], 99.5)) if v.sum() else float("nan")
    return thr, resp


def compute(ctx: Ctx) -> dict:
    cfg = ctx.cfg
    out = {}
    # --- cracks ---
    crack_thr = ctx.extras.get("crack_threshold")
    resp = ctx.extras.get("sato_response")  # on 2x-downsampled image
    curtain = ctx.extras.get("curtaining_deg", float("nan"))
    density = float("nan")
    if crack_thr is not None and resp is not None and np.isfinite(crack_thr):
        ridges = resp >= crack_thr
        ridges &= ctx.valid[::2, ::2]
        skel = skeletonize(ridges)
        lab = label(skel, connectivity=2)
        kept_len = 0
        for p in regionprops(lab):
            if p.area < cfg.get("crack_min_length_px", 80) / 2:
                continue  # skeleton px at 2x
            # mean width of ridge under the skeleton component
            comp = lab == p.label
            width = 2.0
            vals = resp[comp]
            if vals.size:
                # width estimate: local ridge area / skeleton length
                dil = ndi.binary_dilation(comp, iterations=3) & ridges
                width = dil.sum() / max(p.area, 1)
            if width > cfg.get("crack_max_width_px", 6):
                continue
            # dominant orientation of the skeleton component
            coords = p.coords
            if coords.shape[0] >= 5:
                cov = np.cov(coords[:, :2].T)
                evals, evecs = np.linalg.eigh(cov)
                v = evecs[:, np.argmax(evals)]
                ang = abs(np.degrees(np.arctan2(v[0], v[1])))  # 0=vertical
            else:
                ang = 0.0
            if np.isfinite(curtain) and abs(ang - curtain) < cfg.get(
                "curtaining_exclusion_deg", 10
            ):
                continue
            kept_len += p.area * 2  # back to full-res px
        density = float(kept_len * ctx.um_per_px / (ctx.area_um2 / 1e6))
    out["crack_density_um_per_mm2"] = density

    # --- largest void (deep-pore component ECD) ---
    lab = label(ctx.pore & ctx.valid)
    if lab.max():
        amax = max(p.area for p in regionprops(lab))
        out["largest_void_ecd_um"] = float(2 * np.sqrt(amax / np.pi) * ctx.um_per_px)
    else:
        out["largest_void_ecd_um"] = 0.0

    # --- high-Z inclusions ---
    hiz_thr = cfg.get("hiz_threshold", 190)
    hiz_min = cfg.get("hiz_min_px", 50)
    if ctx.bse_s is not None:
        hiz = (ctx.bse_s > hiz_thr) & ctx.valid
        lab = label(hiz)
        n = sum(1 for p in regionprops(lab) if p.area >= hiz_min)
        out["hiz_inclusion_count_mm2"] = float(n / ctx.area_mm2) if ctx.area_mm2 else float("nan")
    else:
        out["hiz_inclusion_count_mm2"] = float("nan")
    return out
