"""Three-class BSE segmentation: deep pore / carbon matrix / Si-candidate."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_multiotsu
from skimage.measure import label
from skimage.morphology import (
    binary_opening,
    disk,
    remove_small_objects,
)


def hysteresis_grow(core: np.ndarray, extended: np.ndarray) -> np.ndarray:
    """Keep extended-mask components that contain >=1 core pixel."""
    lab = label(extended)
    if lab.max() == 0:
        return np.zeros_like(core)
    has_core = np.zeros(lab.max() + 1, dtype=bool)
    has_core[lab[core]] = True
    return has_core[lab]

from .io import Channels


def smooth(img: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian(img, sigma=sigma, preserve_range=True)


def otsu_thresholds(bse_s: np.ndarray, valid: np.ndarray, classes: int = 3):
    t = threshold_multiotsu(bse_s[valid], classes=classes)
    return float(t[0]), float(t[1])


def segment(ch: Channels, cfg: dict, threshold_shift: float = 0.0) -> dict:
    """Return final masks + diagnostics for one sample.

    threshold_shift adds a constant to both Otsu thresholds (used by the
    threshold-sensitivity check in Section 6.3)."""
    sigma = cfg["smoothing_sigma_px"]
    bse_s = smooth(ch.bse, sigma)
    etd_s = smooth(ch.etd, sigma) if ch.etd is not None else None
    valid = ch.valid

    t1, t2 = otsu_thresholds(bse_s, valid, cfg["otsu_classes"])
    t1 += threshold_shift
    t2 += threshold_shift

    pore_core = (bse_s < t1) & valid
    # hysteresis: pixels between t1 and t1+hyst join only if connected to a core
    hyst = cfg["pore_hysteresis_levels"]
    pore_grown = hysteresis_grow(pore_core, (bse_s < t1 + hyst) & valid) & valid
    if etd_s is not None:
        etd_carbon_vals = etd_s[(bse_s >= t1) & (bse_s < t2) & valid]
        etd_p10 = np.percentile(etd_carbon_vals, 10) if etd_carbon_vals.size else 0
        etd_dark = etd_s < etd_p10
        pore = pore_grown & etd_dark
        ambiguous = pore_grown & ~etd_dark
    else:
        pore = pore_grown
        ambiguous = np.zeros_like(pore)
        etd_p10 = float("nan")
    pore = remove_small_objects(pore, min_size=cfg["pore_min_px"]) & valid
    ambiguous = ambiguous & ~pore & valid

    si = (bse_s >= t2) & valid
    si = binary_opening(si, disk(cfg["si_opening_radius_px"]))
    si_counted = remove_small_objects(si, min_size=cfg["si_min_px_count"])
    # interior check: median smoothed BSE inside eroded component >= t2+margin
    margin = cfg["si_interior_margin_levels"]
    lab = label(si_counted)
    keep = np.zeros_like(si_counted)
    for comp in range(1, lab.max() + 1):
        cm = lab == comp
        interior = ndi.binary_erosion(cm, iterations=1)
        src = interior if interior.sum() else cm
        if np.median(bse_s[src]) >= t2 + margin:
            keep |= cm
    si = keep

    carbon = valid & ~pore & ~si

    # shading diagnostic: per-band Otsu drift
    shading_flag = False
    band_thresh = []
    h = bse_s.shape[0]
    for i in range(5):
        band = valid[i * h // 5 : (i + 1) * h // 5]
        bs = bse_s[i * h // 5 : (i + 1) * h // 5]
        if band.sum() > 1000:
            try:
                tt = threshold_multiotsu(bs[band], classes=3)
                band_thresh.append((float(tt[0]), float(tt[1])))
            except Exception:
                pass
    if band_thresh:
        arr = np.array(band_thresh)
        if (arr.max(axis=0) - arr.min(axis=0)).max() > 6:
            shading_flag = True

    return {
        "bse_s": bse_s,
        "etd_s": etd_s,
        "pore": pore,
        "carbon": carbon,
        "si": si,
        "ambiguous": ambiguous,
        "valid": valid,
        "t1": t1,
        "t2": t2,
        "etd_p10": etd_p10,
        "band_thresholds": band_thresh,
        "shading_flag": shading_flag,
    }


def overlay_png(seg: dict, bse: np.ndarray, path, downscale: int = 4):
    """BSE grey + pore blue + Si orange + masked hatched grey."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = bse.shape
    rgb = np.repeat(bse[:, :, None].astype(float) / 255, 3, axis=2)
    pore, si, valid = seg["pore"], seg["si"], seg["valid"]
    rgb[pore] = [0.15, 0.45, 0.95]
    rgb[si] = [1.0, 0.55, 0.1]
    rgb[~valid] = 0.75 * rgb[~valid] + 0.25 * np.array([0.5, 0.5, 0.5])

    fig, ax = plt.subplots(figsize=(w / downscale / 100, h / downscale / 100), dpi=100)
    ax.imshow(rgb[::downscale, ::downscale])
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
