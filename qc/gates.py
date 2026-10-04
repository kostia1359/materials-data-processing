"""Acquisition gates: covariates describing imaging session, not material."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.measure import blur_effect
from skimage.registration import phase_cross_correlation
from skimage.morphology import binary_erosion, disk

from .io import Channels


def n_grey_levels(img: np.ndarray, valid: np.ndarray) -> int:
    return int(np.unique(img[valid]).size)


def comb_step(img: np.ndarray, valid: np.ndarray) -> float:
    levels = np.unique(img[valid])
    levels = levels[(levels >= 20) & (levels <= 200)]
    if levels.size < 3:
        return float("nan")
    return float(np.median(np.diff(levels)))


def noise_sd(img: np.ndarray, carbon_mask: np.ndarray) -> float:
    inner = binary_erosion(carbon_mask, disk(5))
    if inner.sum() < 100:
        inner = carbon_mask
    if inner.sum() < 10:
        return float("nan")
    diff = img.astype(float) - uniform_filter(img.astype(float), size=5)
    return float(diff[inner].std())


def channel_median(img: np.ndarray | None, mask: np.ndarray) -> float:
    if img is None or mask.sum() < 10:
        return float("nan")
    return float(np.median(img[mask]))


def saturation_frac(img: np.ndarray | None, valid: np.ndarray) -> tuple[float, float]:
    if img is None:
        return float("nan"), float("nan")
    v = img[valid]
    return float((v == 0).mean()), float((v == 255).mean())


def row_gradient_pct(img: np.ndarray, valid: np.ndarray) -> float:
    h = img.shape[0]
    eighth = max(1, h // 8)
    vals = img.astype(float)
    overall = vals[valid].mean()
    top = vals[:eighth][valid[:eighth]].mean()
    bot = vals[h - eighth :][valid[h - eighth :]].mean()
    return float((top - bot) / overall * 100)


def registration_shifts(ch: Channels) -> dict:
    shifts = {}
    ref = ch.bse
    for name, img in (("etd", ch.etd), ("inlens", ch.inlens)):
        if img is None:
            shifts[f"shift_{name}_px"] = float("nan")
            continue
        a = ref[::4, ::4].astype(float)
        b = img[::4, ::4].astype(float)
        shift, _, _ = phase_cross_correlation(a, b, upsample_factor=4)
        shifts[f"shift_{name}_px"] = float(4 * np.hypot(*shift))
    return shifts


def compute_gates(
    ch: Channels,
    masks: dict,
    thresholds: tuple[float, float],
) -> dict:
    """All acquisition covariates for one sample."""
    valid = ch.valid
    carbon = masks["carbon"]
    si = masks["si"]
    carbon_er = binary_erosion(carbon, disk(5))
    si_er = binary_erosion(si, disk(2))
    g = {
        "n_grey_levels_bse": n_grey_levels(ch.bse, valid),
        "comb_step_bse": comb_step(ch.bse, valid),
        "noise_sd_bse": noise_sd(ch.bse, carbon),
        "blur_bse": float(blur_effect(ch.bse)),
        "bse_c_med": channel_median(ch.bse, carbon_er),
        "etd_c_med": channel_median(ch.etd, carbon_er),
        "inlens_c_med": channel_median(ch.inlens, carbon_er),
        "bse_si_med": channel_median(ch.bse, si_er),
        "etd_si_med": channel_median(ch.etd, si_er),
        "inlens_si_med": channel_median(ch.inlens, si_er),
        "bse_row_gradient_pct": row_gradient_pct(ch.bse, valid),
        "t1": float(thresholds[0]),
        "t2": float(thresholds[1]),
        "height_px": int(ch.bse.shape[0]),
        "width_px": int(ch.bse.shape[1]),
        "px_nm": ch.px_nm,
        "masked_rows": ch.masked_rows_top + ch.masked_rows_bottom,
    }
    for name, img in (("bse", ch.bse), ("etd", ch.etd), ("inlens", ch.inlens)):
        lo, hi = saturation_frac(img, valid)
        g[f"sat0_{name}"] = lo
        g[f"sat255_{name}"] = hi
    g.update(registration_shifts(ch))
    return g
