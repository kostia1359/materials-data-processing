"""KPI 9 - pore connectivity: percolating share + Euler characteristic."""
from __future__ import annotations

import numpy as np
from skimage.measure import euler_number, label, regionprops

from .common import Ctx


def _percolating_share(mask: np.ndarray, valid: np.ndarray, axis: str) -> float:
    """Share of pore area in components touching both opposite edges (within valid region)."""
    if mask.sum() == 0:
        return float("nan")
    lab = label(mask & valid)
    h, w = mask.shape
    total = float(mask[valid].sum())
    if total == 0:
        return float("nan")
    spanning_area = 0.0
    for p in regionprops(lab):
        minr, minc, maxr, maxc = p.bbox
        if axis == "v" and minr == 0 and maxr == h:
            spanning_area += p.area
        elif axis == "h" and minc == 0 and maxc == w:
            spanning_area += p.area
    return float(spanning_area / total)


def compute(ctx: Ctx) -> dict:
    pore_v = ctx.pore & ctx.valid
    euler = euler_number(pore_v, connectivity=1)
    return {
        "pore_percolating_frac_v": _percolating_share(ctx.pore, ctx.valid, "v"),
        "pore_percolating_frac_h": _percolating_share(ctx.pore, ctx.valid, "h"),
        "pore_euler_density_mm2": float(euler / ctx.area_mm2) if ctx.area_mm2 else float("nan"),
    }
