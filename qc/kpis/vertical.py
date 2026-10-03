"""KPI 12 - vertical gradients (diagnostic only, weight 0)."""
from __future__ import annotations

import numpy as np

from .common import Ctx


def _band_slope(phase: np.ndarray, valid: np.ndarray, n_bands: int = 5) -> float:
    h = phase.shape[0]
    xs, ys = [], []
    for i in range(n_bands):
        v = valid[i * h // n_bands : (i + 1) * h // n_bands]
        p = phase[i * h // n_bands : (i + 1) * h // n_bands]
        if v.sum():
            xs.append(i + 0.5)
            ys.append(float(p[v].mean()))
    if len(xs) < 2:
        return float("nan")
    slope = np.polyfit(xs, ys, 1)[0]
    return float(slope)


def compute(ctx: Ctx) -> dict:
    return {
        "vertical_pore_slope": _band_slope(ctx.pore, ctx.valid),
        "vertical_si_slope": _band_slope(ctx.si, ctx.valid),
    }
