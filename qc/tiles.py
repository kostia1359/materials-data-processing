"""Strips, quadrats and bootstrap resampling for per-sample uncertainty."""
from __future__ import annotations

import numpy as np

from .kpis.common import Ctx


def strip_masks(ctx: Ctx, n_strips: int, width_px: int) -> list[np.ndarray]:
    """Full-height vertical strips; last strip widened to cover remainder."""
    h, w = ctx.valid.shape
    masks = []
    for i in range(n_strips):
        x0 = i * width_px
        x1 = min(w, x0 + width_px) if i < n_strips - 1 else w
        if x0 >= w:
            break
        m = np.zeros((h, w), dtype=bool)
        m[:, x0:x1] = True
        masks.append(m)
    return masks


def strip_ses(strip_rows: list[dict], kpi_names: list[str]) -> dict:
    """SE of each KPI = SD over strips / sqrt(n_strips)."""
    ses = {}
    n = len(strip_rows)
    for k in kpi_names:
        vals = np.array([r.get(k, np.nan) for r in strip_rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        ses[f"{k}_se"] = float(vals.std(ddof=1) / np.sqrt(n)) if vals.size > 1 else float("nan")
    return ses


def bootstrap_intervals(
    strip_rows: list[dict], kpi_names: list[str], n_boot: int, seed: int
) -> dict:
    """Block-bootstrap over strips -> 2.5/97.5% interval per KPI."""
    rng = np.random.default_rng(seed)
    out = {}
    n = len(strip_rows)
    if n < 2:
        return out
    for k in kpi_names:
        vals = np.array([r.get(k, np.nan) for r in strip_rows], dtype=float)
        if not np.isfinite(vals).all():
            out[f"{k}_ci"] = (float("nan"), float("nan"))
            continue
        means = np.array(
            [vals[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
        )
        out[f"{k}_ci"] = (float(np.percentile(means, 2.5)),
                          float(np.percentile(means, 97.5)))
    return out
