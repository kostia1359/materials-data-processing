"""KPI 7 - anisotropy: chord ratios + structure tensor."""
from __future__ import annotations

import numpy as np
from skimage.feature import structure_tensor

from .common import Ctx, chord_lengths


def _ratio(h: np.ndarray, v: np.ndarray) -> float:
    if h.size == 0 or v.size == 0 or v.mean() == 0:
        return float("nan")
    return float(h.mean() / v.mean())


def compute(ctx: Ctx) -> dict:
    pore_h = chord_lengths(ctx.pore, axis=1)
    pore_v = chord_lengths(ctx.pore, axis=0)
    carb_h = chord_lengths(ctx.carbon, axis=1)
    carb_v = chord_lengths(ctx.carbon, axis=0)
    out = {
        "aniso_pore_chord_ratio": _ratio(pore_h, pore_v),
        "aniso_carbon_chord_ratio": _ratio(carb_h, carb_v),
    }
    # structure tensor on smoothed BSE (full res is heavy; use sigma=8 per cfg
    # on a 2x-downsampled image, then scale sigma accordingly)
    if ctx.bse_s is not None:
        sigma = ctx.cfg.get("structure_tensor_sigma_px", 8)
        sub = ctx.bse_s[::2, ::2]
        sub_valid = ctx.valid[::2, ::2]
        try:
            axx, axy, ayy = structure_tensor(sub, sigma=sigma / 2)
        except Exception:
            axx, axy, ayy = structure_tensor(sub, sigma=4)
        # accumulate only valid pixels
        m = sub_valid
        axx_v, axy_v, ayy_v = axx[m].mean(), axy[m].mean(), ayy[m].mean()
        # eigen decomposition of [[axx, axy],[axy, ayy]]
        tr = axx_v + ayy_v
        det = axx_v * ayy_v - axy_v**2
        disc = max(tr * tr / 4 - det, 0.0)
        l1 = tr / 2 + np.sqrt(disc)
        l2 = tr / 2 - np.sqrt(disc)
        coherence = float((l1 - l2) / (l1 + l2)) if (l1 + l2) > 0 else float("nan")
        # dominant orientation = eigenvector of l1
        orient = float(np.degrees(0.5 * np.arctan2(2 * axy_v, axx_v - ayy_v)))
        out["st_coherence"] = coherence
        out["st_orientation_deg"] = orient
    else:
        out["st_coherence"] = float("nan")
        out["st_orientation_deg"] = float("nan")
    return out
