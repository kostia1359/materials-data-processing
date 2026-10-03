"""KPI 5 (pore fractions) and KPI 6 (pore sizes: chords, local thickness)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .common import Ctx, chord_lengths, weighted_quantile


def compute(ctx: Ctx) -> dict:
    a_valid = float(ctx.valid.sum())
    out = {
        "pore_frac_deep": float(ctx.pore.sum() / a_valid) if a_valid else float("nan"),
        "ambiguous_frac": float(ctx.ambiguous.sum() / a_valid) if a_valid else float("nan"),
    }
    # sensitivity: pore fraction when t1 moves +-3 levels (recomputed upstream
    # via ctx.extras['pore_frac_at_t1_plus3' / 'pore_frac_at_t1_minus3'])
    plus = ctx.extras.get("pore_frac_at_t1_plus3")
    minus = ctx.extras.get("pore_frac_at_t1_minus3")
    if plus is not None and minus is not None:
        out["pore_frac_slope_per_level"] = float(
            abs(minus - plus) / 6.0
        )
    else:
        out["pore_frac_slope_per_level"] = float("nan")
    return out


def pore_chords(ctx: Ctx) -> dict:
    """Horizontal and vertical run-length pore chords, px -> um."""
    up = ctx.um_per_px
    ch = chord_lengths(ctx.pore, axis=1) * up   # horizontal runs
    cv = chord_lengths(ctx.pore, axis=0) * up   # vertical runs
    out = {
        "pore_chord_h_mean_um": float(ch.mean()) if ch.size else float("nan"),
        "pore_chord_v_mean_um": float(cv.mean()) if cv.size else float("nan"),
        "pore_chord_h_p90_um": float(np.percentile(ch, 90)) if ch.size else float("nan"),
        "_pore_chords_h_um": ch,
        "_pore_chords_v_um": cv,
    }
    return out


def pore_local_thickness(ctx: Ctx) -> dict:
    """Opening granulometry with disks r=1..40 px -> area-weighted LT D50.

    Opening by disk r is computed via EDT (O(N) per radius): erode = EDT(mask)
    >= r, dilate = EDT(~eroded) <= r. Run on the mask downsampled x2 per the
    brief (porespy does the same); radii are scaled accordingly.
    """
    mask = ctx.pore[::2, ::2] & ctx.valid[::2, ::2]
    if mask.sum() < 50:
        return {"pore_lt_d50_um": float("nan")}
    total = float(mask.sum())
    dt = ndi.distance_transform_edt(mask)
    remaining = []
    for r in range(1, 21):  # 40 full-res px at 2x downsample
        eroded = dt >= r
        if not eroded.any():
            break
        opened = ndi.distance_transform_edt(~eroded) <= r
        remaining.append((r, opened.sum() / total))
        if remaining[-1][1] < 0.02:
            break
    if not remaining:
        return {"pore_lt_d50_um": float("nan")}
    radii = np.array([r for r, _ in remaining])
    fracs = np.array([f for _, f in remaining])
    # fraction of pore area with LT diameter >= 2r (in half-res px); D50 at 50%
    diam_um = 2 * radii * 2 * ctx.um_per_px
    d50 = float(np.interp(0.5, fracs[::-1], diam_um[::-1]))
    return {"pore_lt_d50_um": d50}


def compute_all(ctx: Ctx) -> dict:
    out = compute(ctx)
    out.update(pore_chords(ctx))
    out.update(pore_local_thickness(ctx))
    return out
