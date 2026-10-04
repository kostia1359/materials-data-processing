"""KPI 10 - interfaces: pore-solid perimeter density + Si neighbour contacts."""
from __future__ import annotations

import numpy as np
from skimage.measure import perimeter_crofton
from skimage.morphology import binary_dilation, disk

from .common import Ctx


def compute(ctx: Ctx) -> dict:
    out = {}
    # pore-solid interface length per valid area, px -> um
    per_px = perimeter_crofton(ctx.pore & ctx.valid, directions=4)
    out["interface_pore_solid_um_per_um2"] = float(
        per_px * ctx.um_per_px / ctx.area_um2
    ) if ctx.area_um2 else float("nan")

    # Si contact: 1-px dilation ring of Si classified by neighbour phase
    si = ctx.si & ctx.valid
    ring = binary_dilation(si, disk(1)) & ~si & ctx.valid
    ring_len = float(ring.sum())
    if ring_len > 0:
        out["si_contact_pore_frac"] = float((ring & ctx.pore).sum() / ring_len)
        out["si_contact_carbon_frac"] = float((ring & ctx.carbon).sum() / ring_len)
    else:
        out["si_contact_pore_frac"] = float("nan")
        out["si_contact_carbon_frac"] = float("nan")
    return out
