"""KPI 1 - areal fractions (Delesse: area fraction ~ volume fraction)."""
from __future__ import annotations

from .common import Ctx


def compute(ctx: Ctx) -> dict:
    a_si = float(ctx.si.sum())
    a_carbon = float(ctx.carbon.sum())
    solids = a_si + a_carbon
    return {
        "si_frac_solid": a_si / solids if solids else float("nan"),
        "si_frac_total": a_si / ctx.valid.sum() if ctx.valid.sum() else float("nan"),
        "carbon_frac_total": a_carbon / ctx.valid.sum() if ctx.valid.sum() else float("nan"),
    }
