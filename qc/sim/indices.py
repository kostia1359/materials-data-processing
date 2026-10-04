"""Closed-form relative indices (Section 10.3, ``indices.py``).

Everything here is arithmetic on measured quantities (grade A) or a ratio
whose level is set by a frozen assumption (grade B); the cycle-life index is a
labelled heuristic (grade C).  Ratios to the baseline are formed later from
the per-sample values in ``sim.csv``.
"""
from __future__ import annotations

import numpy as np

from . import PORE

F = 96485.0


def transport_proxies(D_eff_rel: float, ecfg: dict, elcfg: dict) -> dict:
    """Diffusion time t_d = L^2/D_eff and limiting current i_lim (per unit D_eff).

    Uses D_bulk from the electrolyte block; D_eff = D_eff_rel * D_bulk.
    """
    L = ecfg["L_um"] * 1e-6
    D = elcfg["D_bulk_m2s"] * D_eff_rel
    if not np.isfinite(D) or D <= 0:
        return {"t_d_s": float("nan"), "i_lim_Am2": float("nan")}
    t_d = L ** 2 / D
    i_lim = 2 * F * elcfg["c0_molm3"] * D / ((1 - elcfg["t_plus"]) * L)
    return {"t_d_s": float(t_d), "i_lim_Am2": float(i_lim)}


def wetting_index(labels: np.ndarray, um_per_px: float, wcfg: dict) -> dict:
    """Access-limited porosimetry from the top edge (porespy); Lucas-Washburn ratio.

    Skipped with a flag when porespy is unavailable.
    """
    try:
        import porespy as ps
    except Exception:
        return {"wetting_available": False}
    pore = labels == PORE
    if pore.sum() == 0:
        return {"wetting_available": True, "breakthrough_r_um": float("nan"),
                "trapped_frac": float("nan"), "lw_time_rel": float("nan")}
    inlets = np.zeros_like(pore)
    inlets[0, :] = True
    try:
        # porespy >= 2: passing ``inlets`` makes the invasion access-limited
        sizes = ps.filters.porosimetry(pore, inlets=inlets, sizes=25)
    except Exception:
        return {"wetting_available": False}
    invaded = sizes > 0
    trapped = float(1.0 - invaded.sum() / pore.sum())
    # deep pores do not percolate in 2-D (caveat 10), so a true breakthrough radius is
    # undefined; report the median access-limited invasion radius instead
    r_bt = float(np.median(sizes[invaded])) * um_per_px if invaded.any() else float("nan")
    # Lucas-Washburn: t ~ eta L^2 / (gamma cos(theta) r); relative to r = 1 um
    theta = np.radians(wcfg["theta_deg"])
    lw = (wcfg["eta_mPas"] * 1e-3 * (70e-6) ** 2) / (wcfg["gamma_mNm"] * 1e-3 * np.cos(theta) * max(r_bt, 1e-3) * 1e-6)
    return {"wetting_available": True, "breakthrough_r_um": r_bt, "trapped_frac": trapped,
            "lw_time_rel": float(lw)}


def icl_index(si_frac_solid: float, si_interface_len_um_per_mm2: float | None, cfg: dict) -> dict:
    """Initial capacity loss proxy: SEI scales with Si surface; Si share of capacity."""
    f = si_frac_solid
    q_si, q_gr = cfg["q_si_pract"], cfg["q_gr"]
    rho_si, rho_gr = cfg["rho_si"], cfg["rho_gr"]
    m_si = f * rho_si
    m_gr = (1 - f) * rho_gr
    si_cap_share = m_si * q_si / (m_si * q_si + m_gr * q_gr)
    out = {"si_capacity_share": float(si_cap_share)}
    if si_interface_len_um_per_mm2 is not None and np.isfinite(si_interface_len_um_per_mm2):
        out["icl_proxy"] = float(si_cap_share * si_interface_len_um_per_mm2)
    else:
        out["icl_proxy"] = float(si_cap_share)
    return out


def energy_density(pore_frac: float, si_frac_solid: float, cfg: dict) -> dict:
    """Q_vol = (1 - eps) * sum phi_i rho_i q_i with VOLUME fractions (grade A)."""
    eps = pore_frac
    phi_si = si_frac_solid * (1 - cfg.get("cbd_frac_of_solids", 0.08))
    phi_gr = (1 - si_frac_solid) * (1 - cfg.get("cbd_frac_of_solids", 0.08))
    q_vol = (1 - eps) * (phi_si * cfg["rho_si"] * cfg["q_si_pract"] + phi_gr * cfg["rho_gr"] * cfg["q_gr"])
    return {"Q_vol_mAh_cm3": float(q_vol)}


def plating_risk_index(D_eff_rel: float, eps: float) -> dict:
    """Ionic-resistance ratio proxy: R_ion ~ 1/D_eff_rel (MacMullin)."""
    return {"R_ion_rel": float(1.0 / D_eff_rel) if np.isfinite(D_eff_rel) and D_eff_rel > 0 else float("nan")}


def cli_index(si_frac_solid, si_d50_um, buffer_sufficiency, agglomerate_frac, si_contact_pore_frac, w: dict) -> dict:
    """Cycle-life HEURISTIC (grade C): weighted sum of normalised risk terms.

    Each term is clipped to [0, 1]; higher = worse.  Labelled heuristic.
    """
    terms = {
        "f_si": np.clip(si_frac_solid / 0.20, 0, 1),
        "d50": np.clip(si_d50_um / 8.0, 0, 1),
        "buffer": np.clip(1 - (buffer_sufficiency if np.isfinite(buffer_sufficiency) else 0.5), 0, 1),
        "agglomerate": np.clip(agglomerate_frac if np.isfinite(agglomerate_frac) else 0, 0, 1),
        "contact": np.clip(si_contact_pore_frac if np.isfinite(si_contact_pore_frac) else 0, 0, 1),
    }
    score = float(sum(w[k] * terms[k] for k in terms))
    return {"cli_heuristic": score, **{f"cli_term_{k}": float(v) for k, v in terms.items()}}
