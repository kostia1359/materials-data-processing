"""Orchestrate the simulation layer for one processed sample (Section 10.3)."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from skimage.measure import perimeter

from . import CARBON, PORE, SI, UNCERTAIN, assumptions_hash
from .fuse import downsample_labels, fuse_phase_map
from .indices import (cli_index, energy_density, icl_index, plating_risk_index,
                      transport_proxies, wetting_index)
from .laplace import electronic_index, ionic_index
from .swell import swell

GRADES = {
    "D_eff_rel_TP": "B", "D_eff_rel_IP": "B", "aniso_ratio": "B", "tau_p": "B", "N_M": "B",
    "sigma_eff_rel_TP": "C", "t_d_s": "B", "i_lim_Am2": "B", "R_ion_rel": "B",
    "pore_closure_frac_1.6": "B", "constraint_index_2.43": "B", "buffer_sufficiency_frac_2.43": "B",
    "si_touch_frac_2.43": "B", "largest_merged_cluster_ecd_um_2.43": "B",
    "breakthrough_r_um": "B", "trapped_frac": "B", "lw_time_rel": "B",
    "icl_proxy": "B", "Q_vol_mAh_cm3_deep": "A", "Q_vol_mAh_cm3_offset": "A",
    "cli_heuristic": "C",
    "Q_CC_1C_over_Q_C10": "B", "Q_CC_3C_over_Q_C10": "B", "min_eta_sep_1C_V": "B", "min_eta_sep_3C_V": "B",
}
# direction in which a HIGHER value is worse for the cell (for report wording)
HIGHER_IS_WORSE = {"tau_p", "N_M", "t_d_s", "R_ion_rel", "pore_closure_frac_1.6", "constraint_index_2.43",
                   "si_touch_frac_2.43", "largest_merged_cluster_ecd_um_2.43", "trapped_frac", "lw_time_rel",
                   "icl_proxy", "cli_heuristic"}
REPORT_INDICES = ["D_eff_rel_TP", "aniso_ratio", "tau_p", "sigma_eff_rel_TP", "t_d_s", "i_lim_Am2",
                  "pore_closure_frac_1.6", "constraint_index_2.43", "buffer_sufficiency_frac_2.43",
                  "si_touch_frac_2.43", "breakthrough_r_um", "trapped_frac", "icl_proxy",
                  "Q_vol_mAh_cm3_deep", "R_ion_rel", "cli_heuristic",
                  "Q_CC_1C_over_Q_C10", "Q_CC_3C_over_Q_C10", "min_eta_sep_3C_V"]


def _eps_conventions(pore_frac: float, baseline_med_pore: float | None, ecfg: dict) -> dict:
    out = {"deep_pore_as_is": float(pore_frac)}
    ref = baseline_med_pore if baseline_med_pore is not None else pore_frac
    out["offset_baseline_to_0.30"] = float(np.clip(pore_frac + (0.30 - ref), 0.05, 0.6))
    return out


def run_sim(res: dict, cfg: dict, full: bool = False, echo=None) -> dict:
    """Heavy, baseline-independent part: fuse, Laplace, swelling, wetting, ICL, CLI.

    Returns {'row', 'bounds', 'audit', 'maps'}; ``finalize_sim`` adds the
    baseline-dependent porosity conventions, energy density and PyBaMM runs.
    ``full`` adds the D_c sweep and the isotropic swelling variant.
    """
    scfg = cfg["sim"]
    t0 = time.time()
    timings = {}
    kp = res["kpis"]
    ch = res["channels"]
    um_per_px = ch.px_nm / 1000.0

    fused = fuse_phase_map(ch, res["seg"], scfg["fuse"])
    timings["fuse"] = time.time() - t0
    f = scfg["downsample"]["factor"]
    maps = {}
    for name in ("L_mid", "L_solid", "L_pore"):
        maps[name], aud = downsample_labels(fused[name], f, scfg["downsample"]["max_percolating_fraction_change"])
        if name == "L_mid":
            audit = aud
    um_ds = um_per_px * f
    timings["downsample"] = time.time() - t0 - timings["fuse"]

    tr = scfg["transport"]
    solver = tr["solver"]
    row = {"sample_id": res["sample_id"], "batch": res["batch"],
           "uncertain_frac": fused["uncertain_frac"],
           "pore_frac_fused": float((maps["L_mid"] == PORE).mean()),
           "si_frac_fused": float((maps["L_mid"] == SI).mean()),
           "downsample_accepted": bool(audit["accepted"])}
    bounds = {}
    t1 = time.time()
    # the solves are independent; SuperLU releases the GIL so threads give ~3x on a 4-core CPU
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as ex:
        f_ion = ex.submit(ionic_index, maps["L_mid"], tr["D_c"], tr["sigma_si"], solver)
        f_b = {b: ex.submit(ionic_index, maps[b], tr["D_c"], tr["sigma_si"], solver, False)
               for b in ("L_solid", "L_pore")}
        f_sweep = {}
        if full:
            for dc in tr["D_c_sweep"]:
                for b in ("L_mid", "L_solid", "L_pore"):
                    f_sweep[(dc, b)] = ex.submit(ionic_index, maps[b], dc, tr["sigma_si"], solver, False)
        f_el = ex.submit(electronic_index, maps["L_mid"], tr["sigma_si"], solver)
        ion = f_ion.result()
        for k in ("D_eff_rel_TP", "D_eff_rel_IP", "aniso_ratio", "tau_p", "N_M", "flux_balance_TP", "flux_balance_IP"):
            row[k] = ion.get(k, float("nan"))
        row["pore_spans_TP"] = bool(ion["pore_percolation"]["spans"])
        row["pore_n_components"] = ion["pore_percolation"]["n_components"]
        undefined = []
        if ion["undefined_TP"]:
            undefined.append(f"ionic TP: {ion['reason_TP']}")
        for b, fb in f_b.items():
            ib = fb.result()
            bounds[f"D_eff_rel_TP_{b}"] = ib["D_eff_rel_TP"]
            bounds[f"tau_p_{b}"] = ib["tau_p"]
        for (dc, b), fs in f_sweep.items():
            bounds[f"D_eff_rel_TP_Dc{dc}_{b}"] = fs.result()["D_eff_rel_TP"]
        el = f_el.result()
    row["sigma_eff_rel_TP"] = el["sigma_eff_rel_TP"]
    row["carbon_spanning_frac"] = el["carbon_spanning_frac"]
    row["exposed_si_frac"] = el["exposed_si_frac"]
    if el["undefined"]:
        undefined.append(f"electronic TP: {el['reason']}")
    timings["laplace"] = time.time() - t1

    t2 = time.time()
    sw = scfg["swelling"]
    for fa in sw["f_A_scenarios"]:
        r = swell(maps["L_mid"], fa, "pore_first", um_ds, hotspot_um=sw["cluster_hotspot_um"])
        for k, v in r.items():
            if k not in ("labels", "constraint_map"):
                row[f"{k}_{fa}"] = v
        if fa == max(sw["f_A_scenarios"]):
            maps["L_swollen"] = r["labels"]
            maps["constraint_map"] = r.get("constraint_map")
            for b in (("L_solid", "L_pore") if full else ()):
                rb = swell(maps[b], fa, "pore_first", um_ds, hotspot_um=sw["cluster_hotspot_um"])
                for k in ("pore_closure_frac", "constraint_index", "buffer_sufficiency_frac"):
                    bounds[f"{k}_{fa}_{b}"] = rb[k]
            if full:
                ri = swell(maps["L_mid"], fa, "isotropic", um_ds, hotspot_um=sw["cluster_hotspot_um"])
                for k in ("pore_closure_frac", "constraint_index", "buffer_sufficiency_frac"):
                    row[f"{k}_{fa}_isotropic"] = ri[k]
    timings["swell"] = time.time() - t2

    t3 = time.time()
    ecfg, elcfg, encfg = scfg["electrode"], scfg["electrolyte"], scfg["energy"]
    row.update(transport_proxies(row["D_eff_rel_TP"], ecfg, elcfg))
    row.update(wetting_index(maps["L_mid"], um_ds, scfg["wetting"]))
    si_mask = maps["L_mid"] == SI
    si_perim_um = perimeter(si_mask, neighborhood=4) * um_ds
    area_mm2 = si_mask.size * um_ds ** 2 / 1e6
    row["si_interface_um_per_mm2"] = float(si_perim_um / area_mm2)
    row.update(icl_index(kp["si_frac_solid"], row["si_interface_um_per_mm2"] / 1e4, encfg))
    row.update(plating_risk_index(row["D_eff_rel_TP"], row["pore_frac_fused"]))
    fa_max = max(sw["f_A_scenarios"])
    row.update(cli_index(kp["si_frac_solid"], kp.get("si_d50_aw_um", float("nan")),
                         row.get(f"buffer_sufficiency_frac_{fa_max}", float("nan")),
                         kp.get("si_agglomerate_frac", float("nan")),
                         kp.get("si_contact_pore_frac", float("nan")), scfg["cli_weights"]))
    timings["indices"] = time.time() - t3

    row["undefined"] = "; ".join(undefined)
    row["assumptions_hash"] = assumptions_hash(scfg)
    row["sim_time_s"] = time.time() - t0
    row["kpi_si_frac_solid"] = kp["si_frac_solid"]
    row["kpi_pore_frac_deep"] = kp["pore_frac_deep"]
    row["kpi_si_d50_aw_um"] = kp.get("si_d50_aw_um", float("nan"))
    row["se_pore_frac_deep"] = res["kpi_ses"].get("pore_frac_deep_se")
    row["se_si_frac_solid"] = res["kpi_ses"].get("si_frac_solid_se")
    row["se_si_d50_aw_um"] = res["kpi_ses"].get("si_d50_aw_um_se")
    for k, v in timings.items():
        row[f"t_{k}_s"] = v
    if echo:
        echo(f"    sim {res['sample_id']}: " + ", ".join(f"{k} {v:.0f}s" for k, v in timings.items()))
    return {"row": row, "bounds": bounds, "audit": audit, "maps": maps,
            "fused": {"uncertain_frac": fused["uncertain_frac"]}}


def finalize_sim(sim: dict, cfg: dict, baseline_sim: dict | None = None, full: bool = False,
                 with_cell: bool | None = None) -> dict:
    """Baseline-dependent, cheap part: porosity conventions, Q_vol, PyBaMM point (+/-SE, bounds if ``full``).

    Mutates and returns ``sim`` (``row`` and ``bounds``).  Safe to re-run when the
    baseline changes; the heavy part stays cached.
    """
    scfg = cfg["sim"]
    row, bounds = sim["row"], sim["bounds"]
    ecfg, encfg, ccfg = scfg["electrode"], scfg["energy"], scfg["cell"]
    base_pore = None if baseline_sim is None else baseline_sim.get("baseline_med_pore_frac")
    eps_c = _eps_conventions(row["kpi_pore_frac_deep"], base_pore, ecfg)
    row["eps_deep"] = eps_c["deep_pore_as_is"]
    row["eps_offset"] = eps_c["offset_baseline_to_0.30"]
    en = dict(encfg, cbd_frac_of_solids=ecfg["cbd_frac_of_solids"])
    row["Q_vol_mAh_cm3_deep"] = energy_density(eps_c["deep_pore_as_is"], row["kpi_si_frac_solid"], en)["Q_vol_mAh_cm3"]
    row["Q_vol_mAh_cm3_offset"] = energy_density(eps_c["offset_baseline_to_0.30"], row["kpi_si_frac_solid"], en)["Q_vol_mAh_cm3"]
    undefined = [u for u in str(row.get("undefined", "")).split("; ") if u and not u.startswith("cell:")]
    if with_cell is None:
        with_cell = bool(ccfg.get("enabled", False))
    if with_cell and np.isfinite(row.get("tau_p", float("nan"))):
        t4 = time.time()
        from .cell import pybamm_runs
        tau_med = (baseline_sim or {}).get("baseline_med_tau_p") or row["tau_p"]
        tau_se = (abs(bounds["tau_p_L_pore"] - bounds["tau_p_L_solid"]) / 2
                  if np.isfinite(bounds.get("tau_p_L_pore", np.nan)) and np.isfinite(bounds.get("tau_p_L_solid", np.nan)) else None)
        ses = {"eps": row.get("se_pore_frac_deep"), "f_si": row.get("se_si_frac_solid"),
               "r_si": (row.get("se_si_d50_aw_um") or 0) / 2, "tau": tau_se}
        cell_bounds = {b: (eps_c["offset_baseline_to_0.30"], bounds[f"tau_p_{b}"]) for b in ("L_solid", "L_pore")
                       if np.isfinite(bounds.get(f"tau_p_{b}", np.nan))}
        try:
            cr = pybamm_runs(eps_c["offset_baseline_to_0.30"], row["kpi_si_frac_solid"], row["kpi_si_d50_aw_um"] / 2,
                             row["tau_p"], tau_med, ccfg, ecfg, bounds=cell_bounds if full else None,
                             ses=ses if full else None, rates=tuple(ccfg["rates"]))
            row.update(cr["point"])
            for name, r in cr.items():
                if name != "point":
                    for k, v in r.items():
                        bounds[f"{k}_{name}"] = v
        except Exception as e:
            undefined.append(f"cell: {str(e)[:160]}")
        row["t_cell_s"] = time.time() - t4
    row["undefined"] = "; ".join(undefined)
    return sim
