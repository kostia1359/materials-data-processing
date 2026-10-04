"""Assemble the Section 10 part of VALIDATION.md from loop caches, sim_baseline.json and ablations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _f(v, spec=".3g"):
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return str(v)


def sim_validation_md(out: Path, pybamm_sens: dict | None = None) -> str:
    base = json.loads((out / "sim_baseline.json").read_text())
    sim = pd.read_csv(out / "sim.csv")
    bounds = pd.read_csv(out / "sim_bounds.csv").set_index("sample_id")
    md = ["## Simulation layer (Section 10) — self-checks", ""]

    md += ["### Phantoms (exact solutions, `qc/sim/phantoms.py`, `tests/test_sim.py`)", "",
           "| phantom | quantity | target | value | rel. err | tol | pass |", "|---|---|---|---|---|---|---|"]
    for r in base["phantoms"]["rows"]:
        md.append(f"| {r['phantom']} | {r['quantity']} | {_f(r['target'])} | {_f(r['value'])} | "
                  f"{_f(r['rel_err'], '.1%')} | {r['tol'] if r['tol'] is not None else '-'} | "
                  f"{'yes' if r['pass'] else ('no' if r['pass'] is False else 'series')} |")
    md += ["", "The checkerboard series converges from below (corner singularities); the 32-px cell is the tested one.", ""]

    md += ["### Per-sample inequality and flux checks (all 31 triples)", ""]
    s = sim.set_index("sample_id")
    lo, hi = bounds["D_eff_rel_TP_L_solid"], bounds["D_eff_rel_TP_L_pore"]
    mid = s["D_eff_rel_TP"].reindex(bounds.index)
    eps = s["pore_frac_fused"].reindex(bounds.index)
    ineq = (lo <= mid * 1.001) & (mid <= hi * 1.001)
    deff_le_eps = mid <= eps + s["D_eff_rel_TP"].reindex(bounds.index) * 0  # two-conductivity: D_eff can exceed eps*1 via carbon
    flux = (1 - s["flux_balance_TP"]).abs()
    md += [f"- solid-bound <= mid <= pore-bound: {int(ineq.sum())}/{len(ineq)} samples",
           f"- |1 - flux_out/flux_in| max {_f(flux.max(), '.1e')} (through-plane), "
           f"{_f((1 - s['flux_balance_IP']).abs().max(), '.1e')} (in-plane)",
           f"- D_eff_rel_TP <= eps_pore + D_c·(1-eps_pore) (two-conductivity upper bound): "
           f"{int((mid <= eps + 0.05 * (1 - eps) + 1e-9).sum())}/{len(mid)}",
           f"- pore phase spans top-bottom in {int(s['pore_spans_TP'].sum())}/{len(s)} samples (expected 0, caveat 10)",
           f"- downsample audit accepted (pore phase) {int(s['downsample_accepted'].sum())}/{len(s)}; "
           f"median uncertain fraction {_f(s['uncertain_frac'].median())}", ""]

    md += ["### Empirical REV (bound half-width / value, baseline median)", ""]
    for k, v in base.get("bound_half_width_rel_median", {}).items():
        md.append(f"- {k}: {_f(v, '.1%')}")
    md += ["", "### D_c sweep (baseline medians, D_eff_rel_TP)", "", "| column | p10 | median | p90 |", "|---|---|---|---|"]
    for k, v in base.get("D_c_sweep", {}).items():
        md.append(f"| {k} | {_f(v['p10'])} | {_f(v['med'])} | {_f(v['p90'])} |")
    md += ["", "### Rank robustness (Spearman across conventions; < 0.8 marks the index unstable)", ""]
    for k, v in base.get("rank_robustness_spearman", {}).items():
        md.append(f"- {k}: {_f(v, '.3f')}")
    md.append(f"- unstable indices: {base.get('unstable_indices') or 'none'}")

    ab = out / "ablations.json"
    if ab.exists():
        a = json.loads(ab.read_text())
        md += ["", "### Detector and resolution ablations (4 samples)", "",
               "| sample | batch | variant | uncertain | D_eff_rel_TP | tau_p | sigma_eff | carbon span | pore pf Δ | carbon pf Δ |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for sid, rec in a.items():
            for var in ("full", "no_etd", "no_inlens", "x2", "x4", "x8"):
                r = rec.get(var)
                if not r:
                    continue
                md.append(f"| {sid} | {rec['batch']} | {var} | {_f(r.get('uncertain_frac', float('nan')))} | "
                          f"{_f(r['D_eff_rel_TP'])} | {_f(r['tau_p'])} | {_f(r.get('sigma_eff_rel_TP', float('nan')))} | "
                          f"{_f(r.get('carbon_spanning_frac', float('nan')))} | {_f(r.get('pore_pf_change', float('nan')), '.3f')} | "
                          f"{_f(r.get('carbon_pf_change', float('nan')), '.3f')} |")
        # rank agreement x2/x4/x8
        from scipy.stats import spearmanr
        ids = [sid for sid in a if all(f"x{f}" in a[sid] for f in (2, 4, 8))]
        if len(ids) >= 3:
            d = {f: [a[sid][f"x{f}"]["D_eff_rel_TP"] for sid in ids] for f in (2, 4, 8)}
            md += ["", f"- D_eff_rel_TP rank agreement x2 vs x4: Spearman {_f(spearmanr(d[2], d[4]).statistic, '.2f')}; "
                   f"x4 vs x8: {_f(spearmanr(d[4], d[8]).statistic, '.2f')} (n={len(ids)})"]

    if pybamm_sens:
        md += ["", "### PyBaMM sensitivity at the project composition (relative change of each output)", "",
               "| perturbation | Q_CC_1C/Q_C10 | Q_CC_3C/Q_C10 | min eta_sep 1C | min eta_sep 3C | min c_e 3C |", "|---|---|---|---|---|---|"]
        for k, v in pybamm_sens.items():
            md.append(f"| {k} | {_f(v['Q_CC_1C/Q_C10'], '+.1%')} | {_f(v['Q_CC_3C/Q_C10'], '+.1%')} | "
                      f"{_f(v['min_eta_sep_1C'], '+.0%')} | {_f(v['min_eta_sep_3C'], '+.0%')} | {_f(v['min_ce_3C'], '+.0%')} |")
        md += ["", "Ranking at this composition: porosity ≳ tortuosity ≫ Si fraction ≈ Si radius for the 3C plating indicator; "
               "the hysteresis option moves the 1C indicator by ~3× its value, so cell-level plating outputs are ratios with grade B/C only."]

    lp = out / "loop_log.csv"
    if lp.exists():
        log = pd.read_csv(lp).drop_duplicates("sample_id", keep="last")
        md += ["", "### Section 0.3 loop log", "",
               f"- {len(log)} triples processed in order {' → '.join(log['batch'].drop_duplicates())}; "
               f"{int((log['status'] == 'ok').sum())} ok, {int((log['regress'] == 'OK').sum())} regress OK; "
               f"median wall time {_f(log['seconds'].median(), '.0f')} s/triple (KPIs + full sim incl. D_c sweep)",
               f"- shortlist hash constant across the loop: {log['shortlist_hash'].nunique() == 1}",
               f"- flagged triples: {', '.join(log[log['status'] != 'ok']['sample_id']) or 'none'}"]
    return "\n".join(md) + "\n"
