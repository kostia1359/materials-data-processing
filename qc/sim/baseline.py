"""Baseline statistics and per-sample relative block for the simulation layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .run import GRADES, HIGHER_IS_WORSE, REPORT_INDICES

RANK_CHECK_INDEX = "D_eff_rel_TP"


def _num(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype.kind in "fi" and c not in ("sample_id", "batch")]


def sim_baseline(sim_df: pd.DataFrame, bounds_df: pd.DataFrame, cfg: dict, phantoms: dict | None = None,
                 batch: str = "3") -> dict:
    """Medians/MAD on the baseline batch, D_c sweep, bound spread as empirical REV, rank robustness."""
    scfg = cfg["sim"]
    base = sim_df[sim_df["batch"].astype(str) == batch]
    bb = bounds_df[bounds_df["sample_id"].isin(base["sample_id"])] if bounds_df is not None else None
    out = {"n": int(len(base)), "batch": batch, "indices": {}, "grades": GRADES,
           "higher_is_worse": sorted(HIGHER_IS_WORSE)}
    for k in _num(base):
        v = base[k].astype(float).to_numpy()
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        med = float(np.median(v))
        mad = float(1.4826 * np.median(np.abs(v - med)))
        out["indices"][k] = {"med": med, "mad": mad, "n": int(v.size),
                             "min": float(v.min()), "max": float(v.max())}
    out["baseline_med_pore_frac"] = out["indices"].get("eps_deep", {}).get("med")
    out["baseline_med_tau_p"] = out["indices"].get("tau_p", {}).get("med")

    # empirical REV from the L_solid/L_pore bound spread (relative half-width)
    rev = {}
    if bb is not None and len(bb):
        for k in ("D_eff_rel_TP", "tau_p", "pore_closure_frac_2.43", "constraint_index_2.43"):
            lo, hi = f"{k}_L_solid", f"{k}_L_pore"
            if lo in bb.columns and hi in bb.columns and k in base.columns:
                m = base.set_index("sample_id")[k].astype(float)
                b = bb.set_index("sample_id")
                rel = (np.abs(b[hi].astype(float) - b[lo].astype(float)) / 2 / m.reindex(b.index).abs()).replace([np.inf], np.nan)
                rev[k] = float(np.nanmedian(rel))
    out["bound_half_width_rel_median"] = rev

    # D_c sweep (build-baseline only): medians of each sweep column
    sweep = {}
    if bb is not None:
        for dc in scfg["transport"]["D_c_sweep"]:
            for lab in ("L_mid", "L_solid", "L_pore"):
                c = f"D_eff_rel_TP_Dc{dc}_{lab}"
                if c in bb.columns and np.isfinite(bb[c].astype(float)).any():
                    sweep[c] = {"med": float(np.nanmedian(bb[c].astype(float))),
                                "p10": float(np.nanpercentile(bb[c].astype(float), 10)),
                                "p90": float(np.nanpercentile(bb[c].astype(float), 90))}
    out["D_c_sweep"] = sweep

    # rank robustness: Spearman of sample ranking across D_c values and porosity conventions
    rank = {}
    if bb is not None and len(bb) >= 4:
        b = bb.set_index("sample_id")
        cols = [f"D_eff_rel_TP_Dc{dc}_L_mid" for dc in scfg["transport"]["D_c_sweep"]]
        cols = [c for c in cols if c in b.columns]
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                x, y = b[cols[i]].astype(float), b[cols[j]].astype(float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() >= 4:
                    rank[f"{cols[i]} vs {cols[j]}"] = float(spearmanr(x[ok], y[ok]).statistic)
    s = sim_df.set_index("sample_id")
    if "Q_vol_mAh_cm3_deep" in s.columns and "Q_vol_mAh_cm3_offset" in s.columns:
        x, y = s["Q_vol_mAh_cm3_deep"].astype(float), s["Q_vol_mAh_cm3_offset"].astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 4:
            rank["Q_vol deep vs offset"] = float(spearmanr(x[ok], y[ok]).statistic)
    out["rank_robustness_spearman"] = rank
    out["unstable_indices"] = [k for k, r in rank.items() if r < 0.8]
    if phantoms:
        out["phantoms"] = phantoms
    return out


def sim_block(row: dict, bounds: dict, base: dict | None, assumptions_hash: str) -> dict:
    """Verdict JSON ``simulation`` block."""
    rel, bnd, grade = {}, {}, {}
    for k in REPORT_INDICES:
        v = row.get(k)
        if v is None:
            continue
        v = float(v)
        grade[k] = GRADES.get(k, "B")
        e = {"value": v}
        if base and k in base.get("indices", {}):
            med = base["indices"][k]["med"]
            e["baseline_med"] = med
            e["ratio"] = float(v / med) if med not in (0, None) and np.isfinite(v) else None
            mad = base["indices"][k]["mad"]
            e["z_mad"] = float((v - med) / mad) if mad and mad > 0 and np.isfinite(v) else None
            e["worse_than_baseline"] = (None if e["ratio"] is None else
                                        (v > med) if k in HIGHER_IS_WORSE else (v < med))
        rel[k] = e
        lo, hi = bounds.get(f"{k}_L_solid"), bounds.get(f"{k}_L_pore")
        if lo is not None or hi is not None:
            bnd[k] = {"L_solid": lo, "L_pore": hi}
    unstable = (base or {}).get("unstable_indices", [])
    return {"grade": grade, "relative_to_baseline": rel, "bounds": bnd,
            "undefined": [u for u in str(row.get("undefined", "")).split("; ") if u],
            "unstable_indices": unstable, "assumptions_hash": assumptions_hash,
            "downsample_accepted": row.get("downsample_accepted"),
            "uncertain_frac": row.get("uncertain_frac")}
