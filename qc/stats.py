"""Baseline statistics, robust z-scores, verdict and conformal rank."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .kpis import MEANINGS, ROBUSTNESS, SHORTLIST

BASELINE_BATCH = "3"


def baseline_version(sample_ids: list[str]) -> str:
    return hashlib.sha1("\n".join(sorted(sample_ids)).encode()).hexdigest()[:6]


def compute_baseline_stats(
    kpis_df: pd.DataFrame, gates_df: pd.DataFrame, batch: str = BASELINE_BATCH
) -> dict:
    """Per-KPI and per-gate robust stats over baseline (batch 3) rows."""
    base = kpis_df[kpis_df["batch"].astype(str) == batch]
    stats = {"n": int(len(base)), "batch": batch, "kpis": {}, "gates": {}}
    for col in kpis_df.columns:
        if col in ("sample_id", "batch") or col.endswith("_se"):
            continue
        vals = base[col].astype(float).values
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        med = float(np.median(vals))
        mad = float(1.4826 * np.median(np.abs(vals - med)))
        se_col = f"{col}_se"
        if se_col in kpis_df.columns:
            ses = base[se_col].astype(float).values
            within = float(np.median(ses[np.isfinite(ses)])) if np.isfinite(ses).any() else 0.0
        else:
            within = 0.0
        scale = max(mad, within, 0.05 * abs(med))
        stats["kpis"][col] = {
            "n": int(vals.size), "med": med, "mad": mad,
            "within_se": within, "scale": float(scale),
            "robustness": ROBUSTNESS.get(col, "M"),
        }
    for col in gates_df.columns:
        if col in ("sample_id", "batch"):
            continue
        vals = base[col].astype(float).values if col in base.columns else np.array([])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        med = float(np.median(vals))
        mad = float(1.4826 * np.median(np.abs(vals - med)))
        stats["gates"][col] = {"n": int(vals.size), "med": med,
                               "scale": float(max(mad, 0.05 * abs(med), 1e-9))}
    return stats


def z_scores(row: dict, stats: dict, keys: list[str] | None = None) -> dict:
    out = {}
    for k, st in stats["kpis"].items():
        if keys and k not in keys:
            continue
        v = row.get(k)
        if v is None or not np.isfinite(v):
            continue
        out[k] = float((v - st["med"]) / st["scale"]) if st["scale"] else 0.0
    return out


def gate_z_scores(gates_row: dict, stats: dict) -> dict:
    out = {}
    for k, st in stats["gates"].items():
        v = gates_row.get(k)
        if v is None or not np.isfinite(v):
            continue
        out[k] = float((v - st["med"]) / st["scale"]) if st["scale"] else 0.0
    return out


def verdict(row: dict, stats: dict, cfg: dict) -> dict:
    """ACCEPT/INVESTIGATE/REJECT vs baseline + conformal rank."""
    z = z_scores(row, stats, SHORTLIST)
    zones = cfg["zones"]
    zi, zr = zones["investigate"], zones["reject"]
    abs_z = {k: abs(v) for k, v in z.items()}
    n_beyond_multi = sum(1 for v in abs_z.values() if v > zones["multi_count_z"])
    any_reject = any(v > zr for v in abs_z.values())
    defect_breach = False
    for dk in ("crack_density_um_per_mm2", "largest_void_ecd_um",
               "hiz_inclusion_count_mm2"):
        # defect KPI: exceeds baseline max by >3 Poisson SD
        base_max = stats.get("baseline_max", {}).get(dk)
        if base_max is None or base_max == 0:
            continue
        v = row.get(dk)
        if v is not None and np.isfinite(v):
            sd = np.sqrt(base_max + 1)
            if v > base_max + 3 * sd:
                defect_breach = True
    if any_reject or n_beyond_multi >= zones["multi_count_n"] or defect_breach:
        decision = "REJECT"
    elif any(zi < v <= zr for v in abs_z.values()) or sum(
        1 for v in abs_z.values() if v > 2
    ) >= 2:
        decision = "INVESTIGATE"
    else:
        decision = "ACCEPT"

    top3 = sorted(abs_z.values(), reverse=True)[:3]
    S = float(np.mean(top3)) if top3 else 0.0
    drivers = [
        {
            "kpi": k,
            "value": row.get(k),
            "baseline_median": stats["kpis"][k]["med"],
            "baseline_scale": stats["kpis"][k]["scale"],
            "z": z[k],
            "direction": "higher" if z[k] > 0 else "lower",
            "robustness": stats["kpis"][k]["robustness"],
            "meaning": MEANINGS.get(k, ""),
        }
        for k in sorted(z, key=lambda k: -abs_z[k])
    ]
    return {
        "decision": decision,
        "aggregate_score_S": S,
        "n_kpis_beyond_2_5": n_beyond_multi,
        "z": z,
        "drivers": drivers,
        "gate_flags": [],
    }


def conformal_rank(S_new: float, S_baseline: list[float]) -> tuple[float, float]:
    """p_rank and the 1/(n+1) floor."""
    n = len(S_baseline)
    p = (1 + sum(1 for s in S_baseline if s >= S_new)) / (n + 1)
    return float(p), float(1 / (n + 1))


def load_baseline(out_dir: str | Path) -> dict:
    with open(Path(out_dir) / "baseline_stats.json") as f:
        return json.load(f)


def save_baseline(stats: dict, out_dir: str | Path):
    with open(Path(out_dir) / "baseline_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
