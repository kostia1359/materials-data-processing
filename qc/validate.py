"""Self-verification: LOIO, shadow classifier, sensitivity tests -> VALIDATION.md."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .assign import (
    WEIGHT, assign_sample, choose_hyperparams, distances, feature_matrix,
    feature_vector, fit_centroids, probs_from_distances,
)
from .io import discover
from .kpis import ROBUSTNESS, SHORTLIST
from .pipeline import process_sample
from .stats import compute_baseline_stats, verdict, z_scores

GATE_FEATURES = [
    "inlens_c_med", "etd_c_med", "comb_step_bse", "n_grey_levels_bse",
    "noise_sd_bse", "height_px", "blur_bse",
]


def _nearest_centroid_predict(X: np.ndarray, y: np.ndarray, i: int) -> str:
    """LOIO nearest-centroid call for row i of X."""
    mask = np.ones(len(X), dtype=bool)
    mask[i] = False
    cents = {b: np.median(X[mask & (y == b)], axis=0) for b in np.unique(y)}
    d = {b: float(((X[i] - c) ** 2).sum()) for b, c in cents.items()}
    return min(d, key=d.get)


def loio_eval(kpis_df, strips_df, stats, cfg, gates_df=None) -> dict:
    """Leave-one-image-out over labelled images.

    For image i: centroids from all images != i; delta/T chosen by inner LOIO
    on the remaining images; baseline stats from batch-3 != i.
    """
    df = kpis_df[kpis_df["batch"].astype(str).isin(["1", "2", "3"])].reset_index(drop=True)
    records = []
    conf = np.zeros((3, 3))
    nll = 0.0
    brier = 0.0
    idx_of = {"1": 0, "2": 1, "3": 2}
    for i in range(len(df)):
        rest = df.drop(df.index[i])
        rest_base = rest[rest["batch"].astype(str) == "3"]
        # rebuild baseline stats without image i (only matters when i is batch 3)
        rest_gates = (
            gates_df[gates_df["sample_id"].isin(rest_base["sample_id"])]
            if gates_df is not None and len(gates_df) else rest_base
        )
        st = compute_baseline_stats(rest_base, rest_gates)
        delta, T, _ = choose_hyperparams(
            rest, st, cfg["shrinkage_grid"], cfg["temperature_grid"])
        cen = fit_centroids(rest, st, delta)
        row = df.iloc[i].to_dict()
        x = feature_vector(row, st)
        d = distances(x, cen["centroids"])
        p = probs_from_distances(d, T)
        assigned = max(p, key=p.get)
        true = str(row["batch"])
        vv = verdict(row, st, cfg)
        z = vv["z"]
        top3 = sorted(z, key=lambda k: -abs(z[k]))[:3]
        records.append({
            "sample_id": row["sample_id"], "true": true, "assigned": assigned,
            "p1": p.get("1", 0), "p2": p.get("2", 0), "p3": p.get("3", 0),
            "verdict": vv["decision"], "top3": ",".join(top3),
            "S": vv["aggregate_score_S"],
        })
        conf[idx_of[true], idx_of[assigned]] += 1
        nll += -np.log(max(p.get(true, 1e-9), 1e-9))
        brier += sum((p.get(b, 0) - (1.0 if b == true else 0.0)) ** 2
                     for b in ("1", "2", "3"))
    rec_df = pd.DataFrame(records)
    acc = float((rec_df["true"].astype(str) == rec_df["assigned"].astype(str)).mean())
    fa = int(((rec_df["true"] == "3") & (rec_df["verdict"] != "ACCEPT")).sum())
    det = int(((rec_df["true"] != "3") & (rec_df["verdict"] != "ACCEPT")).sum())
    return {
        "confusion": conf, "accuracy": acc, "mean_nll": nll / len(df),
        "mean_brier": brier / len(df), "table": rec_df,
        "false_alarms": fa, "detections": det,
    }


def per_kpi_effects(kpis_df, stats) -> pd.DataFrame:
    """|median_b - med_base|/scale for each batch pair and KPI."""
    rows = []
    for b in ("1", "2"):
        sub = kpis_df[kpis_df["batch"].astype(str) == b]
        for k in SHORTLIST:
            st = stats["kpis"].get(k)
            if not st or not st.get("scale"):
                continue
            med_b = float(np.median(sub[k].astype(float)))
            rows.append({"batch": b, "kpi": k,
                         "effect": (med_b - st["med"]) / st["scale"],
                         "robustness": st["robustness"]})
    return pd.DataFrame(rows)


def shadow_classifier(gates_df) -> dict:
    """Nearest-centroid LOIO on gate covariates only (confound check)."""
    df = gates_df[gates_df["batch"].astype(str).isin(["1", "2", "3"])].reset_index(drop=True)
    feats = [c for c in GATE_FEATURES if c in df.columns]
    X = df[feats].astype(float).values
    # robust standardise
    med = np.nanmedian(X, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(X - med), axis=0)
    mad[mad == 0] = 1.0
    X = np.where(np.isfinite(X), (X - med) / mad, 0.0)
    y = df["batch"].astype(str).values
    preds = [_nearest_centroid_predict(X, y, i) for i in range(len(X))]
    acc = float(np.mean([p == t for p, t in zip(preds, y)]))
    return {"accuracy": acc, "preds": preds, "true": list(y), "features": feats}


def _sens_run(sample, cfg, crack_threshold, kw):
    try:
        r = process_sample(sample, cfg, crack_threshold=crack_threshold, **kw)
        return r["kpis"]
    except Exception as e:
        return {"_error": str(e)}


def sensitivity(df_samples, cfg, stats, out: Path, n_proc: int = 5) -> dict:
    """Recompute KPIs with thresholds +-5 and 2x downsample; deltas in scale units."""
    from concurrent.futures import ProcessPoolExecutor

    crack_thr = stats.get("crack_threshold")
    rows = []
    with ProcessPoolExecutor(n_proc) as ex:
        futs = {}
        for s in df_samples:
            for tag, kw in (("t+5", {"threshold_shift": 5}),
                            ("t-5", {"threshold_shift": -5}),
                            ("ds2", {"downsample": 2})):
                futs[(s.sample_id, tag)] = ex.submit(
                    _sens_run, s, cfg, crack_thr, kw)
        for (sid, tag), f in futs.items():
            res = f.result()
            for k in SHORTLIST:
                v = res.get(k)
                if v is not None and np.isfinite(v):
                    rows.append({"sample_id": sid, "variant": tag, "kpi": k, "value": v})
    return pd.DataFrame(rows)


def run_validation(data: Path, out: Path, cfg: dict):
    """Full validation -> VALIDATION.md."""
    kpis_df = pd.read_csv(out / "kpis.csv")
    strips_df = pd.read_csv(out / "strips.csv") if (out / "strips.csv").exists() else pd.DataFrame()
    gates_df = pd.read_csv(out / "gates.csv") if (out / "gates.csv").exists() else pd.DataFrame()
    stats = json.loads((out / "baseline_stats.json").read_text())

    t0 = time.time()
    loio = loio_eval(kpis_df, strips_df, stats, cfg, gates_df)
    shadow = shadow_classifier(gates_df) if len(gates_df) else {"accuracy": float("nan")}
    effects = per_kpi_effects(kpis_df, stats)

    samples = [s for s in discover(data) if s.batch in ("1", "2", "3")]
    sens = sensitivity(samples, cfg, stats, out)
    if len(sens):
        base = sens[sens["variant"] == "t+5"]  # placeholder index
        # delta in scale units vs the nominal kpis.csv values
        nom = kpis_df.set_index("sample_id")
        sens["nominal"] = sens.apply(
            lambda r: nom.loc[r["sample_id"], r["kpi"]]
            if r["sample_id"] in nom.index else np.nan, axis=1)
        sens["delta_scale"] = sens.apply(
            lambda r: (r["value"] - r["nominal"]) / stats["kpis"][r["kpi"]]["scale"]
            if np.isfinite(r["nominal"]) else np.nan, axis=1)
        sens.to_csv(out / "sensitivity.csv", index=False)

    md = ["# Validation", "",
          f"Run: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} "
          f"({time.time()-t0:.0f}s)", "",
          "## Leave-one-image-out (LOIO)", "",
          f"- n = {len(loio['table'])}, accuracy = {loio['accuracy']:.3f}, "
          f"mean NLL = {loio['mean_nll']:.3f}, mean Brier = {loio['mean_brier']:.3f}",
          f"- false alarms (batch-3 held out -> INVESTIGATE/REJECT): "
          f"{loio['false_alarms']}",
          f"- detections (batch-1/2 held out -> INVESTIGATE/REJECT): "
          f"{loio['detections']}", "",
          "Confusion matrix (rows=true, cols=assigned, order 1,2,3):", "",
          "```", str(loio["confusion"]), "```", "",
          loio["table"].to_markdown(index=False), "",
          "## Per-KPI effect sizes (|delta| in baseline scale units)", "",
          effects.sort_values("effect", key=abs, ascending=False)
          .to_markdown(index=False), "",
          "## Shadow classifier (gate covariates only)", "",
          f"LOIO accuracy on acquisition covariates: **{shadow['accuracy']:.3f}**",
          "",
          "If this is comparable to the material-KPI classifier, batches may be "
          "separable by acquisition session rather than material.", "",
          "## Sensitivity", ""]
    if len(sens):
        summ = (sens.groupby(["variant", "kpi"])["delta_scale"]
                .agg(lambda x: float(np.nanmean(np.abs(x))))
                .reset_index().rename(columns={"delta_scale": "mean_abs_delta_scale"}))
        r_bad = sens[(sens["kpi"].map(ROBUSTNESS) == "R")
                     & (sens["delta_scale"].abs() > 0.5)]
        md += [summ.to_markdown(index=False), "",
               f"R-class KPIs moving > 0.5 scale under perturbation: "
               f"{len(r_bad)} cases", ""]
    with open("VALIDATION.md", "w") as f:
        f.write("\n".join(md))
