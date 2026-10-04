"""Batch assignment: shrunken centroids + softmax, stability, novelty,
signatures (Section 5.3/5.4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .kpis import ROBUSTNESS, SHORTLIST
from .stats import z_scores

WEIGHT = {"R": 1.0, "M": 0.7, "W": 0.4}


def feature_vector(row: dict, stats: dict) -> np.ndarray:
    """Shortlist KPIs -> baseline-z -> robustness-weighted."""
    z = z_scores(row, stats, SHORTLIST)
    return np.array([
        z.get(k, 0.0) * WEIGHT[ROBUSTNESS.get(k, "M")] for k in SHORTLIST
    ])


def feature_matrix(df: pd.DataFrame, stats: dict) -> np.ndarray:
    return np.vstack([feature_vector(r, stats) for r in df.to_dict("records")])


def soft_threshold(x: np.ndarray, delta: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - delta, 0.0)


def fit_centroids(df: pd.DataFrame, stats: dict, delta: float) -> dict:
    """Per-batch shrunken centroids. Returns {batch: vector} plus grand median."""
    X = feature_matrix(df, stats)
    grand = np.median(X, axis=0)
    centroids = {}
    for b in sorted(df["batch"].astype(str).unique()):
        Xb = X[df["batch"].astype(str) == b]
        c = np.median(Xb, axis=0)
        centroids[b] = grand + soft_threshold(c - grand, delta)
    return {"centroids": centroids, "grand": grand}


def distances(x: np.ndarray, centroids: dict) -> dict:
    return {b: float(np.sqrt(((x - c) ** 2).sum())) for b, c in centroids.items()}


def probs_from_distances(d: dict, T: float) -> dict:
    keys = sorted(d)
    e = np.array([np.exp(-d[b] ** 2 / (2 * T)) for b in keys])
    e = e / e.sum()
    return {b: float(v) for b, v in zip(keys, e)}


def assign_sample(
    row: dict,
    stats: dict,
    centroids: dict,
    T: float,
    strip_rows: list[dict] | None = None,
    n_boot: int = 200,
    seed: int = 0,
    id_percentiles: dict | None = None,
) -> dict:
    """Assign one sample to a batch. Always bets."""
    x = feature_vector(row, stats)
    d = distances(x, centroids["centroids"])
    p = probs_from_distances(d, T)
    assigned = max(p, key=p.get)

    # stability: strip-level re-assignment + bootstrap
    stability = float("nan")
    if strip_rows:
        rng = np.random.default_rng(seed)
        agree = 0
        total = 0
        strip_calls = []
        for sr in strip_rows:
            xs = feature_vector(sr, stats)
            ds = distances(xs, centroids["centroids"])
            ps = probs_from_distances(ds, T)
            strip_calls.append(max(ps, key=ps.get))
        n = len(strip_calls)
        for _ in range(n_boot):
            calls = [strip_calls[i] for i in rng.integers(0, n, n)]
            vote = max(set(calls), key=calls.count)
            agree += int(vote == assigned)
            total += 1
        stability = agree / total if total else float("nan")
        # strip-level agreement fraction (strip consistency check, 6.3)
        strip_agree = sum(1 for c in strip_calls if c == assigned) / n

    top_p = p[assigned]
    if top_p >= 0.75 and (not np.isfinite(stability) or stability >= 0.8):
        conf = "confident"
    elif top_p >= 0.5:
        conf = "moderately confident"
    else:
        conf = f"low confidence — betting on batch {assigned}"

    # novelty: distance to nearest centroid vs LOIO within-batch distribution
    min_d = min(d.values())
    id_score = float("nan")
    novelty = False
    if id_percentiles:
        # F = empirical CDF of within-batch distances; id_score = 1 - F(min_d)
        cdf_pts = np.asarray(id_percentiles["distances"])
        id_score = float(1 - np.searchsorted(np.sort(cdf_pts), min_d) / max(len(cdf_pts), 1))
        novelty = id_score < 0.025

    return {
        "assigned_batch": int(assigned),
        "probabilities": {str(k): round(v, 4) for k, v in p.items()},
        "distances": {str(k): round(v, 3) for k, v in d.items()},
        "stability": round(stability, 3) if np.isfinite(stability) else None,
        "strip_agree": round(strip_agree, 3) if strip_rows else None,
        "confidence_label": conf,
        "novelty_flag": bool(novelty),
        "in_distribution_score": round(id_score, 4) if np.isfinite(id_score) else None,
        "priors": "uniform",
    }


def choose_hyperparams(
    df: pd.DataFrame, stats: dict, shrinkage_grid: list, temperature_grid: list
) -> tuple[float, float, dict]:
    """Pick (Delta, T) by inner LOIO NLL on the labelled set given."""
    best = (None, None, np.inf)
    batches = sorted(df["batch"].astype(str).unique())
    for delta in shrinkage_grid:
        for T in temperature_grid:
            nll = 0.0
            n = 0
            for i in range(len(df)):
                rest = df.drop(df.index[i])
                cen = fit_centroids(rest, stats, delta)
                x = feature_vector(df.iloc[i].to_dict(), stats)
                d = distances(x, cen["centroids"])
                p = probs_from_distances(d, T)
                true = str(df.iloc[i]["batch"])
                nll += -np.log(max(p.get(true, 1e-9), 1e-9))
                n += 1
            if nll < best[2]:
                best = (delta, T, nll)
    return best[0], best[1], {"nll": best[2]}


def within_batch_distances(df: pd.DataFrame, stats: dict, delta: float) -> list:
    """LOIO distance of each image to its own batch centroid (for novelty CDF)."""
    out = []
    for i in range(len(df)):
        rest = df.drop(df.index[i])
        cen = fit_centroids(rest, stats, delta)
        x = feature_vector(df.iloc[i].to_dict(), stats)
        true = str(df.iloc[i]["batch"])
        out.append(distances(x, cen["centroids"])[true])
    return out


def batch_signatures(df: pd.DataFrame, stats: dict, baseline_batch: str = "3") -> dict:
    """effect_bk = (median_b,k - med_k)/scale_k with LOIO sign stability."""
    sig = {}
    nonbase = [b for b in df["batch"].astype(str).unique() if b != baseline_batch]
    for b in sorted(nonbase):
        sub = df[df["batch"].astype(str) == b]
        effs = {}
        for k in SHORTLIST:
            st = stats["kpis"].get(k)
            if st is None or st["scale"] == 0:
                continue
            med_b = float(np.median(sub[k].astype(float)))
            eff = (med_b - st["med"]) / st["scale"]
            # LOIO sign stability within batch b
            stable = 0
            for i in range(len(sub)):
                rest = sub.drop(sub.index[i])
                med_r = float(np.median(rest[k].astype(float)))
                eff_r = (med_r - st["med"]) / st["scale"]
                if np.sign(eff_r) == np.sign(eff) and abs(eff_r) > 1:
                    stable += 1
            stab_frac = stable / len(sub)
            effs[k] = {
                "effect": float(eff),
                "sign": int(np.sign(eff)),
                "loo_stability": float(stab_frac),
                "stable": bool(stab_frac >= 0.6 and abs(eff) >= 1.5),
            }
        sig[b] = effs
    return sig


def signature_match(z: dict, signatures: dict) -> dict:
    """signature_match_b = sum w_k sign(effect) z_k / sum w_k over stable KPIs."""
    out = {}
    for b, effs in signatures.items():
        num = den = 0.0
        matched, missed = [], []
        for k, e in effs.items():
            if not e["stable"]:
                continue
            w = WEIGHT[ROBUSTNESS.get(k, "M")]
            zk = z.get(k, 0.0)
            num += w * e["sign"] * zk
            den += w
            if np.sign(zk) == e["sign"] and abs(zk) > 1:
                matched.append(k)
            else:
                missed.append(k)
        out[b] = {
            "score": float(num / den) if den else 0.0,
            "matched": matched,
            "missed": missed,
        }
    return out
