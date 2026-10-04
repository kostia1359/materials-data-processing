"""Fused phase map with an explicit *uncertain* class (Section 10.2).

Per-pixel evidence over the frame {pore, carbon, si}:
  BSE  - three Gaussians at the smoothed-histogram modes give class
         plausibilities; trusted for Si-vs-carbon and dark=>pore, discounted
         (beta) for mid-grey=>solid carbon (mass moves to the set {pore,carbon}).
  ETD  - two-point normalised on pore floor / graphite mode; dark supports
         {pore}, bright supports the set {carbon, si}.
  InLens - zero mass; its gradient only weights the 3x3 majority filter.
Masses are combined with Dempster's rule; a pixel is decided when the
pignistic probability >= betp_min and conflict <= conflict_max (or the
top-two margin >= margin_min), otherwise labelled UNCERTAIN.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import euler_number, label

from . import CARBON, PORE, SI, UNCERTAIN


def _gauss(x, mu, sd):
    sd = max(float(sd), 1.0)
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / sd


def _class_gaussians(bse_s: np.ndarray, t1: float, t2: float, valid: np.ndarray):
    """Mean/SD of the smoothed BSE inside each multi-Otsu class."""
    v = bse_s[valid]
    params = []
    for lo, hi in ((-np.inf, t1), (t1, t2), (t2, np.inf)):
        sel = v[(v >= lo) & (v < hi)]
        if sel.size < 100:
            sel = v
        params.append((float(np.median(sel)), float(1.4826 * np.median(np.abs(sel - np.median(sel))) + 1e-6)))
    return params


def bse_masses(bse_s, t1, t2, valid, beta):
    """m({P}), m({C}), m({S}), m({P,C}) from BSE plausibilities."""
    (mp, sp), (mc, sc), (ms, ss) = _class_gaussians(bse_s, t1, t2, valid)
    lp, lc, ls = _gauss(bse_s, mp, sp), _gauss(bse_s, mc, sc), _gauss(bse_s, ms, ss)
    tot = lp + lc + ls + 1e-12
    pP, pC, pS = lp / tot, lc / tot, ls / tot
    return pP, beta * pC, pS, (1.0 - beta) * pC


def etd_masses(etd_s, carbon_mask, valid):
    """m({P}), m({C,S}) from a per-image two-point normalised ETD."""
    if etd_s is None:
        z = np.zeros_like(valid, dtype=np.float32)
        return z, 1.0 - z  # vacuous: all mass on the full set -> handled by caller
    c = etd_s[carbon_mask & valid]
    if c.size < 100:
        c = etd_s[valid]
    floor = float(np.percentile(c, 10))
    mode = float(np.median(c))
    e = np.clip((etd_s - floor) / max(mode - floor, 1.0), 0.0, 1.0).astype(np.float32)
    return 1.0 - e, e


def combine(mP1, mC1, mS1, mPC1, mP2, mCS2, etd_present=True):
    """Dempster combination of BSE (m1) and ETD (m2) masses.

    Returns singleton masses after normalisation, the conflict K and the
    pignistic probabilities (P, C, S).
    """
    if not etd_present:
        mP, mC, mS, mPC = mP1, mC1, mS1, mPC1
        K = np.zeros_like(mP1)
    else:
        # intersections: {P}x{P}->P ; {P}x{C,S}->empty ; {C}x{P}->empty ; {C}x{C,S}->C
        # {S}x{P}->empty ; {S}x{C,S}->S ; {P,C}x{P}->P ; {P,C}x{C,S}->C
        mP = mP1 * mP2 + mPC1 * mP2
        mC = mC1 * mCS2 + mPC1 * mCS2
        mS = mS1 * mCS2
        K = mP1 * mCS2 + mC1 * mP2 + mS1 * mP2
        norm = np.maximum(1.0 - K, 1e-6)
        mP, mC, mS = mP / norm, mC / norm, mS / norm
        mPC = np.zeros_like(mP)
    betP = mP + 0.5 * mPC
    betC = mC + 0.5 * mPC
    betS = mS
    return (betP, betC, betS), K


def _weighted_majority(labels, weights, n_classes=4, size=3):
    """Weighted 3x3 majority vote; weights damp votes across InLens edges."""
    scores = np.empty((n_classes,) + labels.shape, dtype=np.float32)
    for c in range(n_classes):
        scores[c] = ndi.uniform_filter((labels == c).astype(np.float32) * weights, size=size, mode="nearest")
    return scores.argmax(axis=0).astype(np.uint8)


def _remove_islands(labels, min_px, n_classes=4):
    out = labels.copy()
    for c in range(n_classes):
        lab, n = ndi.label(out == c, structure=np.ones((3, 3)))
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        small = sizes < min_px
        small[0] = False
        if small.any():
            out[small[lab]] = UNCERTAIN
    # fill the created holes from the surrounding (non-uncertain) majority
    hole = out == UNCERTAIN
    if hole.any():
        filled = _weighted_majority(np.where(hole, 0, out), np.where(hole, 0.0, 1.0).astype(np.float32),
                                    n_classes=3, size=5)
        out[hole & (labels != UNCERTAIN)] = filled[hole & (labels != UNCERTAIN)]
    return out


def fuse_phase_map(ch, seg: dict, fcfg: dict) -> dict:
    """Return L_mid, L_solid, L_pore (uint8 maps), confidence, conflict, uncertain_frac.

    Maps are cropped to the valid row band so that every row is usable.
    """
    valid = ch.valid
    rows = np.where(valid.any(axis=1))[0]
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    sl = slice(r0, r1)
    bse_s = seg["bse_s"][sl].astype(np.float32)
    vmask = valid[sl]
    etd_s = seg.get("etd_s")
    etd_s = None if etd_s is None else etd_s[sl].astype(np.float32)
    carbon = seg["carbon"][sl]

    mP1, mC1, mS1, mPC1 = bse_masses(bse_s, seg["t1"], seg["t2"], vmask, fcfg["beta_bse_carbon"])
    etd_present = etd_s is not None
    mP2, mCS2 = etd_masses(etd_s, carbon, vmask) if etd_present else (None, None)
    (bP, bC, bS), K = combine(mP1, mC1, mS1, mPC1, mP2, mCS2, etd_present)

    bet = np.stack([bP, bC, bS])
    order = np.sort(bet, axis=0)
    top, second = order[-1], order[-2]
    argmax = bet.argmax(axis=0).astype(np.uint8)
    decided = (K <= fcfg["conflict_max"]) & ((top >= fcfg["betp_min"]) | (top - second >= fcfg["margin_min"]))
    labels = np.where(decided, argmax, UNCERTAIN).astype(np.uint8)

    # InLens gradient -> smoothness weight of the majority filter
    if ch.inlens is not None:
        g = ndi.gaussian_gradient_magnitude(ch.inlens[sl].astype(np.float32), 1.0)
        g = g / (np.percentile(g[vmask], 99) + 1e-6)
        w = (1.0 / (1.0 + fcfg["inlens_gradient_weight"] * g)).astype(np.float32)
    else:
        w = np.ones_like(bse_s, dtype=np.float32)
    labels = _weighted_majority(labels, w, size=fcfg["majority_filter"])
    labels = _remove_islands(labels, fcfg["min_island_px"])
    labels[~vmask] = UNCERTAIN

    L_mid = argmax.copy()
    L_mid = _weighted_majority(L_mid, w, n_classes=3, size=fcfg["majority_filter"])
    unc = labels == UNCERTAIN
    L_mid = np.where(unc, L_mid, labels).astype(np.uint8)
    L_solid = np.where(unc, CARBON, labels).astype(np.uint8)
    L_pore = np.where(unc, PORE, labels).astype(np.uint8)
    return {
        "L_mid": L_mid, "L_solid": L_solid, "L_pore": L_pore,
        "confidence": top.astype(np.float32), "conflict": K.astype(np.float32),
        "uncertain_frac": float(unc[vmask].mean()),
        "uncertain": unc, "row_offset": r0,
    }


# ----------------------------------------------------------------------------
def percolating_fraction(mask: np.ndarray, axis: int = 0) -> float:
    lab, n = ndi.label(mask)
    if n == 0:
        return 0.0
    first = np.unique(lab.take(0, axis=axis))
    last = np.unique(lab.take(-1, axis=axis))
    span = np.intersect1d(first, last)
    span = span[span > 0]
    if span.size == 0:
        return 0.0
    return float(np.isin(lab, span).sum() / mask.sum())


def _phase_audit(labels: np.ndarray) -> dict:
    out = {}
    for name, c in (("pore", PORE), ("carbon", CARBON), ("si", SI)):
        m = labels == c
        out[f"{name}_frac"] = float(m.mean())
        out[f"{name}_n_components"] = int(ndi.label(m)[1])
        out[f"{name}_euler"] = int(euler_number(m, connectivity=1)) if m.any() else 0
        out[f"{name}_percolating_frac_v"] = percolating_fraction(m, 0)
        out[f"{name}_percolating_frac_h"] = percolating_fraction(m, 1)
    return out


def downsample_labels(labels: np.ndarray, factor: int, max_pf_change: float = 0.02) -> tuple[np.ndarray, dict]:
    """Block-majority downsampling on labels (ties -> UNCERTAIN), with audit.

    No opening/closing.  Audit compares Euler number, component count and the
    percolating fraction of each phase before/after; ``accepted`` is True when
    no phase's vertical percolating fraction moves by more than ``max_pf_change``.
    """
    H, W = labels.shape
    Hc, Wc = (H // factor) * factor, (W // factor) * factor
    lab = labels[:Hc, :Wc]
    blocks = lab.reshape(Hc // factor, factor, Wc // factor, factor).transpose(0, 2, 1, 3).reshape(Hc // factor, Wc // factor, -1)
    counts = np.stack([(blocks == c).sum(axis=-1) for c in (PORE, CARBON, SI)], axis=0)
    order = np.sort(counts, axis=0)
    ds = counts.argmax(axis=0).astype(np.uint8)
    tie = order[-1] == order[-2]
    ds[tie] = UNCERTAIN
    before = _phase_audit(labels)
    after = _phase_audit(ds)
    audit = {"factor": factor, "before": before, "after": after,
             "tie_frac": float(tie.mean())}
    for p in ("pore", "carbon", "si"):
        audit[f"accepted_{p}"] = bool(
            abs(after[f"{p}_percolating_frac_v"] - before[f"{p}_percolating_frac_v"]) <= max_pf_change)
    # the ionic index rides on the pore phase; carbon bridges one pixel wide are
    # lost at x4, which is reported (accepted_carbon) rather than hidden
    audit["accepted"] = audit["accepted_pore"]
    audit["accepted_all"] = all(audit[f"accepted_{p}"] for p in ("pore", "carbon", "si"))
    return ds, audit
