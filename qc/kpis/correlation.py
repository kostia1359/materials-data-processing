"""KPI 8 - two-point correlation (S2) lengths + image-representativity SE."""
from __future__ import annotations

import numpy as np

from .common import Ctx


def _autocorr(indicator: np.ndarray, valid: np.ndarray):
    """Normalised autocovariance of a 0/1 indicator over valid pixels.

    C(r) = <(I - p)(I' - p)> / Var(I) with C(0)=1, computed via FFT with
    proper normalisation by the number of valid pixel pairs at each lag.
    """
    f = np.where(valid, indicator.astype(float), 0.0)
    w = valid.astype(float)
    n = float(f[valid].sum())
    denom = float(valid.sum())
    if denom == 0 or n == 0 or n == denom:
        return None, None
    p = n / denom
    fc = f - p * w  # zero-mean, zero outside valid
    # unnormalised autocovariances
    ac_f = np.fft.ifft2(np.abs(np.fft.fft2(fc)) ** 2).real
    ac_w = np.fft.ifft2(np.abs(np.fft.fft2(w)) ** 2).real
    ac_w[ac_w < 1] = np.nan
    c = ac_f / ac_w
    var = np.nanmean(c[0, 0]) if np.isnan(c[0, 0]) else c[0, 0]
    var = c[0, 0]
    if var <= 0:
        return None, None
    return c / var, p


def _corr_length(profile: np.ndarray) -> float:
    """First lag where C < 1/e (linear interpolation), in px."""
    thr = 1.0 / np.e
    for i, v in enumerate(profile):
        if i == 0:
            continue
        if v < thr:
            prev = profile[i - 1]
            if prev == v:
                return float(i)
            return float(i - 1 + (prev - thr) / (prev - v))
    return float("nan")


def compute(ctx: Ctx) -> dict:
    out = {}
    # pore indicator
    c_p, p_p = _autocorr(ctx.pore, ctx.valid)
    if c_p is not None:
        h, w = c_p.shape
        out["s2_len_pore_h_px"] = _corr_length(c_p[0, : w // 2])
        out["s2_len_pore_v_px"] = _corr_length(c_p[: h // 2, 0])
        # integral range: positive central lobe sum
        cpos = np.where(c_p > 0, c_p, 0.0)
        # zero out lags beyond first negative crossing in each direction
        hh = int(np.argmax(c_p[0] < 0) or w // 2)
        vv = int(np.argmax(c_p[:, 0] < 0) or h // 2)
        a2 = float(cpos[: max(vv, 1), : max(hh, 1)].sum())
        out["s2_integral_range_pore_px2"] = a2
        out["pore_frac_se_imagerep"] = float(
            np.sqrt(max(p_p * (1 - p_p), 0) * a2 / ctx.valid.sum())
        )
    else:
        for k in ("s2_len_pore_h_px", "s2_len_pore_v_px",
                  "s2_integral_range_pore_px2", "pore_frac_se_imagerep"):
            out[k] = float("nan")
    # si indicator -> representative SE of si_frac
    c_s, p_s = _autocorr(ctx.si, ctx.valid)
    if c_s is not None:
        h, w = c_s.shape
        lh = _corr_length(c_s[0, : w // 2])
        lv = _corr_length(c_s[: h // 2, 0])
        out["s2_len_si_px"] = float(np.nanmean([lh, lv]))
        cpos = np.where(c_s > 0, c_s, 0.0)
        hh = int(np.argmax(c_s[0] < 0) or w // 2)
        vv = int(np.argmax(c_s[:, 0] < 0) or h // 2)
        a2 = float(cpos[: max(vv, 1), : max(hh, 1)].sum())
        out["si_frac_se_imagerep"] = float(
            np.sqrt(max(p_s * (1 - p_s), 0) * a2 / ctx.valid.sum())
        )
    else:
        out["s2_len_si_px"] = float("nan")
        out["si_frac_se_imagerep"] = float("nan")
    return out
