"""Per-sample processing: load -> segment -> gates -> KPIs -> strips."""
from __future__ import annotations

import numpy as np
from skimage.filters import gaussian

from .config import load_config
from .gates import compute_gates
from .io import Channels, Sample, load_sample
from .kpis import MODULES, ROBUSTNESS
from .kpis.common import Ctx
from .kpis.common import chord_lengths
from .segment import segment
from .tiles import bootstrap_intervals, strip_masks, strip_ses

# KPIs whose list-valued internals must not leak into CSV rows
_PRIVATE = lambda k: k.startswith("_")


def _downsample_channels(ch: Channels, f: int) -> Channels:
    ds = lambda a: None if a is None else a[::f, ::f]
    return Channels(
        bse=ds(ch.bse), etd=ds(ch.etd), inlens=ds(ch.inlens),
        valid=ds(ch.valid), px_nm=ch.px_nm * f,
        px_nm_assumed=ch.px_nm_assumed,
        masked_rows_top=ch.masked_rows_top // f,
        masked_rows_bottom=ch.masked_rows_bottom // f,
        shape=ds(ch.bse).shape,
    )


def _extras(ch: Channels, seg: dict, cfg: dict) -> dict:
    """Image-level extras needed by KPIs (Sato response, curtaining, pore slope)."""
    from .kpis.defects import curtaining_angle_deg, sato_threshold_for

    extras = {}
    if ch.etd is not None:
        etd_s = seg["etd_s"]
        valid = ch.valid
        bse_s = seg["bse_s"]
        t1 = seg["t1"]
        carbon = seg["carbon"]
        etd_carbon_vals = etd_s[carbon]
        etd_p10 = np.percentile(etd_carbon_vals, 10) if etd_carbon_vals.size else 0
        etd_dark = etd_s < etd_p10
        for delta, key in ((3, "pore_frac_at_t1_plus3"), (-3, "pore_frac_at_t1_minus3")):
            frac = float(((bse_s < t1 + delta) & etd_dark & valid).sum() / valid.sum())
            extras[key] = frac
    extras["curtaining_deg"] = curtaining_angle_deg(ch.inlens)
    thr, resp = sato_threshold_for(seg["bse_s"], ch.valid, cfg)
    extras["sato_response"] = resp
    extras["sato_p995_self"] = thr  # baseline stores median of these
    return extras


def compute_all_kpis(ctx: Ctx) -> dict:
    out = {}
    si_lab = None
    for mod in MODULES:
        name = mod.__name__.rsplit(".", 1)[-1]
        if name == "fractions":
            out.update(mod.compute(ctx))
        elif name == "sizes":
            out.update(mod.compute(ctx))
            si_lab = out.get("_si_lab")
            out.update(mod.compute_shape(ctx, si_lab) if si_lab is not None else {})
        elif name == "dispersion":
            out.update(mod.compute(ctx, si_lab))
        elif name == "pores":
            out.update(mod.compute_all(ctx))
        else:
            out.update(mod.compute(ctx))
    return {k: v for k, v in out.items() if not _PRIVATE(k)}, out


def process_sample(
    sample: Sample,
    cfg: dict | None = None,
    crack_threshold: float | None = None,
    threshold_shift: float = 0.0,
    downsample: int = 1,
) -> dict:
    """Full pipeline for one sample. Returns dict with kpis/strip rows/gates."""
    cfg = cfg or load_config()
    ch = load_sample(sample, cfg)
    if downsample > 1:
        ch = _downsample_channels(ch, downsample)
        cfg = dict(cfg)
        for k in ("si_min_px_count", "si_min_px_size", "pore_min_px", "hiz_min_px"):
            cfg[k] = max(1, int(cfg[k] / downsample ** 2))
        cfg["strip_width_px"] = cfg["strip_width_px"] // downsample
        cfg["quadrat_px"] = cfg["quadrat_px"] // downsample
        cfg["crack_min_length_px"] = cfg["crack_min_length_px"] // downsample
    seg = segment(ch, cfg, threshold_shift=threshold_shift)

    extras = _extras(ch, seg, cfg)
    extras["noise_sd_bse"] = None  # filled after gates
    if crack_threshold is not None:
        extras["crack_threshold"] = crack_threshold
    else:
        extras["crack_threshold"] = extras["sato_p995_self"]

    masks = {"carbon": seg["carbon"], "si": seg["si"]}
    gates = compute_gates(ch, masks, (seg["t1"], seg["t2"]))
    extras["noise_sd_bse"] = gates["noise_sd_bse"]

    ctx = Ctx(
        pore=seg["pore"], carbon=seg["carbon"], si=seg["si"],
        ambiguous=seg["ambiguous"], valid=ch.valid, bse_s=seg["bse_s"],
        px_nm=ch.px_nm, cfg=cfg, extras=extras,
    )
    kpi_row, _full = compute_all_kpis(ctx)

    # strips
    smasks = strip_masks(ctx, cfg["strips"], cfg["strip_width_px"])
    strip_rows = []
    for i, sm in enumerate(smasks):
        sctx = ctx.subregion(sm)
        if sctx.valid.sum() < 1000:
            continue
        row, _ = compute_all_kpis(sctx)
        row["strip"] = i
        strip_rows.append(row)
    ses = strip_ses(strip_rows, [k for k in kpi_row if isinstance(kpi_row[k], float)])
    cis = bootstrap_intervals(
        strip_rows, [k for k in kpi_row if isinstance(kpi_row[k], float)],
        cfg["bootstrap_n"], cfg["seed"],
    )

    return {
        "sample_id": sample.sample_id,
        "batch": sample.batch,
        "channels": ch,
        "seg": seg,
        "ctx": ctx,
        "kpis": kpi_row,
        "kpi_ses": ses,
        "kpi_cis": cis,
        "strip_rows": strip_rows,
        "gates": gates,
        "warnings": list(sample.warnings) + (["shading_flag"] if seg["shading_flag"] else []),
        "etd_label_in_filename": sample.etd_label_in_filename,
        "has_batch_prefix": sample.has_batch_prefix,
    }
