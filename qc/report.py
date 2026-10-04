"""Reports: verdict JSON, report.md, report.html, figures, DM text."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .assign import (
    assign_sample, feature_matrix, feature_vector, distances,
    probs_from_distances, signature_match,
)
from .kpis import MEANINGS, ROBUSTNESS, SHORTLIST
from .stats import conformal_rank, gate_z_scores, verdict, z_scores

CAVEATS = [
    "Bright particles are identified by backscatter Z-contrast only (no EDS); they are reported as Si-candidates and may include SiOx or Si-C composite.",
    "Pores appear open (not resin-infiltrated); the deep-pore fraction is a lower bound on porosity, not a porosity measurement.",
    "Graphite, carbon black and binder cannot be separated in these images; the 'carbon matrix' class contains all three.",
    "Coating thickness, surface roughness and collector delamination are not assessable: the fields lie entirely inside the coating.",
    "Anisotropy and transport indices are 2-D section quantities compared like-with-like against the baseline, not 3-D values; the through-plane direction is assumed vertical in the frame.",
    "Pixel size (25 nm/px) is taken from the TIFF resolution tags; vendor metadata is absent, so it is unverified.",
    "With N baseline images the smallest achievable rank p-value is 1/(N+1); verdicts are effect-size judgements against a small baseline.",
    "Image grey levels have been remapped after acquisition (comb histograms); any method relying on raw intensities would be confounded - this system uses BSE phase identity after smoothing and treats ETD/InLens levels as acquisition covariates.",
    "Transport and mechanics indices are 2-D effective-medium quantities computed with a fixed matrix diffusivity D_c = 0.05 and fixed moduli; they are ratios to the baseline under identical assumptions, not electrode tortuosity, conductivity or stress values.",
    "The deep-pore phase does not percolate in 2-D, so no pore-only tortuosity is reported; the two-conductivity index's absolute level is set by D_c and only its ratios are meaningful.",
    "Cell-level outputs come from a PyBaMM composite graphite-Si model with a frozen LG-M50-type parameter set and are relative rate-capability and plating-indicator shifts, not predictions of the real cell.",
]

NOT_MEASURABLE = [
    "coating thickness", "surface roughness", "collector delamination",
    "binder/carbon-black distribution", "Si vs SiOx identity",
    "true (total) porosity",
]


def evaluate_result(res: dict, stats: dict, kpis_df: pd.DataFrame, cfg: dict,
                    sim: dict | None = None, sim_base: dict | None = None) -> dict:
    """Full verdict object (Section 7.2) from a process_sample result (+ Section 10 block)."""
    row = dict(res["kpis"])
    v = verdict(row, stats, cfg)

    gz = gate_z_scores(res["gates"], stats)
    gate_flags = [k for k, z in gz.items() if abs(z) > 4]
    gate_status = "caution" if gate_flags else "ok"

    centroids = {b: np.array(c) for b, c in stats["centroids"].items()}
    T = stats["temperature"]
    strip_rows = res.get("strip_rows")
    id_perc = {"distances": stats.get("id_distances", [])}
    a = assign_sample(row, stats, {"centroids": centroids}, T,
                      strip_rows=strip_rows, id_percentiles=id_perc)

    # signature match
    z = v["z"]
    sig_path = Path(str(stats.get("_sig_path", ""))) if stats.get("_sig_path") else None
    try:
        signatures = json.loads((Path(stats["_sig_path"])).read_text())
    except Exception:
        signatures = stats.get("signatures", {})
    sm = signature_match(z, signatures)
    a["signature_match"] = {b: round(s["score"], 2) for b, s in sm.items()}
    a["signature_kpis_matched"] = {b: s["matched"] for b, s in sm.items()}
    a["signature_kpis_missed"] = {b: s["missed"] for b, s in sm.items()}
    a["calibration_note"] = (
        f"T and shrinkage chosen by nested leave-one-image-out on "
        f"{len(kpis_df)} labelled images"
    )

    # conformal rank: S_i per baseline image (held out vs the rest)
    base = kpis_df[kpis_df["batch"].astype(str) == stats.get("batch", "3")]
    S_b = []
    for _, r in base.iterrows():
        zz = z_scores(r.to_dict(), stats, SHORTLIST)
        top3 = sorted((abs(x) for x in zz.values()), reverse=True)[:3]
        S_b.append(float(np.mean(top3)) if top3 else 0.0)
    p_rank, p_floor = conformal_rank(v["aggregate_score_S"], S_b)

    if gate_flags:
        v["drivers"] = drivers_with_lead = v["drivers"]

    simulation = None
    if sim is not None:
        from .sim.baseline import sim_block
        simulation = sim_block(sim["row"], sim["bounds"], sim_base, sim["row"]["assumptions_hash"])
    return {
        "sample_id": res["sample_id"],
        "simulation": simulation,
        "baseline": {"batch": int(stats.get("batch", 3)),
                     "n_images": stats["n"], "version": stats.get("version")},
        "acquisition_gates": {"status": gate_status, "flags": gate_flags, "z": gz},
        "verdict_vs_baseline": {
            "decision": v["decision"],
            "aggregate_score_S": v["aggregate_score_S"],
            "conformal_rank_p": p_rank,
            "p_floor": p_floor,
            "n_kpis_beyond_2_5": v["n_kpis_beyond_2_5"],
            "drivers": v["drivers"],
        },
        "batch_assignment": a,
        "kpis": {k: {"value": row.get(k), "se": res["kpi_ses"].get(f"{k}_se"),
                     "z": z.get(k), "unit": "see report"} for k in row},
        "not_measurable": NOT_MEASURABLE,
        "caveats": CAVEATS,
        "strip_assignments": a.get("strip_agree"),
        "warnings": res.get("warnings", []),
    }


# ---------------- figures ----------------

def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return buf.getvalue()


def fig_signatures(signatures: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    batches = sorted(signatures)
    if not batches:
        fig, axes = plt.subplots(1, 1, figsize=(7, 2))
        axes.text(0.5, 0.5, "no signatures (single labelled batch)",
                  ha="center", va="center")
        axes.axis("off")
        fig.savefig(path)
        plt.close(fig)
        return
    fig, axes = plt.subplots(1, len(batches), figsize=(7 * len(batches), 6),
                             squeeze=False)
    for ax, b in zip(axes[0], batches):
        effs = signatures[b]
        ks = sorted(effs, key=lambda k: -abs(effs[k]["effect"]))
        vals = [effs[k]["effect"] for k in ks]
        colors = ["#1f77b4" if effs[k]["stable"] else "#bbbbbb" for k in ks]
        ax.barh(range(len(ks)), vals, color=colors)
        ax.set_yticks(range(len(ks)), ks, fontsize=7)
        ax.axvline(0, color="k", lw=0.5)
        ax.axvline(1.5, color="r", lw=0.5, ls="--")
        ax.axvline(-1.5, color="r", lw=0.5, ls="--")
        ax.set_title(f"Batch {b} signature (effect vs baseline, sigma units)")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_kpi_boxplots(kpis_df: pd.DataFrame, stats: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(SHORTLIST)
    fig, axes = plt.subplots(4, 4, figsize=(20, 14))
    for ax, k in zip(axes.ravel(), SHORTLIST):
        data = [kpis_df[kpis_df["batch"].astype(str) == b][k].dropna()
                for b in ("1", "2", "3")]
        ax.boxplot(data, tick_labels=["B1", "B2", "B3"])
        ax.set_title(k, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_zbars(z: dict, stats: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = sorted(z, key=lambda k: -abs(z[k]))
    vals = [z[k] for k in ks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvspan(-2.5, 2.5, color="#d4edda", alpha=0.6)
    ax.axvspan(-4, -2.5, color="#fff3cd", alpha=0.6)
    ax.axvspan(2.5, 4, color="#fff3cd", alpha=0.6)
    ax.barh(range(len(ks)), vals,
            color=["#d62728" if abs(v) > 4 else "#ff7f0e" if abs(v) > 2.5 else "#1f77b4"
                   for v in vals])
    ax.set_yticks(range(len(ks)), ks, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("robust z vs baseline")
    ax.set_title("KPI z-scores")
    fig.savefig(path)
    plt.close(fig)


def fig_pca(kpis_df: pd.DataFrame, stats: dict, new_x: np.ndarray,
            strip_xs: list[np.ndarray] | None, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    X = feature_matrix(kpis_df[kpis_df["batch"].astype(str).isin(["1", "2", "3"])],
                       stats)
    labels = kpis_df[kpis_df["batch"].astype(str).isin(["1", "2", "3"])][
        "batch"].astype(str).values
    allx = np.vstack([X, new_x[None]])
    p2 = PCA(n_components=2).fit_transform(allx)
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = {"1": "#d62728", "2": "#2ca02c", "3": "#1f77b4"}
    for b in ("1", "2", "3"):
        m = labels == b
        ax.scatter(p2[: len(X)][m, 0], p2[: len(X)][m, 1],
                   c=cmap[b], label=f"batch {b}", s=40)
    ax.scatter(p2[-1, 0], p2[-1, 1], c="k", marker="*", s=300, label="new sample")
    if strip_xs:
        xs = p2 = None
        stx = np.vstack(strip_xs)
        stx = PCA(n_components=2).fit(np.vstack([X, new_x[None]])).transform(stx)
        ax.scatter(stx[:, 0], stx[:, 1], c="k", s=10, alpha=0.4)
    ax.legend()
    ax.set_title("Standardised KPI space (PCA)")
    fig.savefig(path)
    plt.close(fig)


def fig_overlay(res: dict, path: Path):
    from .segment import overlay_png
    overlay_png(res["seg"], res["channels"].bse, path)


# ---------------- text outputs ----------------

PHASE_COLORS = np.array([[20, 20, 20], [120, 120, 120], [255, 200, 40], [200, 40, 200]], dtype=np.uint8)


def fig_sim_maps(sim: dict, path: Path):
    """Fused phase map (x4), swollen map at the largest f_A, per-particle constraint map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    maps = sim.get("maps") or {}
    if "L_mid" not in maps:
        return
    L, S, C = maps["L_mid"], maps.get("L_swollen"), maps.get("constraint_map")
    fig, ax = plt.subplots(3, 1, figsize=(14, 3 * 14 * L.shape[0] / L.shape[1] + 1))
    ax[0].imshow(PHASE_COLORS[np.clip(L, 0, 3)]); ax[0].set_title("fused phase map L_mid (black pore, grey carbon, yellow Si, magenta uncertain)")
    if S is not None:
        ax[1].imshow(PHASE_COLORS[np.clip(S, 0, 3)]); ax[1].set_title("after swelling (largest f_A scenario, pore-first)")
    if C is not None:
        im = ax[2].imshow(C, cmap="magma", vmin=0, vmax=1); ax[2].set_title("per-particle constraint index (fraction of growth into solid)")
        fig.colorbar(im, ax=ax[2], fraction=0.02)
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def fig_dc_sweep(sim_base: dict | None, sim: dict | None, path: Path):
    """D_eff_rel_TP vs D_c: baseline band (p10-p90, L_mid) + L_solid/L_pore medians; sample point with bounds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    sweep = (sim_base or {}).get("D_c_sweep", {})
    dcs = sorted({float(k.split("_Dc")[1].split("_")[0]) for k in sweep})
    if dcs:
        med = [sweep[f"D_eff_rel_TP_Dc{d:g}_L_mid"]["med"] for d in dcs]
        p10 = [sweep[f"D_eff_rel_TP_Dc{d:g}_L_mid"]["p10"] for d in dcs]
        p90 = [sweep[f"D_eff_rel_TP_Dc{d:g}_L_mid"]["p90"] for d in dcs]
        ax.fill_between(dcs, p10, p90, alpha=0.25, label="baseline p10-p90 (L_mid)")
        ax.plot(dcs, med, "k-o", label="baseline median (L_mid)")
        for lab, ls in (("L_solid", "--"), ("L_pore", ":")):
            ys = [sweep.get(f"D_eff_rel_TP_Dc{d:g}_{lab}", {}).get("med", np.nan) for d in dcs]
            ax.plot(dcs, ys, "k" + ls, label=f"baseline median ({lab})")
    if sim is not None:
        row, b = sim["row"], sim["bounds"]
        dc0 = 0.05
        v = row.get("D_eff_rel_TP", np.nan)
        lo, hi = b.get("D_eff_rel_TP_L_solid", v), b.get("D_eff_rel_TP_L_pore", v)
        ax.errorbar([dc0], [v], yerr=[[max(v - lo, 0)], [max(hi - v, 0)]], fmt="rs", capsize=4,
                    label="this sample at D_c=0.05 [L_solid, L_pore]")
        own = sorted((float(k.split("_Dc")[1].split("_")[0]), b[k]) for k in b if k.startswith("D_eff_rel_TP_Dc") and k.endswith("_L_mid"))
        if own:
            ax.plot([x for x, _ in own], [y for _, y in own], "r-", label="this sample sweep")
    ax.set_xscale("log"); ax.set_xlabel("matrix diffusivity D_c (frozen assumption)"); ax.set_ylabel("D_eff_rel through-plane")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def sim_table_rows(simulation: dict) -> list[tuple]:
    rows = []
    for k, e in simulation["relative_to_baseline"].items():
        b = simulation["bounds"].get(k, {})
        rows.append((k, simulation["grade"].get(k, "B"), e["value"], e.get("baseline_med"), e.get("ratio"),
                     b.get("L_solid"), b.get("L_pore")))
    return rows


def _g(v, spec: str = ".4g") -> str:
    """Format a float or return 'nan' for None/NaN."""
    return format(v, spec) if v is not None and np.isfinite(v) else "nan"


def _fmt_probs(p: dict) -> str:
    return " | ".join(f"B{b} {v:.2f}" for b, v in sorted(p.items()))


def dm_block(ev: dict) -> str:
    a = ev["batch_assignment"]
    v = ev["verdict_vs_baseline"]
    probs = a["probabilities"]
    assigned = a["assigned_batch"]
    others = _fmt_probs({b: p for b, p in probs.items() if b != str(assigned)})
    top_drivers = v["drivers"][:2]
    why = "; ".join(
        f"{d['kpi']} {d['value']:.3g} vs {d['baseline_median']:.3g}+/-{d['baseline_scale']:.2g} (z {d['z']:+.1f})"
        for d in top_drivers if d.get("value") is not None
    )
    sig = a.get("signature_match", {})
    reads = ""
    if sig:
        best = max(sig, key=lambda b: sig[b])
        if abs(sig[best]) >= 0.5:
            reads = f"signature: batch {best} {sig[best]:+.1f} sigma"
    novelty = ("YES - far from all known batches "
               f"(id-score {a.get('in_distribution_score')}); betting on nearest"
               if a["novelty_flag"] else
               f"no (in-distribution score {a.get('in_distribution_score')})")
    lines = [
        f"{ev['sample_id']} -> Batch {assigned} (p={probs[str(assigned)]:.2f} | {others}) - {a['confidence_label']}, stability {a.get('stability')}",
        f"  vs baseline: {v['decision']} ({v['n_kpis_beyond_2_5']}/16 KPIs > 2.5sigma; rank-p {v['conformal_rank_p']:.3f}, floor {v['p_floor']:.3f})",
        f"  why: {why}",
    ]
    if reads:
        lines.append(f"  {reads}")
    lines.append(f"  novelty: {novelty}")
    return "\n".join(lines)


def lead_paragraph(ev: dict, stats: dict) -> str:
    a = ev["batch_assignment"]
    v = ev["verdict_vs_baseline"]
    probs = a["probabilities"]
    assigned = a["assigned_batch"]
    sig = a.get("signature_match", {})
    sig_txt = ""
    nonzero = {b: s for b, s in sig.items() if abs(s) >= 0.5}
    if nonzero:
        best = max(nonzero, key=lambda b: nonzero[b])
        sig_txt = (f" It sits {nonzero[best]:+.1f} sigma along batch "
                   f"{best}'s signature.")
    elif sig:
        sig_txt = " No batch signature is stable enough to read."
    p_floor_note = (f"Baseline has {stats['n']} images, so the smallest reportable "
                    f"rank p-value is {v['p_floor']:.3f}; this verdict is an "
                    f"effect-size judgement, not a significance test.")
    gates = ev["acquisition_gates"]
    gate_warn = ""
    if gates["status"] == "caution":
        gate_warn = (" **Acquisition drift or artefact suspected - compare with "
                     f"caution** (gates: {', '.join(gates['flags'])}).")
    return (
        f"**{ev['sample_id']} is most consistent with Batch {assigned}** "
        f"(probability {probs[str(assigned)]:.2f}; "
        + "; ".join(f"Batch {b}: {p:.2f}" for b, p in sorted(probs.items()))
        + (f"; assignment stable in {a['stability']:.0%} of strip-bootstrap "
           f"resamples" if a.get("stability") is not None else "")
        + f"; {a['confidence_label']})."
        + sig_txt
        + f" **Relative to the Batch-{stats.get('batch',3)} baseline the verdict is "
        f"{v['decision']}**: {v['n_kpis_beyond_2_5']} of 16 shortlist KPIs exceed "
        f"2.5 robust sigma; aggregate score S={v['aggregate_score_S']:.2f}, "
        f"rank-p={v['conformal_rank_p']:.3f}."
        + gate_warn + " " + p_floor_note
    )


def write_report(res: dict, stats: dict, kpis_df: pd.DataFrame,
                 strips_df: pd.DataFrame, outdir: Path, cfg: dict,
                 signatures_path: Path | None = None,
                 sim: dict | None = None, sim_base: dict | None = None) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    if signatures_path is None:
        signatures_path = Path(stats.get("_sig_path") or "")
    stats = dict(stats)
    sp = Path(str(outdir)).parent.parent / "signatures.json"
    stats["_sig_path"] = str(sp) if sp.exists() else ""

    ev = evaluate_result(res, stats, kpis_df, cfg, sim=sim, sim_base=sim_base)
    (outdir / "verdict.json").write_text(json.dumps(ev, indent=2, default=str))

    # figures
    figs = outdir / "figs"
    figs.mkdir(exist_ok=True)
    fig_overlay(res, figs / "overlay.png")
    fig_zbars(ev["verdict_vs_baseline"]["drivers"] and ev["verdict_vs_baseline"]
              and {d["kpi"]: d["z"] for d in ev["verdict_vs_baseline"]["drivers"]},
              stats, figs / "z_bars.png")
    new_x = feature_vector(res["kpis"], stats)
    strip_xs = [feature_vector(sr, stats) for sr in res.get("strip_rows", [])]
    fig_pca(kpis_df, stats, new_x, strip_xs, figs / "pca.png")
    sim_md, sim_html = [], ""
    if ev.get("simulation"):
        fig_sim_maps(sim, figs / "sim_maps.png")
        fig_dc_sweep(sim_base, sim, figs / "dc_sweep.png")
        sm = ev["simulation"]
        sim_md = ["", "## Simulation layer (relative indices, frozen assumptions "
                  f"{sm['assumptions_hash']})", "",
                  "Grades: A arithmetic on measured quantities; B direction supported, level set by an "
                  "assumption (ratios only); C labelled heuristic. Bounds = value on L_solid / L_pore.", "",
                  "| index | grade | value | baseline med | ratio | L_solid | L_pore |", "|---|---|---|---|---|---|---|"]
        for k, gr, v, med, ratio, lo, hi in sim_table_rows(sm):
            sim_md.append(f"| {k} | {gr} | {_g(v)} | {_g(med)} | {_g(ratio, '.3g')} | {_g(lo)} | {_g(hi)} |")
        if sm["undefined"]:
            sim_md += ["", "Undefined: " + "; ".join(sm["undefined"])]
        if sm["unstable_indices"]:
            sim_md += ["", "Rank-unstable across conventions: " + ", ".join(sm["unstable_indices"])]
        sim_md += ["", f"Downsample audit (pore phase) accepted: {sm['downsample_accepted']}; "
                   f"uncertain fraction {sm['uncertain_frac']:.3f}", "",
                   "![sim maps](figs/sim_maps.png)", "![D_c sweep](figs/dc_sweep.png)"]
        sim_html = "<h2>Simulation layer</h2><p>Frozen assumptions " + sm["assumptions_hash"] + \
            "; grades A/B/C as in report.md.</p><table><tr><th>index</th><th>grade</th><th>value</th>" \
            "<th>baseline med</th><th>ratio</th><th>L_solid</th><th>L_pore</th></tr>" + "".join(
            f"<tr><td>{k}</td><td>{gr}</td><td>{_g(v)}</td><td>{_g(med)}</td><td>{_g(ratio, '.3g')}</td>"
            f"<td>{_g(lo)}</td><td>{_g(hi)}</td></tr>" for k, gr, v, med, ratio, lo, hi in sim_table_rows(sm)) + \
            "</table>" + (f"<p>Undefined: {'; '.join(sm['undefined'])}</p>" if sm["undefined"] else "") + \
            (f"<p>Rank-unstable: {', '.join(sm['unstable_indices'])}</p>" if sm["unstable_indices"] else "")

    # markdown report
    md = [f"# QC report: {ev['sample_id']}", "", "## Verdict", "",
          lead_paragraph(ev, stats), "",
          "## Acquisition gates", "",
          f"Status: **{ev['acquisition_gates']['status']}**"
          + (f" - flags: {', '.join(ev['acquisition_gates']['flags'])}"
             if ev['acquisition_gates']['flags'] else ""), "",
          "## KPI table", "",
          "| KPI | value | SE | baseline med +/- scale | z | class |",
          "|---|---|---|---|---|---|"]
    z = {d["kpi"]: d for d in ev["verdict_vs_baseline"]["drivers"]}
    for k in SHORTLIST:
        d = z.get(k)
        kv = res["kpis"].get(k)
        se = res["kpi_ses"].get(f"{k}_se")
        st = stats["kpis"].get(k, {})
        md.append(
            f"| {k} | {_g(kv)} | {_g(se, '.3g')} | "
            f"{_g(st.get('med'))} +/- {_g(st.get('scale'), '.3g')} | "
            f"{_g(d['z'], '+.2f') if d else 'nan'} | {ROBUSTNESS.get(k,'M')} |"
        )
    md += ["", "## Drivers", ""]
    for d in ev["verdict_vs_baseline"]["drivers"][:8]:
        md.append(f"- **{d['kpi']}**: {d['value']:.4g} vs baseline "
                  f"{d['baseline_median']:.4g} +/- {d['baseline_scale']:.3g} "
                  f"(z={d['z']:+.2f}, {d['direction']}, class {d['robustness']}) - "
                  f"{d['meaning']}")
    a = ev["batch_assignment"]
    md += ["", "## Batch assignment", "",
           f"Assigned: **batch {a['assigned_batch']}** "
           f"({a['confidence_label']}, stability {a.get('stability')})",
           "",
           f"probabilities: {_fmt_probs(a['probabilities'])}",
           f"distances: {_fmt_probs(a['distances'])}",
           f"signature match: {a.get('signature_match')}",
           f"novelty: {a['novelty_flag']} (in-distribution score "
           f"{a.get('in_distribution_score')})",
           f"priors: {a['priors']}",
           f"{a.get('calibration_note','')}", "",
           "## Figures", "",
           "![overlay](figs/overlay.png)", "![z-scores](figs/z_bars.png)",
           "![pca](figs/pca.png)", "",
           "## Not measurable", ""]
    md += [f"- {x}" for x in NOT_MEASURABLE]
    md += sim_md
    md += ["", "## Caveats", ""]
    md += [f"- {c}" for c in CAVEATS]
    (outdir / "report.md").write_text("\n".join(md))

    # self-contained HTML
    imgs = ""
    for f in ("overlay.png", "z_bars.png", "pca.png", "sim_maps.png", "dc_sweep.png"):
        p = figs / f
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            imgs += f'<figure><img src="data:image/png;base64,{b64}" ' \
                    f'style="max-width:100%"><figcaption>{f}</figcaption></figure>'
    kpi_rows = "".join(
        f"<tr><td>{k}</td><td>{res['kpis'].get(k, float('nan')):.4g}</td>"
        f"<td>{(res['kpi_ses'].get(k + '_se') or float('nan')):.3g}</td>"
        f"<td>{z[k]['z']:+.2f}</td><td>{ROBUSTNESS.get(k,'M')}</td></tr>"
        for k in SHORTLIST if k in z
    )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>QC report {ev['sample_id']}</title>
<style>body{{font-family:sans-serif;max-width:1100px;margin:2em auto;color:#222}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:3px 8px;font-size:13px}}
figure{{margin:1em 0}}figcaption{{font-size:12px;color:#666}}</style></head><body>
<h1>QC report: {ev['sample_id']}</h1>
<h2>Verdict</h2><p>{lead_paragraph(ev, stats)}</p>
<h2>Acquisition gates</h2><p>Status: {ev['acquisition_gates']['status']}
{('; flags: ' + ', '.join(ev['acquisition_gates']['flags'])) if ev['acquisition_gates']['flags'] else ''}</p>
<h2>KPI table</h2><table><tr><th>KPI</th><th>value</th><th>SE</th><th>z</th><th>class</th></tr>{kpi_rows}</table>
<h2>Batch assignment</h2>
<p>Assigned: batch {a['assigned_batch']} ({a['confidence_label']},
stability {a.get('stability')}); probabilities {_fmt_probs(a['probabilities'])};
signature match {a.get('signature_match')}; novelty {a['novelty_flag']}
(id-score {a.get('in_distribution_score')}).</p>
<h2>Figures</h2>{imgs}
<h2>Not measurable</h2><ul>{''.join(f'<li>{x}</li>' for x in NOT_MEASURABLE)}</ul>
{sim_html}
<h2>Caveats</h2><ul>{''.join(f'<li>{c}</li>' for c in CAVEATS)}</ul>
</body></html>"""
    (outdir / "report.html").write_text(html)
    return ev
