"""Command-line interface (Section 3 contract)."""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import typer

from .config import load_config
from .io import discover, sha1_of, write_manifest
from .pipeline import process_sample

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _cfg(ctx: typer.Context) -> dict:
    return ctx.obj or load_config()


@app.callback()
def main(ctx: typer.Context, config: str = typer.Option(None, "--config")):
    ctx.obj = load_config(config)


def _cache_path(out: Path, sample_id: str) -> Path:
    return out / "cache" / sample_id / "result.json"


def _sim_key(cfg: dict):
    scfg = cfg.get("sim") or {}
    if not scfg.get("enabled"):
        return None
    from .sim import SIM_VERSION, assumptions_hash
    return [SIM_VERSION, assumptions_hash(scfg)]


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o.item() if hasattr(o, "item") else str(o)


def _load_sim_baseline(out: Path) -> dict | None:
    p = out / "sim_baseline.json"
    return json.loads(p.read_text()) if p.exists() else None


def _process_cached(sample, cfg, out: Path, crack_threshold=None, full_sim: bool = False):
    """Cache per-sample results keyed on (sha1_bse, pipeline_version, sim_version+assumptions_hash).

    The heavy, baseline-independent simulation part (``run_sim``) is cached with
    the KPIs; ``finalize_sim`` is re-applied by the caller against the current baseline.
    """
    from . import PIPELINE_VERSION

    cp = _cache_path(out, sample.sample_id)
    sha = sha1_of(sample.path_bse) if sample.path_bse else "none"
    key = [sha, PIPELINE_VERSION, _sim_key(cfg)]
    if cp.exists():
        rec = json.loads(cp.read_text())
        if rec.get("key") == key and crack_threshold is None and (
                not full_sim or (rec["data"].get("sim") or {}).get("full")):
            return rec["data"], False
    res = process_sample(sample, cfg, crack_threshold)
    data = {
        "sample_id": res["sample_id"], "batch": res["batch"],
        "kpis": res["kpis"], "kpi_ses": res["kpi_ses"],
        "strip_rows": res["strip_rows"], "gates": res["gates"],
        "warnings": res["warnings"],
        "etd_label_in_filename": res["etd_label_in_filename"],
        "has_batch_prefix": res["has_batch_prefix"],
        "t1": res["seg"]["t1"], "t2": res["seg"]["t2"],
        "sato_p995_self": res["ctx"].extras.get("sato_p995_self"),
        "curtaining_deg": res["ctx"].extras.get("curtaining_deg"),
        "masked_rows_top": res["channels"].masked_rows_top,
        "masked_rows_bottom": res["channels"].masked_rows_bottom,
        "height": res["channels"].bse.shape[0], "width": res["channels"].bse.shape[1],
        "px_nm": res["channels"].px_nm, "px_nm_assumed": res["channels"].px_nm_assumed,
        "shading_flag": res["seg"]["shading_flag"],
    }
    if key[2] is not None:
        from .sim.run import run_sim
        sim = run_sim(res, cfg, full=full_sim, echo=typer.echo)
        data["sim"] = {"row": sim["row"], "bounds": sim["bounds"], "audit": sim["audit"], "full": full_sim}
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({"key": key, "data": data}, default=_jsonable))
    return data, True


def _sim_frames(sims: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sim_df = pd.DataFrame([sm["row"] for sm in sims])
    bounds_df = pd.DataFrame([{"sample_id": sm["row"]["sample_id"], **sm["bounds"]} for sm in sims])
    return sim_df, bounds_df


def _rows_from_results(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    kpi_rows, strip_rows, gate_rows = [], [], []
    for r in results:
        row = {"sample_id": r["sample_id"], "batch": r["batch"]}
        row.update(r["kpis"])
        row.update(r["kpi_ses"])
        kpi_rows.append(row)
        grow = {"sample_id": r["sample_id"], "batch": r["batch"]}
        grow.update(r["gates"])
        gate_rows.append(grow)
        for sr in r["strip_rows"]:
            srow = {"sample_id": r["sample_id"], "batch": r["batch"]}
            srow.update(sr)
            strip_rows.append(srow)
    return pd.DataFrame(kpi_rows), pd.DataFrame(strip_rows), pd.DataFrame(gate_rows)


@app.command("build-baseline")
def build_baseline(
    ctx: typer.Context,
    data: Path = typer.Option("./data", "--data"),
    out: Path = typer.Option("./out", "--out"),
    force: bool = typer.Option(False, "--force"),
):
    """Manifest, kpis.csv, strips.csv, baseline_stats.json, signatures, overlays."""
    from .assign import (
        batch_signatures, choose_hyperparams, fit_centroids, within_batch_distances,
    )
    from .stats import baseline_version, compute_baseline_stats, save_baseline
    from .segment import overlay_png
    from .io import load_sample
    from .segment import segment

    cfg = _cfg(ctx)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "baseline_stats.json").exists() and not force:
        raise typer.Exit(
            "baseline_stats.json already exists; baselines are append-only. "
            "Use --force to rebuild."
        )
    samples = [s for s in discover(data) if s.batch != "unknown" and s.path_bse]
    write_manifest(samples, out / "manifest.csv")
    typer.echo(f"{len(samples)} labelled samples discovered")

    # pass 1: per-image crack thresholds (sato p99.5) to derive the baseline's
    crack_self = []
    results = []
    for s in samples:
        t0 = time.time()
        r, fresh = _process_cached(s, cfg, out, full_sim=True)
        results.append(r)
        if r.get("sato_p995_self") is not None and np.isfinite(r["sato_p995_self"]):
            crack_self.append(r["sato_p995_self"])
        typer.echo(f"  {r['sample_id']} batch {r['batch']} ({time.time()-t0:.0f}s, {'fresh' if fresh else 'cached'})")
    crack_thr = float(np.median(crack_self)) if crack_self else None

    kpis_df, strips_df, gates_df = _rows_from_results(results)
    kpis_df.to_csv(out / "kpis.csv", index=False)
    strips_df.to_csv(out / "strips.csv", index=False)
    gates_df.to_csv(out / "gates.csv", index=False)

    stats = compute_baseline_stats(kpis_df, gates_df)
    base = kpis_df[kpis_df["batch"].astype(str) == "3"]
    stats["baseline_max"] = {
        k: float(base[k].max()) for k in
        ("crack_density_um_per_mm2", "largest_void_ecd_um", "hiz_inclusion_count_mm2")
        if k in base.columns and np.isfinite(base[k].astype(float)).any()
    }
    stats["crack_threshold"] = crack_thr
    stats["version"] = baseline_version(list(base["sample_id"]))
    stats["baseline_sample_ids"] = sorted(base["sample_id"])

    labelled = kpis_df[kpis_df["batch"].astype(str).isin(["1", "2", "3"])]
    delta, T, _ = choose_hyperparams(
        labelled, stats, cfg["shrinkage_grid"], cfg["temperature_grid"])
    stats["delta"], stats["temperature"] = delta, T
    cen = fit_centroids(labelled, stats, delta)
    stats["centroids"] = {b: list(map(float, c)) for b, c in cen["centroids"].items()}
    stats["grand"] = list(map(float, cen["grand"]))
    id_d = within_batch_distances(labelled, stats, delta)
    stats["id_distances"] = id_d
    save_baseline(stats, out)

    sig = batch_signatures(labelled, stats)
    (out / "signatures.json").write_text(json.dumps(sig, indent=2))

    # figures: signatures + overlays
    figs = out / "figs"
    figs.mkdir(exist_ok=True)
    from .report import fig_signatures, fig_kpi_boxplots
    fig_signatures(sig, figs / "signatures.png")
    fig_kpi_boxplots(kpis_df, stats, figs / "kpi_boxplots.png")
    typer.echo(f"baseline: n={stats['n']} version={stats['version']} delta={delta} T={T}")

    if _sim_key(cfg):
        from .sim.baseline import sim_baseline
        from .sim.phantoms import phantom_table
        from .sim.run import finalize_sim
        from .report import fig_dc_sweep
        sims = [r["sim"] for r in results if r.get("sim")]
        for sm in sims:
            finalize_sim(sm, cfg, None, full=False, with_cell=False)
        base_sim = sim_baseline(*_sim_frames(sims), cfg)
        for sm in sims:
            t0 = time.time()
            finalize_sim(sm, cfg, base_sim, full=True)
            typer.echo(f"  sim finalize {sm['row']['sample_id']} ({time.time()-t0:.0f}s)")
        sim_df, bounds_df = _sim_frames(sims)
        base_sim = sim_baseline(sim_df, bounds_df, cfg, phantoms=phantom_table())
        sim_df.to_csv(out / "sim.csv", index=False)
        bounds_df.to_csv(out / "sim_bounds.csv", index=False)
        (out / "sim_baseline.json").write_text(json.dumps(base_sim, indent=2, default=_jsonable))
        fig_dc_sweep(base_sim, None, figs / "dc_sweep.png")
        typer.echo(f"sim baseline: n={base_sim['n']} unstable={base_sim['unstable_indices']} "
                   f"phantoms_pass={base_sim['phantoms']['all_pass']}")


def _sim_for(res: dict, cfg: dict, baseline: Path):
    """Heavy + finalized simulation for a freshly processed sample (evaluate/predict)."""
    if not _sim_key(cfg):
        return None, None
    from .sim.run import finalize_sim, run_sim
    base_sim = _load_sim_baseline(baseline)
    sim = run_sim(res, cfg, full=False, echo=typer.echo)
    finalize_sim(sim, cfg, base_sim, full=False)
    return sim, base_sim


@app.command("evaluate")
def evaluate(
    ctx: typer.Context,
    sample: Path = typer.Option(..., "--sample"),
    baseline: Path = typer.Option("./out", "--baseline"),
    sample_id: str = typer.Option("", "--id", help="pick this sample_id when the folder holds several"),
):
    """Verdict JSON + report.md + report.html + figures for one sample folder."""
    from .report import write_report
    from .stats import load_baseline

    cfg = _cfg(ctx)
    stats = load_baseline(baseline)
    samples = discover(sample)
    # allow pointing at a folder containing one sample, or the sample dir itself
    if not samples:
        # maybe files live directly in `sample`
        samples = discover(sample.parent)
        samples = [s for s in samples if str(s.path_bse).startswith(str(sample))]
    if sample_id:
        samples = [x for x in samples if x.sample_id == sample_id]
    if not samples:
        raise typer.Exit(f"no sample found under {sample}")
    s = samples[0]
    res = process_sample(s, cfg, crack_threshold=stats.get("crack_threshold"))
    sim, base_sim = _sim_for(res, cfg, baseline)
    kpis_df = pd.read_csv(baseline / "kpis.csv")
    strips_df = pd.read_csv(baseline / "strips.csv")
    write_report(res, stats, kpis_df, strips_df, baseline / "reports" / s.sample_id, cfg,
                 sim=sim, sim_base=base_sim)
    typer.echo(f"report: {baseline / 'reports' / s.sample_id}")


@app.command("predict")
def predict(
    ctx: typer.Context,
    folder: Path = typer.Option(..., "--folder"),
    baseline: Path = typer.Option("./out", "--baseline"),
):
    """Evaluate every sample in a folder -> predictions.json + dm.txt."""
    from .report import dm_block, evaluate_result
    from .stats import load_baseline

    cfg = _cfg(ctx)
    stats = load_baseline(baseline)
    kpis_df = pd.read_csv(baseline / "kpis.csv")
    preds = []
    dm_lines = []
    for s in discover(folder):
        if not s.path_bse:
            continue
        res = process_sample(s, cfg, crack_threshold=stats.get("crack_threshold"))
        sim, base_sim = _sim_for(res, cfg, baseline)
        ev = evaluate_result(res, stats, kpis_df, cfg, sim=sim, sim_base=base_sim)
        preds.append(ev)
        dm_lines.append(dm_block(ev))
        typer.echo(dm_lines[-1])
    (baseline / "predictions.json").write_text(json.dumps(preds, indent=2))
    (baseline / "dm.txt").write_text("\n\n".join(dm_lines))
    typer.echo(f"predictions: {baseline / 'predictions.json'}\ndm: {baseline / 'dm.txt'}")


@app.command("score")
def score(
    ctx: typer.Context,
    predictions: Path = typer.Option(..., "--predictions"),
    truth: Path = typer.Option(..., "--truth"),
    out: Path = typer.Option("./out", "--out"),
):
    """Accuracy/log-loss on revealed labels, appended to VALIDATION.md."""
    preds = json.loads(predictions.read_text())
    truth_df = pd.read_csv(truth, dtype={"batch": str})
    truth_map = dict(zip(truth_df["sample_id"], truth_df["batch"]))
    rows = []
    nll = 0.0
    correct = 0
    n = 0
    for p in preds:
        sid = p["sample_id"]
        t = truth_map.get(sid)
        if t is None:
            continue
        probs = p["batch_assignment"]["probabilities"]
        hit = str(p["batch_assignment"]["assigned_batch"]) == str(t)
        correct += int(hit)
        nll += -np.log(max(probs.get(str(t), 1e-9), 1e-9))
        rows.append({"sample_id": sid, "truth": t,
                     "assigned": p["batch_assignment"]["assigned_batch"],
                     "correct": hit, **{f"p{k}": v for k, v in probs.items()}})
        n += 1
    df = pd.DataFrame(rows)
    acc = correct / n if n else float("nan")
    mean_nll = nll / n if n else float("nan")
    txt = (
        f"\n## Test-drop score ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})\n\n"
        f"- n = {n}, accuracy = {acc:.3f}, mean NLL = {mean_nll:.3f}\n\n"
        + df.to_markdown(index=False) + "\n"
    )
    with open("VALIDATION.md", "a") as f:
        f.write(txt)
    typer.echo(txt)


@app.command("quicklook")
def quicklook(
    ctx: typer.Context,
    data: Path = typer.Option("./data", "--data"),
    out: Path = typer.Option("./out/quicklook", "--out"),
):
    """Downscaled 3-channel PNG + histograms per sample."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .io import load_sample
    from .segment import overlay_png, segment

    cfg = _cfg(ctx)
    out.mkdir(parents=True, exist_ok=True)
    for s in discover(data):
        if not s.path_bse:
            continue
        ch = load_sample(s, cfg)
        fig, axes = plt.subplots(2, 4, figsize=(28, 8))
        for j, (name, img) in enumerate(
            (("BSE", ch.bse), ("ETD", ch.etd), ("InLens", ch.inlens))
        ):
            ax = axes[0, j]
            if img is not None:
                ax.imshow(img[::8, ::8], cmap="gray", vmin=0, vmax=255)
            ax.set_title(name)
            ax.axis("off")
            axh = axes[1, j]
            if img is not None:
                vals = img[ch.valid][:: max(1, ch.valid.sum() // 200000)]
                axh.hist(vals, bins=128, range=(0, 255))
            axh.set_title(f"{name} histogram")
        seg = segment(ch, cfg)
        ax = axes[0, 3]
        overlay_rgb = np.repeat(ch.bse[:, :, None].astype(float) / 255, 3, axis=2)
        overlay_rgb[seg["pore"]] = [0.15, 0.45, 0.95]
        overlay_rgb[seg["si"]] = [1.0, 0.55, 0.1]
        overlay_rgb[~ch.valid] *= 0.3
        ax.imshow(overlay_rgb[::8, ::8])
        ax.set_title("segmentation")
        ax.axis("off")
        axes[1, 3].axis("off")
        fig.suptitle(f"{s.sample_id} (batch {s.batch})")
        fig.tight_layout()
        fig.savefig(out / f"{s.sample_id}.png", dpi=80)
        plt.close(fig)
        typer.echo(f"quicklook {s.sample_id}")


@app.command("validate")
def validate(ctx: typer.Context, data: Path = typer.Option("./data", "--data"),
             out: Path = typer.Option("./out", "--out")):
    """LOIO + synthetic + confound tests -> VALIDATION.md."""
    from .validate import run_validation

    cfg = _cfg(ctx)
    run_validation(data, out, cfg)


@app.command("regress")
def regress(
    ctx: typer.Context,
    data: Path = typer.Option("./data", "--data"),
    out: Path = typer.Option("./out", "--out"),
    accept: bool = typer.Option(False, "--accept"),
    only: str = typer.Option("", "--only", help="comma-separated sample_ids to compare"),
):
    """Recompute KPIs and diff against last accepted snapshot."""
    from .regress import run_regress

    cfg = _cfg(ctx)
    run_regress(data, out, cfg, accept, only=[x for x in only.split(",") if x] or None)


if __name__ == "__main__":
    app()


def _triple_checks(r: dict, cfg: dict) -> dict:
    """Per-triple acceptance checks logged by ``qc loop`` (Section 0.3)."""
    from .kpis import SHORTLIST
    k = r["kpis"]
    n_fin = sum(1 for x in SHORTLIST if np.isfinite(k.get(x, float("nan"))))
    c = {
        "height": r["height"], "width": r["width"], "px_nm": round(r["px_nm"], 2),
        "px_ok": abs(r["px_nm"] - 25.0) < 0.5 and not r["px_nm_assumed"],
        "etd_label": r["etd_label_in_filename"],
        "shortlist_finite": n_fin, "shortlist_total": len(SHORTLIST),
        "shading_flag": bool(r["shading_flag"]), "n_warnings": len(r.get("warnings", [])),
    }
    sm = (r.get("sim") or {}).get("row")
    if sm:
        c.update({
            "sim_s": round(sm["sim_time_s"]),
            "uncertain_frac": round(sm["uncertain_frac"], 3),
            "ds_pore_ok": bool(sm["downsample_accepted"]),
            "ds_all_ok": bool(r["sim"]["audit"].get("accepted_all")),
            "flux_ok": abs(1 - sm["flux_balance_TP"]) < 1e-6 if np.isfinite(sm["flux_balance_TP"]) else False,
            "bounds_ok": (r["sim"]["bounds"]["D_eff_rel_TP_L_solid"] <= sm["D_eff_rel_TP"] * 1.001
                          <= r["sim"]["bounds"]["D_eff_rel_TP_L_pore"] * 1.001 + 1e-12),
            "sim_undefined": sm["undefined"],
        })
    # an undefined *electronic* index (carbon phase does not span) is a designed refusal, not a failure;
    # an undefined ionic index would be, because the per-triple transport checks protect that path
    undefined = (sm.get("undefined") or []) if sm else []
    if isinstance(undefined, str):
        undefined = [u.strip() for u in undefined.split(";") if u.strip()]
    ionic_undefined = [u for u in undefined if not str(u).startswith("electronic")]
    c["checks_pass"] = bool(c["px_ok"] and c["shortlist_finite"] >= c["shortlist_total"] - 2
                            and (not sm or (c["flux_ok"] and c["bounds_ok"] and not ionic_undefined)))
    return c


@app.command("loop")
def loop(
    ctx: typer.Context,
    data: Path = typer.Option("./data", "--data"),
    out: Path = typer.Option("./out", "--out"),
    manifest: Path = typer.Option("./drive_manifest.csv", "--manifest"),
    first: str = typer.Option("img_0grcilhi", "--first"),
    limit: int = typer.Option(0, "--limit", help="stop after N new triples (0 = all)"),
    no_download: bool = typer.Option(False, "--no-download"),
):
    """Section 0.3 loop: fetch one triple, process (+sim), check, regress, log; resumable."""
    import hashlib
    from .download import ensure_samples, ingestion_order, load_manifest
    from .kpis import SHORTLIST
    from .regress import run_regress

    cfg = _cfg(ctx)
    out.mkdir(parents=True, exist_ok=True)
    m = load_manifest(manifest)
    order = ingestion_order(m, first)
    log_path = out / "loop_log.csv"
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    done = set(log[log["status"] == "ok"]["sample_id"]) if len(log) else set()
    sl_hash = hashlib.sha1(",".join(SHORTLIST).encode()).hexdigest()[:10]
    n = 0
    for batch, sid in order:
        if sid in done:
            continue
        t0 = time.time()
        if not no_download:
            st = ensure_samples(m, data, [sid], echo=typer.echo)
            if not st["ok"].all():
                typer.echo(f"  {sid}: download failed"); continue
        ss = [x for x in discover(data) if x.sample_id == sid and x.path_bse]
        if not ss:
            typer.echo(f"  {sid}: not discoverable"); continue
        r, fresh = _process_cached(ss[0], cfg, out, full_sim=True)
        checks = _triple_checks(r, cfg)
        try:
            run_regress(data, out, cfg, accept=False, only=sorted(done | {sid}))
            regress = "OK"
        except SystemExit:
            regress = "FAIL"
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": len(done) + 1, "batch": batch, "sample_id": sid,
               "fresh": fresh, "seconds": round(time.time() - t0), "regress": regress,
               "status": "ok" if checks["checks_pass"] and regress == "OK" else "flag",
               "shortlist_hash": sl_hash, **checks}
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
        log.to_csv(log_path, index=False)
        typer.echo(f"[{row['n']:2d}/31] {batch} {sid} {row['status']} regress={regress} "
                   f"{row['seconds']}s finite={checks['shortlist_finite']}/{checks['shortlist_total']} "
                   f"sim={checks.get('sim_s','-')}s unc={checks.get('uncertain_frac','-')} "
                   f"ds={checks.get('ds_pore_ok','-')}/{checks.get('ds_all_ok','-')} undefined='{checks.get('sim_undefined','')}'")
        if row["status"] == "ok":
            done.add(sid)
        n += 1
        if limit and n >= limit:
            break
    typer.echo(f"loop: {len(done)} / {len(order)} triples ok; log {log_path}")
