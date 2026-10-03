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


def _process_cached(sample, cfg, out: Path, crack_threshold=None):
    """Cache per-sample results keyed on (sha1_bse, pipeline_version)."""
    from . import PIPELINE_VERSION

    cp = _cache_path(out, sample.sample_id)
    sha = sha1_of(sample.path_bse) if sample.path_bse else "none"
    if cp.exists():
        rec = json.loads(cp.read_text())
        if rec.get("key") == [sha, PIPELINE_VERSION] and crack_threshold is None:
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
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({"key": [sha, PIPELINE_VERSION], "data": data}))
    return data, True


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
        r, fresh = _process_cached(s, cfg, out)
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


@app.command("evaluate")
def evaluate(
    ctx: typer.Context,
    sample: Path = typer.Option(..., "--sample"),
    baseline: Path = typer.Option("./out", "--baseline"),
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
    if not samples:
        raise typer.Exit(f"no sample found under {sample}")
    s = samples[0]
    res = process_sample(s, cfg, crack_threshold=stats.get("crack_threshold"))
    kpis_df = pd.read_csv(baseline / "kpis.csv")
    strips_df = pd.read_csv(baseline / "strips.csv")
    write_report(res, stats, kpis_df, strips_df, baseline / "reports" / s.sample_id, cfg)
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
        ev = evaluate_result(res, stats, kpis_df, cfg)
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
):
    """Recompute KPIs and diff against last accepted snapshot."""
    from .regress import run_regress

    cfg = _cfg(ctx)
    run_regress(data, out, cfg, accept)


if __name__ == "__main__":
    app()
