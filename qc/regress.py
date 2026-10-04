"""qc regress: recompute KPIs, diff vs last accepted snapshot."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import discover
from .kpis import SHORTLIST


def run_regress(data: Path, out: Path, cfg: dict, accept: bool, only: list[str] | None = None):
    """``only``: restrict to these sample_ids (the loop passes the triples processed so far)."""
    snap_dir = out / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    snaps = sorted(snap_dir.glob("kpis_*.csv"))
    stats = json.loads((out / "baseline_stats.json").read_text()) \
        if (out / "baseline_stats.json").exists() else None

    # recompute all samples fresh
    from .cli import _process_cached  # uses pipeline-version keyed cache
    from . import PIPELINE_VERSION
    results = []
    for s in discover(data):
        if not s.path_bse or (only is not None and s.sample_id not in only):
            continue
        r, fresh = _process_cached(s, cfg, out)
        results.append(r)
    rows = []
    for r in results:
        row = {"sample_id": r["sample_id"], "batch": r["batch"]}
        row.update(r["kpis"])
        rows.append(row)
    new = pd.DataFrame(rows).set_index("sample_id").sort_index()

    n = len(snaps)
    if not snaps or accept:
        new.reset_index().to_csv(snap_dir / f"kpis_{n}.csv", index=False)
        print(f"snapshot kpis_{n}.csv written ({len(new)} samples)")
        return

    old = pd.read_csv(snaps[-1]).set_index("sample_id").sort_index()
    common = old.index.intersection(new.index)
    fails = []
    for sid in common:
        for k in SHORTLIST:
            if k not in old.columns or k not in new.columns:
                continue
            o, nw = old.loc[sid, k], new.loc[sid, k]
            if not (np.isfinite(o) and np.isfinite(nw)):
                continue
            scale = stats["kpis"][k]["scale"] if stats else max(abs(o) * 0.05, 1e-9)
            frac_ok = k.endswith("frac") or "frac" in k
            if abs(nw - o) >= 0.1 * scale and not (
                frac_ok and abs(nw - o) < 0.01
            ):
                fails.append((sid, k, o, nw))
    if fails:
        print(f"REGRESS FAILED: {len(fails)} behaviour changes")
        for f_ in fails[:40]:
            print("  ", f_)
        raise SystemExit(1)
    print(f"regress OK vs {snaps[-1].name} ({len(common)} samples)")
