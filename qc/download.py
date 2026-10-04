"""Manifest-driven download: drive_manifest.csv -> data/<batch>/<file>.

Each row carries a Drive file id and the verified byte size; a file is
re-fetched only when missing or size-mismatched (resume), via gdown by id.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

MANIFEST_COLS = ["batch", "sample_id", "channel", "filename", "file_id", "bytes"]


def load_manifest(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path)
    missing = [c for c in MANIFEST_COLS if c not in m.columns]
    if missing:
        raise ValueError(f"manifest lacks columns {missing}")
    return m


def status(m: pd.DataFrame, data: Path) -> pd.DataFrame:
    rows = []
    for r in m.itertuples():
        p = data / r.batch / r.filename
        have = p.stat().st_size if p.exists() else 0
        rows.append({"batch": r.batch, "sample_id": r.sample_id, "filename": r.filename,
                     "expected": int(r.bytes), "on_disk": have,
                     "ok": have == int(r.bytes)})
    return pd.DataFrame(rows)


def fetch_file(file_id: str, dest: Path, expected: int, retries: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(retries):
        subprocess.run(
            [sys.executable, "-m", "gdown", "--quiet", "--fuzzy",
             f"https://drive.google.com/uc?id={file_id}", "-O", str(dest)],
            check=False,
        )
        if dest.exists() and dest.stat().st_size == expected:
            return True
        if dest.exists():
            dest.unlink()
    return False


def ensure_samples(m: pd.DataFrame, data: Path, sample_ids: list[str] | None = None,
                   echo=print) -> pd.DataFrame:
    """Download (or verify) every file of the requested samples; returns status."""
    sub = m if sample_ids is None else m[m.sample_id.isin(sample_ids)]
    st = status(sub, data)
    for r in sub.itertuples():
        p = data / r.batch / r.filename
        if p.exists() and p.stat().st_size == int(r.bytes):
            continue
        echo(f"  fetching {r.batch}/{r.filename} ({int(r.bytes)/1e6:.1f} MB)")
        ok = fetch_file(r.file_id, p, int(r.bytes))
        if not ok:
            echo(f"  FAILED {r.filename}")
    return status(sub, data)


def ingestion_order(m: pd.DataFrame, first: str | None = None) -> list[tuple[str, str]]:
    """(batch, sample_id) in Section 0.3 order: attached triple, Batch_3, Batch_1, Batch_2."""
    order = []
    for b in ("Batch_3", "Batch_1", "Batch_2"):
        ids = sorted(m[m.batch == b].sample_id.unique())
        order += [(b, s) for s in ids]
    if first:
        order.sort(key=lambda t: t[1] != first)
    return order
