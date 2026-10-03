"""Loader/discovery unit tests."""
from __future__ import annotations

import re

import numpy as np

from qc.io import SAMPLE_RE, batch_from_path, detect_edge_bands


def test_filename_regex():
    m = SAMPLE_RE.match("img_abc123_BSE.tif")
    assert m and m.group("sid") == "abc123" and m.group("det") == "BSE"
    m = SAMPLE_RE.match("Batch_3_img_x9_Inlens.tif")
    assert m and m.group("pb") == "3" and m.group("det").upper() == "INLENS"
    m = SAMPLE_RE.match("img_q_SE.tif")
    assert m and m.group("det") == "SE"
    assert not SAMPLE_RE.match("readme.txt")
    assert not SAMPLE_RE.match("img_x.png")


def test_batch_from_path():
    from pathlib import Path
    assert batch_from_path(Path("data/Batch_2/img_x_BSE.tif")) == "2"
    assert batch_from_path(Path("data/batch3/img_x_BSE.tif")) == "3"
    assert batch_from_path(Path("data/test/img_x_BSE.tif")) == "unknown"


def test_edge_bands_clean_image():
    rng = np.random.default_rng(0)
    il = rng.normal(70, 5, (200, 400)).astype(np.uint8)
    etd = rng.normal(80, 10, (200, 400)).astype(np.uint8)
    top, bottom = detect_edge_bands(il, etd)
    assert top == 0 and bottom == 0


def test_edge_bands_nodular():
    rng = np.random.default_rng(0)
    il = rng.normal(70, 5, (200, 400)).astype(np.uint8)
    etd = rng.normal(80, 10, (200, 400)).astype(np.uint8)
    il[:15] = 10  # dark InLens rows
    # high variance ETD rows: checkerboard
    etd[:15] = (np.indices((15, 400)).sum(0) % 2 * 200).astype(np.uint8)
    top, bottom = detect_edge_bands(il, etd)
    assert top >= 10
    assert bottom == 0
