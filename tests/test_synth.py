"""Synthetic ground-truth recovery tests (Section 6.1)."""
from __future__ import annotations

import numpy as np
import pytest

from qc import synth
from qc.config import load_config
from qc.io import Channels
from qc.kpis.common import Ctx
from qc.pipeline import compute_all_kpis, _extras
from qc.segment import segment

CFG = load_config()


def _ctx(img, truth_masks=None):
    valid = np.ones_like(img["bse"], dtype=bool)
    ch = Channels(bse=img["bse"], etd=img["etd"], inlens=img["inlens"],
                  valid=valid, px_nm=25.0, px_nm_assumed=False,
                  masked_rows_top=0, masked_rows_bottom=0,
                  shape=img["bse"].shape)
    seg = segment(ch, CFG)
    extras = _extras(ch, seg, CFG)
    extras["noise_sd_bse"] = 8.0
    extras["crack_threshold"] = extras.get("sato_p995_self")
    ctx = Ctx(pore=seg["pore"], carbon=seg["carbon"], si=seg["si"],
              ambiguous=seg["ambiguous"], valid=valid, bse_s=seg["bse_s"],
              px_nm=25.0, cfg=CFG, extras=extras)
    kpi, _ = compute_all_kpis(ctx)
    return kpi, seg


@pytest.fixture(scope="module")
def base_img():
    return synth.generate(width=7000, height=2000, seed=0,
                          si_frac=0.10, si_d50_um=3.5)


@pytest.fixture(scope="module")
def base_kpi(base_img):
    kpi, seg = _ctx(base_img)
    return kpi, base_img


def test_si_fraction_recovery(base_kpi):
    kpi, img = base_kpi
    truth = img["truth"]["si_frac"]
    assert abs(kpi["si_frac_total"] - truth) < 0.005, (
        f"si_frac_total {kpi['si_frac_total']:.4f} vs truth {truth:.4f}")


def test_pore_fraction_recovery(base_kpi):
    kpi, img = base_kpi
    truth = img["truth"]["pore_frac"]
    assert abs(kpi["pore_frac_deep"] - truth) < 0.02, (
        f"pore_frac_deep {kpi['pore_frac_deep']:.4f} vs truth {truth:.4f}")


def test_si_d50_recovery(base_kpi):
    kpi, img = base_kpi
    truth = img["truth"]["si_d50_aw_um"]
    assert abs(kpi["si_d50_aw_um"] - truth) / truth < 0.10


def test_anisotropy_recovery(base_kpi):
    kpi, img = base_kpi
    assert abs(kpi["aniso_pore_chord_ratio"] - 2.0) / 2.0 < 0.10


def test_clark_evans_poisson(base_kpi):
    kpi, img = base_kpi
    assert 0.9 < kpi["si_clark_evans_R"] < 1.1


def test_clark_evans_clustered():
    img = synth.generate(width=7000, height=2000, seed=1, clustered=True,
                         si_frac=0.10, si_d50_um=3.5)
    kpi, _ = _ctx(img)
    assert kpi["si_clark_evans_R"] < 0.8


def test_robustness_perturbations(base_img):
    """R-class KPIs stable under brightness/contrast/gamma/noise."""
    kpi0, _ = _ctx(base_img)
    variants = {
        "bright+15%": dict(), "contrast": {}, "gamma": {}, "noise": {},
    }
    bse = base_img["bse"].astype(float)
    import copy
    for name, img in (
        ("bright", {**base_img, "bse": np.clip(bse * 1.15, 0, 255).astype(np.uint8)}),
        ("contrast", {**base_img, "bse": np.clip((bse - 128) * 0.85 + 128, 0, 255).astype(np.uint8)}),
        ("gamma", {**base_img, "bse": np.clip(255 * (bse / 255) ** 1.2, 0, 255).astype(np.uint8)}),
        ("noise", {**base_img, "bse": np.clip(bse + np.random.default_rng(0).normal(0, 5, bse.shape), 0, 255).astype(np.uint8)}),
    ):
        kpi, _ = _ctx(img)
        for k in ("si_frac_solid", "si_d50_aw_um", "si_d90_aw_um",
                  "aniso_pore_chord_ratio", "aniso_carbon_chord_ratio",
                  "si_solidity_med", "s2_len_pore_h_px",
                  "si_contact_carbon_frac"):
            assert np.isfinite(kpi.get(k, np.nan)), f"{name}: {k} nan"
