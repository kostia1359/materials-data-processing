import numpy as np
import pytest

from qc.sim import CARBON, PORE, SI, UNCERTAIN
from qc.sim.laplace import electronic_index, fv_laplace, ionic_index, percolation_check
from qc.sim.phantoms import keller_checkerboard, serpentine, straight_channels, tilted_channel
from qc.sim.fuse import downsample_labels
from qc.sim.swell import swell


def _cond(lab, pore=1.0, carbon=0.0, si=0.0):
    c = np.zeros(lab.shape)
    c[lab == PORE] = pore
    c[lab == CARBON] = carbon
    c[lab == SI] = si
    return c


def test_straight_channels_tau_one():
    lab = straight_channels()
    r = fv_laplace(_cond(lab), axis=0)
    eps = (lab == PORE).mean()
    assert not r["undefined"]
    assert abs(r["D_eff_rel"] / eps - 1) < 1e-6
    assert abs(r["flux_balance"] - 1) < 1e-6


def test_tilted_channel():
    lab = tilted_channel(theta_deg=30)
    r = fv_laplace(_cond(lab), axis=0)
    eps = (lab == PORE).mean()
    tau = eps / r["D_eff_rel"]
    assert abs(tau - 1 / np.cos(np.radians(30)) ** 2) / (1 / np.cos(np.radians(30)) ** 2) < 0.08


def test_serpentine():
    lab, ratio = serpentine()
    r = fv_laplace(_cond(lab), axis=0)
    eps = (lab == PORE).mean()
    tau = eps / r["D_eff_rel"]
    assert abs(tau - ratio ** 2) / ratio ** 2 < 0.12


def test_keller_checkerboard():
    # corner singularities make the FV estimate converge slowly from below;
    # 16 x 16 cells of 32 px give ~5 %, finer grids approach sqrt(s1*s2).
    lab = keller_checkerboard(512, 512, 32)
    s1, s2 = 1.0, 0.1
    r = fv_laplace(_cond(lab, pore=s1, carbon=s2), axis=0)
    assert abs(r["D_eff_rel"] - np.sqrt(s1 * s2)) / np.sqrt(s1 * s2) < 0.06


def test_percolation_refusal():
    lab = np.full((100, 100), SI, dtype=np.uint8)
    lab[20:40, :] = PORE  # horizontal slab: spans in-plane only
    assert not percolation_check(lab == PORE, 0)["spans"]
    assert percolation_check(lab == PORE, 1)["spans"]
    r = fv_laplace(_cond(lab), axis=0)
    assert r["undefined"] and np.isnan(r["D_eff_rel"])


def test_inequalities_bounds_and_reciprocity():
    rng = np.random.default_rng(0)
    lab = np.where(rng.random((120, 160)) < 0.3, PORE, CARBON).astype(np.uint8)
    lab[rng.random(lab.shape) < 0.05] = SI
    unc = rng.random(lab.shape) < 0.1
    L_solid = np.where(unc, CARBON, lab).astype(np.uint8)
    L_pore = np.where(unc, PORE, lab).astype(np.uint8)
    L_mid = lab
    d = [ionic_index(L, D_c=0.05, in_plane=False)["D_eff_rel_TP"] for L in (L_solid, L_mid, L_pore)]
    assert d[0] <= d[1] + 1e-12 <= d[2] + 2e-12
    eps_rel = (L_pore == PORE).mean() + 0.05 * (L_pore != PORE).mean()
    assert d[2] <= eps_rel + 1e-9  # D_eff/D <= volume-average bound
    r = ionic_index(L_mid, D_c=0.05)
    assert abs(r["flux_balance_TP"] - 1) < 1e-6
    flipped = fv_laplace(_cond(L_mid, carbon=0.05)[::-1, :], axis=0)["D_eff_rel"]
    assert abs(flipped - r["D_eff_rel_TP"]) < 1e-8


def test_electronic_index_runs():
    rng = np.random.default_rng(1)
    lab = np.where(rng.random((80, 100)) < 0.2, PORE, CARBON).astype(np.uint8)
    lab[30:40, 30:40] = SI
    r = electronic_index(lab)
    assert not r["undefined"] and 0 < r["sigma_eff_rel_TP"] <= 1
    assert 0 <= r["exposed_si_frac"] <= 1


def test_downsample_audit():
    rng = np.random.default_rng(2)
    lab = np.where(rng.random((400, 400)) < 0.3, PORE, CARBON).astype(np.uint8)
    ds, audit = downsample_labels(lab, 4)
    assert ds.shape == (100, 100)
    assert set(np.unique(ds)) <= {PORE, CARBON, SI, UNCERTAIN}
    assert "pore_euler" in audit["before"] and "accepted" in audit


def test_swelling_area_conservation():
    lab = np.full((200, 200), CARBON, dtype=np.uint8)
    yy, xx = np.indices(lab.shape)
    lab[(yy - 100) ** 2 + (xx - 100) ** 2 < 20 ** 2] = SI
    lab[(yy - 100) ** 2 + (xx - 100) ** 2 < 20 ** 2] = SI
    lab[(yy - 100) ** 2 + (xx - 140) ** 2 < 10 ** 2] = PORE
    a0 = (lab == SI).sum()
    r = swell(lab, 1.6, mode="pore_first", um_per_px=0.1)
    a1 = (r["labels"] == SI).sum()
    assert abs(a1 / a0 - 1.6) < 0.08
    assert 0 <= r["pore_closure_frac"] <= 1
    assert r["si_touch_frac"] == 0.0


def test_pybamm_tau_equals_bruggeman():
    """tau = eps^-0.5 via 'tortuosity factor' must reproduce the Bruggeman-1.5 default (dQ_CC < 0.5 %)."""
    pybamm = pytest.importorskip("pybamm")
    from qc.sim.cell import _params
    eps, f_si, r_um, L = 0.30, 0.09, 2.2, 70.0
    ccfg = {"parameter_set": "Chen2020_composite", "si_radius_factor": 1.273, "cbd": 0.08}
    opts = {"particle phases": ("2", "1")}
    res = {}
    for name, extra in (("brugg", {}), ("tau", {"transport efficiency": "tortuosity factor"})):
        model = pybamm.lithium_ion.DFN({**opts, **extra})
        pv = pybamm.ParameterValues("Chen2020_composite")
        if extra:
            pv = _params(pv, eps, f_si, r_um, eps ** -0.5, ccfg, L)
        else:
            pv.update({"Negative electrode porosity": eps,
                       "Primary: Negative electrode active material volume fraction": (1 - eps) * 0.92 * (1 - f_si),
                       "Secondary: Negative electrode active material volume fraction": (1 - eps) * 0.92 * f_si,
                       "Secondary: Negative particle radius [m]": r_um * 1e-6 * 1.273,
                       "Negative electrode thickness [m]": L * 1e-6}, check_already_exists=False)
        exp = pybamm.Experiment([("Discharge at C/10 until 2.5 V", "Rest for 15 minutes", "Charge at 3C until 4.2 V")])
        sol = pybamm.Simulation(model, parameter_values=pv, experiment=exp, solver=pybamm.IDAKLUSolver()).solve()
        st = sol.cycles[0].steps
        q = lambda s: abs(s["Discharge capacity [A.h]"].entries[-1] - s["Discharge capacity [A.h]"].entries[0])
        res[name] = q(st[2]) / q(st[0])
    assert abs(res["tau"] - res["brugg"]) / res["brugg"] < 0.005, res
