"""PyBaMM composite graphite-Si DFN runs (Section 10.3, ``cell.py``).

Relative rate-capability and plating-indicator shifts under a frozen
LG-M50-type parameter set (Chen2020_composite); never absolute predictions
(caveat 11).  Tortuosity enters via "transport efficiency": "tortuosity
factor" with the sample's tau_p rescaled so the baseline median maps to
Bruggeman 1.5 (eps^-0.5).
"""
from __future__ import annotations

import numpy as np

_OPTS_BASE = {"particle phases": ("2", "1"), "transport efficiency": "tortuosity factor"}


def _params(pv, eps, f_si, r_si_um, tau, ccfg, L_um):
    pv.update({
        "Negative electrode porosity": eps,
        "Negative electrode tortuosity factor (electrolyte)": tau,
        # Chen2020 uses Bruggeman 0 for the solid phase (efficiency 1 = eps_s / tau_s) -> tau_s = eps_s
        "Negative electrode tortuosity factor (electrode)": (1 - eps) * (1 - ccfg["cbd"]),
        "Positive electrode tortuosity factor (electrolyte)": pv["Positive electrode porosity"] ** -0.5,
        "Positive electrode tortuosity factor (electrode)": pv["Positive electrode active material volume fraction"],
        "Separator tortuosity factor (electrolyte)": pv["Separator porosity"] ** -0.5,
        "Primary: Negative electrode active material volume fraction": (1 - eps) * (1 - ccfg["cbd"]) * (1 - f_si),
        "Secondary: Negative electrode active material volume fraction": (1 - eps) * (1 - ccfg["cbd"]) * f_si,
        "Secondary: Negative particle radius [m]": r_si_um * 1e-6 * ccfg["si_radius_factor"],
        "Negative electrode thickness [m]": L_um * 1e-6,
    }, check_already_exists=False)
    return pv


def pybamm_run(eps: float, f_si: float, r_si_um: float, tau: float, ccfg: dict, L_um: float,
               rates=(1, 3), hysteresis: bool = False) -> dict:
    import pybamm
    pybamm.set_logging_level("ERROR")
    opts = dict(_OPTS_BASE)
    opts["open-circuit potential"] = (("single", "current sigmoid" if hysteresis else "single"), "single")
    model = pybamm.lithium_ion.DFN(opts)
    pv = _params(pybamm.ParameterValues(ccfg["parameter_set"]), eps, f_si, r_si_um, tau, ccfg, L_um)
    steps = []
    for r in rates:
        steps += ["Discharge at C/10 until 2.5 V", "Rest for 15 minutes",
                  f"Charge at {r}C until 4.2 V", "Rest for 15 minutes"]
    sim = pybamm.Simulation(model, parameter_values=pv, experiment=pybamm.Experiment([tuple(steps)]),
                            solver=pybamm.IDAKLUSolver())
    sol = sim.solve()
    st = sol.cycles[0].steps
    q10 = abs(st[0]["Discharge capacity [A.h]"].entries[-1] - st[0]["Discharge capacity [A.h]"].entries[0])
    out = {"Q_C10_Ah": float(q10)}
    for i, r in enumerate(rates):
        s = st[4 * i + 2]
        qc = abs(s["Discharge capacity [A.h]"].entries[-1] - s["Discharge capacity [A.h]"].entries[0])
        out[f"Q_CC_{r}C_over_Q_C10"] = float(qc / q10)
        eta = s["Negative electrode surface potential difference at separator interface [V]"].entries
        out[f"min_eta_sep_{r}C_V"] = float(eta.min())
        cap = np.abs(s["Discharge capacity [A.h]"].entries - s["Discharge capacity [A.h]"].entries[0])
        cross = np.where(eta <= 0)[0]
        out[f"cap_at_eta0_{r}C_over_Q_C10"] = float(cap[cross[0]] / q10) if cross.size else float("nan")
        out[f"min_ce_{r}C_molm3"] = float(s["Electrolyte concentration [mol.m-3]"].entries.min())
    # N/P at the end of formation is a model constant here; report the input share
    return out


def pybamm_runs(eps, f_si, r_si_um, tau_tp_index, tau_baseline_med, ccfg, ecfg, bounds: dict | None = None,
                ses: dict | None = None, rates=(1, 3)) -> dict:
    """Point run + +/-SE runs on eps, f_Si, R_Si, tau (Section 10.3).

    ``tau`` fed to PyBaMM = eps^-0.5 * (tau_tp_index / tau_baseline_med), i.e. the
    baseline median maps to Bruggeman 1.5.
    """
    def tau_of(t, e):
        ratio = t / tau_baseline_med if (tau_baseline_med and np.isfinite(t)) else 1.0
        return e ** -0.5 * ratio

    L_um = ecfg["L_um"]
    ccfg = dict(ccfg, cbd=ecfg["cbd_frac_of_solids"])
    out = {"point": pybamm_run(eps, f_si, r_si_um, tau_of(tau_tp_index, eps), ccfg, L_um, rates)}
    ses = ses or {}
    for name, (val, se) in {
        "eps": (eps, ses.get("eps")), "f_si": (f_si, ses.get("f_si")),
        "r_si": (r_si_um, ses.get("r_si")), "tau": (tau_tp_index, ses.get("tau")),
    }.items():
        if se is None or not np.isfinite(se) or se == 0:
            continue
        for sgn in (-1, 1):
            kw = {"eps": eps, "f_si": f_si, "r_si_um": r_si_um, "tau": tau_tp_index}
            kw[name if name != "r_si" else "r_si_um"] = val + sgn * se
            kw["eps"] = float(np.clip(kw["eps"], 0.05, 0.6))
            try:
                out[f"{name}{'+' if sgn > 0 else '-'}SE"] = pybamm_run(
                    kw["eps"], kw["f_si"], kw["r_si_um"], tau_of(kw["tau"], kw["eps"]), ccfg, L_um, rates)
            except Exception as e:  # solver cut-offs are reported, not hidden
                out[f"{name}{'+' if sgn > 0 else '-'}SE"] = {"error": str(e)[:200]}
    if bounds:
        for k, (e_b, t_b) in bounds.items():
            try:
                out[f"bound_{k}"] = pybamm_run(e_b, f_si, r_si_um, tau_of(t_b, e_b), ccfg, L_um, rates)
            except Exception as e:
                out[f"bound_{k}"] = {"error": str(e)[:200]}
    return out
