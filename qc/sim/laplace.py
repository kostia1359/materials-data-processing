"""5-point finite-volume Laplace solver on the fused label map (Section 10.3).

TauFactor conventions: Dirichlet planes at half-pixel spacing (C=1 inlet,
C=0 outlet), no-flux lateral sides, harmonic-mean interface conductances,
D_eff/D = Q*L/(dC*A), tau = eps*D/D_eff.  A percolation check runs first and
the solve is refused (``undefined``) when no conducting path spans the frame.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage as ndi

from . import CARBON, PORE, SI, UNCERTAIN

_FLOOR = 1e-9  # relative conductance given to insulating pixels so A is SPD


def percolation_check(mask: np.ndarray, axis: int = 0) -> dict:
    """Does any 4-connected component of ``mask`` touch both planes along ``axis``?"""
    lab, n = ndi.label(mask)
    if n == 0:
        return {"spans": False, "n_components": 0, "spanning_frac": 0.0,
                "reason": "empty conducting set"}
    first = np.unique(lab.take(0, axis=axis))
    last = np.unique(lab.take(-1, axis=axis))
    span = np.intersect1d(first, last)
    span = span[span > 0]
    if span.size == 0:
        return {"spans": False, "n_components": int(n), "spanning_frac": 0.0,
                "reason": f"{n} components, none spanning along axis {axis}"}
    frac = float(np.isin(lab, span).sum() / mask.sum())
    return {"spans": True, "n_components": int(n), "spanning_frac": frac, "reason": ""}


def _assemble(cond: np.ndarray):
    H, W = cond.shape
    n = H * W
    idx = np.arange(n).reshape(H, W)

    def harm(a, b):
        return 2.0 * a * b / (a + b)

    gv = harm(cond[:-1, :], cond[1:, :]).ravel()
    iv, jv = idx[:-1, :].ravel(), idx[1:, :].ravel()
    gh = harm(cond[:, :-1], cond[:, 1:]).ravel()
    ih, jh = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    rows = np.concatenate([iv, jv, iv, jv, ih, jh, ih, jh])
    cols = np.concatenate([jv, iv, iv, jv, jh, ih, ih, jh])
    vals = np.concatenate([-gv, -gv, gv, gv, -gh, -gh, gh, gh])
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    # Dirichlet at half-pixel: boundary cell to plane conductance = 2*cond
    d = np.zeros(n)
    b = np.zeros(n)
    top, bot = idx[0, :], idx[-1, :]
    d[top] = 2.0 * cond[0, :]
    b[top] = 2.0 * cond[0, :] * 1.0
    d[bot] = 2.0 * cond[-1, :]
    A = (A + sp.diags(d)).tocsr()
    return A, b, top, bot


def fv_laplace(cond_map: np.ndarray, axis: int = 0, solver: str = "spsolve", tol: float = 1e-8) -> dict:
    """Solve steady diffusion with unit concentration drop along ``axis``.

    ``cond_map`` holds relative conductivities (0 = insulating).  Returns
    D_eff_rel = D_eff/D (dimensionless, in [0, 1]), inlet/outlet fluxes and
    the solution field.  Refuses (``undefined``) when the conducting set does
    not span.
    """
    cond = np.asarray(cond_map, dtype=np.float64)
    if axis == 1:
        cond = cond.T
    perc = percolation_check(cond > 0, axis=0)
    if not perc["spans"]:
        return {"undefined": True, "reason": perc["reason"], "percolation": perc,
                "D_eff_rel": float("nan"), "Q_in": float("nan"), "Q_out": float("nan")}
    H, W = cond.shape
    c = np.where(cond > 0, cond, _FLOOR * max(cond.max(), 1e-30))
    A, b, top, bot = _assemble(c)
    if solver == "amg_cg":
        import pyamg
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=500)
        x = ml.solve(b, tol=tol, accel="cg", maxiter=500)
    else:
        x = spla.spsolve(A.tocsc(), b)
    Q_in = float(np.sum(2.0 * c[0, :] * (1.0 - x[top])))
    Q_out = float(np.sum(2.0 * c[-1, :] * (x[bot] - 0.0)))
    # D_eff/D = Q * L / (dC * A) with L = H, A = W, dC = 1 (pixel units, unit D)
    D_eff_rel = Q_in * H / W
    field = x.reshape(H, W)
    if axis == 1:
        field = field.T
    return {"undefined": False, "D_eff_rel": float(D_eff_rel), "Q_in": Q_in, "Q_out": Q_out,
            "flux_balance": float(Q_out / Q_in) if Q_in else float("nan"),
            "percolation": perc, "field": field, "n_unknowns": int(H * W)}


def _cond_from_labels(labels, pore, carbon, si, uncertain):
    c = np.zeros(labels.shape, dtype=np.float64)
    c[labels == PORE] = pore
    c[labels == CARBON] = carbon
    c[labels == SI] = si
    c[labels == UNCERTAIN] = uncertain
    return c


def ionic_index(labels: np.ndarray, D_c: float = 0.05, sigma_si: float = 0.0,
                solver: str = "spsolve", in_plane: bool = True) -> dict:
    """Two-conductivity ionic transport index: pore 1, carbon D_c, Si sigma_si.

    tau_p = eps / D_eff_rel (TauFactor definition with eps = pore fraction);
    N_M = 1 / D_eff_rel (MacMullin number).  Absolute levels are set by D_c;
    only ratios to the baseline and the in-plane/through-plane ratio are
    meaningful (caveat 10).
    """
    cond = _cond_from_labels(labels, 1.0, D_c, sigma_si, D_c)
    eps = float((labels == PORE).mean())
    tp = fv_laplace(cond, axis=0, solver=solver)
    out = {"eps_pore": eps, "D_c": D_c,
           "pore_percolation": percolation_check(labels == PORE, 0),
           "D_eff_rel_TP": tp["D_eff_rel"], "flux_balance_TP": tp.get("flux_balance", float("nan")),
           "undefined_TP": tp["undefined"], "reason_TP": tp.get("reason", "")}
    if in_plane:
        ip = fv_laplace(cond, axis=1, solver=solver)
        out.update({"D_eff_rel_IP": ip["D_eff_rel"], "flux_balance_IP": ip.get("flux_balance", float("nan")),
                    "undefined_IP": ip["undefined"]})
        out["aniso_ratio"] = ip["D_eff_rel"] / tp["D_eff_rel"] if tp["D_eff_rel"] else float("nan")
    if np.isfinite(tp["D_eff_rel"]) and tp["D_eff_rel"] > 0:
        out["tau_p"] = eps / tp["D_eff_rel"]
        out["N_M"] = 1.0 / tp["D_eff_rel"]
    else:
        out["tau_p"] = out["N_M"] = float("nan")
    return out


def electronic_index(labels: np.ndarray, sigma_si: float = 0.0, solver: str = "spsolve") -> dict:
    """Electronic index through the carbon matrix: carbon 1, pore 0, Si sigma_si."""
    cond = _cond_from_labels(labels, 0.0, 1.0, sigma_si, 1.0)
    tp = fv_laplace(cond, axis=0, solver=solver)
    carbon = labels == CARBON
    perc = percolation_check(carbon | (labels == UNCERTAIN), 0)
    si = labels == SI
    if si.any():
        ring = ndi.binary_dilation(si, structure=ndi.generate_binary_structure(2, 1)) & ~si
        exposed = float((ring & (labels == PORE)).sum() / max(ring.sum(), 1))
    else:
        exposed = float("nan")
    return {"sigma_eff_rel_TP": tp["D_eff_rel"], "undefined": tp["undefined"],
            "reason": tp.get("reason", ""), "flux_balance": tp.get("flux_balance", float("nan")),
            "carbon_spanning_frac": perc["spanning_frac"], "exposed_si_frac": exposed}
