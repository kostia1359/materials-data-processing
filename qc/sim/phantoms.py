"""Exact-answer phantoms for the Laplace solver (Section 10.3)."""
from __future__ import annotations

import numpy as np

from . import CARBON, PORE, SI


def straight_channels(H=200, W=200, n=5, width=10):
    """Vertical pore channels in an insulating solid: tau = 1, D_eff/D = eps."""
    lab = np.full((H, W), SI, dtype=np.uint8)
    for i in range(n):
        x0 = int((i + 0.5) * W / n) - width // 2
        lab[:, x0:x0 + width] = PORE
    return lab


def tilted_channel(H=400, W=400, theta_deg=30.0, width=12):
    """One channel tilted by theta from the through-plane axis: tau = 1/cos^2(theta)."""
    lab = np.full((H, W), SI, dtype=np.uint8)
    t = np.tan(np.radians(theta_deg))
    ys = np.arange(H)
    xc = W / 2 + (ys - H / 2) * t
    for y in range(H):
        lo, hi = int(round(xc[y] - width / 2)), int(round(xc[y] + width / 2))
        lab[y, max(lo, 0):min(hi, W)] = PORE
    return lab


def serpentine(H=300, W=300, width=12, n_turns=3):
    """A single serpentine channel of path length L_path: tau = (L_path/L)^2."""
    lab = np.full((H, W), SI, dtype=np.uint8)
    seg_h = H // (n_turns + 1)
    x = width
    path = 0
    y = 0
    for k in range(n_turns + 1):
        y1 = min(y + seg_h, H)
        lab[y:y1, x:x + width] = PORE
        path += y1 - y
        if k < n_turns:
            x_new = W - 2 * width if k % 2 == 0 else width
            xl, xr = sorted((x, x_new))
            lab[y1 - width:y1, xl:xr + width] = PORE
            path += abs(x_new - x)
            x = x_new
        y = y1
    lab[y - 1:, x:x + width] = PORE
    return lab, path / H


def keller_checkerboard(H=256, W=256, cell=4):
    """Two-phase checkerboard: sigma_eff = sqrt(sigma1*sigma2) (Keller)."""
    yy, xx = np.indices((H, W))
    board = ((yy // cell + xx // cell) % 2).astype(np.uint8)
    return np.where(board == 0, PORE, CARBON).astype(np.uint8)


def phantom_table() -> dict:
    """Exact-solution checks used in VALIDATION.md and sim_baseline.json."""
    from .laplace import fv_laplace
    from . import PORE, CARBON

    def cond(lab, pore=1.0, carbon=0.0):
        c = np.zeros(lab.shape, float)
        c[lab == PORE] = pore
        c[lab == CARBON] = carbon
        return c

    rows = []
    lab = straight_channels()
    r = fv_laplace(cond(lab), axis=0)
    eps = float((lab == PORE).mean())
    rows.append({"phantom": "straight channels", "quantity": "tau", "target": 1.0,
                 "value": float(eps / r["D_eff_rel"]), "tol": 0.01})
    theta = 30.0
    lab = tilted_channel(theta_deg=theta)
    r = fv_laplace(cond(lab), axis=0)
    eps = float((lab == PORE).mean())
    rows.append({"phantom": f"tilted channel {theta:g} deg", "quantity": "tau",
                 "target": float(1 / np.cos(np.radians(theta)) ** 2),
                 "value": float(eps / r["D_eff_rel"]), "tol": 0.08})  # 12 px staircase channel
    lab, lpath = serpentine()
    r = fv_laplace(cond(lab), axis=0)
    eps = float((lab == PORE).mean())
    rows.append({"phantom": "serpentine", "quantity": "tau", "target": float(lpath ** 2),
                 "value": float(eps / r["D_eff_rel"]), "tol": 0.10})
    s1, s2 = 1.0, 0.1
    for cell in (8, 16, 32):
        lab = keller_checkerboard(512, 512, cell)
        r = fv_laplace(cond(lab, pore=s1, carbon=s2), axis=0)
        rows.append({"phantom": f"Keller checkerboard cell={cell}px", "quantity": "sigma_eff",
                     "target": float(np.sqrt(s1 * s2)), "value": float(r["D_eff_rel"]),
                     "tol": 0.06 if cell == 32 else None})
    for row in rows:
        row["rel_err"] = float(abs(row["value"] - row["target"]) / row["target"])
        row["pass"] = (row["rel_err"] <= row["tol"]) if row["tol"] is not None else None
    return {"rows": rows, "all_pass": all(r["pass"] for r in rows if r["pass"] is not None)}
