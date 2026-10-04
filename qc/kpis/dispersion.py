"""KPI 4 - Si dispersion: quadrat CV, Clark-Evans, agglomerate share."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import regionprops
from skimage.morphology import binary_closing, disk, label, remove_small_objects

from .common import Ctx


def compute(ctx: Ctx, si_lab: np.ndarray | None = None) -> dict:
    q = ctx.cfg.get("quadrat_px", 700)
    valid = ctx.valid
    si = ctx.si

    # quadrat CV of Si fraction over q x q grid (floor division)
    h, w = si.shape
    fracs = []
    for y0 in range(0, h - q + 1, q):
        for x0 in range(0, w - q + 1, q):
            vq = valid[y0 : y0 + q, x0 : x0 + q]
            if vq.sum() < 0.5 * q * q:
                continue
            fracs.append(si[y0 : y0 + q, x0 : x0 + q][vq].mean())
    fracs = np.asarray(fracs)
    cv = float(fracs.std() / fracs.mean()) if fracs.size and fracs.mean() > 0 else float("nan")

    # Clark-Evans R on sized-component centroids
    ce = float("nan")
    cents = np.array([])
    if si_lab is not None and si_lab.max() >= 2:
        cents = np.array([p.centroid for p in regionprops(si_lab)])
        tree = cKDTree(cents)
        d, _ = tree.query(cents, k=2)
        nn = d[:, 1] * ctx.um_per_px
        n = len(cents)
        lam = n / ctx.area_um2
        # Donnelly edge correction for finite windows
        ys, xs = np.where(ctx.valid)
        per_um = 2 * ((ys.max() - ys.min() + 1) + (xs.max() - xs.min() + 1)) * ctx.um_per_px
        edge_corr = (0.0514 + 0.0412 / np.sqrt(n)) * per_um / n
        ce = float(nn.mean() / (0.5 / np.sqrt(lam) + edge_corr)) if lam > 0 else float("nan")

    # agglomerate share: Si area in components > 3x median component area
    agg = float("nan")
    closed = binary_closing(si, disk(2))
    lab2 = label(closed)
    lab2 = remove_small_objects(lab2, min_size=ctx.cfg.get("si_min_px_size", 500))
    if lab2.max() > 0:
        areas = np.array([p.area for p in regionprops(lab2)], dtype=float)
        big = areas > 3 * np.median(areas)
        agg = float(areas[big].sum() / areas.sum())

    return {
        "si_quadrat_cv": cv,
        "si_clark_evans_R": ce,
        "si_agglomerate_frac": agg,
        "_si_centroids_um": cents * ctx.um_per_px if cents.size else cents,
    }
