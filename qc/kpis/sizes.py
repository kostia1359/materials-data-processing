"""KPI 2 - Si-candidate particle sizes (watershed split, area-weighted ECD)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import h_maxima, remove_small_objects
from skimage.segmentation import watershed

from .common import Ctx, weighted_quantile


def si_components(si_mask: np.ndarray, min_px: int = 500):
    """Split touching particles; return labelled image of components >= min_px."""
    if si_mask.sum() == 0:
        return np.zeros_like(si_mask, dtype=int)
    edt = ndi.distance_transform_edt(si_mask)
    markers = label(h_maxima(edt, h=2))
    if markers.max() == 0:
        markers, _ = ndi.label(si_mask)
    split = watershed(-edt, markers, mask=si_mask)
    split = remove_small_objects(split, min_size=min_px)
    return split


def ecd_um(area_px: np.ndarray, um_per_px: float) -> np.ndarray:
    return 2 * np.sqrt(area_px / np.pi) * um_per_px


def compute(ctx: Ctx) -> dict:
    min_size = ctx.cfg.get("si_min_px_size", 500)
    min_count = ctx.cfg.get("si_min_px_count", 100)
    lab = si_components(ctx.si, min_size)
    props = regionprops(lab)
    out = {}
    if not props:
        for k in ("si_d50_aw_um", "si_d90_aw_um", "si_d10_aw_um",
                  "si_span", "si_num_density_mm2"):
            out[k] = float("nan")
        out["_si_ecd_um"] = np.array([])
        return out
    areas = np.array([p.area for p in props], dtype=float)
    ecd = ecd_um(areas, ctx.um_per_px)
    d10, d50, d90 = weighted_quantile(ecd, areas, [0.1, 0.5, 0.9])
    # number density uses components >= si_min_px_count (smaller gate)
    lab_count = remove_small_objects(label(ctx.si), min_size=min_count)
    n_count = lab_count.max()
    out.update({
        "si_d50_aw_um": float(d50),
        "si_d90_aw_um": float(d90),
        "si_d10_aw_um": float(d10),
        "si_span": float((d90 - d10) / d50) if d50 else float("nan"),
        "si_num_density_mm2": float(n_count / ctx.area_mm2) if ctx.area_mm2 else float("nan"),
        "_si_ecd_um": ecd,
        "_si_lab": lab,
    })
    return out


def compute_shape(ctx: Ctx, lab: np.ndarray) -> dict:
    """KPI 3 - Si-candidate shape/texture fingerprints."""
    if lab.max() == 0 or ctx.bse_s is None:
        return {k: float("nan") for k in
                ("si_solidity_med", "si_aspect_med", "si_circularity_med",
                 "si_interior_texture")}
    noise_sd = ctx.extras.get("noise_sd_bse", 1.0) or 1.0
    sol, asp, circ, tex = [], [], [], []
    for p in regionprops(lab):
        sol.append(p.solidity)
        asp.append(p.axis_major_length / max(p.axis_minor_length, 1e-9))
        per = p.perimeter_crofton
        circ.append(4 * np.pi * p.area / max(per * per, 1e-9))
        interior = ndi.binary_erosion(lab == p.label, iterations=1)
        src = interior if interior.sum() > 4 else (lab == p.label)
        tex.append(float(ctx.bse_s[src].std() / noise_sd))
    return {
        "si_solidity_med": float(np.median(sol)),
        "si_aspect_med": float(np.median(asp)),
        "si_circularity_med": float(np.median(circ)),
        "si_interior_texture": float(np.median(tex)),
    }
