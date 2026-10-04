"""Geometric Si swelling scenarios on the fused label map (Section 10.3).

Every Si component grows by iterative 1-px rings until its area reaches
f_A x A0, consuming pore pixels first (``pore_first``) or isotropically
(``isotropic``); carbon pixels that would be displaced are counted.
Particles touching the frame edge (``edge_hit``) are excluded from the
per-particle statistics.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import CARBON, PORE, SI, UNCERTAIN

_S4 = ndi.generate_binary_structure(2, 1)


def swell(labels: np.ndarray, f_A: float, mode: str = "pore_first", um_per_px: float = 0.1,
          max_iter: int = 60, hotspot_um: float = 5.0) -> dict:
    lab = labels.copy()
    si0 = lab == SI
    comp0, n0 = ndi.label(si0, structure=_S4)
    if n0 == 0:
        return {"labels": lab, "pore_closure_frac": float("nan"), "constraint_index": float("nan"),
                "constraint_index_aw": float("nan"), "buffer_sufficiency_frac": float("nan"),
                "si_touch_frac": float("nan"), "si_merges": 0, "largest_merged_cluster_ecd_um": float("nan"),
                "si_area_in_clusters_gt_hotspot": float("nan"), "dH_H_bound": float("nan"), "n_particles": 0}
    areas0 = np.bincount(comp0.ravel())[1:].astype(float)
    target = f_A * areas0
    H, W = lab.shape
    edge_ids = np.unique(np.concatenate([comp0[0], comp0[-1], comp0[:, 0], comp0[:, -1]]))
    edge_hit = np.zeros(n0 + 1, bool)
    edge_hit[edge_ids] = True
    edge_hit[0] = False

    owner = comp0.copy()  # which particle each swollen pixel belongs to
    displaced_carbon = np.zeros(n0 + 1, float)
    consumed_pore = np.zeros(n0 + 1, float)
    pore0 = float((lab == PORE).sum())
    cur = areas0.copy()
    for _ in range(max_iter):
        need = cur < target
        if not need.any():
            break
        need_full = np.concatenate([[False], need])
        active = need_full[owner]
        ring = ndi.binary_dilation(active, structure=_S4) & (owner == 0)
        if not ring.any():
            break
        # assign ring pixels to the nearest active owner via grey dilation of ids
        ids = ndi.grey_dilation(np.where(active, owner, 0), footprint=_S4)
        ring_ids = ids[ring]
        ry, rx = np.where(ring)
        if mode == "pore_first":
            # order: pore pixels first, then carbon/uncertain; cap per particle
            is_pore = lab[ry, rx] == PORE
            order = np.argsort(~is_pore, kind="stable")
        else:
            order = np.arange(len(ry))
        rem = np.maximum((target - cur).astype(int), 0)
        # cap per particle: within the priority order, take the first rem[pid] ring pixels of each pid
        pid_ord = ring_ids[order]
        valid = pid_ord > 0
        sub = np.argsort(pid_ord[valid], kind="stable")
        pv = pid_ord[valid][sub]
        first = np.r_[0, np.flatnonzero(np.diff(pv)) + 1]
        rank = np.arange(len(pv)) - np.repeat(first, np.diff(np.r_[first, len(pv)]))
        take_sorted = rank < rem[pv - 1]
        taken = np.zeros(len(ry), bool)
        idx_valid = order[valid]
        taken[idx_valid[sub]] = take_sorted
        ty, tx = ry[taken], rx[taken]
        pid = ring_ids[taken]
        was = lab[ty, tx]
        np.add.at(displaced_carbon, pid, (was == CARBON) | (was == UNCERTAIN))
        np.add.at(consumed_pore, pid, was == PORE)
        owner[ty, tx] = pid
        lab[ty, tx] = SI
        cur = np.bincount(owner.ravel(), minlength=n0 + 1)[1:].astype(float)

    grown = cur - areas0
    with np.errstate(invalid="ignore", divide="ignore"):
        ci = np.where(grown > 0, displaced_carbon[1:] / grown, 0.0)  # fraction of growth into solid
    keep = ~edge_hit[1:]
    si_after = lab == SI
    comp1, n1 = ndi.label(si_after, structure=_S4)
    merges = int(n0 - n1)
    areas1 = np.bincount(comp1.ravel())[1:]
    # a merged cluster: component after that contains > 1 original particle
    flat0 = comp0.ravel()
    _, first_px = np.unique(flat0, return_index=True)
    first_px = first_px[1:] if flat0.min() == 0 else first_px  # drop background
    n_orig_in = np.bincount(comp1.ravel()[first_px], minlength=n1 + 1)
    merged = n_orig_in[1:] > 1
    largest = float(np.sqrt(4 * areas1[merged].max() / np.pi) * um_per_px) if merged.any() else 0.0
    hot_px = (np.pi / 4) * (hotspot_um / um_per_px) ** 2
    big = areas1 > hot_px
    ci_full = np.concatenate([[0.0], ci])[owner].astype(np.float32)
    ci_full[owner == 0] = np.nan
    return {
        "labels": lab,
        "constraint_map": ci_full,
        "pore_closure_frac": float(consumed_pore.sum() / pore0) if pore0 else float("nan"),
        "constraint_index": float(ci[keep].mean()) if keep.any() else float("nan"),
        "constraint_index_aw": float(np.average(ci[keep], weights=areas0[keep])) if keep.any() else float("nan"),
        "buffer_sufficiency_frac": float((ci[keep] < 0.5).mean()) if keep.any() else float("nan"),
        "si_touch_frac": float(merged.sum() / max(n1, 1)),
        "si_merges": merges,
        "largest_merged_cluster_ecd_um": largest,
        "si_area_in_clusters_gt_hotspot": float(areas1[big].sum() / max(areas1.sum(), 1)),
        "dH_H_bound": float(displaced_carbon.sum() / (H * W)),
        "n_particles": int(keep.sum()),
        "f_A_achieved": float(cur[keep].sum() / areas0[keep].sum()) if keep.any() else float("nan"),
    }
