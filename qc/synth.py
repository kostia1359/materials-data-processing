"""Synthetic ground-truth image generator (Section 6.1)."""
from __future__ import annotations

import numpy as np
from skimage.draw import disk as draw_disk, ellipse as draw_ellipse


def generate(
    width: int = 7000, height: int = 2000, seed: int = 0,
    pore_frac: float = 0.15, si_frac: float = 0.07,
    si_d50_um: float = 4.0, pore_aspect: float = 2.0,
    clustered: bool = False, carbon_grey: float = 58, pore_grey: float = 12,
    si_grey: float = 112, noise_sd: float = 8.0, comb_step: int = 3,
    gamma: float = 0.9, px_nm: float = 25.0,
) -> dict:
    """Synthesize BSE/ETD/InLens with known phase masks (truth).

    Si discs are placed first (non-overlapping, log-normal ECD, D50 =
    si_d50_um); pore ellipses (aspect pore_aspect:1) fill around them.
    Returns dict with 'bse', 'etd', 'inlens' uint8 images, 'pore', 'si',
    'carbon' boolean masks and 'truth' with realized distribution stats.
    """
    rng = np.random.default_rng(seed)
    um_px = px_nm / 1000.0
    area_px = width * height

    pore = np.zeros((height, width), bool)
    si = np.zeros((height, width), bool)

    # --- Si: discs, log-normal ECD; rim overlaps allowed, engulfing not ---
    sigma_ln = 0.5
    mu = np.log(si_d50_um)
    target = si_frac * area_px
    placed = 0
    tries = 0
    max_tries = 40000 if clustered else 6000
    discs = []  # (cy, cx, r_px) of placed discs
    if clustered:
        n_cl = 8
        centres = np.column_stack(
            [rng.uniform(0, height, n_cl), rng.uniform(0, width, n_cl)])
    while placed < target and tries < max_tries:
        tries += 1
        ecd_um = rng.lognormal(mu, sigma_ln)
        r_px = (ecd_um / um_px) / 2
        if clustered:
            c = centres[rng.integers(0, len(centres))]
            cy = c[0] + rng.normal(0, 140)
            cx = c[1] + rng.normal(0, 140)
        else:
            cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        rr, cc = draw_disk((cy, cx), r_px, shape=(height, width))
        si[rr, cc] = True
        discs.append((cy, cx, r_px))
        placed = int(si.sum())

    # --- pores: ellipses, pore_aspect:1 horizontal, avoid Si ---
    target = pore_frac * area_px
    placed = 0
    tries = 0
    while placed < target and tries < 60000:
        tries += 1
        a = rng.uniform(0.2, 2.5) / um_px / 2  # horizontal semi-axis, px
        b = a / pore_aspect
        cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        rr, cc = draw_ellipse(cy, cx, b, a, shape=(height, width))
        # keep ellipses isolated so the aspect truth is recoverable
        if (pore[rr, cc]).sum() > 0 or (si[rr, cc]).sum() > 0:
            continue
        pore[rr, cc] = True
        placed += int(rr.size)

    carbon = ~(pore | si)

    # realized truth stats
    from skimage.measure import label, regionprops
    from .kpis.common import weighted_quantile
    tlab = label(si)
    ta = np.array([p.area for p in regionprops(tlab)], dtype=float)
    tecd = 2 * np.sqrt(ta / np.pi) * um_px  # um
    truth = {
        "pore_frac": float(pore.sum() / area_px),
        "si_frac": float(si.sum() / area_px),
        "si_frac_solid": float(si.sum() / (si | carbon).sum()),
        "si_d50_aw_um": float(weighted_quantile(tecd, ta, [0.5])[0]),
        "pore_aspect": pore_aspect,
    }

    # --- images ---
    bse = np.zeros((height, width), float)
    bse[carbon] = carbon_grey
    bse[pore] = pore_grey
    bse[si] = si_grey
    bse += rng.normal(0, noise_sd, bse.shape)
    # gamma + comb LUT (keep every comb_step-th level)
    bse = 255.0 * (np.clip(bse, 0, 255) / 255.0) ** gamma
    bse = np.round(bse / comb_step) * comb_step
    bse = np.clip(bse, 0, 255).astype(np.uint8)

    etd = np.where(pore, 15.0, 80.0) + rng.normal(0, 10, bse.shape)
    etd = np.clip(etd, 0, 255).astype(np.uint8)
    inlens = np.where(carbon, 70.0, 120.0) + rng.normal(0, 12, bse.shape)
    inlens = np.clip(inlens, 0, 255).astype(np.uint8)

    return {"bse": bse, "etd": etd, "inlens": inlens,
            "pore": pore, "si": si, "carbon": carbon, "truth": truth}
