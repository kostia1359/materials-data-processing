"""Shared context and helpers for KPI computation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Ctx:
    """Everything a KPI function needs for one region (image or strip)."""
    pore: np.ndarray
    carbon: np.ndarray
    si: np.ndarray
    ambiguous: np.ndarray
    valid: np.ndarray
    bse_s: np.ndarray | None = None
    px_nm: float = 25.0
    cfg: dict = field(default_factory=dict)
    # global (image-level) extras, e.g. crack threshold, curtaining angle
    extras: dict = field(default_factory=dict)

    @property
    def um_per_px(self) -> float:
        return self.px_nm / 1000.0

    @property
    def area_um2(self) -> float:
        return float(self.valid.sum()) * self.um_per_px ** 2

    @property
    def area_mm2(self) -> float:
        return self.area_um2 / 1e6

    def subregion(self, mask: np.ndarray) -> "Ctx":
        """Restrict all masks to a sub-region (strip/quadrat), cropped to its
        bounding box so FFT/EDT work scales with the region, not the frame."""
        v = self.valid & mask
        ys, xs = np.where(v)
        if ys.size == 0:
            return Ctx(v, v, v, v, v, None, self.px_nm, self.cfg, dict(self.extras))
        r0, r1 = ys.min(), ys.max() + 1
        c0, c1 = xs.min(), xs.max() + 1
        crop = lambda a: None if a is None else a[r0:r1, c0:c1]
        extras = dict(self.extras)
        resp = extras.get("sato_response")
        if resp is not None:
            extras["sato_response"] = resp[r0 // 2 : (r1 + 1) // 2,
                                         c0 // 2 : (c1 + 1) // 2]
        return Ctx(
            pore=crop(self.pore) & crop(v), carbon=crop(self.carbon) & crop(v),
            si=crop(self.si) & crop(v), ambiguous=crop(self.ambiguous) & crop(v),
            valid=crop(v), bse_s=crop(self.bse_s),
            px_nm=self.px_nm, cfg=self.cfg, extras=extras,
        )


def chord_lengths(mask: np.ndarray, axis: int) -> np.ndarray:
    """Run-length chord sizes in px. axis=1 -> horizontal runs (per row);
    axis=0 -> vertical runs (per column). Chords touching the frame edge
    are dropped. Vectorized version of Appendix B's formula."""
    m = mask if axis == 1 else mask.T
    m = np.ascontiguousarray(m)
    w = m.shape[1]
    padded = np.zeros((m.shape[0], w + 2), dtype=np.int8)
    padded[:, 1:-1] = m
    d = np.diff(padded, axis=1)
    _, starts = np.where(d == 1)
    _, ends = np.where(d == -1)
    # within each row, starts/ends interleave and np.where preserves order, so
    # elementwise pairing is correct. Drop chords touching the frame edge:
    # start==0 means pixel 0 is in the phase; end==w means pixel w-1 is.
    keep = (starts > 0) & (ends < w)
    return (ends - starts)[keep].astype(float)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, qs) -> np.ndarray:
    """Weighted quantiles (area-weighted D-values)."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return np.interp(qs, cw, v)
