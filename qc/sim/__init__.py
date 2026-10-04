"""Section 10 simulation layer: relative performance indices from the fused phase map.

Additive to the KPI pipeline: reads masks, writes its own tables and a
separate ``simulation`` block.  Everything is a ratio to the baseline under one
frozen assumption set (``config.yaml -> sim``), hashed into every verdict.
"""
from __future__ import annotations

import hashlib
import json

PORE, CARBON, SI, UNCERTAIN = 0, 1, 2, 3
SIM_VERSION = "0.1.0"


def assumptions_hash(sim_cfg: dict) -> str:
    s = json.dumps(sim_cfg, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:10]
