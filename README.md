# anode-qc — SEM batch QC for Si–graphite anodes

Interpretable, no-black-box QC pipeline for co-registered SEM detector triples
(BSE / ETD|SE / InLens TIFFs) of silicon–graphite anode cross-sections.

Given a baseline batch (Batch_3 = nominal process) it:

1. segments each triple into **carbon / pore / Si-candidate** phases (BSE-only
   phase identity, hysteresis deep-pore grow on ETD-confirmed darks);
2. extracts ~40 interpretable KPIs with per-sample strip-SE and bootstrap CIs;
3. issues an **ACCEPT / INVESTIGATE / REJECT** verdict vs the baseline
   (robust z-scores, zones 2.5σ/4σ, conformal rank over baseline aggregates);
4. assigns the sample to **batch 1 / 2 / 3** (always bets) with calibrated
   probability, strip-vote stability, a novelty flag, and signature explanations;
5. self-validates: LOIO accuracy/NLL, synthetic ground truth, shadow
   (acquisition-only) classifier, sensitivity to thresholds/downsampling.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e .
```

Python ≥ 3.11. Everything runs on CPU; `evaluate` is < 3 min/sample.

## Data layout

```
data/
  Batch_1/img_<id>_{BSE,ETD,Inlens}.tif
  Batch_2/img_<id>_{BSE,ETD|SE,Inlens}.tif
  Batch_3/img_<id>_{BSE,ETD|SE,Inlens}.tif   # baseline batch
```

`SE.tif` is accepted as an alias for ETD (`etd_label_in_filename` covariate).
Pixel size is read from TIFF resolution tags (25 nm/px); images missing tags
default to 25 nm/px with a warning.

## Commands

```bash
qc build-baseline --data ./data --out ./out          # one-time; append-only
qc evaluate --sample ./new/img_abc123 --baseline ./out
qc predict --folder ./test_drop --baseline ./out     # predictions.json + dm.txt
qc score --predictions ./out/predictions.json --truth truth.csv
qc validate --data ./data --out ./out                # writes VALIDATION.md
qc quicklook --data ./data --out ./out/quicklook     # QC PNGs per sample
qc regress --data ./data --out ./out [--accept] [--only id1,id2]  # KPI drift vs snapshot
qc loop --data ./data --out ./out --manifest drive_manifest.csv  # Section 0.3: one triple at a time
```

`loop` is the development loop of the brief: for each triple in order
(attached triple → Batch_3 → Batch_1 → Batch_2) it downloads the three TIFFs
from `drive_manifest.csv` (gdown, byte-exact verify, skip if present), runs the
KPI pipeline plus the full simulation layer, applies per-triple checks
(pixel size, 16/16 finite shortlist KPIs, flux balance, bound ordering,
downsample audit), runs `qc regress --only <processed so far>` and appends one
line to `out/loop_log.csv`. It is resumable; re-running skips `ok` triples.

### Simulation layer (`qc/sim/`, Section 10 — additive, relative indices only)

`build-baseline` also writes `sim.csv` (one row per sample, every index on the
fused L_mid map), `sim_bounds.csv` (L_solid / L_pore values, D_c sweep, PyBaMM
±SE runs) and `sim_baseline.json` (baseline medians, D_c sweep band, empirical
REV, phantom results, rank robustness). `evaluate`/`predict` add a
`"simulation"` block to the verdict JSON (grade A/B/C per index, ratio to the
baseline median, bounds, undefined list, `assumptions_hash`) and the reports
gain one table and two figures (fused/swollen/constraint maps; D_c sweep over
the baseline band). Frozen assumptions live in `config.yaml → sim:`; every
result carries their hash. Nothing in this layer feeds the KPI shortlist, the
verdict or the batch classifier (`qc regress` proves it).

`evaluate` writes `verdict.json`, `report.md` and a standalone `report.html`
(base64 figures) under `<baseline>/reports/<sample_id>/`.

## Repo map

- `qc/io.py` — discovery, naming regex, TIFF loader, bad-pixel hygiene, edge bands
- `qc/gates.py` — acquisition-gate covariates (comb step, noise, blur, drift, registration…)
- `qc/segment.py` — multi-Otsu + hysteresis deep-pore + Si opening
- `qc/kpis/` — KPI groups: fractions, sizes, dispersion, pores, anisotropy,
  correlation, connectivity, contacts, defects, vertical
- `qc/stats.py` — baseline robust stats, z-scores, verdict + conformal rank
- `qc/assign.py` — shrunken centroids, softmax calibration, stability, novelty, signatures
- `qc/validate.py` — LOIO, shadow classifier, sensitivity
- `qc/synth.py` — synthetic ground-truth generator
- `qc/report.py` — verdict JSON, report.md/html, dm.txt block
- `qc/download.py` — manifest-driven gdown fetch with byte-exact verification
- `qc/sim/` — `fuse` (BSE/ETD evidence → L_mid/L_solid/L_pore + uncertain),
  `laplace` (percolation check, FV two-conductivity solve, ionic/electronic
  indices), `swell` (pore-first Si growth, constraint/buffer), `indices`
  (transport, wetting, ICL, Q_vol, plating-risk, cycle-life heuristic),
  `cell` (PyBaMM composite DFN, relative), `phantoms`, `baseline`, `run`
- `config.yaml` — every threshold (Appendix A)
- `DECISIONS.md`, `DATA_FACTS.md`, `VALIDATION.md` — audit trail

## Status against the brief's definition of done (2026-10-04)

- Section 0.3 loop: 31/31 triples processed one at a time (Batch_3 → Batch_1 →
  Batch_2) with per-triple checks and `qc regress` after each; `out/loop_log.csv`
  has every row, shortlist hash constant, no KPI code changed after the first
  Batch_1 triple. The last ten triples ran without any code change.
- `python -m pytest -q`: 21 passed (11 synthetic ground truth + 10 simulation
  phantoms/invariants); `qc regress` OK vs the 31-sample snapshot.
- Cold `qc evaluate` on a 2316-row triple: ~150 s on 4 CPU cores (KPIs ~80 s,
  simulation ~60 s), peak RSS 2.8 GB.
- Simulation self-checks (VALIDATION.md): phantoms within tolerance, bound
  ordering 31/31, flux balance < 1e-9, D_c-sweep rank Spearman ≥ 0.985, both
  porosity conventions Spearman 0.996, no unstable index; electronic index is
  resolution-sensitive and undefined on 3/31 samples (carbon does not span),
  graded C and reported as undefined rather than imputed.
- Test-drop (B1+B2 as unseen): 9/14, NLL 0.825; LOIO 0.42 / 1.07 — unchanged,
  the simulation layer does not feed the classifier.

`evidence/` holds the small outputs of the pass above so they can be reviewed
without re-running it: `loop_log.csv`, `sim.csv`, `sim_bounds.csv`,
`sim_baseline.json`, `ablations.json`, `pybamm_sensitivity.json` and the
Markdown report of the attached triple. Everything else lives under `out/`
(ignored) and is reproduced by `qc loop` + `qc build-baseline --force`.

## Honest wording

Outputs say "Si-candidate (Z-contrast)" not "silicon"; "deep-pore fraction
(lower bound)"; batches are "different/shifted", never "defective" unless a
defect KPI fires. Eight fixed caveat strings (Appendix C) ship in every report.
