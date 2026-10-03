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
qc regress --data ./data --out ./out [--accept]      # KPI drift vs snapshot
```

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
- `config.yaml` — every threshold (Appendix A)
- `DECISIONS.md`, `DATA_FACTS.md`, `VALIDATION.md` — audit trail

## Honest wording

Outputs say "Si-candidate (Z-contrast)" not "silicon"; "deep-pore fraction
(lower bound)"; batches are "different/shifted", never "defective" unless a
defect KPI fires. Eight fixed caveat strings (Appendix C) ship in every report.
