# DECISIONS.md — judgment calls, one line each

- Data source: user-supplied Drive folder `12UnB4HYDElXzoR4I0mG7NZ0buSr4QXF6` (folder "Hackathon-Polaron"), not the `data_all` ID in the brief; sample IDs match the brief's list. `drive_manifest.csv` rebuilt from a manual Drive listing (brief's manifest was never attached).
- Research reports 1 & 2 referenced by the brief were not attached; all parameter choices follow the brief's own config/stats sections; any conflict resolved in favour of the brief.
- `min_size`-style APIs: pinned to skimage behaviour at install time (0.26 deprecation warnings silenced via `warnings` filter at CLI level only; semantics unchanged — brief requires ≥N px, code uses `remove_small_objects(min_size=N)`).
- `load_grey` drops bad **columns first, then rows**: two camera-dead columns poisoned every row on one file; row-drop-first would have zeroed the image.
- Registration verified on Batch_3_img_0grcilhi (ETD/InLens vs BSE shift = 0.0 px); shifts still measured per-sample as a gate.
- Si "rim-only" overlap allowed in synthetic generator (centres may approach to 0.85×(r1+r2)); fully-free overlap collapses components and breaks `si_clark_evans_R` recovery on sparse fields.
- Clark–Evans R uses the Donnelly finite-window edge correction; uncorrected R overestimates ~8% on bounded fields.
- CE/si_clark_evans computed on `si_components` (watershed-split ≥500 px), not raw connected components, per brief shortlist intent.
- `SE.tif` treated as alias for ETD (`etd_label_in_filename` covariate kept).
- Scale bar absent in all inspected files → px_nm read from TIFF XResolution tags; samples missing tags default to 25.0 nm/px and get a `px_nm_default` warning.
- Baseline is append-only: `build-baseline` refuses to overwrite an existing `baseline.json` unless `--force`.
- Verdict wording: KPIs are reported as "different / shifted" and signatures describe process shifts; "defect" language only fires on defect-KPIs (cracks, HIZ, curtaining) per brief.
- `qc regress` compares the frozen 16-KPI shortlist per sample; |Δ| < 0.1·scale_k = insignificant.
- `si_contact_pore_frac` and `pore_percolating_frac_v` are structurally zero on all 31 samples (hysteresis pore mask terminates ≥2 px inside carbon; no vertically-percolating pore network at this FOV). They stay in the frozen shortlist but carry z=0; noted honestly rather than silently dropped.
- LOIO baseline stats are rebuilt excluding the held-out image (uses rebuilt `st` for z/centroids/hyperparams, not the full-baseline stats — fixed a mild leakage).
- **Sensitivity runs report `nan` delta for all-zero KPIs** (the two dead
  shortlist KPIs): delta/scale = 0/0. Reported as nan, counted as stable.
- **Sensitivity workers are module-level `_sens_run`** so ProcessPoolExecutor
  can pickle them; each variant subprocess re-runs `process_sample` with a
  patched cfg — ~20s/sample on a 5-worker pool.
- **VALIDATION.md carries a written interpretation**: shadow-classifier
  confound, ~40% B3 verdict false-alarm rate explained as real internal
  outliers, and resolution-sensitivity of chord/correlation-length KPIs named
  as the pipeline's weakest property.
