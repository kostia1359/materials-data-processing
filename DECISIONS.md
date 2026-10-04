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

## Simulation layer (Section 10) — additive, core shortlist/classifier untouched
- Sim results are cached with the KPIs under key `(sha1_bse, PIPELINE_VERSION, SIM_VERSION+assumptions_hash)`; only the heavy, baseline-independent part (`run_sim`) is cached — porosity conventions, Q_vol and PyBaMM runs (`finalize_sim`) are re-applied against the current `sim_baseline.json`, so a baseline rebuild never forces a Laplace re-solve.
- ×4 block-majority audit is reported **per phase**: `accepted` (used for the ionic index) tracks the pore phase; carbon bridges one pixel wide are lost at ×4 (carbon spanning fraction can drop e.g. 0.90→0.41), so `accepted_carbon`/`accepted_all` are logged and the electronic index is read as ×4-resolution-relative only.
- Deep pores never percolate in 2-D, so the ionic index is always the two-conductivity (D_c = 0.05) solve; a pore-only tortuosity is not reported (caveat 10). `tau_p = eps_fused / D_eff_rel` is a 2-D index, ratios only.
- Phantom tolerances: straight 1 %, tilted 30° 8 % (12-px staircase channel), serpentine 10 %, Keller checkerboard 6 % at 32-px cells (FV converges from below through the corner singularities: 0.278/0.294/0.30 at 8/16/32 px vs 0.316) — the convergence series is stored in `sim_baseline.json → phantoms`, not hidden behind a loose tolerance.
- Wetting: porespy 3.x `porosimetry(im, inlets=...)` is access-limited by construction; since deep pores do not span, "breakthrough radius" is undefined and the **median access-limited invasion radius** is reported under that name, plus the trapped fraction.
- Swelling bounds (L_solid / L_pore) and the isotropic variant are computed only in `full` mode (loop / build-baseline); `evaluate` runs the two pore-first scenarios to stay inside the 3-minute budget.
- PyBaMM τ mapping: `transport efficiency: tortuosity factor` with electrolyte τ = ε^-0.5·(tau_p / baseline median) and solid τ_s = ε_s (Chen2020 Bruggeman 0 ⇒ efficiency 1) reproduces the Bruggeman default to < 0.5 % in Q_CC (test). Experiment steps are one tuple = one cycle (PyBaMM 26 treats bare strings as separate cycles).
- PyBaMM sensitivity at the project composition (ε 0.30, f_Si 9 %, R_Si 2.2 µm, L 70 µm): τ +30 % and ε −0.05 move the 3C plating indicator by +35 % / +51 %, f_Si +30 % by +6 %, R_Si +30 % by +5 %; the hysteresis option moves the 1C indicator by ~3× → cell-level plating outputs are graded **C**-leaning B and the hysteresis case is reported as a sensitivity, not the default.
- Energy density Q_vol uses frame **volume** fractions with CBD 8 % of solids; both porosity conventions are reported (`_deep`, `_offset`) and their rank agreement (Spearman) gates `unstable_indices`.

## Section 0.3 loop (`qc loop`)

- The loop runs against the real baseline directory (`out/baseline`), so each `qc regress --only <processed so far>` compares the freshly processed triple with the 31-sample snapshot `kpis_0.csv` that the first PR froze; a separate one-sample snapshot would have made regress trivially green.
- Per-triple checks are gates on *facts* (pixel size 25 nm, 16/16 finite shortlist KPIs, flux balance, L_solid <= L_mid <= L_pore ordering, pore-phase downsample audit). `ds_all_ok` (carbon spanning fraction after x4) is logged but does not flag the triple: the ionic index that the checks protect is pore-driven and the carbon loss is a known, documented resolution effect (below).
- Cache hygiene: swelling `constraint_map`s are kept in memory for the report figures but never pickled into `out/<base>/cache/<id>/result.json` (they were 14 MB/sample).
- The shortlist was frozen before the first Batch_1 triple (`shortlist_hash` in `loop_log.csv` is constant across all 31 rows); only the simulation layer and the loop plumbing changed during the pass, never a KPI.

## Ablations (detector / resolution), `out/baseline/ablations.json`

- Resolution: on the fused L_mid map the ionic index `D_eff_rel_TP` agrees to within 3 % between x2 and x4 block-majority downsampling on all four ablation samples (B3, B3, B1, B2); x8 drifts by up to 20 % on the low-porosity Batch_1 sample, so x4 is the frozen factor. The electronic index `sigma_eff_rel_TP` falls with coarsening because thin carbon bridges are lost (carbon spanning fraction 0.71 → 0.50 → 0.20 on img_0grcilhi) and is undefined (carbon does not span) on img_4ih2ggld at every factor. **`sigma_eff_rel_TP` is therefore graded C** and reported as a ratio only; the x4 audit stays pore-driven.
- Detector: removing ETD evidence collapses the uncertain fraction (17 % → 0.7 % on img_0grcilhi) and lowers the pore fraction — BSE-only is over-confident about pores, which is the reason the fusion keeps ETD as independent evidence rather than a tie-breaker. Removing InLens changes the indices by < 1 % (it only weights the smoothing).
- The independent Laplace solves (L_mid TP/IP, L_solid, L_pore, electronic, D_c sweep) run in a thread pool; SuperLU releases the GIL, so a cold `evaluate` dropped from 188 s to 147 s with bit-identical results (checked against the serial loop rows).
