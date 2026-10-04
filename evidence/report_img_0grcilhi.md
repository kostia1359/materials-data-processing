# QC report: img_0grcilhi

## Verdict

**img_0grcilhi is most consistent with Batch 3** (probability 0.74; Batch 1: 0.05; Batch 2: 0.21; Batch 3: 0.74; assignment stable in 93% of strip-bootstrap resamples; moderately confident). No batch signature is stable enough to read. **Relative to the Batch-3 baseline the verdict is REJECT**: 4 of 16 shortlist KPIs exceed 2.5 robust sigma; aggregate score S=3.74, rank-p=0.111. Baseline has 17 images, so the smallest reportable rank p-value is 0.056; this verdict is an effect-size judgement, not a significance test.

## Acquisition gates

Status: **ok**

## KPI table

| KPI | value | SE | baseline med +/- scale | z | class |
|---|---|---|---|---|---|
| si_frac_solid | 0.06268 | 0.0102 | 0.06673 +/- 0.00833 | -0.49 | M |
| si_d50_aw_um | 3.372 | 0.406 | 3.232 +/- 0.333 | +0.42 | R |
| si_d90_aw_um | 6.435 | 0.586 | 6.103 +/- 0.838 | +0.40 | R |
| si_num_density_mm2 | 1.561e+04 | 1.27e+03 | 1.815e+04 +/- 3.77e+03 | -0.67 | M |
| si_solidity_med | 0.8824 | 0.0193 | 0.8847 +/- 0.0442 | -0.05 | R |
| si_quadrat_cv | 0.8581 | 0.118 | 0.7514 +/- 0.176 | +0.61 | M |
| pore_frac_deep | 0.1614 | 0.00997 | 0.1211 +/- 0.0114 | +3.54 | W |
| pore_chord_h_mean_um | 0.8985 | 0.0631 | 0.6445 +/- 0.0747 | +3.40 | R |
| pore_chord_v_mean_um | 0.7381 | 0.0649 | 0.5406 +/- 0.0534 | +3.70 | R |
| aniso_pore_chord_ratio | 1.217 | 0.0206 | 1.177 +/- 0.0588 | +0.68 | R |
| aniso_carbon_chord_ratio | 1.198 | 0.0118 | 1.202 +/- 0.0601 | -0.05 | R |
| s2_len_pore_h_px | 53.23 | 5.16 | 33.31 +/- 5.01 | +3.98 | R |
| pore_percolating_frac_v | 0 | 0 | 0 +/- 0 | +0.00 | M |
| si_contact_pore_frac | 0 | 0 | 0 +/- 0 | +0.00 | R |
| si_contact_carbon_frac | 1 | 0 | 1 +/- 0.05 | +0.00 | R |
| crack_density_um_per_mm2 | 3056 | 1.07e+03 | 3099 +/- 1.51e+03 | -0.03 | M |

## Drivers

- **s2_len_pore_h_px**: 53.23 vs baseline 33.31 +/- 5.01 (z=+3.98, higher, class R) - Correlation length of the pore network (horizontal)
- **pore_chord_v_mean_um**: 0.7381 vs baseline 0.5406 +/- 0.0534 (z=+3.70, higher, class R) - Vertical pore size: through-plane wetting and fast-charge paths
- **pore_frac_deep**: 0.1614 vs baseline 0.1211 +/- 0.0114 (z=+3.54, higher, class W) - Deep-pore fraction: lower bound on porosity; governs ionic transport and wetting
- **pore_chord_h_mean_um**: 0.8985 vs baseline 0.6445 +/- 0.0747 (z=+3.40, higher, class R) - Horizontal pore size: calendering shrinks pores
- **aniso_pore_chord_ratio**: 1.217 vs baseline 1.177 +/- 0.0588 (z=+0.68, higher, class R) - Pore alignment parallel to collector: raises through-plane tortuosity, slows fast-charge; consistent with calendering
- **si_num_density_mm2**: 1.561e+04 vs baseline 1.815e+04 +/- 3.77e+03 (z=-0.67, lower, class M) - Si-candidate particle density: change at constant fraction = size change
- **si_quadrat_cv**: 0.8581 vs baseline 0.7514 +/- 0.176 (z=+0.61, higher, class M) - Si spatial uniformity: agglomeration at constant loading = mixing problem
- **si_frac_solid**: 0.06268 vs baseline 0.06673 +/- 0.00833 (z=-0.49, lower, class M) - Si-candidate share of solids: sets capacity, swelling and first-cycle loss

## Batch assignment

Assigned: **batch 3** (moderately confident, stability 0.93)

probabilities: B1 0.05 | B2 0.21 | B3 0.74
distances: B1 8.09 | B2 7.39 | B3 6.66
signature match: {'1': 0.0, '2': 0.0}
novelty: False (in-distribution score 0.129)
priors: uniform
T and shrinkage chosen by nested leave-one-image-out on 31 labelled images

## Figures

![overlay](figs/overlay.png)
![z-scores](figs/z_bars.png)
![pca](figs/pca.png)

## Not measurable

- coating thickness
- surface roughness
- collector delamination
- binder/carbon-black distribution
- Si vs SiOx identity
- true (total) porosity

## Simulation layer (relative indices, frozen assumptions e62b8644bf)

Grades: A arithmetic on measured quantities; B direction supported, level set by an assumption (ratios only); C labelled heuristic. Bounds = value on L_solid / L_pore.

| index | grade | value | baseline med | ratio | L_solid | L_pore |
|---|---|---|---|---|---|---|
| D_eff_rel_TP | B | 0.06081 | 0.07916 | 0.768 | 0.0459 | 0.08157 |
| aniso_ratio | B | 1.66 | 1.235 | 1.34 | nan | nan |
| tau_p | B | 4.153 | 3.014 | 1.38 | 3.639 | 4.064 |
| sigma_eff_rel_TP | C | 0.01256 | 0.05497 | 0.228 | nan | nan |
| t_d_s | B | 455.2 | 349.7 | 1.3 | nan | nan |
| i_lim_Am2 | B | 40.06 | 52.15 | 0.768 | nan | nan |
| pore_closure_frac_1.6 | B | 0.1113 | 0.06123 | 1.82 | nan | nan |
| constraint_index_2.43 | B | 0.7628 | 0.7639 | 0.998 | nan | nan |
| buffer_sufficiency_frac_2.43 | B | 0.1848 | 0.1848 | 1 | nan | nan |
| si_touch_frac_2.43 | B | 0.2862 | 0.2584 | 1.11 | nan | nan |
| breakthrough_r_um | B | 0.2083 | 0.1 | 2.08 | nan | nan |
| trapped_frac | B | 0.8129 | 0.9252 | 0.879 | nan | nan |
| icl_proxy | B | 13.05 | 5.516 | 2.37 | nan | nan |
| Q_vol_mAh_cm3_deep | A | 779.9 | 827.4 | 0.943 | nan | nan |
| R_ion_rel | B | 16.44 | 12.63 | 1.3 | nan | nan |
| cli_heuristic | C | 0.4653 | 0.4653 | 1 | nan | nan |
| Q_CC_1C_over_Q_C10 | B | 0.7018 | 0.7264 | 0.966 | nan | nan |
| Q_CC_3C_over_Q_C10 | B | 0.3094 | 0.3531 | 0.876 | nan | nan |
| min_eta_sep_3C_V | B | -0.09264 | -0.087 | 1.06 | nan | nan |

Downsample audit (pore phase) accepted: True; uncertain fraction 0.170

![sim maps](figs/sim_maps.png)
![D_c sweep](figs/dc_sweep.png)

## Caveats

- Bright particles are identified by backscatter Z-contrast only (no EDS); they are reported as Si-candidates and may include SiOx or Si-C composite.
- Pores appear open (not resin-infiltrated); the deep-pore fraction is a lower bound on porosity, not a porosity measurement.
- Graphite, carbon black and binder cannot be separated in these images; the 'carbon matrix' class contains all three.
- Coating thickness, surface roughness and collector delamination are not assessable: the fields lie entirely inside the coating.
- Anisotropy and transport indices are 2-D section quantities compared like-with-like against the baseline, not 3-D values; the through-plane direction is assumed vertical in the frame.
- Pixel size (25 nm/px) is taken from the TIFF resolution tags; vendor metadata is absent, so it is unverified.
- With N baseline images the smallest achievable rank p-value is 1/(N+1); verdicts are effect-size judgements against a small baseline.
- Image grey levels have been remapped after acquisition (comb histograms); any method relying on raw intensities would be confounded - this system uses BSE phase identity after smoothing and treats ETD/InLens levels as acquisition covariates.
- Transport and mechanics indices are 2-D effective-medium quantities computed with a fixed matrix diffusivity D_c = 0.05 and fixed moduli; they are ratios to the baseline under identical assumptions, not electrode tortuosity, conductivity or stress values.
- The deep-pore phase does not percolate in 2-D, so no pore-only tortuosity is reported; the two-conductivity index's absolute level is set by D_c and only its ratios are meaningful.
- Cell-level outputs come from a PyBaMM composite graphite-Si model with a frozen LG-M50-type parameter set and are relative rate-capability and plating-indicator shifts, not predictions of the real cell.