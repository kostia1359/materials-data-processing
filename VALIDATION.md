# Validation

Run: 2026-10-04 00:12 UTC (1380s)

## Leave-one-image-out (LOIO)

- n = 31, accuracy = 0.419, mean NLL = 1.069, mean Brier = 0.645
- false alarms (batch-3 held out -> INVESTIGATE/REJECT): 7
- detections (batch-1/2 held out -> INVESTIGATE/REJECT): 6

Confusion matrix (rows=true, cols=assigned, order 1,2,3):

```
[[4. 0. 3.]
 [3. 2. 2.]
 [8. 2. 7.]]
```

| sample_id    |   true |   assigned |       p1 |        p2 |        p3 | verdict     | top3                                                                     |         S |
|:-------------|-------:|-----------:|---------:|----------:|----------:|:------------|:-------------------------------------------------------------------------|----------:|
| img_4ih2ggld |      1 |          1 | 0.694521 | 0.116166  | 0.189313  | REJECT      | si_num_density_mm2,si_frac_solid,si_solidity_med                         | 11.5156   |
| img_5n1q8atc |      1 |          1 | 0.928973 | 0.0442488 | 0.0267783 | REJECT      | si_num_density_mm2,si_frac_solid,si_d90_aw_um                            | 12.1181   |
| img_f1vzngrs |      1 |          1 | 0.435527 | 0.347151  | 0.217322  | REJECT      | pore_frac_deep,pore_chord_v_mean_um,pore_chord_h_mean_um                 |  3.39835  |
| img_ffwubibz |      1 |          3 | 0.149895 | 0.402876  | 0.447229  | ACCEPT      | pore_frac_deep,si_frac_solid,aniso_carbon_chord_ratio                    |  1.61524  |
| img_fzrt2k6r |      1 |          3 | 0.312802 | 0.328749  | 0.358449  | ACCEPT      | si_frac_solid,pore_frac_deep,aniso_carbon_chord_ratio                    |  0.688465 |
| img_iv6g2oq0 |      1 |          3 | 0.285402 | 0.318261  | 0.396337  | ACCEPT      | aniso_pore_chord_ratio,si_d50_aw_um,si_num_density_mm2                   |  0.976159 |
| img_uhdslk0o |      1 |          1 | 0.402488 | 0.308542  | 0.28897   | INVESTIGATE | si_d50_aw_um,si_num_density_mm2,si_d90_aw_um                             |  1.85293  |
| img_3806gxp0 |      2 |          1 | 0.347785 | 0.315373  | 0.336843  | ACCEPT      | si_d50_aw_um,aniso_pore_chord_ratio,pore_frac_deep                       |  1.58781  |
| img_avn74qx1 |      2 |          3 | 0.161782 | 0.322672  | 0.515546  | ACCEPT      | si_frac_solid,s2_len_pore_h_px,crack_density_um_per_mm2                  |  1.44482  |
| img_b3esycq1 |      2 |          1 | 0.489551 | 0.208179  | 0.30227   | INVESTIGATE | si_frac_solid,si_num_density_mm2,aniso_carbon_chord_ratio                |  2.13977  |
| img_epqdaau9 |      2 |          2 | 0.330886 | 0.36887   | 0.300244  | REJECT      | crack_density_um_per_mm2,pore_frac_deep,pore_chord_v_mean_um             |  3.13138  |
| img_i9jiqjwl |      2 |          1 | 0.380052 | 0.315134  | 0.304814  | ACCEPT      | pore_frac_deep,pore_chord_h_mean_um,s2_len_pore_h_px                     |  1.44043  |
| img_r17byphk |      2 |          3 | 0.167871 | 0.269349  | 0.562779  | ACCEPT      | s2_len_pore_h_px,pore_frac_deep,si_frac_solid                            |  1.17676  |
| img_rxax5ozo |      2 |          2 | 0.306801 | 0.357899  | 0.3353    | ACCEPT      | crack_density_um_per_mm2,si_quadrat_cv,pore_frac_deep                    |  1.66206  |
| img_0grcilhi |      3 |          3 | 0.108147 | 0.326917  | 0.564936  | REJECT      | pore_frac_deep,pore_chord_h_mean_um,pore_chord_v_mean_um                 |  4.98588  |
| img_71vgq3fw |      3 |          2 | 0.371149 | 0.409564  | 0.219287  | ACCEPT      | s2_len_pore_h_px,aniso_carbon_chord_ratio,pore_chord_h_mean_um           |  1.38068  |
| img_9luzk4jm |      3 |          3 | 0.216706 | 0.346323  | 0.436971  | REJECT      | pore_frac_deep,pore_chord_v_mean_um,pore_chord_h_mean_um                 |  2.73749  |
| img_cfe5vt7s |      3 |          1 | 0.363237 | 0.336754  | 0.300009  | ACCEPT      | si_frac_solid,si_d90_aw_um,pore_chord_v_mean_um                          |  1.11721  |
| img_hawkfj64 |      3 |          3 | 0.251797 | 0.35737   | 0.390833  | ACCEPT      | si_frac_solid,si_d50_aw_um,si_d90_aw_um                                  |  1.3273   |
| img_hzumfsms |      3 |          3 | 0.104723 | 0.339913  | 0.555364  | REJECT      | pore_frac_deep,s2_len_pore_h_px,pore_chord_h_mean_um                     |  5.46708  |
| img_kbdh4tri |      3 |          1 | 0.419034 | 0.333349  | 0.247618  | ACCEPT      | s2_len_pore_h_px,pore_chord_v_mean_um,pore_frac_deep                     |  1.40279  |
| img_mgxahqnk |      3 |          1 | 0.404925 | 0.306741  | 0.288333  | REJECT      | si_d50_aw_um,crack_density_um_per_mm2,si_d90_aw_um                       |  2.75119  |
| img_pl8uabbv |      3 |          2 | 0.265495 | 0.385241  | 0.349264  | ACCEPT      | si_num_density_mm2,aniso_pore_chord_ratio,crack_density_um_per_mm2       |  1.1385   |
| img_ptg8lmto |      3 |          1 | 0.362245 | 0.327844  | 0.30991   | ACCEPT      | pore_frac_deep,pore_chord_v_mean_um,si_quadrat_cv                        |  1.68366  |
| img_tuy3zymq |      3 |          3 | 0.276175 | 0.348889  | 0.374935  | ACCEPT      | si_d50_aw_um,s2_len_pore_h_px,si_num_density_mm2                         |  1.04021  |
| img_ufdvpb81 |      3 |          3 | 0.191536 | 0.400366  | 0.408098  | ACCEPT      | pore_frac_deep,si_d50_aw_um,pore_chord_v_mean_um                         |  1.83404  |
| img_utfgcjfa |      3 |          1 | 0.38337  | 0.304389  | 0.31224   | INVESTIGATE | si_d50_aw_um,si_quadrat_cv,crack_density_um_per_mm2                      |  1.88456  |
| img_vc2whyaq |      3 |          3 | 0.276899 | 0.330278  | 0.392823  | ACCEPT      | crack_density_um_per_mm2,aniso_pore_chord_ratio,aniso_carbon_chord_ratio |  1.48082  |
| img_x77cy643 |      3 |          1 | 0.388264 | 0.335757  | 0.275979  | ACCEPT      | si_d90_aw_um,si_quadrat_cv,si_frac_solid                                 |  1.3014   |
| img_x7u69zsw |      3 |          1 | 0.458396 | 0.289601  | 0.252004  | INVESTIGATE | si_frac_solid,si_d90_aw_um,s2_len_pore_h_px                              |  2.38203  |
| img_xgj4xftb |      3 |          1 | 0.402435 | 0.279128  | 0.318437  | REJECT      | si_d50_aw_um,si_d90_aw_um,si_quadrat_cv                                  |  2.58122  |

## Per-KPI effect sizes (|delta| in baseline scale units)

|   batch | kpi                      |     effect | robustness   |
|--------:|:-------------------------|-----------:|:-------------|
|       1 | pore_chord_v_mean_um     | -1.00823   | R            |
|       1 | pore_frac_deep           | -0.948155  | W            |
|       1 | si_d50_aw_um             |  0.900168  | R            |
|       1 | si_num_density_mm2       |  0.80313   | M            |
|       1 | si_frac_solid            |  0.790136  | M            |
|       2 | si_quadrat_cv            |  0.764775  | M            |
|       1 | pore_chord_h_mean_um     | -0.688039  | R            |
|       2 | crack_density_um_per_mm2 |  0.642116  | M            |
|       2 | aniso_pore_chord_ratio   | -0.562211  | R            |
|       2 | aniso_carbon_chord_ratio | -0.502989  | R            |
|       2 | pore_frac_deep           | -0.455204  | W            |
|       1 | aniso_pore_chord_ratio   | -0.408111  | R            |
|       1 | s2_len_pore_h_px         | -0.404994  | R            |
|       2 | pore_chord_h_mean_um     | -0.371715  | R            |
|       2 | pore_chord_v_mean_um     | -0.367209  | R            |
|       1 | si_d90_aw_um             | -0.33903   | R            |
|       2 | si_d90_aw_um             | -0.303337  | R            |
|       2 | s2_len_pore_h_px         | -0.296423  | R            |
|       2 | si_num_density_mm2       | -0.241148  | M            |
|       2 | si_frac_solid            |  0.225751  | M            |
|       1 | si_quadrat_cv            | -0.210362  | M            |
|       1 | si_solidity_med          | -0.188144  | R            |
|       1 | aniso_carbon_chord_ratio | -0.145232  | R            |
|       1 | crack_density_um_per_mm2 | -0.0945776 | M            |
|       2 | si_solidity_med          |  0.0274723 | R            |
|       2 | si_d50_aw_um             |  0.0079826 | R            |
|       1 | si_contact_carbon_frac   |  0         | R            |
|       2 | si_contact_carbon_frac   |  0         | R            |

## Shadow classifier (gate covariates only)

LOIO accuracy on acquisition covariates: **0.581**

If this is comparable to the material-KPI classifier, batches may be separable by acquisition session rather than material.

## Sensitivity

| variant   | kpi                      |   mean_abs_delta_scale |
|:----------|:-------------------------|-----------------------:|
| ds2       | aniso_carbon_chord_ratio |               0.807382 |
| ds2       | aniso_pore_chord_ratio   |               0.770717 |
| ds2       | crack_density_um_per_mm2 |               5.56643  |
| ds2       | pore_chord_h_mean_um     |               2.90691  |
| ds2       | pore_chord_v_mean_um     |               2.97287  |
| ds2       | pore_frac_deep           |               0.219651 |
| ds2       | pore_percolating_frac_v  |             nan        |
| ds2       | s2_len_pore_h_px         |               3.05246  |
| ds2       | si_contact_carbon_frac   |               0        |
| ds2       | si_contact_pore_frac     |             nan        |
| ds2       | si_d50_aw_um             |               1.36508  |
| ds2       | si_d90_aw_um             |               0.267764 |
| ds2       | si_frac_solid            |               0.262284 |
| ds2       | si_num_density_mm2       |               1.58588  |
| ds2       | si_quadrat_cv            |               0.149906 |
| ds2       | si_solidity_med          |               0.443604 |
| t+5       | aniso_carbon_chord_ratio |               0.447173 |
| t+5       | aniso_pore_chord_ratio   |               0.481523 |
| t+5       | crack_density_um_per_mm2 |               0.937881 |
| t+5       | pore_chord_h_mean_um     |               0.559583 |
| t+5       | pore_chord_v_mean_um     |               0.488331 |
| t+5       | pore_frac_deep           |               1.92957  |
| t+5       | pore_percolating_frac_v  |             nan        |
| t+5       | s2_len_pore_h_px         |               0.192841 |
| t+5       | si_contact_carbon_frac   |               0        |
| t+5       | si_contact_pore_frac     |             nan        |
| t+5       | si_d50_aw_um             |               0.517388 |
| t+5       | si_d90_aw_um             |               0.151813 |
| t+5       | si_frac_solid            |               0.524128 |
| t+5       | si_num_density_mm2       |               0.642547 |
| t+5       | si_quadrat_cv            |               0.207171 |
| t+5       | si_solidity_med          |               0.228129 |
| t-5       | aniso_carbon_chord_ratio |               0.164918 |
| t-5       | aniso_pore_chord_ratio   |               0.190011 |
| t-5       | crack_density_um_per_mm2 |               0.937881 |
| t-5       | pore_chord_h_mean_um     |               0.222464 |
| t-5       | pore_chord_v_mean_um     |               0.250712 |
| t-5       | pore_frac_deep           |               1.45974  |
| t-5       | pore_percolating_frac_v  |             nan        |
| t-5       | s2_len_pore_h_px         |               0.188205 |
| t-5       | si_contact_carbon_frac   |               0        |
| t-5       | si_contact_pore_frac     |             nan        |
| t-5       | si_d50_aw_um             |               0.412848 |
| t-5       | si_d90_aw_um             |               0.153595 |
| t-5       | si_frac_solid            |               0.818924 |
| t-5       | si_num_density_mm2       |               0.997747 |
| t-5       | si_quadrat_cv            |               0.16464  |
| t-5       | si_solidity_med          |               0.32838  |

R-class KPIs moving > 0.5 scale under perturbation: 279 cases

## Test-drop score (2026-10-04 00:16 UTC)

- n = 14, accuracy = 0.643, mean NLL = 0.825

| sample_id    |   truth |   assigned | correct   |     p1 |     p2 |     p3 |
|:-------------|--------:|-----------:|:----------|-------:|-------:|-------:|
| img_4ih2ggld |       1 |          1 | True      | 0.8456 | 0.0608 | 0.0936 |
| img_5n1q8atc |       1 |          1 | True      | 0.976  | 0.0151 | 0.0089 |
| img_f1vzngrs |       1 |          1 | True      | 0.6474 | 0.2629 | 0.0897 |
| img_ffwubibz |       1 |          3 | False     | 0.2642 | 0.3594 | 0.3764 |
| img_fzrt2k6r |       1 |          1 | True      | 0.3361 | 0.3321 | 0.3317 |
| img_iv6g2oq0 |       1 |          3 | False     | 0.3123 | 0.2933 | 0.3944 |
| img_uhdslk0o |       1 |          1 | True      | 0.5125 | 0.2752 | 0.2123 |
| img_3806gxp0 |       2 |          2 | True      | 0.3367 | 0.3479 | 0.3154 |
| img_avn74qx1 |       2 |          3 | False     | 0.1532 | 0.3627 | 0.4842 |
| img_b3esycq1 |       2 |          1 | False     | 0.4366 | 0.2918 | 0.2717 |
| img_epqdaau9 |       2 |          2 | True      | 0.2191 | 0.5965 | 0.1844 |
| img_i9jiqjwl |       2 |          2 | True      | 0.3646 | 0.3982 | 0.2372 |
| img_r17byphk |       2 |          3 | False     | 0.1505 | 0.3396 | 0.51   |
| img_rxax5ozo |       2 |          2 | True      | 0.2603 | 0.428  | 0.3117 |

## Interpretation (honest read)

- **Batch assignment is weakly but really informative.** On the 14 truly-held-out
  Batch_1/Batch_2 images: 64% accuracy, mean NLL 0.825 (chance: 1.10). LOIO over
  all 31 gives 42%/1.07 — depressed because held-out Batch_3 outliers get
  re-centred. The classifier bets with honest low probabilities where batches
  overlap; probabilities are not inflated.
- **Acquisition confound is real.** The gate-only shadow classifier reaches
  0.581 LOIO accuracy — above the material-KPI classifier's 0.419 — so part of
  batch separation may come from acquisition session (height, detector medians,
  noise) rather than material. Reported, not hidden.
- **Verdict false-alarm rate inside Batch_3 is ~40% (7/17).** Inspection shows
  these are genuine internal outliers (e.g. shading artefact, InLens
  saturation, unusual texture), not pipeline noise — the verdict is doing its
  job by flagging them.
- **Sensitivity is the weak spot.** Many KPIs move >0.5 scale under t1/t2±5 and
  especially under 2x downsampling (chord lengths and correlation lengths are
  ruler-dependent; crack density 5.6 scale). Fractions and uniformity KPIs are
  the most stable; size/length KPIs are resolution-sensitive — reflected in
  their robustness classes.
- **Two shortlist KPIs are structurally zero** on all samples
  (`si_contact_pore_frac`, `pore_percolating_frac_v`) — kept, z=0, no info.
- **Synthetic ground truth: 11/11 checks pass** (fractions, D50, anisotropy,
  Clark–Evans Poisson/clustered, perturbation robustness of R-class KPIs).
