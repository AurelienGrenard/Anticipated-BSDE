# Conditional-CVaR engines in the direct MKABSDE

All methods use identical outer paths, Hermite regressions, and backward
updates. Only the conditional-risk estimator changes.

| Product | Method | Price (mean +/- seed std) | MVA (% of BS; mean +/- seed std) | GPU time | Mean difference vs quadrature |
|---|---|---:|---:|---:|---:|
| butterfly | quadrature | 0.31142686 +/- 8.04e-07 | 0.774071% +/- 2.60e-04 pp | 14.90 s | +0.00e+00 |
| butterfly | nested_mc | 0.31143805 +/- 2.41e-05 | 0.777692% +/- 7.80e-03 pp | 60.10 s | +1.12e-05 |
| butterfly | learning | 0.31146298 +/- 1.34e-05 | 0.785762% +/- 4.32e-03 pp | 51.94 s | +3.61e-05 |
| call | quadrature | 2.19277761 +/- 3.24e-05 | 0.858553% +/- 1.49e-03 pp | 5.22 s | +0.00e+00 |
| call | nested_mc | 2.19279564 +/- 2.63e-05 | 0.859383% +/- 1.21e-03 pp | 43.63 s | +1.80e-05 |
| call | learning | 2.19282870 +/- 6.31e-05 | 0.860903% +/- 2.90e-03 pp | 51.16 s | +5.11e-05 |

Agreement concerns the computed time-zero price at the retained grid;
it is not a comparison with an unavailable exact MKABSDE solution.
