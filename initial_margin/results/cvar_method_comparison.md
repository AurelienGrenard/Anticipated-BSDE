# Conditional-CVaR engines in the direct MKABSDE

All methods use identical outer paths, Hermite regressions, and backward
updates. Only the conditional-risk estimator changes.

| Product | Method | Price (mean +/- seed std) | MVA (% of BS; mean +/- seed std) | GPU time | Mean difference vs quadrature |
|---|---|---:|---:|---:|---:|
| butterfly | quadrature | 0.31143172 +/- 8.07e-07 | 0.775644% +/- 2.61e-04 pp | 53.37 s | +0.00e+00 |
| butterfly | nested_mc | 0.31143851 +/- 2.41e-05 | 0.777843% +/- 7.81e-03 pp | 61.16 s | +6.80e-06 |
| butterfly | learning | 0.31146300 +/- 1.28e-05 | 0.785767% +/- 4.15e-03 pp | 54.95 s | +3.13e-05 |
| call | quadrature | 2.19277835 +/- 3.24e-05 | 0.858587% +/- 1.49e-03 pp | 37.69 s | +0.00e+00 |
| call | nested_mc | 2.19279652 +/- 2.64e-05 | 0.859423% +/- 1.21e-03 pp | 44.31 s | +1.82e-05 |
| call | learning | 2.19282394 +/- 5.68e-05 | 0.860684% +/- 2.61e-03 pp | 56.29 s | +4.56e-05 |

Agreement concerns the computed time-zero price at the retained grid;
it is not a comparison with an unavailable exact MKABSDE solution.
