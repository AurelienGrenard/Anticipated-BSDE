# Conditional-CVaR engines in the direct MKABSDE

All methods use identical outer paths, degree-two MVA/control regressions
in the paper runs, and the same backward update. Only the risk fit changes.

| Product | Horizon (years) | Method | Price (mean +/- seed std) | MVA (% of BS; mean +/- seed std) | GPU time | Mean difference vs quadrature |
|---|---:|---|---:|---:|---:|---:|
| butterfly | 0.02 | quadrature | 0.31143172 +/- 8.07e-07 | 0.775644% +/- 2.61e-04 pp | 52.91 s | +0.00e+00 |
| butterfly | 0.02 | nested_mc | 0.31143851 +/- 2.41e-05 | 0.777843% +/- 7.81e-03 pp | 59.73 s | +6.80e-06 |
| butterfly | 0.02 | learning | 0.31146300 +/- 1.28e-05 | 0.785767% +/- 4.15e-03 pp | 49.94 s | +3.13e-05 |
| call | 0.02 | quadrature | 2.19277835 +/- 3.24e-05 | 0.858587% +/- 1.49e-03 pp | 37.96 s | +0.00e+00 |
| call | 0.02 | nested_mc | 2.19279652 +/- 2.64e-05 | 0.859423% +/- 1.21e-03 pp | 42.88 s | +1.82e-05 |
| call | 0.02 | learning | 2.19282394 +/- 5.68e-05 | 0.860684% +/- 2.61e-03 pp | 48.59 s | +4.56e-05 |

Agreement concerns the computed time-zero price at the retained grid;
it is not a comparison with an unavailable exact MKABSDE solution.
