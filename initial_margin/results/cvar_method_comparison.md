# Conditional-CVaR engines in the direct MKABSDE

All methods use identical outer paths, Hermite regressions, and backward
updates. Only the conditional-risk estimator changes.

| Product | Method | Price | Seed std | IM correction | GPU time | Mean difference vs quadrature |
|---|---|---:|---:|---:|---:|---:|
| butterfly | quadrature | 0.31142686 | 8.04e-07 | 0.00239215 | 14.90 s | +0.00e+00 |
| butterfly | nested_mc | 0.31143805 | 2.41e-05 | 0.00240334 | 60.10 s | +1.12e-05 |
| butterfly | learning | 0.31146298 | 1.34e-05 | 0.00242828 | 51.94 s | +3.61e-05 |
| call | quadrature | 2.19277761 | 3.24e-05 | 0.01866591 | 5.22 s | +0.00e+00 |
| call | nested_mc | 2.19279564 | 2.63e-05 | 0.01868394 | 43.63 s | +1.80e-05 |
| call | learning | 2.19282870 | 6.31e-05 | 0.01871700 | 51.16 s | +5.11e-05 |

Agreement concerns the computed time-zero price at the retained grid;
it is not a comparison with an unavailable exact MKABSDE solution.
