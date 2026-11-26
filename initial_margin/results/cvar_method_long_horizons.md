# Conditional-CVaR engines in the direct MKABSDE

All methods use identical outer paths, degree-two MVA/control regressions
in the paper runs, and the same backward update. Only the risk fit changes.

| Product | Horizon (years) | Method | Price (mean +/- seed std) | MVA (% of BS; mean +/- seed std) | GPU time | Mean difference vs quadrature |
|---|---:|---|---:|---:|---:|---:|
| butterfly | 0.10 | quadrature | 0.31364582 +/- 2.40e-06 | 1.492103% +/- 7.77e-04 pp | 49.47 s | +0.00e+00 |
| butterfly | 0.10 | nested_mc | 0.31365622 +/- 4.49e-05 | 1.495467% +/- 1.45e-02 pp | 56.20 s | +1.04e-05 |
| butterfly | 0.10 | learning | 0.31369801 +/- 4.67e-06 | 1.508991% +/- 1.51e-03 pp | 49.38 s | +5.22e-05 |
| butterfly | 0.50 | quadrature | 0.31525987 +/- 3.04e-06 | 2.014389% +/- 9.85e-04 pp | 31.71 s | +0.00e+00 |
| butterfly | 0.50 | nested_mc | 0.31526878 +/- 5.37e-05 | 2.017272% +/- 1.74e-02 pp | 35.67 s | +8.91e-06 |
| butterfly | 0.50 | learning | 0.31544579 +/- 3.62e-05 | 2.074550% +/- 1.17e-02 pp | 50.37 s | +1.86e-04 |
| call | 0.10 | quadrature | 2.20710843 +/- 6.54e-05 | 1.517711% +/- 3.01e-03 pp | 34.79 s | +0.00e+00 |
| call | 0.10 | nested_mc | 2.20712830 +/- 8.88e-05 | 1.518625% +/- 4.09e-03 pp | 39.59 s | +1.99e-05 |
| call | 0.10 | learning | 2.20723429 +/- 1.18e-04 | 1.523500% +/- 5.45e-03 pp | 49.33 s | +1.26e-04 |
| call | 0.50 | quadrature | 2.21520833 +/- 9.10e-05 | 1.890272% +/- 4.18e-03 pp | 23.43 s | +0.00e+00 |
| call | 0.50 | nested_mc | 2.21522408 +/- 1.59e-04 | 1.890997% +/- 7.30e-03 pp | 26.42 s | +1.58e-05 |
| call | 0.50 | learning | 2.21586903 +/- 3.23e-04 | 1.920662% +/- 1.48e-02 pp | 47.01 s | +6.61e-04 |

Agreement concerns the computed time-zero price at the retained grid;
it is not a comparison with an unavailable exact MKABSDE solution.
