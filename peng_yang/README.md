# Peng--Yang ABSDE benchmarks

This experiment applies the local Hermite regression scheme to Examples 4.3,
5.3 (I), and 5.3 (II) of Peng and Yang. Their explicit solutions provide
independent values of both `Y` and `Z`, so the implementation can measure the
two squared errors appearing in the paper's time-discretization estimate.

## Numerical method

On a uniform grid, the solver works backward from the prescribed process on
`[T, T + delta]`. At every grid date it performs four scalar conditional
regressions:

```text
Ytilde_n = E_n[Y_(n+1)]
Z_n      = E_n[(Y_(n+1) - Ytilde_n) Delta_W_n / h]
A_n      = E_n[anticipated future quantity]
Y_n      = Ytilde_n + h f(t_n, A_n)
```

All four regressions use the shared normalized probabilists' Hermite basis

```text
H0(x) = 1
H1(x) = x
H2(x) = (x^2 - 1) / sqrt(2)
H3(x) = (x^3 - 3x) / sqrt(6)
```

with `x = W_t / sqrt(t)`, an FP64 ridge solve, and coefficients stored in the
configured output dtype. Only the constant feature is active at time zero.
Exact solutions are isolated in `problems.py` and are used only during
out-of-sample evaluation.

## Layout

```text
peng_yang/
├── config.py          grids, paths, seeds, and delay validation
├── problems.py        three ABSDEs and their evaluation-only exact solutions
├── simulation.py      time grid and Brownian simulation
├── regression.py      local per-date model containers and state normalization
├── solver.py          four-regression backward scheme
├── evaluation.py      grid and left-constant reconstruction errors
├── checkpoints.py     compact model persistence
├── run_campaign.py    complete three-example campaign
├── make_figures.py    six manuscript figures
├── notebooks/         executed result report
├── results/           raw rows, summaries, and three representative models
└── figures/           generated PNG files
```

The Hermite polynomials and ridge solve live in
`hermite/regression.py`; no second implementation exists here.

## Reproduce the paper results

From the repository root, the complete command is:

```bash
python -m peng_yang.run_campaign \
  --example all \
  --steps 5 10 20 40 80 160 \
  --seeds 42 43 44 \
  --train-paths 1000000 \
  --grid-test-paths 200000 \
  --reconstruction-test-paths 100000 \
  --subdivisions 8 \
  --device cuda
```

The command retrains every grid/seed pair, evaluates on independent paths,
saves the representative `N=20`, seed-42 checkpoint, and replaces the raw and
aggregated JSON files. To run only one example, pass `--example 4.3`,
`--example 5.3.1`, or `--example 5.3.2`.

Regenerate the six figures from the newly trained checkpoints and summaries:

```bash
python -m peng_yang.make_figures
```

The quantile figures use 50,000 independent evaluation paths. The convergence
figures display the two measured squared errors without adding a fitted or
synthetic reference slope.

## Interpretation

The theorem bounds the squared errors by `C h`; it therefore corresponds to an
`O(sqrt(h))` bound in norm. These polynomial examples lie inside the degree-3
Hermite approximation space. They are clean checks of the backward time scheme,
but not stress tests of spatial approximation outside the chosen basis.
