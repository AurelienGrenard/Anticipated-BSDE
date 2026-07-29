# CVaR MKABSDE experiment

This autonomous experiment applies the local Hermite backward scheme directly
to the one-dimensional initial-margin MKABSDE of the associated article, then
solves its non-linear and linear small-margin-period approximations.

## Financial case

The code covers the article's at-the-money European call and symmetric
butterfly. The butterfly payoff is

```text
(S-(K-2))^+ - 2(S-K)^+ + (S-(K+2))^+.
```

The default market case is:

```text
S0 = 20, K = 20, T = 1
sigma = 0.25, r = mu = 0.02
funding spread R = 0.02
CVaR confidence alpha = 0.99
margin period Delta = 0.02
```

The direct equation contains

```text
CVaR_t(Y_t - Y_(t+Delta))
  = Y_t + CVaR_t(-Y_(t+Delta)).
```

The current value is therefore handled implicitly but exactly, by one scalar
division at each date. For a call, the future value is monotone in the future
spot. The conditional CVaR of its negative is evaluated by Gauss-Legendre
quadrature over the lower `1-alpha` Gaussian tail. This avoids a prohibitively
noisy nested Monte Carlo at 99% confidence.

Because unconstrained Hermite polynomials can oscillate in an extreme tail,
future call values used by the CVaR quadrature are projected onto the financial
shape constraints `0 <= value <= 1.05 * spot` and monotonicity in spot. This
projection is applied only to tail evaluation; all four backward conditional
regressions remain the declared Hermite regressions.

For the butterfly, this monotone specialization is deliberately disabled.
Its delta changes sign, so the worst future losses can come from either side
of the conditional Gaussian law. The direct solver evaluates the complete law
by deterministic quadrature, sorts the resulting losses, and averages exactly
the weighted worst `1-alpha` probability mass. It enforces only the payoff
bounds `0 <= value <= 1.05 * 2`, never monotonicity.

## Layout

```text
cvar_mkabsde/
├── cvar_mkabsde/
│   ├── config.py       financial and numerical parameters
│   ├── problem.py      Black-Scholes model, payoff, and analytic controls
│   ├── hermite.py      normalized Hermite conditional regressions
│   ├── simulation.py   exact geometric-Brownian simulation
│   ├── solvers.py      direct, non-linear, and linear BSDE solvers
│   └── evaluation.py   prices, deltas, errors, and convergence orders
├── run_experiment.py   command-line experiment
├── tests/
└── results/
```

## Run

From the repository root:

```bash
python cvar_mkabsde/run_experiment.py \
  --product call \
  --steps 50 100 200 \
  --paths 200000 \
  --seeds 42 43 44
```

Use `--product butterfly --quadrature-nodes 64` for the non-monotone case.
More nodes are useful there because the CVaR boundary must be located after
sorting the full conditional distribution rather than known in advance.

The default local basis is the same normalized Hermite degree-three family as
the main ABSDE examples.

`Delta / h` must be an integer. The JSON output keeps the discretization study
separate from the small-`Delta` approximation error.

## Interpretation

At fixed `Delta`, refinement in `h` diagnoses the direct numerical scheme.
Distances between the direct MKABSDE and the two proxy BSDEs contain an
irreducible model error in `Delta` and should plateau as `h` becomes small.
They must not be used alone to claim a time-discretization order.

## Retained ATM-call results

With 200,000 paths, seeds 42/43/44, Hermite degree 3, and 24 CVaR quadrature
nodes, the Black-Scholes-control-variate prices are:

| time steps | direct MKABSDE | non-linear proxy | linear proxy |
|---:|---:|---:|---:|
| 50 | 2.192023 | 2.196519 | 2.196019 |
| 100 | 2.191963 | 2.196463 | 2.195972 |
| 200 | 2.191948 | 2.196442 | 2.195948 |

The article reports 2.1959 for the non-linear ATM-call proxy and 2.1741 without
initial margin. The direct value has no independent oracle in the article; its
stability under time refinement is therefore evidence, not a proof of
correctness. The raw common-random-number paths exhibit ordinary Monte Carlo
bias, so the reported values add each learned IM correction to the analytic
Black-Scholes no-IM price, following the paper's difference-control idea.

## Retained ATM-butterfly results

The full non-monotone campaign uses 200,000 paths, seeds 42/43/44, Hermite
degree 3, and 64 full-law quadrature nodes:

| time steps | direct MKABSDE | non-linear proxy | linear proxy |
|---:|---:|---:|---:|
| 50 | 0.310358 | 0.310245 | 0.311271 |
| 100 | 0.310362 | 0.310240 | 0.311293 |
| 200 | 0.310355 | 0.310232 | 0.311304 |

These are Black-Scholes-control-variate prices. The article gives 0.3090
without initial margin and 0.3112 for its reported ATM butterfly approximation.
The linear proxy agrees closely. Degree 3 under-resolves the localized,
sign-changing endogenous `Z` of the non-linear proxy: at `N=50`, exploratory
degree-5 and degree-7 runs move its corrected price to 0.31067 and 0.31091.
Those higher-degree global polynomials also oscillate strongly in remote
states, so they are diagnostics rather than retained production settings.

The direct MKABSDE settles near 0.31036. Its adjacent three-grid differences
are already at a few parts in one million and are not monotone, so this
experiment does not support an empirical `O(h)` claim from only these three
levels. It does show numerical stability and the expected two-sided state
dependence without using the call's monotonicity shortcut.
