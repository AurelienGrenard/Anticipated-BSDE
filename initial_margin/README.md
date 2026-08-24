# Initial-margin CVaR MKABSDE

This experiment treats the one-dimensional initial-margin equation of Agarwal
et al. It computes the direct anticipated CVaR equation and the non-linear and
linear small-margin-period proxy BSDEs for an at-the-money call and a symmetric
butterfly.

## Financial case

```text
S0 = K = 20, T = 1
sigma = 0.25, r = mu = 0.02
funding spread R = 0.02
CVaR confidence alpha = 0.99
margin period Delta = 0.02
```

The butterfly payoff is

```text
(S-(K-2))^+ - 2(S-K)^+ + (S-(K+2))^+.
```

## Direct equation and CVaR target

Conditional cash invariance gives

```text
CVaR_t(Y_t - Y_(t+Delta))
  = Y_t + CVaR_t(-Y_(t+Delta)).
```

Only the future-loss term is regressed. The remaining dependence on `Y_t` is
handled exactly by one scalar division in the implicit update.

For a call, monotonicity identifies the lower Gaussian `1-alpha` tail. A
Gauss--Legendre rule integrates that tail directly. For the butterfly, whose
delta changes sign, the complete conditional Gaussian law is discretized by
Gauss--Legendre nodes; the weighted losses are sorted and the worst
`1-alpha` probability mass is averaged. No monotonicity shortcut is used.

The two proxies no longer contain an anticipated component. Their numerical
solver therefore reduces to the classical implicit backward BSDE scheme, with
conditional expectations evaluated by the same Hermite regressions.

## Black--Scholes difference formulation

All three IM equations are solved for the small difference from the analytic
no-IM Black--Scholes value. Thus the regressions learn only
`D = V - V_BS` and `Z_D = Z - Z_BS`; the analytic value and hedge are restored
inside the non-linear driver and the CVaR target. This is essential for the
butterfly because its signed, localized Black--Scholes delta is poorly
represented by a low-degree global polynomial before taking its absolute
value. The retained basis remains normalized Hermite of degree three.

In dimension one, `deterministic_reference.py` supplies a same-grid check of
the direct solver. Both implementations use the Black--Scholes difference
formulation, the same implicit time step, and the same Gauss--Legendre CVaR
rule. Only the conditional functions change: the main solver uses Monte Carlo
paths and Hermite regressions, whereas the reference stores them on a fine
log-spot grid and integrates the Gaussian transition by Gauss--Hermite
quadrature with interpolation.

## Layout

```text
initial_margin/
├── config.py          financial and numerical parameters
├── problem.py         payoff and analytic Black--Scholes controls
├── simulation.py      exact geometric-Brownian simulation
├── risk.py            direct conditional-CVaR targets
├── deterministic_reference.py  same-grid deterministic validation
├── solvers.py         direct, non-linear proxy, linear proxy, and no-IM control
├── evaluation.py      prices, deltas, errors, and article controls
├── run_campaign.py    complete call and butterfly campaign
├── notebooks/         executed result report
└── results/           raw rows and aggregated JSON summaries
```

The Hermite basis and ridge solve are imported from
`hermite/regression.py`.

## Reproduce the paper results

From the repository root:

```bash
python -m initial_margin.run_campaign \
  --product all \
  --steps 50 100 200 \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --paths 200000 \
  --degree 3 \
  --margin-period 0.02 \
  --device cuda
```

The default quadrature sizes are 24 nodes for the monotone call tail and 64
nodes for the complete butterfly law. The ten independent seeds quantify the
combined Monte Carlo and regression variability. Pass `--quadrature-nodes Q`
only for a deliberate sensitivity test. The command writes one JSON artifact
per product.

Validate the direct equation independently of the regressions with:

```bash
python -m initial_margin.deterministic_reference \
  --product all \
  --steps 50 100 200 \
  --space-steps 6000 \
  --expectation-nodes 24 \
  --output initial_margin/results/deterministic_reference.json
```

The product-specific default CVaR rules use 64 call-tail nodes and 256 nodes
for the complete butterfly law. Increase the spatial or quadrature sizes for a
deliberate sensitivity check.

An inexpensive end-to-end check is:

```bash
python -m initial_margin.run_campaign --quick --device cpu
```

## Interpretation

At fixed margin period, refinement in `h` diagnoses the direct numerical
scheme. The deterministic reference isolates Monte Carlo and regression effects
at each retained time grid, and the price-level refinements are stable. The
distance between the direct MKABSDE and either proxy also contains the proxy
model error. Moreover, a time-zero price comparison alone is not a direct test
of the full squared-error estimate, so no theorem-level `O(h)` claim is inferred
from these three values.

The proxy values can be compared with the article's analytic and finite-
difference references. The article has no direct MKABSDE reference; the
one-dimensional deterministic reference included here supplies that missing
same-grid validation without claiming a closed-form exact solution.
