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

## Layout

```text
initial_margin/
├── config.py          financial and numerical parameters
├── problem.py         payoff and analytic Black--Scholes controls
├── simulation.py      exact geometric-Brownian simulation
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
  --seeds 42 43 44 \
  --paths 200000 \
  --degree 3 \
  --margin-period 0.02 \
  --device cuda
```

The default quadrature sizes are 24 nodes for the monotone call tail and 64
nodes for the complete butterfly law. Pass `--quadrature-nodes Q` only for a
deliberate sensitivity test. The command writes one JSON artifact per product.

An inexpensive end-to-end check is:

```bash
python -m initial_margin.run_campaign --quick --device cpu
```

## Interpretation

At fixed margin period, refinement in `h` diagnoses the direct numerical
scheme. The distance between the direct MKABSDE and either proxy also contains
the proxy model error and cannot by itself establish a time-discretization
order. The retained three-grid study is stable but does not exhibit a robust
`O(h)` slope; regression and Monte Carlo noise may already dominate the small
adjacent-grid differences.

The linear proxy can be compared with the article's reported values. The direct
MKABSDE has no independent oracle in that article, so its refinement stability
must be described as numerical evidence, not validation of the true solution.
