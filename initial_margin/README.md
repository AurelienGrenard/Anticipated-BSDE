# Initial-margin CVaR MKABSDE

This experiment treats the one-dimensional initial-margin equation of Agarwal
et al. It computes the direct anticipated CVaR equation and the nonlinear and
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

## Backward formulation

Conditional cash invariance gives

```text
CVaR_t(V_t - V_(t+Delta))
  = V_t + CVaR_t(-V_(t+Delta)).
```

For variance reduction, the solver represents
`MVA_t = V_t - V_BS(t,S_t)` and its control on a normalized Hermite basis of
degree two. It separately regresses the conditional future-loss CVaR. The
affine dependence on the current total value is resolved by one scalar division
in the implicit update. Total values are always reconstructed before a future
loss is formed, so the target remains the original MKABSDE rather than an
equation for the MVA alone.

The two proxies contain no anticipated component. Their implementation reduces
to the standard implicit backward BSDE scheme with the same Hermite conditional
expectations. The nonlinear proxy uses the reconstructed `|Z|`; the linear
proxy freezes this source at the analytic Black--Scholes hedge.

## Three interchangeable CVaR estimators

The direct solver accepts one of three engines from `initial_margin/risk/`:

1. `quadrature`: Gauss--Legendre integration of the conditional Gaussian law,
   followed by the Rockafellar--Uryasev CVaR calculation;
2. `nested_mc`: antithetic inner simulations, exact empirical upper-tail
   averaging, then Hermite regression across current states;
3. `learning`: two-stage conditional VaR/expected-shortfall learning following
   Barrera et al., with a Softplus network for VaR and a frozen-feature linear
   ES regression.

For the monotone call, quadrature integrates the lower Gaussian `1-alpha` tail
directly. For the non-monotone butterfly, it discretizes the complete law,
sorts the `Q` weighted losses, and accumulates exactly the worst `1-alpha`
mass. The sort therefore costs `O(Q log Q)` per current state; it never acts on
the number of outer paths.

Quadrature is the retained paper method because dimension one reduces the
conditional law to one scalar integral. Nested MC and learning are independent
audits of that construction: at the calibrated settings all three produce the
same time-zero direct price to better than `1e-4`, while quadrature is faster
and adds no inner-sampling or optimization noise.

## Layout

```text
initial_margin/
├── config.py                    financial and time-grid parameters
├── problem.py                   payoff and analytic Black--Scholes controls
├── simulation.py                exact geometric-Brownian paths
├── risk/
│   ├── common.py                loss reconstruction and weighted CVaR
│   ├── config.py                validated estimator settings
│   ├── quadrature.py            deterministic Gauss--Legendre estimator
│   ├── nested.py                nested conditional Monte Carlo estimator
│   └── learning.py              two-stage conditional VaR/ES learner
├── solvers.py                   direct MKABSDE, two proxies, and no-IM control
├── evaluation.py                prices, deltas, and proxy distances
├── run_campaign.py              retained quadrature and proxy campaign
├── compare_risk_estimators.py   common-path three-engine audit
├── deterministic_reference.py  same-grid state-space validation
├── notebooks/                   executed result report
└── results/                     raw and aggregated JSON artifacts
```

The shared Hermite basis and FP64 ridge solve live in `hermite/`.

## Reproduce the paper campaign

From the repository root:

```bash
python -m initial_margin.run_campaign \
  --product all \
  --steps 50 100 200 \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --paths 200000 \
  --degree 2 \
  --margin-period 0.02 \
  --device cuda
```

The default quadrature sizes are 24 nodes for the call tail and 64 nodes for
the complete butterfly law. Ten seeds quantify combined outer Monte Carlo and
regression variability. Pass `--quadrature-nodes Q` only for a sensitivity
test.

Compare the three direct-MKABSDE risk engines on identical outer paths with:

```bash
python -m initial_margin.compare_risk_estimators \
  --product all \
  --steps 200 \
  --seeds 42 43 44 \
  --paths 200000 \
  --degree 2 \
  --device cuda
```

The retained audit uses `32,768` antithetic inner samples at `2,048` current
states for nested MC. Learning uses three Softplus layers of width 32, a batch
size of `131,072`, 2,048 initial Adam updates, and 128 warm-started updates per
earlier date. The inexpensive `--quick` mode checks code paths only; its
learning output is deliberately not an accuracy result.

Validate the quadrature solution independently of Monte Carlo regressions with:

```bash
python -m initial_margin.deterministic_reference \
  --product all \
  --steps 50 100 200 \
  --space-steps 6000 \
  --expectation-nodes 24 \
  --output initial_margin/results/deterministic_reference.json
```

This reference uses the same time-discrete equation and CVaR rule, but stores
conditional functions on a fine log-spot grid and evaluates transitions by
Gauss--Hermite quadrature. It is a fixed-grid regression check, not a closed
form or continuous-time exact solution.

## Interpretation

The retained tables report
`100 * MVA_0 / V_BS(0,S_0)`: the funding-induced price adjustment as a
percentage of the no-initial-margin Black--Scholes price. The raw MVA and the
total price remain available in the JSON artifacts.

The three CVaR engines agree on the direct MKABSDE price, so the observed gap
from the linear and nonlinear proxies is not a quadrature artifact. Relative
to the proxy MVA, the direct MVA is about 15% lower for the call and about 5%
higher for the butterfly.

No exact direct solution is available. Price refinement and same-grid
validation establish numerical stability, but do not by themselves identify
the full theorem-level `O(h)` squared-error rate in `(V,Z)`.
