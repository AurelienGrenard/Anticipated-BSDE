# Markovian anticipated BSDEs — minimal numerical project

This directory is the compact numerical companion to:

> Cyril Bénézet, Stéphane Crépey, and Aurélien Grenard,  
> *Markovian Anticipated BSDEs: Well-posedness, Comparison, Feynman-Kac
> Representation, and Approximation*, 2026.

It implements the local Hermite regression scheme used for Examples 4.3,
5.3 (I), and 5.3 (II), and reproduces the two convergence diagnostics associated
with Theorem 5.5.

## Minimal layout

```text
./
├── ABSDEs.pdf                       associated article
├── hermite_absde/
│   ├── config.py                    experiment and time-grid parameters
│   ├── problems.py                  three ABSDEs and evaluation oracles
│   ├── hermite_regression.py        Hermite-3 basis and local regressors
│   ├── brownian.py                  time grid and Brownian simulation
│   ├── solver.py                    backward regression scheme
│   ├── convergence.py               independent error measurements
│   └── checkpoints.py               checkpoint persistence
├── run_campaign.py                  complete result-generation pipeline
├── make_figures.py                  six paper figures
├── notebooks/                       short executed reports
├── results/                         checkpoints and convergence JSON
├── figures/                         generated PNG figures
└── paper/numerical_experiments.tex  numerical section source
```

The numerical implementation has a single source of truth: `hermite_absde/`.
The notebooks contain no private solver or metric implementation.

## Numerical method

On a uniform grid, the solver works backward from the prescribed terminal
processes on `[T, T + delta]`. At each date it fits four conditional expectations
in this order:

```text
Ytilde_n = E_n[Y_(n+1)]
Z_n      = E_n[(Y_(n+1) - Ytilde_n) Delta_W_n / h]
A_n      = E_n[anticipated future quantity]
Y_n      = Ytilde_n + h f(t_n, A_n)
```

All four fits use the normalized Hermite basis

```text
H0(x) = 1
H1(x) = x
H2(x) = (x^2 - 1) / sqrt(2)
H3(x) = (x^3 - 3x) / sqrt(6)
```

with `x = W(t_n) / sqrt(t_n)`, ridge penalty `1e-10`, and an FP64 linear
solve. At time zero only the constant feature is active.

Closed-form solutions are isolated in `problems.py` and are used only by
out-of-sample evaluation. Training sees only the prescribed terminal data on
`[T, T + delta]`.

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduce one complete campaign

Run from the repository root:

```bash
python run_campaign.py \
  --example 5.3.1 \
  --steps 5 10 20 40 80 160 \
  --seeds 42 43 44 \
  --train-paths 1000000 \
  --grid-test-paths 200000 \
  --reconstruction-test-paths 100000 \
  --subdivisions 8
```

The script performs the complete published workflow:

1. train every grid and seed;
2. evaluate errors at grid dates on independent paths;
3. evaluate the left-constant reconstruction on eight substeps;
4. save the representative `N=20`, seed-42 checkpoint;
5. aggregate means, sample standard deviations, and adjacent-grid orders;
6. write both the raw campaign and reconstruction-summary JSON files.

The defaults select CUDA when available and otherwise use CPU. The complete
million-path campaign is computationally intensive. A quick smoke run is:

```bash
python run_campaign.py \
  --example 4.3 \
  --steps 5 10 \
  --seeds 42 \
  --train-paths 10000 \
  --grid-test-paths 5000 \
  --reconstruction-test-paths 5000 \
  --output /tmp/absde-smoke
```

## Reproduce figures and notebooks

Once all three summary JSON files and representative checkpoints exist:

```bash
python make_figures.py
jupyter execute notebooks/Example_4_3.ipynb
jupyter execute notebooks/Example_5_3_I.ipynb
jupyter execute notebooks/Example_5_3_II.ipynb
```

The quantile figures use 50,000 independent paths. The convergence figures plot
the two errors separately and do not add a synthetic reference slope.

## Interpretation of the order

Theorem 5.5 bounds squared errors by `C h`; the corresponding norm convergence is
therefore of order `sqrt(h)`. The supplied examples are polynomial in the
Brownian state and lie inside the Hermite-3 approximation space. They are clean
scheme checks, but they do not test spatial approximation outside that basis.

## Reproducibility

- Seeds, grids, path counts, dtype, ridge penalty, and subdivision count are
  written to the raw campaign JSON.
- Checkpoints are outputs used for the quantile figures, never training inputs.
- Summary JSON is strict JSON: the undefined first local order is stored as
  `null`.
- CPU and CUDA reductions may differ slightly because of floating-point
  reduction order.
