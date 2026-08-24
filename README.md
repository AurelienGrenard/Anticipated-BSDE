# Markovian anticipated BSDEs — reproducibility project

This repository contains the reproducibility code and retained numerical
outputs for:

> Cyril Bénézet, Stéphane Crépey, and Aurélien Grenard,  
> *Markovian Anticipated BSDEs: Well-posedness, Comparison, Feynman-Kac
> Representation, and Approximation*, 2026.

Two independent numerical studies share one implementation of normalized
Hermite regression: the three Peng--Yang benchmarks used to test the time
scheme, and the initial-margin MKABSDE with conditional CVaR.

## Project layout

```text
./
├── hermite/                shared Hermite basis and FP64 ridge regression
├── peng_yang/              three explicit ABSDE benchmarks
│   ├── README.md
│   ├── notebooks/
│   ├── results/
│   └── figures/
├── initial_margin/         direct CVaR MKABSDE and its two proxy BSDEs
│   ├── README.md
│   ├── notebooks/
│   └── results/
├── tests/                  common, solver, artifact, and benchmark checks
├── reproducibility/        artifact checksums and figure correspondence
├── pyproject.toml          package metadata and command-line entry points
├── requirements.txt        exact Python dependency versions
├── CITATION.cff
└── LICENSE
```

The shared package is intentionally small. It contains only the orthonormal
Hermite basis, the scalar ridge solve, and prediction from fitted coefficients.
State normalization, equations, simulations, and backward updates remain inside
their respective experiments.

The manuscript and the source articles are intentionally kept outside this
code repository.

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

The pinned environment is the reference environment. Complete reproduction is
intended for a CUDA-capable GPU; all tests and smoke campaigns also run on CPU.

## Complete reproduction

Run from the repository root:

```bash
python -m peng_yang.run_campaign --device cuda
python -m peng_yang.make_figures
python -m initial_margin.run_campaign --device cuda
python -m initial_margin.oracle \
  --output initial_margin/results/deterministic_oracle.json
pytest -q
sha256sum --check reproducibility/artifacts.sha256
```

The first command retrains all three Peng--Yang examples on six time grids and
three seeds. The third command recomputes both initial-margin products on three
grids and ten seeds. Neither command reads a checkpoint as a training input.

For inexpensive pipeline checks:

```bash
python -m peng_yang.run_campaign --quick --device cpu
python -m initial_margin.run_campaign --quick --device cpu
```

Smoke outputs are written under `/tmp` and never replace the retained paper
artifacts.

## Executed result notebooks

- `peng_yang/notebooks/peng_yang_results.ipynb` records the full command,
  convergence tables, and six paper figures.
- `initial_margin/notebooks/initial_margin_results.ipynb` records the full
  command and the call/butterfly initial-margin correction tables.

The notebooks are presentation layers only: they read committed JSON and PNG
artifacts and contain no solver implementation.

## Integrity and interpretation

`reproducibility/artifacts.sha256` identifies the exact checkpoints, JSON files,
figures, and executed notebooks retained with this release. Recompute the
manifest only after a deliberate full reproduction.

Fixed seeds make each campaign repeatable on a fixed software stack and GPU.
Floating-point reductions may vary slightly across devices and PyTorch/CUDA
versions. The initial-margin direct solution has no external closed form; its
independent one-dimensional state-grid oracle validates the implementation
numerically without turning that reference into an exact solution.

The code is released under the BSD 3-Clause License. Citation metadata are in
`CITATION.cff`.
