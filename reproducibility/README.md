# Reproducibility records

`artifacts.sha256` fixes the exact numerical outputs retained with the paper:
Peng--Yang checkpoints, JSON summaries and figures, initial-margin JSON
results (including the five larger-margin-period campaigns), and both executed
notebooks. `figure-map.csv` maps every Peng--Yang
panel to its numerical input and generated image.

From the repository root, verify every retained byte with:

```bash
python tools/verify_artifacts.py
```

Regenerate the checksum manifest only after deliberately rerunning the complete
commands in the top-level README and reviewing the new numerical results. GPU
runs are reproducible for the recorded seeds on a fixed stack; small floating-
point differences may remain across GPU architectures and PyTorch/CUDA builds.

Every initial-margin campaign records package/device metadata and the SHA-256
of every solver source file at execution time. The `runtime` entry is stored
inside the main JSON files and inside `metadata.json` for the larger-margin
campaigns. These fingerprints identify the numerical implementation more
precisely than the base Git revision alone. For the larger-margin table,
`runs.jsonl` is the source of the means in `summary.json` and the generated
`table_rows.tex`; tests recompute both.

Checksums verify the retained files, not statistical agreement of a fresh run.
New runs contain new timings and provenance, so even numerically identical
prices need not produce byte-identical JSON. Reproduce into separate output
directories and compare the numerical quantities, not those new file hashes.
The manuscript itself and source articles remain outside the public repository.
