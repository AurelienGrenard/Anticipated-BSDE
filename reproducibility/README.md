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

For the larger-margin-period table, `metadata.json` also records the SHA-256
of every solver source file at execution time, even if the working tree has
uncommitted edits. These fingerprints are more precise than the recorded base
Git revision alone. `runs.jsonl` is the source of the means in
`summary.json` and the generated `table_rows.tex`; tests recompute both.
The manuscript itself and source articles remain outside the public repository.
