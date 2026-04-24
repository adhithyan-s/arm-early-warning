# ADR 003: CI Linting Configuration

## Status
Resolved

## Context
After setting up GitHub Actions CI with ruff linting, the pipeline consistently failed despite local linting passing cleanly. The error messages showed paths like `src/amr-early-warning/src/models/train.py` which don't exist in the repository.

## Root Cause
When GitHub Actions checks out a repo named `arm-early-warning`, the runner workspace path becomes:
`/home/runner/work/arm-early-warning/arm-early-warning/`

Running `ruff check src/ tests/` from this directory causes ruff to report paths relative to the parent workspace, making them appear as `src/amr-early-warning/src/...`. 
This is a display artifact, but the real issue was that `pip install -e .` creates an `amr_early_warning.egg-info` 
folder inside `src/` which contains cached copies of source files. 
Ruff was picking up these cached copies which reflected an older version of the code before unused imports were removed.

## Decision
Replace `ruff check src/ tests/` with explicit subdirectory paths:
ruff check src/api/ src/data/ src/features/ src/models/ tests/
This explicitly targets only real source directories, skipping anything the package installer creates inside `src/`.

## What I Learned
- `pip install -e .` (editable install) creates egg-info artifacts inside `src/` that linters can pick up
- GitHub Actions path display is relative to the runner workspace parent, not the repo root — this can make error paths look misleading
- Always run `ruff check` with explicit paths in CI rather than broad directory globs when using editable installs
- Local lint passing does not guarantee CI lint passing if the install artifacts differ between environments

## Consequences
- CI now passes consistently
- Linting is scoped to exactly the directories we own
- If new top-level source directories are added, they must be explicitly added to the lint command in ci.yml