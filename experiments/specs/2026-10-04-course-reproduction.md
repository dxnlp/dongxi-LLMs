# Course routes and isolated CPU reproduction

DXI-17 specification, frozen before the new installation/checks. This is a
material-readiness experiment, not learner progress or a model capability run.

## Hypothesis and interventions

One declarative registry can describe fifteen chapters, twenty-eight day routes
and optional extensions without silently admitting missing, unknown or empty
notebooks. A separate locked CPU environment can execute the repository's tests,
source checks and a representative notebook panel without using Spark CUDA
packages, contacting a model hub or changing the existing platform environment.

Intervene on duplicate/unknown paths, day/chapter mismatch, missing routes,
dependency cycles, placeholder cells and required explanatory previews. These
malformations must fail closed. Preserve learner exercise cells: unfinished
explicit exercises may be skipped in execution copies, complete references may
not be skipped. The registry is a routing contract, not an assertion of mastery.

## Environment and bounded execution

- Python3.12; a new temporary environment, never the existing Spark environment.
- Resolve package wheels through public package registries; use an explicit
  Linux CPU-only PyTorch source. No model/dataset acquisition or GPU job.
- Freeze the resolved dependency graph in `uv.lock`; install with `--locked`.
- Run under `CUDA_VISIBLE_DEVICES=''`, `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1`. Retain exact versions, lock/source hashes, platform,
  commands, exit codes, failures and actual scope.
- Selected fresh-kernel panel: Day1 checkpoint interface, Day3 logits/softmax,
  Day10 grading, Day15 judge audit and Day20 probability accounting. Five
  notebooks, per-cell timeout240seconds; no full-course or Mac pass inferred.
- Limit local reproduction to20minutes wall time via an external timeout; keep
  the existing25GiB pre-notebook host-memory guard. Package installation is
  permitted only in the new environment and leaves the shared kernel untouched.

## Acceptance and exclusions

All schema/failure tests and repository tests pass; source math/navigation
checks pass; every selected notebook has a new retained output/hash/failure
manifest. No source notebook or learner work is overwritten. CI definitions
have immutable action pins, read-only repository permissions, an offline CPU
execution lane and retained evidence/failures.

Linux actual evidence, Mac verification pending, and CI source readiness are
separate rows. No hosted CI run, live GitHub math rendering, pretrained learning,
Mac binary compatibility or full notebook rerun is proved by this check.
