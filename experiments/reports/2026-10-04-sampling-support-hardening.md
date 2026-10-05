# DXI-13 — numerical acceptance hardening

Status: passed bounded CPU reference,2026-10-04. This supplements the
[initial report](2026-10-04-sampling-support.md), preserving its historical
JSON rather than overwriting it. [New portable evidence](2026-10-04-sampling-support-hardening.json)
records revised module/test/specification hashes, repeated normal fixture values,
all check commands and outputs, and the fresh notebook reference identity.

## Failures and repair

Independent acceptance review found two failures in the original instrument:

- Float64 logits `[0,-1000]` exponentiate to a numerical zero. The full-support
  k3 and importance-weighted audit then returned NaNs, although its inputs were
  finite. The hardened audit rejects target/reference probability underflow,
  unrepresentable log probabilities, overflowing k3 ratios and nonfinite values
  or gradients explicitly. It does not silently reinterpret underflow as a
  deliberate support filter. Importance ratios use saved log probabilities to
  avoid an unnecessary reciprocal of a tiny positive behavior probability.
- A target supported only on action0 and reference logits `[1e308,-1e308]`
  returned NaN through an excluded `0 * infinity`. Exact KL now masks excluded
  target and reference log probabilities before subtraction. This case returns
  zero KL and finite zero target gradients. Including action1 instead rejects
  the unrepresentable retained reference log probability.

Four added regression tests also cover underflow in either audit distribution,
overflowing k3 with still-positive float64 probabilities, matching distributions
with representable subnormal probabilities, and retained target underflow in
exact KL. An explicitly one-action conditional objective remains different from
the original full-support objective; it is not an automatic underflow repair.

## Repeated evidence

- All21 focused sampling-likelihood tests pass.
- All218 repository tests pass with CUDA hidden. This count describes the
  inspected shared-tree revision; it is not a claim about later edits.
- The book source check passes:54 Markdown files,1099 expressions,zero issues.
- All three Day20 notebooks pass in fresh `dgx-spark-native` CPU kernels. The
  support notebook executes all7 code cells and produces4 figures; the existing
  two companions each execute4 code cells and produce2 figures.
- All8 recorded source hashes and all4 regenerated preview hashes match their
  current files. The KL-gradient preview was inspected again and remains
  readable. Notebook source and learner cells were not rewritten.
- The normal three-action fixture's expectation values and all KL gradients
  match the initial report exactly in this rerun. No teaching objective changed.

The fresh manifest is `/tmp/dongxi-course-check-2pj73gw8/manifest.json`; its hash,
kernel, notebook source hash and executed-reference path are captured in the
new portable report. Temporary executed notebooks are diagnostics, not durable
course source. The report command uses an unused output path and records both
the focused and entire shared regression suite, not only a favorable subset.

## Boundary

These are exact finite categorical CPU checks, not pretrained model-scale or
GPU training evidence. No model/data acquisition, installation, external API,
animation, persistent server or external publication occurred. The shared suite
includes temporary local HTTP test fixtures, not a course service. The temperature-one/full-support Qwen
baseline and normal fixture objectives remain unchanged. Mac execution and live
GitHub math rendering remain unverified; learner progress is not advanced.
