# DXI-13 — measured sampling support and probability accounting

Status: passed bounded CPU reference,2026-10-04. The preregistered
[specification](../specs/2026-10-04-sampling-support.md) precedes measurements.
[Portable JSON evidence](2026-10-04-sampling-support.json) stores actual values,
gradient vectors, sampled actions/behavior log probabilities, source hashes,
commands, environment, check outputs and fresh-kernel identity.

This initial17-test report is historical. Independent acceptance review found
and repaired extreme-logit numerical failures; the
[21-test hardening report](2026-10-04-sampling-support-hardening.md) records the
current source hashes and repeated notebook/shared checks without overwriting
these original measurements.

## What was implemented and observed

The original [finite categorical module](../../src/dongxi_llms/sampling_likelihood_lab.py)
separates raw model, actual transformed collection and declared target.
Filtering order is temperature→top-k→renormalize→top-p→renormalize; ties use
smaller action ID and top-p includes the crossing action. Collection records
detach and include all16 seeded actions and their actual likelihoods.

For raw probabilities0.55,0.30,0.15 and rewards0,1,4:

| Quantity | Measured finite value |
|---|---|
| T=0.5, k=2, p=0.9 behavior | 0.770700637,0.229299363,0 |
| T=1 conditional target on fixed support | 0.647058824,0.352941176,0 |
| Matched temperature/support ratios | 1,1 |
| T=1 conditional-target/behavior ratios | 0.839572193,1.539215686 |
| Direct conditional expected reward | 0.352941176 |
| Correct behavior denominator | 0.352941176 |
| Deliberately wrong raw-model denominator | 0.269763957 |
| Raw-target mass excluded by collection | 0.15 |
| Raw expected reward contribution excluded | 0.60 |

Direct and corrected current-logit gradients agree:
`[−0.228373702,+0.228373702,0]`. The wrong denominator gives
`[−0.174553148,+0.174553148,0]`. These are exact enumeration/autograd outputs,
not estimates from the16 sampled draws. The latter check stored record identity.

Requesting the original raw full-support target rejects action2 as missing
behavior support. Zeroing its ratio is not a repair. Conditional fixed support
is an explicitly different objective; temperature must still be accounted for.
The original raw objective can instead use a full-support collector. None of
these statements establishes general autoregressive off-policy correctness.

## KL values versus KL loss gradients

With full-support reference0.20,0.50,0.30, exact forward KL and both k1/k3
expectations have value0.299160737nats at the fresh state. Their gradients
under different differentiation contracts are:

| Contract | Current-logit gradient |
|---|---|
| Exact categorical KL | +0.391842096,−0.242995908,−0.148846188 |
| Frozen-sample k1 term alone | 0,0,0 |
| Frozen-sample k3 term alone | +0.350000000,−0.200000000,−0.150000000 |
| Entire importance-weighted k1 expectation | Matches exact KL |
| Entire importance-weighted k3 expectation | Matches exact KL |

This isolates the derivative of the action distribution at a fixed state.
Correct KL values do not identify the intended gradient. The weighting identity
does not settle trajectory state-distribution gradients or prescribe every PPO
regularizer. The chapter states the objective, direction and detach conventions.

## Acceptance checks

- 17 independent focused tests pass: analytic/autograd agreement, wrong
  denominator, missing support, matched ratio1, temperature dependence,
  deterministic ties, filter normalization/order, invalid/overflow/underflow
  inputs, detached collection/reward/old/reference/advantage tensors and live
  current gradients.
- EOS is included as a sampled action; prompt, padding and post-stop positions
  are excluded. EOS-as-padding, first-action EOS, cap truncation and unclassified
  early end are tested. Bootstrap permission is a continuing-task convention;
  finite-horizon tasks may declare the cap terminal instead.
- Padding with NaN likelihood placeholders remains safe because valid positions
  are selected before arithmetic; multiplying NaN by zero is not used.
- The [new Day20 notebook](../../notebooks/day-20/03_behavior_probabilities_and_support.ipynb)
  executes all7 code cells in a fresh CPU kernel and produces4 original figures.
  The two existing Day20 notebooks also pass. Notebook sources were not overwritten
  by execution; completed reference copies remain in the verification directory.
- All4 saved previews were directly inspected: probability comparison,
  expected-reward/gradient comparison, KL value/gradient comparison, and response
  boundary mask. Axes, common scales, numerical annotations and captions explain
  what is measured versus constructed. No animation was produced.
- The recorded book source check passes; source check is not live GitHub rendering.
  Chapter14, worked solutions11–13 and the lab route integrate the probability
  bridge without adding a chapter or changing learner progress.

An initial development test exposed an incompatible positional argument to
Torch's stable argsort; changing it to the explicit keyword form corrected all
eight affected checks. A report command initially used a nonexistent temporary
manifest path and failed before report creation. Rerunning with the actually
printed fresh-kernel manifest path produced this successful evidence. Neither
failure is silently interpreted as experiment success.

## Environment and reproduction

Spark Linux/aarch64; Python3.12.14, Torch2.13.0+cu130, Matplotlib3.10.8. Tensors
and execution are CPU float64; seed2020. A CUDA-capable installed Torch does not
mean CUDA was used. Actual Git base/dirty identity and sampled resource
measurements are in JSON. Notebook MemAvailable observations are pre-execution
samples, not continuous memory monitoring; report-process RSS excludes a claim
about GPU or system-wide peak memory.

```bash
PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m unittest discover -s tests -p test_sampling_likelihood_lab.py -v
/home/dongxi/dgx-spark-dongxi/.venv/bin/python scripts/verify_course_notebooks.py --days 20 --kernel dgx-spark-native
# Substitute the actual freshly printed manifest and an unused report path.
PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m dongxi_llms.sampling_likelihood_lab --report /tmp/NEW-sampling-support-report.json --notebook-manifest /tmp/ACTUAL-notebook-run/manifest.json --export-previews
python3 scripts/check_book_math.py
```

The report exporter updates only this lesson's generated PNG previews and
refuses to overwrite historical JSON evidence. All new prose, source, fixtures
and figures are original; the
[pinned sampling-mask reference](https://github.com/rasbt/reasoning-from-scratch/blob/a788466dc85cfe8617b6c0b809ed4ab9084485de/ch07/03_rlvr_grpo_scripts_advanced/7_7_improvements/deepseek_v32_style.py)
is a mechanism source, not a copied implementation or imported result.

## Boundary

This completes the CPU probability instrument, not a full PPO implementation,
universal off-policy correction, learned critic, model-scale stability claim or
reasoning capability experiment. No model/data acquisition, API, GPU training,
environment installation, server, animation production or external publication
was performed. Mac execution and live GitHub rendering remain unverified.
The existing temperature-one/full-support Qwen baseline is unchanged. Learner
position remains Day9; prepared Day20 material does not advance mastery.
