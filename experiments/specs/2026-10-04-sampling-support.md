# DXI-13 — sampling likelihood, support and gradient boundaries

Premeasurement specification,2026-10-04. Mode: bounded CPU reference. No model
weights, datasets, APIs, CUDA jobs, installations or services are required.
Original finite categorical fixtures; no upstream prose or implementation copied.

## Question and declared objectives

Distinguish raw model probabilities, transformed collection probabilities and
the separately declared target objective. Hypothesis: exact importance weighting
reproduces a finite target expectation and its gradient only when the target's
support is covered and the recorded behavior denominator is correct.

Use raw probabilities `[0.55,0.30,0.15]`, rewards `[0,1,4]`, temperature0.5,
top-k2 then top-p0.9. Freeze the resulting collection support. Compare the raw
full-support target with a declared conditional target on that support, using
temperature1 or0.5 explicitly. The excluded third action deliberately has
nonzero target mass and a high reward: this is a missing-support failure, not
an acceptable zero-probability convention.

Predict before enumeration:

- Behavior and an identically transformed frozen-support target have ratio1.
- A conditional target with temperature1 needs a nontrivial ratio, even when
  its logits and support match collection. Saving only a mask is insufficient.
- Correct recorded denominators recover the conditional expectation/gradient;
  replacing them with untransformed model likelihoods changes both.
- Full-support target optimization cannot be recovered from truncated behavior
  by masking its probabilities. Reject it or explicitly change the objective.
- At a fresh full-support policy, k1 and k3 have equal expected KL values but
  their frozen-sample autodiff gradients differ. Exact categorical KL and a
  fully differentiated importance-weighted expectation agree instead.
- Response masks include sampled EOS, omit prompt/pad/post-stop positions and
  distinguish terminal EOS from a token-limit boundary.

## Implementation and acceptance

Source: `src/dongxi_llms/sampling_likelihood_lab.py`. Independent unit tests:
`tests/test_sampling_likelihood_lab.py`. Visual reference:
`notebooks/day-20/03_behavior_probabilities_and_support.ipynb`.

Require exact float64 enumeration, analytic/autograd agreement, deterministic
filter tie order, top-p crossing-token retention, invalid-input rejection,
recorded detached behavior likelihoods and support validation. Test old-policy,
reference and advantage detachment; current-policy gradients must remain live.
Test EOS/prompt/padding/truncation masks, including EOS-as-padding and an
unclassified early stop rejected rather than silently called truncation.

Run focused tests and a fresh `dgx-spark-native` CPU kernel with CUDA hidden.
Save actual fixture vectors, values, gradients, source hashes, Python/Torch
identity, base Git commit/dirty state, command, exit results and limitations.
Render original white-canvas probability, expectation/gradient and KL-gradient
plots; inspect previews. The notebook and report do not train an LLM or establish
model-scale stability, quality or speed. Existing temperature-one/full-support
Qwen RLVR behavior remains unchanged. Mac compatibility is not verified here.

## Acceptance hardening,2026-10-04

An independent acceptance review subsequently found two numerical failures in
the first implementation, not new evidence for the normal fixture. Finite
float64 logits `[0,-1000]` lost mathematical full support after exponentiation
and contaminated the audit's k3/importance terms. A one-action conditional target
with reference logits `[1e308,-1e308]` multiplied an excluded zero by infinity.

The hardening contract explicitly rejects unrepresentable retained probabilities,
reference log probabilities, k3 ratios or audit gradients. Excluded reference
actions must be masked before arithmetic, so that same conditional one-action
KL is zero with finite zero target gradients. Tiny matching distributions whose
probabilities remain representable must still pass, using saved-log-probability
ratios. Add regression checks in both target/reference directions; the original
three-action objectives, temperature-one/full-support Qwen baseline and initial
historical report remain unchanged. Store a new report with revised source/test
hashes, a fresh CPU notebook execution and the shared regression suite.

Primary mechanism reference, inspected rather than copied:
[retained sampling masks in the pinned reasoning companion](https://github.com/rasbt/reasoning-from-scratch/blob/a788466dc85cfe8617b6c0b809ed4ab9084485de/ch07/03_rlvr_grpo_scripts_advanced/7_7_improvements/deepseek_v32_style.py).
