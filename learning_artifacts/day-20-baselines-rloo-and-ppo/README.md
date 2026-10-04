# Day 20 — What changes the expectation, variance, and update restraint?

Book placement: [Chapter 12](../../book/chapters/12-language-generation-as-a-policy.md).
Material prepared as part of the user's complete-course request on 2026-10-04.
This record describes the syllabus and reference evidence. Learner predictions,
explanations, and mastery have not yet been assessed in a live lesson.

## Deliberate learning arc

1. Prove conditional action-independent baseline unbiasedness.
2. Derive the score-norm-weighted optimal scalar baseline.
3. Enumerate RLOO and inclusive-mean scaling for iid groups.
4. Distinguish old rollout policy from fixed reference.
5. Inspect sign-dependent PPO clipping, exact trajectory versus local ratios, and KL estimator support/variance.

## CPU learning route

- [01 baselines and rloo](../../notebooks/day-20/01_baselines_and_rloo.ipynb)
- [02 ppo clipping and kl](../../notebooks/day-20/02_ppo_clipping_and_kl.ipynb)

Begin with a deep prediction, run the adjacent complete reference, perturb one
specified control, and explain which measurement changed. Static saved PNGs
are labeled reference outputs and regenerate from notebook code. Python needs
PyTorch and Matplotlib; an existing Jupyter kernel is optional. Default machine:
Mac Studio. The same bounded reference computations were executed on Spark CPU;
that does not verify a Mac environment or install dependencies there.

## Recorded evidence

Enumerated RLOO mean equals the exact gradient; inclusive mean scales it 2/3 at G3. k1/k3 exact KL=.675806; k2=.829174. Common positive support is part of this evidence.

Canonical [predeclared specification](../../experiments/specs/2026-10-04-preference-policy-cpu.md)
and [JSON-backed report](../../experiments/reports/2026-10-04-preference-policy-cpu.md).
Code and notebook readiness do not mark this learning day complete.

## Spark extension and handoff

Chapter13 extends the finite policy into model-generated grouped rollouts. Before model-scale use, freeze a verifier, data and checkpoint identity, sampling support, masks, reward/KL contract, memory guard, and rollout/update synchronization. This day launches no GPU service.

Use the course machine-switching contract only when the learner requests a
handoff; Git synchronization and current progress remain owned by the root
course trackers.

## Animation and article capture

Candidate source: roadmap/agent; production state: concept only. Canonical
mechanism: Exclude-one group baseline -> unbiased gradient; clipping by advantage sign; signed KL sample -> control variate and rare tails.

A useful article can build on this mechanism after chapter review. It must keep
synthetic results separate from language-model and human evidence. No article
publication or animation rendering is authorized by material preparation;
production belongs on Mac Studio after approval.

## Live discussion to record later

Ask the reader to defend the relevant assumptions and identify one tempting but
unsupported conclusion. Record their initial model, refined explanation,
prediction, perturbation result, and unresolved edge in this topic directory.
The completed reference answer supplies a study aid; it is not the learner's
answer.
