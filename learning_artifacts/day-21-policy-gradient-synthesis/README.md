# Day 21 — Compare sample efficiency and optimizer work separately

Book placement: [Chapter 12](../../book/chapters/12-language-generation-as-a-policy.md).
Material prepared as part of the user's complete-course request on 2026-10-04.
This record describes the syllabus and reference evidence. Learner predictions,
explanations, and mastery have not yet been assessed in a live lesson.

## Deliberate learning arc

1. Run four estimators on identical six-prompt modulo task and three declared seeds.
2. Use exact success probabilities and separate rollout/sample/gradient clocks.
3. Read a range band as three seeds, not a confidence interval.
4. Compare supervision contracts for SFT, DPO, REINFORCE, RLOO, and PPO.
5. Defend a future neural rollout comparison with independent capabilities and verifier checks.

## CPU learning route

- [01 policy algorithm comparison](../../notebooks/day-21/01_policy_algorithm_comparison.ipynb)

Begin with a deep prediction, run the adjacent complete reference, perturb one
specified control, and explain which measurement changed. Static saved PNGs
are labeled reference outputs and regenerate from notebook code. Python needs
PyTorch and Matplotlib; an existing Jupyter kernel is optional. Default machine:
Mac Studio. The same bounded reference computations were executed on Spark CPU;
that does not verify a Mac environment or install dependencies there.

## Recorded evidence

All declared finite arms learned high success; PPO uses 480 gradient passes against 160 for the others at 3840 samples each. Its exact value baseline excludes critic error. Independent prompt logits exclude unseen task generalization.

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
mechanism: Same sampled completions -> unequal gradient passes -> changed efficiency ranking; transition to grouped neural rollouts.

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
