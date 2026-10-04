# Day 15 — Preferences need a rubric, a likelihood, and a noise model

Book placement: [Chapter 10](../../book/chapters/10-preferences-and-reward-models.md).
Material prepared as part of the user's complete-course request on 2026-10-04.
This record describes the syllabus and reference evidence. Learner predictions,
explanations, and mastery have not yet been assessed in a live lesson.

## Deliberate learning arc

1. Derive strength-normalized pair probability and stable soft-target likelihood.
2. Trace opposite score gradients into shared score parameters.
3. Explain reward-offset freedom, repeated votes, finite noisy optimum, ties, and cyclic judgments.
4. Separate probability calibration from ranking and fit a calibration mapping only on its assigned split.

## CPU learning route

- [01 bradley terry margin](../../notebooks/day-15/01_bradley_terry_margin.ipynb)
- [02 disagreement and calibration](../../notebooks/day-15/02_disagreement_and_calibration.ipynb)

Begin with a deep prediction, run the adjacent complete reference, perturb one
specified control, and explain which measurement changed. Static saved PNGs
are labeled reference outputs and regenerate from notebook code. Python needs
PyTorch and Matplotlib; an existing Jupyter kernel is optional. Default machine:
Mac Studio. The same bounded reference computations were executed on Spark CPU;
that does not verify a Mac environment or install dependencies there.

## Recorded evidence

BT loss/gradient and gauge agree exactly in the CPU tests. The vote and reliability plots are synthetic population results; no human calibration evidence is claimed.

Canonical [predeclared specification](../../experiments/specs/2026-10-04-preference-policy-cpu.md)
and [JSON-backed report](../../experiments/reports/2026-10-04-preference-policy-cpu.md).
Code and notebook readiness do not mark this learning day complete.

## Spark extension and handoff

A neural reward model would add transformer endpoints, pooled representations, labeled provenance, and model-selected adversarial responses. First verify a small independent annotation contract; no pretrained reward model or GPU run was executed by this CPU lesson.

Use the course machine-switching contract only when the learner requests a
handoff; Git synchronization and current progress remain owned by the root
course trackers.

## Animation and article capture

Candidate source: roadmap/agent; production state: concept only. Canonical
mechanism: BT margin -> preference probability -> opposing gradients; finite noisy optimum versus unanimous margin growth.

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
