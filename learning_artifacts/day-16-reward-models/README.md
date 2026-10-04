# Day 16 — Which feature does the reward actually reward?

Book placement: [Chapter 10](../../book/chapters/10-preferences-and-reward-models.md).
Material prepared as part of the user's complete-course request on 2026-10-04.
This record describes the syllabus and reference evidence. Learner predictions,
explanations, and mastery have not yet been assessed in a live lesson.

## Deliberate learning arc

1. Fit the explicit quality/length/format scorer.
2. Change training confounding while freezing architecture, seed, budget, and held-out population.
3. Read coefficient plots, held-out NLL, expected Brier, and the declared adversary.
4. Define outcome versus process reward boundaries and reward-head endpoint masks.

## CPU learning route

- [01 reward model bias audit](../../notebooks/day-16/01_reward_model_bias_audit.ipynb)

Begin with a deep prediction, run the adjacent complete reference, perturb one
specified control, and explain which measurement changed. Static saved PNGs
are labeled reference outputs and regenerate from notebook code. Python needs
PyTorch and Matplotlib; an existing Jupyter kernel is optional. Default machine:
Mac Studio. The same bounded reference computations were executed on Spark CPU;
that does not verify a Mac environment or install dependencies there.

## Recorded evidence

The confounded scorer assigned nuisance coefficients and preferred the prescribed worse-but-longer adversary with probability .9625; balanced arm .1369. This is a measured synthetic result, not a neural or human reward verdict.

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
mechanism: Correlated feature credit -> optimization shortcut -> adversarial inversion; matched-length/format audit.

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
