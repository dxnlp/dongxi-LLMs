# Day 18 — A favorable preference margin is a diagnostic, not the final task score

Book placement: [Chapter 11](../../book/chapters/11-direct-preference-optimization.md).
Material prepared as part of the user's complete-course request on 2026-10-04.
This record describes the syllabus and reference evidence. Learner predictions,
explanations, and mastery have not yet been assessed in a live lesson.

## Deliberate learning arc

1. Run categorical DPO, label-flipped DPO, and chosen-SFT from one reference.
2. Train the actual tiny decoder DPO/SFT controls and measure an independent held-out desired token.
3. Interpret ratio improvement with falling absolute chosen probability.
4. Prepare a bounded optional Spark DPO run from a full or merged Chapter9 SFT checkpoint.
5. Freeze independent evaluation and preserve all generations, including regressions.

## CPU learning route

- [01 dpo controlled comparison](../../notebooks/day-18/01_dpo_controlled_comparison.ipynb)

Begin with a deep prediction, run the adjacent complete reference, perturb one
specified control, and explain which measurement changed. Static saved PNGs
are labeled reference outputs and regenerate from notebook code. Python needs
PyTorch and Matplotlib; an existing Jupyter kernel is optional. Default machine:
Mac Studio. The same bounded reference computations were executed on Spark CPU;
that does not verify a Mac environment or install dependencies there.

## Recorded evidence

Tiny decoder DPO reached mean margin 6.9845 but held-out P(A) fell .08129 to .000492. The matched chosen-SFT control rose to .9981. These are synthetic one-token outcomes; Qwen DPO remains proposed.

Canonical [predeclared specification](../../experiments/specs/2026-10-04-preference-policy-cpu.md)
and [JSON-backed report](../../experiments/reports/2026-10-04-preference-policy-cpu.md).
Code and notebook readiness do not mark this learning day complete.

## Spark extension and handoff

The optional Chapter11 HF runner takes a local full/merged SFT checkpoint, pinned tokenizer, authored preference fixture, and disjoint independent generation records. Qwen smoke, memory profiling, training, and independent quality assessment remain proposed. Stop an existing inference service before approved training.

Use the course machine-switching contract only when the learner requests a
handoff; Git synchronization and current progress remain owned by the root
course trackers.

## Animation and article capture

Candidate source: roadmap/agent; production state: concept only. Canonical
mechanism: Pair-ratio improvement while both answers lose mass to a third continuation; reference versus old-policy identity.

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
