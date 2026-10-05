# Day 18 — A favorable preference margin is a diagnostic, not the final task score

Book placement: [Chapter 11](../../book/chapters/11-direct-preference-optimization.md).
Material prepared as part of the user's complete-course request on 2026-10-04.
This record describes the syllabus and reference evidence. Learner predictions,
explanations, and mastery have not yet been assessed in a live lesson.

## Deliberate learning arc

1. Run categorical DPO, label-flipped DPO, and chosen-SFT from one reference.
2. Train the actual tiny decoder DPO/SFT controls and measure an independent held-out desired token.
3. Interpret ratio improvement with falling absolute chosen probability.
   Contrast that CPU failure mechanism with the actual native case below,
   where chosen likelihood improves in both arms but exact answering differs.
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

Historical CPU reference (2026-10-04): tiny decoder DPO reached mean margin
6.9845 but held-out P(A) fell .08129 to .000492. The matched chosen-SFT control
rose to .9981. These are synthetic one-token outcomes; native Qwen DPO was
proposed at that historical preparation boundary, not measured by this CPU run.

Canonical [predeclared specification](../../experiments/specs/2026-10-04-preference-policy-cpu.md)
and [JSON-backed report](../../experiments/reports/2026-10-04-preference-policy-cpu.md).
Code and notebook readiness do not mark this learning day complete.

The 2026-10-05 [actual native comparison](../../experiments/reports/2026-10-05-native-preference-comparison.md)
now supplies a separate model-scale case: both fixed 100-update pilots and the
common evaluation pass their retained technical checks. Both pilots present
the same 2,047 chosen targets with identical actual replacement draws and token
masks, but unequal rejected/reference/forward work. Both improve chosen
likelihood; DPO's larger validation relative margin accompanies1/4 exact common
location answers, versus 4/4 for chosen-only. All three arms retain 120/120
instruction answers; the annotated reasoning panel changes 5/20→6/20 in both
descendants. These populations and precision paths must not be pooled.

Focused lesson: [Margin versus the generated answer](margin-versus-generated-answer.md).
Read the existing receipts/figures rather than rerunning exclusive model jobs.
This is course evidence/material readiness, not an observed live learner answer,
Day 18 mastery or advancement from the learner's active Day 9 position.

## Spark extension and handoff

The cross-day [durable-boundary note](../day-12-sft-mechanics/durable-training-boundaries.md)
explains why the original DPO reference must survive recovery and why an HF
export alone is not optimizer continuation. Its source/CPU checks do not
establish pretrained or Spark replay.

The optional Chapter11 HF runner takes a local full/merged SFT checkpoint,
pinned tokenizer, authored preference fixture and independent generation
records. Its earlier preparation recorded native smoke/profiling/training as
proposed. The separate fixed2026-10-05 native recovery, pilots and common
comparison now have actual evidence linked above; that is not blanket authority
for new models, sweeps or concurrent inference/training services.

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

Reuse **CAND-ANIM-025** for the actual matched-exposure lesson: separate absolute
chosen/rejected logp, unscaled reference-relative margin, strict generated
answers, natural stops and retention. In this native example, animate both
chosen likelihoods improving rather than reusing the CPU falling-mass story.
Possible X article: “The preference margin improved. Why didn't the answer?”
This remains a suggestion, not a commissioned article or approved film.

## Live discussion to record later

Ask the reader to defend the relevant assumptions and identify one tempting but
unsupported conclusion. Record their initial model, refined explanation,
prediction, perturbation result, and unresolved edge in this topic directory.
The completed reference answer supplies a study aid; it is not the learner's
answer.
