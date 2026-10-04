# Day 17 — From a KL optimum to response log-probability ratios

Book placement: [Chapter 11](../../book/chapters/11-direct-preference-optimization.md).
Material prepared as part of the user's complete-course request on 2026-10-04.
This record describes the syllabus and reference evidence. Learner predictions,
explanations, and mastery have not yet been assessed in a live lesson.

## Deliberate learning arc

1. Derive the constrained finite exponential-tilt optimum and positive-support assumptions.
2. Cancel within-prompt partition terms in Bradley–Terry likelihood.
3. Derive beta-scaled DPO gradients and distinguish beta from inference temperature.
4. Audit a single causal shift, prompt/response/EOS/pad boundaries, and sum versus average.
5. Enforce immutable reference identity, dropout state, and caching provenance.

## CPU learning route

- [01 kl regularized optimum](../../notebooks/day-17/01_kl_regularized_optimum.ipynb)
- [02 sequence likelihood and masks](../../notebooks/day-17/02_sequence_likelihood_and_masks.ipynb)

Begin with a deep prediction, run the adjacent complete reference, perturb one
specified control, and explain which measurement changed. Static saved PNGs
are labeled reference outputs and regenerate from notebook code. Python needs
PyTorch and Matplotlib; an existing Jupyter kernel is optional. Default machine:
Mac Studio. The same bounded reference computations were executed on Spark CPU;
that does not verify a Mac environment or install dependencies there.

## Recorded evidence

Exact finite stationarity, masked logits gradients, and frozen-reference derivative tests passed. Large HF tokenizer-template compatibility is not inferred from these tensors.

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
mechanism: Reference distribution -> exponential tilt -> partition cancellation -> DPO margin; target shift and mask.

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
