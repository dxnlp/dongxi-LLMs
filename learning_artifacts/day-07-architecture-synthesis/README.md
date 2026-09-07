# Day 7 — Architecture synthesis

- Prepared: 2026-09-07 at the learner's request to add Day 7 notebooks.
- State: material preparation; guided study and independent defense pending.
- The optional recurrence notebook already existed. Added the missing core
  `notebooks/day-07/02_architecture_defense.ipynb`, preserving the older path.
- Study order: core defense first, optional recurrence second. Canonical route:
  [Day 7 notebook index](../../notebooks/day-07/README.md).

## Learning objectives and evidence

Explain a complete token-to-logit path using actual module shapes; distinguish
projection output from head split; count tied Parameters once; compare compact
cache tensor bytes with analytical payload; distinguish finite gradients from
useful learned roles and backward from optimizer updates. Diagnose controlled
wrong-offset and full-time-normalization faults without treating a diagnostic
signature as a uniquely identified cause for every possible implementation.

End with a comparison proposal and an explanation-back. The reference proposal
holds stored parameters fixed for one versus two shared applications, explicitly
not FLOPs or wall-clock time. Concrete data revisions, budgets, seed values,
measurement procedures, and approval are needed before a training run.

No new large model, long training, quality comparison, animation, or article is
commissioned by material creation. Day 6 prose remains preserved. Prior Day 4/5
mastery gaps are not silently completed.

Implementation: `src/dongxi_llms/decoder_audit.py`, existing decoder/visual helpers,
and focused tests. Specification/report:
`experiments/{specs,reports}/2026-09-07-day7-architecture-defense.md`.

## Coherent chapter synthesis

The learner also requested integration into the book, not just a ready notebook.
[Chapter 5](../../book/chapters/05-building-a-modern-decoder.md#523-read-the-model-as-a-connected-argument)
now concludes the Days 5–7 arc with sections 5.23–5.27 and
[worked answers 25–30](../../book/solutions/05-decoder-notebook-solutions.md#day-7-architecture-defense--worked-conceptual-solutions).

- Trace actual module boundaries and distinguish residual features, head axes,
  and vocabulary candidates even where their numerical dimensions coincide.
- Connect twelve averaged token losses to accumulated parameter gradients;
  distinguish backward connectivity from optimizer updates and useful learning.
- Use independent finite/causal/cache checks to explain known wrong-offset and
  time-centering failures. Diagnostic signatures narrow causes, not identify
  every possible bug uniquely.
- Defend GQA using unique-parameter and compact-cache evidence without claiming
  unmeasured peak-memory, latency, or trained-quality gains.
- Distinguish fixed-parameter, training-compute, and wall-clock comparison
  contracts, then bridge to controlled pretraining.

Canonical explanation-back rubric: choice → mechanism → tensor shapes →
evidence → trade-off → failure risk → next experiment. The prose reuses the
verified fixture and its saved budget/diagnostic figures. It is material
preparation, not a new learner answer, a passed defense, or a completed
recurrence comparison. Learner understanding of this conclusion is unassessed.

### Chapter verification — 2026-09-07

A read-only CPU float64 recheck using seed 505 and the existing teaching batch
passed: actual projection/MLP/logit shapes; `(p-q)/12` against retained logit
gradients; 24 connected finite parameter gradients; unchanged weights; the
four-versus-two-KV-head budgets; and all three controlled diagnosis signatures.
Loss and maximum errors reproduced the notebook report. This recheck took no
optimizer step and introduced no new training claim.

Navigation review passed 93 local links and six heading anchors across the
chapter, solutions, book/lab indexes, and this artifact. Numbering covers 27
chapter sections and 30 worked answers. All twelve notebook hashes still match
the existing Day 7 verification manifest. The 53-test and full notebook-suite
results belong to that prior execution; this prose-only update did not rerun
the suite or overwrite notebooks.

- Chapter SHA256: `35f0f2034293ddf15ecbd15a987ba335403f42a912e72197a770cf5e2e197c70`
- Solutions SHA256: `5e8f50a983f88bdc47039fe495ab6a745ac314cc12d126432128674d6f902fcd`

## Portable reuse

Reuse the existing attention/embedding/CE gradient animations and
CAND-ANIM-014/016/017 for normalization, cache positions, and accounting.
CAND-ANIM-011 and X-LOOP-001 remain the recurrence mechanism/article reminders.
The new audit can supply a future failure-diagnosis storyboard, but no extra
animation is necessary before approval; all rendering remains on Mac Studio.

Next: start guided Day 6 notebook 01 unless the learner explicitly chooses the
Day 7 review first. This preparation request is not evidence of a completed day.
