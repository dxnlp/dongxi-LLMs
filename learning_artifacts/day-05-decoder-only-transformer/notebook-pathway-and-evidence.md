# Chapter 5 Hands-On Pathway and Evidence

- Date: 2026-09-06.
- Learner request: separate hands-on notebooks for the architecture components,
  then build the entire sequence now so we can study it one session at a time.
- Delivered: seven baseline notebooks, three modern-architecture notebooks, and
  an optional fixed recurrent-depth notebook. Each has safe attempt cells,
  predictions, adjacent runnable solutions, explanations, and controlled changes.
- Canonical index: `notebooks/day-05/README.md`.
- Book placement: `book/labs/05-building-a-modern-decoder.md` and
  `book/solutions/05-decoder-notebook-solutions.md`; Day 5 foundation now in
  `book/chapters/05-building-a-modern-decoder.md`, to extend during Days 6–7.

## Mechanisms made inspectable

Lookup rows and their repeated-token gradients; positions versus causal
structure; per-head/vectorized attention; residual identity and cancellation;
LayerNorm versus wrong-axis normalization; pre/post-norm placement; positionwise
MLPs and affine collapse; full-model shapes, tying, causality, cache replay and
label shifting; bounded one-batch fitting; RMSNorm and SwiGLU ablations; RoPE
geometry and cache-offset failure; GQA, QK norm, parameter/FLOP/cache accounting;
and fixed repeated-block gradient accumulation.

The modern tiny decoder is not a Qwen checkpoint loader. Its configuration
mapping uses a pinned Qwen3 primary source and explicitly separates residual
width from concatenated query-head width. The recurrence notebook implements
fixed sharing only, not adaptive routing or a trained architecture comparison.

## Evidence and learning state

- Eleven fresh-kernel reference runs passed, totaling 93 executed code cells.
- 38 repository tests passed, including 12 new decoder tests.
- Fixed 160-step seed-505 CPU experiment: CE 2.774792432785034 to
  0.0007856183219701052, training-token accuracy 1.0. This verifies fitting the
  declared batch, not language capability or generalization.
- Specification/report: `experiments/{specs,reports}/2026-09-06-decoder-notebooks.md`.
- Learner has not yet completed the pathway. Day 4 practice stays deferred.
- Existing Day 3/4 learner edits were preserved and excluded from this work.
- Original next action was notebook 01. Current position (2026-09-07): live
  explanations reached notebook 07, and the learner requested foundation prose.
  Review its evidence boundary before the Day 6 transition; mastery remains open.

## Live notebook 01 discussion — 2026-09-07

- Learner asked why nn.Embedding(16,8) uses 16: clarified toy vocabulary capacity
  (IDs 0–15), separate from input length and embedding feature width. No real
  tokenizer or word mapping is defined for these teaching IDs.
- Learner requested an accessible explanation of gradients. Introduce one scalar
  parameter e=0.2 with illustrative L=(e-1)^2: derivative -1.6, and an SGD step
  with learning rate 0.1 gives e=0.36. A gradient measures local loss sensitivity;
  subtracting it defines the SGD update. This target-vector toy is not the real
  next-token objective used to learn embeddings.
- Connect to the notebook's sum-of-looked-up-features objective: each occurrence
  contributes 1 per coordinate; ID 2 occurs four times across the two sequences,
  so row 2 accumulates gradient 4 per coordinate. ID 5 and ID 7 each occur twice.
  Lookup-unused rows get zero on this path, while output tying can add a separate
  dense gradient. backward() accumulates gradients but does not update weights.
- These are explanations offered, not demonstrated learner mastery. No learner
  notebook edits or outputs are changed. Reuse ANIM-EMB-001 for repeated-row
  gradient accumulation; no new animation or production approval is needed.

### Learner connects gradients to end-to-end LLM training

- Learner restated that embeddings, FFNs, and attention weights are trained by
  gradient descent to minimize cross-entropy on a training corpus. This supports
  the central end-to-end optimization connection, not independent mastery of
  every derivative or notebook exercise.
- Refine attention terminology: W_Q/W_K/W_V/W_O are stored learned parameters;
  Q/K/V, attention scores, and softmax attention weights are input-dependent
  activations. Gradients pass through activations to reach the parameters.
- Scope: full-model training optimizes all designated trainable parameters;
  frozen-parameter or adapter training updates only a subset. AdamW is a common
  gradient-based optimizer, not the literal plain-SGD update.
- Corpus next tokens are observed training labels, not guaranteed factual or
  uniquely valid gold truth. Lower training CE measures improved fit to those
  labels; it does not alone establish generalization or truthfulness.
- Reuse the existing embedding/attention/loss animation packets for this
  end-to-end connection; no new production task or render is authorized.

## Reuse and portability

### Architecture visuals — 2026-09-07

The learner additionally requested architecture diagrams and approved their
implementation. Each of the eleven Chapter 5 notebooks now opens with a
whole-model map highlighting its current component, followed by a close-up of
operations, branches, and tensor dimensions. The modern-block lesson includes
an extra SwiGLU diagram: 23 new schematics, 48 total previews. Their adjacent
code redraws the schematic; numerical plots continue to use lesson tensors.

Preserved all 299 cells present before this addition. Verified all eleven
notebooks (152 code cells, 48 images) and 47 tests. Report:
`experiments/reports/2026-09-07-decoder-architecture-visuals.md`.
This is a durable notebook-design preference, now recorded in AGENTS.md and
LEARNING_MEMORY.md. Existing animation proposals cover these mechanisms;
static architecture diagrams do not authorize Mac animation production or
mark any learning checkpoint complete.

### Visual enrichment request — 2026-09-07

The learner found text/code-only notebooks boring and asked for Python visuals
that explain the key concepts. Added 25 data-backed static figures across all
eleven Chapter 5 sessions, plus fixed reference previews and regeneration cells.
Preserved 238 original cells (source, metadata, execution counts, and outputs),
including active Notebook 01/02 learner edits. The figures show mechanisms rather
than decorative illustrations; see `docs/NOTEBOOK_VISUALS.md` and
`experiments/reports/2026-09-07-decoder-notebook-visuals.md`.

This preference is now in AGENTS.md and LEARNING_MEMORY.md for future sessions.
Existing animation candidates cover these mechanisms; this request authorizes
static notebook plots, not Mac animation production. The learner is at Notebook
02 and has not independently demonstrated every Notebook 01 mechanism.

### Optimizer and learning-rate discussion — 2026-09-07

The learner asked for accessible concrete examples after connecting gradients
to end-to-end training. Reuse the illustrative scalar L=(e-1)^2, e=0.2,
gradient=-1.6. Plain SGD e_new=e-lr*gradient gives 0.216 at lr=.01, 0.36 at
lr=.1, and 2.6 at lr=1.5. The last increases loss from .64 to 2.56: a locally
helpful direction need not remain helpful over a large step. This is a toy
quadratic, not LLM cross-entropy or a recommended LLM learning-rate range.

Distinguish backward gradient calculation, optimizer update rule, and the
learning rate scaling that rule. AdamW tracks gradient history and squared
gradient history for adaptive updates; its step is not plain lr times raw
gradient. Weight decay is a separate regularizing shrinkage; the course's
one-batch AdamW experiment explicitly sets it to zero. Explain zero_grad(),
backward(), step() without implying every batch step lowers loss. This is
introduced material, not independent learner mastery.

Capture CAND-ANIM-018 for matched-start learning-rate trajectories; deeper
optimizer treatment belongs to Chapter 6/Days 8–9. No rendering authorized.

Math opportunity check expands CAND-ANIM-011/012/013 and captures 014–017 for
normalization, MLP gates, RoPE, and grouped KV accounting. Existing embedding and
attention packets remain canonical. All media production requires approval and
stays on Mac Studio. The looped-transformer article reminder remains X-LOOP-001.

CPU-only references require Torch, nbformat/nbclient for automated verification,
and a working Jupyter kernel for interactive study. Spark's kernel is
`dgx-spark-native`; Mac users should choose their local Python kernel. No Mac
execution is claimed. No notebook server was launched for this production task.
