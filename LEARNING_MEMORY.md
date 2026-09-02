# Learning Memory and Production Queue

This ledger preserves the learner's evolving mental models and turns promising
discussions into portable production tasks. It is designed to survive chat
compaction, new Codex sessions, and work split across the DGX Spark and a local
Mac.

It is not a transcript and it does not replace the book:

- `BOOK.md` defines the reader-facing narrative.
- `ROADMAP.md` defines planned learning outcomes.
- `PROGRESS.md` records the active day and exact next action.
- `learning_artifacts/` preserves deep discussions by day and topic.
- This file indexes that understanding and records public-content ideas and
  cross-machine task packets.

## Maintenance contract

Create or update the relevant topic in `learning_artifacts/` when a discussion
produces at least one of the following:

1. a durable explanation or corrected misconception;
2. evidence that the learner can explain or calculate a mechanism;
3. an unresolved question worth carrying into a later day;
4. a possible article, animation, diagram, experiment, or exercise;
5. work that may be executed in another session or on another machine.

Record the learner's current model and the important distinction, not every turn
of dialogue. Update this file's index or production queue when the discussion
opens a new topic or reusable task. Never silently upgrade "understood in
conversation" to "verified by experiment" or "mastered." Link empirical claims
to specifications, reports, or stored outputs when those exist.

Every portable production task must identify its learning objective, source
material, claims that must remain precise, expected outputs, acceptance checks,
dependencies, preferred machine, and current status.

## Learner profile and working preferences

- Learns best through prediction, a concrete example, a short explanation, and
  an immediate check for understanding.
- Prefers profound, mechanism-level discussion over calculation quizzes. Use
  arithmetic as supporting evidence or executable verification, not as the main
  conversational teaching format; favor conceptual tensions, implications, and
  connections to real model behavior for understanding checks.
- Wants interactive mechanism notebooks that make mathematical ideas executable.
  Structure them around a deep question, prediction before execution, a small
  learner-written implementation, deliberate perturbation or broken variant,
  interpretation, and an explicit evidence boundary. Guide these notebooks
  collaboratively cell by cell rather than presenting them as passive demos.
- Wants the correct runnable solution and a concise mechanism explanation
  attached directly after each notebook exercise. Keep the attempt first, but
  make the notebook self-contained so routine syntax lookup does not interrupt
  the conceptual learning flow.
- The DGX Spark notebook kernel is `Python (DGX Spark Native)`, backed by
  `/home/dongxi/dgx-spark-dongxi/.venv/bin/python`. Recreate it with the platform
  repository's `scripts/setup_jupyter.sh`; it adds locked Jupyter components to
  the validated CUDA 13 environment without replacing PyTorch. Launch JupyterLab
  from the course root and keep it bound to `127.0.0.1` unless remote access is
  deliberately secured.
- Wants mechanisms explained beneath convenient APIs: bytes and BPE merges before
  token IDs, embedding rows before contextual states, and masks before trainer
  abstractions.
- Values multilingual comparisons, especially English, Chinese, and Swedish.
- Wants course learning to become a coherent technical book, not chronological
  notes.
- Wants strong discussions reused as public X articles and mathematical
  animations after the canonical book treatment is stable.
- Wants animation opportunities surfaced proactively during learning and course
  development. Either the user or an agent may suggest an idea; record it in
  `visuals/animations/PROPOSALS.md` and wait for explicit approval before
  production.
- Wants explicit mathematics that is important to LLM mechanisms automatically
  marked as a potential animation without needing to ask. Central objectives,
  probability transformations, gradients, tensor operations, masking rules,
  optimization updates, and multi-step derivations trigger candidate capture;
  related equations should be consolidated into coherent concepts rather than
  generating duplicate proposals. Automatic marking never authorizes production.
- Uses the DGX Spark for model- and GPU-dependent work. All animation production
  and rendering belongs on the Mac Studio; Spark sessions only identify, record,
  specify, and review animation concepts unless the learner explicitly changes
  that assignment. The Mac Studio may also handle design, editing, and publishing.
- Works across multiple machines and may push course changes from either one.
  Before every new or resumed learning session, inspect the branch and working
  tree, fast-forward pull the remote branch when clean, and reread the durable
  project state before starting work. Never discard dirty local work merely to
  synchronize.
- Prefers course animations with a white canvas, Arial normal-weight English,
  Songti SC Chinese, minimal text, stable geometric alignment, and continuous
  mechanism-first motion. The reusable specification is
  `visuals/animations/STYLE_GUIDE.md`.
- Uses a 5:2 aspect ratio for X Article cover images by default; prefer 2000×800
  px with generous safe margins. Inline article figures may retain the aspect
  ratio required by their mechanism. The reusable publication rule is
  `publications/x-articles/README.md`.

## Learning-artifact index

Detailed knowledge state lives in `learning_artifacts/` so future sessions can
load only the active day and relevant topic rather than rereading one growing
ledger.

| Day | Topic | Status | Artifact index |
|---:|---|---|---|
| 1 | Evidence before optimization | complete | `learning_artifacts/day-01-evidence-before-optimization/README.md` |
| 2 | Text, tokens, and embeddings | complete | `learning_artifacts/day-02-text-tokens-and-embeddings/README.md` |
| 3 | Probabilities and next-token loss | in progress | `learning_artifacts/day-03-probabilities-and-next-token-loss/README.md` |

At the start of each new day, create its directory and index. During the lesson,
update the relevant focused topic whenever the learner states a prediction,
demonstrates understanding, encounters a correction, or identifies an open edge.

## Public-content production queue

| ID | Type | Topic | Status | Preferred machine | Dependency |
|---|---|---|---|---|---|
| `X-BPE-001` | X article | What is a token? Unicode → BPE → model IDs | bilingual local packages prepared; editorial review pending | Mac | Chapter 2 tokenizer-mechanics enrichment complete |
| `X-EMB-001` | X article | How transformer embedding tables are actually trained | ready for Mac drafting | Mac | Chapter 2 and embedding labs complete |
| `ANIM-BPE-001` | Animation | Bytes → characters → Chinese word/phrase tokens | minimal Manim style approved and committed | Mac Studio | Day 2 explanation complete |
| `ANIM-EMB-001` | Animation | End-to-end embedding training and tied gradient paths | continuous-animation Mac handoff ready | Mac Studio | Day 2 embedding lab and Day 3 loss derivation |
| `ANIM-CE-001` | Animation | LLM target probability → per-token NLL → masked mean cross-entropy | expanded concept approved; Mac Studio production waits for Day 3 evidence | Mac Studio | Day 3 derivation, target alignment, and PyTorch verification |
| `ANIM-NTP-001` | Animation | One-hot next-token supervision → `p-q` gradient → distribution learning across examples | concept approved; Mac Studio production waits for Day 3 evidence | Mac Studio | Gradient verification and controlled target-frequency experiment |
| `ANIM-LOGLOSS-001` | Animation | Why `-log p_target`: additive sequence surprise, confident-error gradients, and proper probability reporting | three-act concept approved; Mac Studio production waits for Day 3 evidence | Mac Studio | Chain-rule, gradient, and expected-scoring verification |

## Production-system tasks

| ID | Type | Objective | Status | Durable output |
|---|---|---|---|---|
| `ANIM-SYSTEM-001` | Workflow | Create a two-way, approval-gated mechanism for user- and agent-suggested animations | complete | `visuals/animations/PROPOSALS.md`; animation-opportunity check in `AGENTS.md` and `ROADMAP.md` |
| `ANIM-SYSTEM-002` | Workflow | Automatically capture animation candidates for explicit LLM mathematics while retaining approval-gated Mac Studio production | complete | Math trigger in `AGENTS.md`, `ROADMAP.md`, and `visuals/animations/PROPOSALS.md` |

Animation suggestions begin in `visuals/animations/PROPOSALS.md`. Approved
concepts are promoted into the public-content production queue and receive a
complete `ANIM-*` task packet below. Candidate capture does not expand the active
day or authorize media production.

### Task packet: `X-BPE-001`

**Working title:** What Is a Token, Really? From Unicode Bytes to BPE and Token
IDs

**Learning promise:** A reader should be able to distinguish orthographic words,
grapheme clusters, code points, UTF-8 bytes, subword or byte tokens, and token
IDs; explain BPE training separately from runtime encoding with trained merge
ranks; trace byte fallback for `数`; and
explain how normalization, spaces, special tokens, and chat templates alter the
model's input representation without implying understanding.

**Narrative spine:**

1. Open with the verified surprise: Qwen3 represented `下一个` as one token while
   splitting the Swedish word `språkmodellen` into five.
2. Separate orthographic words, grapheme clusters, code points, UTF-8 bytes,
   subword or byte tokens, and token IDs.
3. Show the standard Tokenization pipeline: raw text → normalization →
   pre-tokenization → tokenization model → post-processing → token IDs. Show
   Chat Template serialization as an earlier step for message-based input.
4. Establish that IDs are meaningful only under the tokenizer name, revision,
   configuration, vocabulary, and matching model input embedding matrix.
5. Use the tested `hug/hugs/hugging` corpus to show pair counts, a tied maximum,
   three deterministic BPE rounds, and runtime use of the trained Merge Rank.
6. State the scope explicitly: byte-level BPE is the main mechanism; WordPiece
   and Unigram receive only a compact comparison.
7. Show `数 → E6 95 B0` and the coverage-to-compression ladder through `数据库`.
8. Use the pinned NFC/NFD, grapheme, and leading-space results to make
   preprocessing visible, including the failed exact NFD source round trip.
9. Compare raw `Hello` at one token with the pinned one-message chat template at
   nine positions; keep deep masking mechanics for later chapters.
10. Close with the measured multilingual example and the distinction among
    coverage, compression, context occupancy, and model understanding.

**Required precision:**

- Qualify the example as byte-level BPE; do not generalize it to every tokenizer.
- Do not describe encoding as learning new merges at runtime.
- Do not claim that BPE understands Chinese words; it exploits repeated adjacent
  patterns.
- Preserve the difference between an observed Qwen3 segmentation and a universal
  rule.
- Distinguish exact code-point equality from Unicode canonical equivalence.
- Do not treat chat-template overhead as subword learning.
- Do not generalize the BPE mechanism to WordPiece, Unigram, or tokenizers without
  complete byte fallback.

**Source material:**
`learning_artifacts/day-02-text-tokens-and-embeddings/bpe-training-and-byte-coverage.md`;
`learning_artifacts/day-02-text-tokens-and-embeddings/unicode-normalization-pretokenization-and-chat-packaging.md`;
the fixed multilingual specification and report;
`experiments/reports/2026-08-31-tokenizer-mechanics.md`;
`book/chapters/02-text-tokens-and-embeddings.md`;
`ANIM-BPE-001`.

**Expected output:** An English X Article or thread draft, plus a short Chinese
adaptation only after the English claims are reviewed. Decide article versus
thread at production time rather than maintaining two premature versions.

**Acceptance checks:** A reader can explain (1) why an orthographic word,
grapheme, code point, byte, token, and Token ID are different; (2) how BPE
training differs from runtime
encoding; (3) why `数` can be encoded without a learned Chinese merge; (4) why
normalization and Chat Template serialization affect the final model input; and
(5) why coverage and compression do not establish understanding.

**Current output:** The reviewable English package is under
`publications/x-articles/x-bpe-001/`. It contains the approximately 1,500-word canonical draft,
claim-to-evidence map, separate cover, five inline figures, editable figure
source, a responsive `review.html` with all media in reading order, body-only
HTML/Markdown transfer files, and a validated reverse image insertion plan. No
content has been transferred to X or published. A complete natural-Chinese
adaptation now lives under `publications/x-articles/x-bpe-001/zh/`, with a
localized 5:2 cover, five localized figures, evidence map, browser review, and
transfer package. `publications/x-articles/x-bpe-001/terminology.md` is the
canonical bilingual terminology map; reader-facing explanations use standard
NLP stages and avoid software-architecture metaphors. The Chinese text was
scanned for the prohibited
`不是……而是……` construction and variants. Review both language versions before
any X editor transfer.

### Task packet: `X-EMB-001`

**Working title:** How Are Embeddings Actually Trained Inside a Transformer?

**Hook:** An embedding table looks like a dictionary of vectors, but no teacher
normally supplies the "correct vector" for `cat`, `sat`, or `dog`. The vectors
become useful because next-token prediction sends gradients all the way back to
the table.

**Learning promise:** A reader should be able to trace one next-token training
example from token IDs to embedding lookup, transformer state, logits, loss,
backpropagation, and an optimizer update. They should also understand why the
answer to "which embedding rows are updated?" changes under weight tying.

**Narrative spine:**

1. Begin with the apparent missing label: if training data contains text rather
   than target vectors, where does an embedding's meaning come from?
2. Define `E ∈ R^(V×d)` and show IDs selecting rows: `[B,T] → [B,T,d]`.
3. Follow selected vectors through the transformer to contextual state `h` and
   next-token loss.
4. Reverse the computation: the loss gradient flows through the output head and
   transformer into the selected lookup rows.
5. Use repeated ID sequence `[2,5,2]` to show that two position-level gradient
   contributions add into shared row `E[2]`.
6. Reveal the weight-tying subtlety. With untied weights, only selected input rows
   receive lookup-path embedding gradients. With `logits = hE^T`, the shared `E`
   also receives output-classifier gradients across the vocabulary.
7. End with the conceptual distinction: identical input embeddings can become
   different contextual hidden states, and "meaning" is distributed across the
   complete trained network rather than stored only in one row.

**Required equations and shapes:**

```text
E.shape = [V, d]
input_ids.shape = [B, T]
X = E[input_ids], X.shape = [B, T, d]
logit_i = h · E[i]                         # tied output weights
dL/dE[i] |_output = (p_i - 1[i = y]) h
E[2] lookup gradient = contribution_at_1 + contribution_at_3
```

**Required precision:** Say "direct lookup-path gradient" rather than claiming
that unused rows always receive zero total gradient; distinguish tied and untied
output weights; do not imply that the embedding layer is trained with a separate
semantic objective; distinguish input embeddings from contextual hidden states;
label `E ← E - η∇E` as an SGD sketch when the actual optimizer is AdamW.

**Source material:**
`learning_artifacts/day-02-text-tokens-and-embeddings/embeddings-context-and-gradients.md`;
`experiments/reports/2026-08-31-embedding-gradient-paths.md`;
`experiments/reports/2026-08-31-qwen3-embedding-inspection.md`;
`book/chapters/02-text-tokens-and-embeddings.md`; `ANIM-EMB-001`.

**Expected output:** An English X Article or thread draft with the continuous
embedding-training animation as its primary visual. Consider a compact Chinese
adaptation only after the canonical English claims and equations pass review.

**Mac handoff:** Use branch `content/x-transformer-embeddings`. Start from the
latest `origin/main` containing this packet, record its exact commit, and keep the
article draft separate from canonical Chapter 2 until review.

**Acceptance checks:** A reader can explain where embedding supervision comes
from, calculate the lookup output shape, explain repeated-row gradient addition,
and distinguish the rows updated through lookup from those updated through a tied
classifier. Every empirical statement links to the later Day 2 lab; unverified
claims remain labeled.

### Task packet: `ANIM-BPE-001`

**Learning objective:** Make the difference between universal byte coverage and
learned compression visible in one sequence.

**Existing baseline:**

- Source: `visuals/animations/bpe_byte_merges.py`
- Render: `visuals/animations/rendered/bpe-byte-merges.gif`
- Reproducible environment: `visuals/animations/pyproject.toml` and `uv.lock`

**Current Mac review candidate:**

- Base commit: `26d631c69e740bf0bf1a4324e25528f092b9d4fd`
- Branch: `codex/visuals/manim-bpe`
- Source: `visuals/animations/manim_bpe_byte_merges.py`
- Outputs: 1920×1080 H.264 MP4, 960×540 GIF preview, and PNG still under
  `visuals/animations/rendered/`
- Renderer: Manim Community `0.21.0`, Python `3.12.11`, Cairo renderer
- Status: minimal motion-first visual style approved on 2026-08-31; commit and
  final content integration remain pending; the Matplotlib baseline is unchanged

**Approved reusable style:** Follow `visuals/animations/STYLE_GUIDE.md`. Use a
white canvas, Arial normal-weight English, Songti SC Chinese, semantic colors,
stable geometric anchors, and the least text that preserves accuracy. The
mechanism should be carried by continuous object motion; avoid prose blocks,
merge tables, space-padded alignment, and decorative transitions. Apply this
system by default to `ANIM-EMB-001` and later course animations.

**Storyboard for a Mac/Manim refinement:**

1. Present one visible glyph, `数`, then reveal its UTF-8 bytes `E6 95 B0` as
   three separate reusable byte tokens.
2. Replay two learned BPE merges that compress those bytes into the character
   token `数`. Label these as frozen rules learned offline.
3. Expand to `[数] [据] [库]`; merge `[数] [据]` into `[数据]`, then merge
   `[数据] [库]` into `[数据库]`.
4. Keep a small persistent scale underneath: bytes → characters → words/phrases.
5. Close with the paired message: base bytes guarantee coverage; corpus-learned
   merges improve efficiency.

**Required precision:** Old constituent tokens remain in the vocabulary after a
merge; frequency makes a merge more likely but competes with other pairs and a
finite budget; encoding replays rules and does not perform tokenizer training;
byte coverage says nothing by itself about model understanding.

**Expected outputs:** Editable Manim source, a 16:9 H.264 MP4 suitable for the
book/site, a lightweight GIF preview, the exact render command, and a note of the
Manim/Python revisions. Do not replace the baseline until the new render passes
content and legibility review.

**Acceptance checks:** Chinese glyphs render correctly; every merge direction is
unambiguous; the final frame is readable on a phone; no frame implies that a
single occurrence creates a new token online.

### Task packet: `ANIM-EMB-001`

**Learning objective:** Show that an embedding table is normally trained end to
end by the next-token loss, and distinguish the lookup gradient path from the
additional classifier path created by tied input/output weights.

**Animation form:** A continuous animation rather than a sequence of disconnected
slides or a manually stepped interface. Preserve the identity of the shared
matrix `E` throughout the forward and backward motion so weight tying is visible,
not merely stated.

**Source material:**
`learning_artifacts/day-02-text-tokens-and-embeddings/embeddings-context-and-gradients.md`;
`experiments/reports/2026-08-31-embedding-gradient-paths.md`;
`experiments/reports/2026-08-31-qwen3-embedding-inspection.md`;
`book/chapters/02-text-tokens-and-embeddings.md`; the Day 3 cross-entropy
derivation when available.

**Continuous-motion storyboard:**

1. Let IDs `[cat=1, sat=2]` move to rows `E[1]` and `E[2]` in
   `E ∈ R^(V×d)`. Copies of those vectors lift out of the table while the original
   rows remain visibly part of `E`.
2. Move the selected vectors continuously through a transformer block and
   compress the last-position computation into contextual state `h`.
3. Keep the same `E` on screen and visually reuse or transpose it as the tied
   output projection. Animate `hE^T` into one logit per vocabulary row.
4. Transform logits into probabilities, highlight target `dog`, and collapse the
   target probability into scalar loss `L = -log p(dog)`.
5. Reverse the direction of motion for backpropagation. Let the loss signal split
   into two clearly labeled streams: the output-classifier path reaches all rows
   of `E`, while the input lookup path travels back through the transformer and
   reaches only rows `cat` and `sat`.
6. Recombine the two streams on the selected input rows so their accumulated
   gradients are visually distinct from output-only gradients on unused input
   rows.
7. Apply an optimizer update. Move the affected row vectors from their old
   coordinates or values to their new ones, then leave the updated `E` ready for
   the next batch.
8. End with a brief untied contrast only if it remains legible: the classifier
   gradient updates a separate `W_out`, while lookup gradients update selected
   rows of `E_in`.

**Motion language:** Forward computation moves consistently toward the loss;
backpropagation travels in the opposite direction. Use one stable visual identity
for `E`, one for input-path gradients, and another for output-path gradients.
Avoid jump cuts that make the output projection look like an unrelated matrix.

**Required precision:** Do not say that only selected rows receive total gradient
when weights are tied; call selected-row contributions through lookup "direct
input-path gradients"; identify `E ← E - η∇E` as an SGD sketch rather than the
exact AdamW rule; do not derive cross-entropy before Day 3's canonical treatment.
Do not animate `p(dog)` improving after the update unless that change is computed
and verified in the same toy model.

**Expected outputs:** Editable animation source, 16:9 H.264 MP4, lightweight GIF,
render command, dependency revisions, and a still diagram for Chapter 2. The Mac
is preferred for final media rendering.

**Mac handoff:** Use branch `visuals/manim-embedding-training`. Start from the
latest `origin/main` containing this packet and record its exact commit before
work begins. Limit changes to the embedding-animation source, rendered previews,
and their render instructions under `visuals/animations/`; do not edit the
canonical Chapter 2 prose in the rendering branch.

**Acceptance checks:** The input IDs and matrix shapes remain visible; tied versus
untied behavior is unambiguous; input rows visibly accumulate two contributions
under tying; the target and loss direction agree with the later numerical lab;
the animation reads as one continuous computation; the final frame remains
readable on a phone. Return the commit hash, changed files, exact render command,
dependency revisions, outputs, and known limitations.

### Task packet: `ANIM-CE-001`

**Learning objective:** Place negative log-likelihood and cross-entropy inside the
LLM training computation. Show that one-hot token cross-entropy equals
`-ln p_target`, then aggregate per-position NLL terms across valid causal targets
into the reported mean loss.

**Approval and ownership:** Dongxi approved this expanded concept during Day 3.
All production and rendering belongs on the Mac Studio. DGX Spark work is limited
to the canonical derivation, executable evidence, task specification, and review.

**Existing preview:**

- Source: `visuals/animations/cross_entropy_curve.py`
- Render: `visuals/animations/rendered/cross-entropy-curve.gif`

**Storyboard for the canonical Day 3 version:**

1. Begin with one causal position, its next-token target, and a softmax
   distribution. Preserve the identity of the target while selecting `p_target`.
2. Synchronize that probability with the natural negative-log curve and transform
   it into one per-token NLL tile, `-ln p_target`.
3. Briefly reveal the full one-hot cross-entropy sum and collapse its zero-weighted
   terms to the same `-ln p_target`, making equality rather than conversion the
   central point.
4. Repeat the mechanism across several causally aligned next-token positions.
5. Mark padding or ignored labels visually and remove their NLL tiles from both
   numerator and denominator.
6. Aggregate the remaining tiles into the mean token cross-entropy reported by
   the training loop. Distinguish this mean from the summed sequence NLL.
7. End with perplexity only if the Day 3 derivation shows that the extra transition
   remains legible; otherwise reserve it for a separate visual.

**Dependency:** Do not finalize the narrative before Day 3 derives softmax,
negative log-likelihood, cross-entropy, causal shifting, and label alignment. The
current animation is an intentionally limited preview.

**Required precision:** Use the natural logarithm; identify the observed target
explicitly; do not imply that cross-entropy selects the model's largest
probability; show that one-hot token cross-entropy and token NLL are numerically
the same objective; distinguish summed sequence NLL from the mean over valid
tokens; do not include padding or ignored positions in either the numerator or
denominator; do not imply that low cross-entropy proves truthfulness or general
capability.

**Expected outputs:** Editable source, MP4, GIF preview, render command, revision
manifest, and a still frame suitable for Chapter 3.

**Acceptance checks:** Displayed probabilities sum to one within rounding; the
marker agrees numerically with `-ln(p)`; the one-hot cross-entropy collapses to
that same value; masked positions contribute neither loss nor count; the final
mean agrees with the later manual and PyTorch calculations; behavior near zero is
described as a limit rather than evaluating `ln(0)`.

### Task packet: `ANIM-NTP-001`

**Learning objective:** Show how standard one-hot next-token training converts a
predicted distribution `p` and observed target distribution `q` into the logit
gradient `dL/dz = p-q`, and how repeated samples—not one isolated target—shape the
learned conditional distribution.

**Approval and ownership:** Dongxi approved this concept during Day 3. All
production and rendering belongs on the Mac Studio. The DGX Spark supplies only
the derivation, verified numerical examples, tiny-model evidence, task packet,
and later content review.

**Narrative spine:**

1. Hold a small candidate vocabulary fixed and display the model distribution
   `p` as probability bars.
2. Reveal the one-hot observed target `q` without describing other candidates as
   linguistically invalid.
3. Transform the aligned bars into `p-q`. Keep signs and token identities visible.
4. Reverse from gradient to gradient-descent motion: the target logit rises while
   non-target logits fall, with the update size controlled by the optimizer.
5. Emphasize that one example provides no special protection for valid but
   unobserved alternatives.
6. Replay a verified sequence of examples whose targets vary among several valid
   continuations. Accumulate their pressures and show the model distribution
   approaching the empirical conditional frequencies rather than collapsing to
   the last target.
7. Close with the evidence boundary: matching the training distribution is not
   the same as truth, calibration on a shifted domain, or general capability.

**Required precision:** Use `dL/dz_i = p_i-q_i`; distinguish gradient sign from
the direction of a gradient-descent parameter update; do not imply that all
logits change by the same magnitude; do not imply one update produces a one-hot
prediction; state that alternatives recover support through other observations,
shared generalization, or different supervision; use an explicitly verified
optimizer and target-frequency example.

**Expected outputs:** Editable Manim source, 16:9 H.264 MP4, lightweight GIF,
render command, dependency revisions, metadata, and one still suitable for
Chapter 3. The Mac Studio decides whether the final form is standalone or a
companion segment to `ANIM-CE-001`, without changing the canonical equations.

**Acceptance checks:** Every candidate retains a stable visual identity; `p` and
`q` each sum to one; displayed gradient values equal `p-q`; target/non-target
update directions are correct; repeated-example frequencies and final
probabilities agree with the committed Day 3 numerical evidence; no frame claims
that low training loss establishes truthfulness.

### Task packet: `ANIM-LOGLOSS-001`

**Learning objective:** Explain three independent reasons negative log-likelihood
fits next-token language modeling: it converts sequence probability products into
additive token surprise, preserves strong corrective gradients for confident
softmax errors, and acts as a proper scoring rule whose expected value is
minimized by matching the full target distribution.

**Approval and ownership:** Dongxi approved all three acts during Day 3. All
animation design, production, and rendering belongs on the Mac Studio. DGX Spark
work is limited to derivation, numerical/autograd verification, experiment
evidence, task specification, and later review.

**Three-act structure:**

1. **Products become sums.** Build a causal sequence probability from conditional
   factors, then apply `-log` so the product unfolds into additive per-token
   surprise. Preserve the difference between summed sequence NLL and mean token
   loss.
2. **Confident errors keep a correction.** Compare log loss with the intuitive
   alternative `1-p_target`. Synchronize probability, loss, and derivative with
   respect to the target logit. As `p_target` approaches zero, show the log-loss
   gradient approaching `-1` while the alternative's softmax gradient approaches
   zero.
3. **Honest distributions win in expectation.** Fix a small categorical data
   distribution `q`; vary model distribution `p`; show expected log loss reaching
   its minimum at `p=q`. Contrast the linear alternative, whose expected optimum
   concentrates probability on the modal outcome.

**Packaging decision:** The Mac Studio may produce one coherent three-act film or
three visually coordinated shorts. All three remain one approved learning package
and must share notation, candidate colors, typography, and evidence.

**Required precision:** Use conditional sequence probabilities; use natural logs
and call their units nats; distinguish derivative with respect to probability
from derivative with respect to logits; do not confuse large `1-p_target` loss
with its vanishing softmax gradient near zero; compare scoring rules in
expectation over `q`, not from one sample; distinguish empirical distribution
matching from factual truth, calibration under shift, and general capability.

**Expected outputs:** Editable Manim source or coordinated sources, 16:9 H.264
MP4 output, lightweight GIF previews, stills for Chapter 3, exact render commands,
dependency revisions, metadata, and a mapping from each displayed number to the
committed verification evidence.

**Acceptance checks:** Product and sum sequence values agree within displayed
precision; all derivative arrows agree with analytical and PyTorch results; the
proper-scoring minimum occurs at the fixed verified `p=q`; the linear comparison
uses the same `q`; no act implies that low training loss proves truthfulness;
phone-scale text remains readable under the animation style guide.

## Cross-machine execution protocol

### Machine roles

- **DGX Spark:** CUDA/model experiments, checkpoint-dependent inspection,
  empirical reports, and canonical course integration.
- **Mac Studio:** all Manim and media production/rendering, typography and layout
  review, X article editing, and other tasks that do not require DGX-local models
  or data.

These are preferred lanes, not claims that either machine is incapable of the
other work.

### Git handoff rules

1. Uncommitted files are invisible to the other machine. Commit and push the
   required inputs before beginning remote work.
2. Use one bounded branch per portable task, such as `visuals/manim-bpe` or
   `content/x-bpe-byte-fallback`.
3. State the base commit in the task handoff and avoid editing the same files on
   both machines concurrently.
4. The producing machine returns a commit hash, changed-file list, exact render or
   validation command, outputs, and known limitations.
5. Review content correctness before merging aesthetic work into the canonical
   book branch.
6. Update the task status and new evidence in this file after integration.

### Portable task handoff template

```markdown
Task ID:
Owner/session:
Machine:
Status: queued | in progress | review | complete | blocked
Base commit:
Branch:
Learning objective:
Inputs:
Files allowed to change:
Expected outputs:
Claims that must remain precise:
Validation and acceptance checks:
Return: commit hash, changed files, commands, evidence, limitations
```
