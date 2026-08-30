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
- Wants mechanisms explained beneath convenient APIs: bytes and BPE merges before
  token IDs, embedding rows before contextual states, and masks before trainer
  abstractions.
- Values multilingual comparisons, especially English, Chinese, and Swedish.
- Wants course learning to become a coherent technical book, not chronological
  notes.
- Wants strong discussions reused as public X articles and mathematical
  animations after the canonical book treatment is stable.
- Uses the DGX Spark for model- and GPU-dependent work and may use a local Mac for
  animation, design, editing, and publishing tasks.

## Learning-artifact index

Detailed knowledge state lives in `learning_artifacts/` so future sessions can
load only the active day and relevant topic rather than rereading one growing
ledger.

| Day | Topic | Status | Artifact index |
|---:|---|---|---|
| 1 | Evidence before optimization | complete | `learning_artifacts/day-01-evidence-before-optimization/README.md` |
| 2 | Text, tokens, and embeddings | in progress | `learning_artifacts/day-02-text-tokens-and-embeddings/README.md` |

At the start of each new day, create its directory and index. During the lesson,
update the relevant focused topic whenever the learner states a prediction,
demonstrates understanding, encounters a correction, or identifies an open edge.

## Public-content production queue

| ID | Type | Topic | Status | Preferred machine | Dependency |
|---|---|---|---|---|---|
| `X-BPE-001` | X article | Why an English byte tokenizer can still encode `数` | queued | Mac | Stable Chapter 2 draft |
| `ANIM-BPE-001` | Animation | Bytes → characters → Chinese word/phrase tokens | baseline rendered; Manim refinement queued | Mac | Day 2 explanation complete |
| `ANIM-EMB-001` | Animation | End-to-end embedding training and tied gradient paths | interactive explanation drafted; reusable animation queued | Mac | Day 2 embedding lab and Day 3 loss derivation |
| `ANIM-CE-001` | Animation | Correct-token probability → negative-log loss | preview rendered; canonical version deferred | Mac | Day 3 derivation |

### Task packet: `X-BPE-001`

**Working title:** How an English-Trained Tokenizer Can Still Read the Bytes of
`数`

**Learning promise:** A reader should be able to predict what happens when an
English-only byte-level BPE tokenizer receives a Chinese character, and explain
why encodability is different from understanding.

**Narrative spine:**

1. Open with the apparent paradox: the tokenizer never learned Chinese, yet it
   does not need `<unk>`.
2. Show the concrete reversible mapping `数` → `E6 95 B0`.
3. Distinguish the 256-byte base alphabet from merges learned offline.
4. Show the compression ladder: three byte tokens → one character token →
   multi-character tokens such as `数据` and `数据库`.
5. Explain that corpus mixture and vocabulary budget decide which compression is
   worth learning.
6. End with the distinction `can encode ≠ can understand`.

**Required precision:**

- Qualify the example as byte-level BPE; do not generalize it to every tokenizer.
- Do not describe encoding as learning new merges at runtime.
- Do not claim that BPE understands Chinese words; it exploits repeated adjacent
  patterns.
- Preserve the difference between an observed Qwen3 segmentation and a universal
  rule.

**Source material:**
`learning_artifacts/day-02-text-tokens-and-embeddings/bpe-training-and-byte-coverage.md`;
the fixed multilingual specification; the Chapter 2 draft when available;
`ANIM-BPE-001`.

**Expected output:** An English X Article or thread draft, plus a short Chinese
adaptation only after the English claims are reviewed. Decide article versus
thread at production time rather than maintaining two premature versions.

**Acceptance checks:** A reader can answer both of these after reading: (1) why
can the tokenizer encode `数` without a Chinese vocabulary entry? (2) why might
the resulting language model still not understand Chinese?

### Task packet: `ANIM-BPE-001`

**Learning objective:** Make the difference between universal byte coverage and
learned compression visible in one sequence.

**Existing baseline:**

- Source: `visuals/animations/bpe_byte_merges.py`
- Render: `visuals/animations/rendered/bpe-byte-merges.gif`
- Reproducible environment: `visuals/animations/pyproject.toml` and `uv.lock`

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

**Source material:**
`learning_artifacts/day-02-text-tokens-and-embeddings/embeddings-context-and-gradients.md`;
the planned Day 2 embedding lab; the Day 3 cross-entropy derivation when
available.

**Storyboard:**

1. Start with token IDs `[cat=1, sat=2]` selecting rows `E[1]` and `E[2]` from
   `E ∈ R^(V×d)`.
2. Move the selected vectors through a transformer into a contextual state `h`.
3. Reuse the tied matrix to produce `logits = hE^T`, identify target `dog`, and
   create the next-token loss.
4. Run the backward pass along two visually distinct paths: lookup-path gradients
   to selected input rows, and output-classifier gradients to all vocabulary rows.
5. Accumulate both contributions on input rows, then show an optimizer update.
6. Contrast briefly with untied weights, where the all-row classifier gradient
   updates a separate output matrix instead of the input embedding table.

**Required precision:** Do not say that only selected rows receive total gradient
when weights are tied; call selected-row contributions through lookup "direct
input-path gradients"; identify `E ← E - η∇E` as an SGD sketch rather than the
exact AdamW rule; do not derive cross-entropy before Day 3's canonical treatment.

**Expected outputs:** Editable animation source, 16:9 H.264 MP4, lightweight GIF,
render command, dependency revisions, and a still diagram for Chapter 2. The Mac
is preferred for final media rendering.

**Acceptance checks:** The input IDs and matrix shapes remain visible; tied versus
untied behavior is unambiguous; input rows visibly accumulate two contributions
under tying; the target and loss direction agree with the later numerical lab;
the final frame remains readable on a phone.

### Task packet: `ANIM-CE-001`

**Learning objective:** Connect the probability assigned to the correct next
token with its one-hot cross-entropy contribution, `L = -ln p_correct`.

**Existing preview:**

- Source: `visuals/animations/cross_entropy_curve.py`
- Render: `visuals/animations/rendered/cross-entropy-curve.gif`

**Storyboard for the canonical Day 3 version:** Keep a next-token probability
distribution and the negative-log curve synchronized. Move probability toward
and away from the correct token while a marker traces the loss. Later add only
the minimum transition needed to connect logits → softmax probabilities → the
correct-token loss.

**Dependency:** Do not finalize the narrative before Day 3 derives softmax,
negative log-likelihood, cross-entropy, causal shifting, and label alignment. The
current animation is an intentionally limited preview.

**Required precision:** Use the natural logarithm; identify the correct target
explicitly; do not imply that cross-entropy uses only the largest predicted
probability; show that the one-hot target reduces the token loss to
`-ln p_correct`; do not include padded positions as targets.

**Expected outputs:** Editable source, MP4, GIF preview, render command, revision
manifest, and a still frame suitable for Chapter 3.

**Acceptance checks:** Displayed probabilities sum to one within rounding; the
marker agrees numerically with `-ln(p)`; behavior near zero is described as a
limit rather than evaluating `ln(0)`; the animation agrees with the later manual
and PyTorch calculations.

## Cross-machine execution protocol

### Machine roles

- **DGX Spark:** CUDA/model experiments, checkpoint-dependent inspection,
  empirical reports, and canonical course integration.
- **Local Mac:** Manim and media rendering, typography and layout review, X article
  editing, and other tasks that do not require DGX-local models or data.

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
