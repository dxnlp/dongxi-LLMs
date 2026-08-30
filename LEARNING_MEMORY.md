# Learning Memory and Production Queue

This ledger preserves the learner's evolving mental models and turns promising
discussions into portable production tasks. It is designed to survive chat
compaction, new Codex sessions, and work split across the DGX Spark and a local
Mac.

It is not a transcript and it does not replace the book:

- `BOOK.md` defines the reader-facing narrative.
- `ROADMAP.md` defines planned learning outcomes.
- `PROGRESS.md` records the active day and exact next action.
- This file records demonstrated understanding, unresolved edges, public-content
  ideas, and cross-machine task packets.

## Maintenance contract

Update this file when a discussion produces at least one of the following:

1. a durable explanation or corrected misconception;
2. evidence that the learner can explain or calculate a mechanism;
3. an unresolved question worth carrying into a later day;
4. a possible article, animation, diagram, experiment, or exercise;
5. work that may be executed in another session or on another machine.

Record the learner's current model and the important distinction, not every turn
of dialogue. Never silently upgrade "understood in conversation" to "verified by
experiment" or "mastered." Link empirical claims to specifications, reports, or
stored outputs when those exist.

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

## Knowledge map

Status vocabulary:

- `introduced`: discussed but not yet explained back or calculated;
- `demonstrated`: explained or calculated correctly in the interactive lesson;
- `observed`: measured during an interactive session but not yet preserved as a
  reproducible report;
- `verified`: supported by preserved executable evidence;
- `open`: a named gap remains.

### Day 1 — Evidence before optimization

- `demonstrated` — Distinguishes observations, unsupported claims, and supported
  interpretations.
- `demonstrated` — Understands that a successful process exit is necessary but
  not sufficient when a precommitted safety criterion fails.
- `demonstrated` — Separates report evidence (exit code, measured minimum
  `MemAvailable`, observations and interpretations) from environment and run
  specifications (GPU/driver, package versions, Git commit, starting memory).
- `verified` — The Qwen3-0.6B smoke run and its limits are preserved in
  `experiments/reports/2026-08-29-qwen3-0.6b-sft-smoke.md`.

### Day 2 — Text, tokens, and embeddings

#### Tokenizer identity and multilingual behavior

- `demonstrated` — Token IDs are meaningful only under the tokenizer that created
  them. Sending IDs from another tokenizer can map the same integers to unrelated
  strings and therefore unrelated embedding rows.
- `demonstrated` — Tokenization depends on corpus composition and weighting,
  normalization and pre-tokenization, subword algorithm and merge priority,
  vocabulary budget, and special-token policy. Model size alone does not determine
  the tokenizer.
- `observed` — On the fixed examples in
  `experiments/specs/2026-08-30-qwen3-multilingual-tokenization.yaml`, observed
  Qwen3 counts were Chinese 9, English 11, and Swedish 20. The learner's original
  prediction, Swedish < Chinese < English, was falsified rather than rewritten.
  Re-run these measurements in the planned lab before promoting the claim to
  `verified`.
- `observed` — Qwen3 tokenized `数据库` as one token in the inspected revision,
  while its substrings `数据` and `库` also had tokens. This is an observation
  about that tokenizer revision, not a universal Chinese rule; it still requires
  preservation in the Day 2 report.

#### BPE training versus encoding

- `demonstrated` — BPE has an offline training phase: prepare a representative
  corpus, count eligible adjacent pairs, add the selected merged symbol, rewrite
  the representation, and repeat under a merge/vocabulary budget.
- `demonstrated` — Encoding is a frozen online phase. It replays learned merge
  priorities and maps the final pieces to IDs; it does not learn a new token when
  an unfamiliar word arrives.
- `demonstrated` — Vocabulary membership alone is not a complete description of
  segmentation. Pre-tokenization boundaries and ordered merge ranks also matter.
- `demonstrated` — Chinese BPE is statistical rather than inherently
  morphological. Learned pieces may be part of a UTF-8 character, one character,
  a multi-character word such as `数据`, or a frequent expression such as
  `数据库`.

#### Byte-level coverage

- `demonstrated` — In UTF-8, `数` is the three-byte sequence `E6 95 B0`. A true
  byte-level tokenizer trained only on English can still encode it using base byte
  tokens, likely three tokens if no relevant merges were learned.
- `demonstrated` — Sufficient Chinese exposure may learn byte merges that form
  `数`, followed by character merges that form `数据` or `数据库`.
- `demonstrated` — Perfect byte-level representability does not imply linguistic
  understanding. An English-only model can mechanically preserve the bytes while
  having learned little about their Chinese meaning.
- `demonstrated` — A tokenizer without complete byte coverage or byte fallback may
  instead emit `<unk>` for an unsupported character.

#### Embeddings and contextual states

- `demonstrated` — IDs have shape `[B, T]`; embedding lookup through
  `E ∈ R^(V×d)` returns `[B, T, d]`.
- `demonstrated` — For IDs `[2, 5, 2]`, three vectors are returned but only rows 2
  and 5 receive direct embedding gradients; the two contributions to row 2 add.
- `demonstrated` — Repeated occurrences of the same ID retrieve identical input
  embeddings. Their later hidden states generally differ because position and
  context affect transformer computation.
- `introduced` — Qwen3 applies positional information through RoPE inside
  attention rather than by adding a learned absolute-position embedding here.

#### Padding, masks, and loss

- `demonstrated` — An attention mask value of 1 marks a real token position and 0
  marks padding to exclude from attention use.
- `demonstrated` — Attention masking and loss masking have different jobs. Padding
  labels commonly use `-100` so cross-entropy ignores those positions; otherwise
  training rewards prediction of the artificial padding ID.
- `demonstrated` — `<eos>` is a meaningful target that teaches stopping, whereas
  padding is batch formatting and is normally excluded from the loss.

#### Remaining Day 2 gaps

- `open` — Verify embedding lookup, gradient accumulation, padding masks, and
  input/output weight tying in executable code.
- `open` — Build and report the complete tokenizer exploration lab.
- `open` — Integrate the discussion and evidence into Chapter 2 with exercises
  and solutions.

## Public-content production queue

| ID | Type | Topic | Status | Preferred machine | Dependency |
|---|---|---|---|---|---|
| `X-BPE-001` | X article | Why an English byte tokenizer can still encode `数` | queued | Mac | Stable Chapter 2 draft |
| `ANIM-BPE-001` | Animation | Bytes → characters → Chinese word/phrase tokens | baseline rendered; Manim refinement queued | Mac | Day 2 explanation complete |
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

**Source material:** This ledger's Day 2 BPE sections; the fixed multilingual
specification; the Chapter 2 draft when available; `ANIM-BPE-001`.

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
