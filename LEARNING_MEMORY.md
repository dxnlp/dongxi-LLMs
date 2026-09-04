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
- Treats guided notebooks as first-class course material rather than temporary
  session scratchpads. Preserve, validate, index, and link them from the
  corresponding chapter or solutions while keeping reusable logic in `src/`.
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
| 3 | Probabilities and next-token loss | complete | `learning_artifacts/day-03-probabilities-and-next-token-loss/README.md` |
| 4 | Attention and the causal information boundary | in progress | `learning_artifacts/day-04-attention-and-causal-information-boundary/README.md` |

At the start of each new day, create its directory and index. During the lesson,
update the relevant focused topic whenever the learner states a prediction,
demonstrates understanding, encounters a correction, or identifies an open edge.

## Scheduled frontier modules

| ID | Topic | Planned placement | Status | Durable source |
|---|---|---|---|---|
| `ARCH-LOOP-001` | Recurrent depth and looped Transformers: fixed stack reuse, variable recurrence, adaptive token-level routing, and latent versus token-space computation | Chapter 5; Days 6–7 | scheduled; primary sources captured; Astra attribution remains unverified | `learning_artifacts/day-04-attention-and-causal-information-boundary/future-recurrent-depth-and-looped-transformers.md` |

This queue preserves worthwhile, time-sensitive architecture topics without
expanding the active learning day. Recheck primary sources when the module is
taught; keep model-vendor rumors out of the factual architecture narrative until
they are corroborated.

## Public-content production queue

| ID | Type | Topic | Status | Preferred machine | Dependency |
|---|---|---|---|---|---|
| `X-BPE-001` | X article | What is a token? Unicode → BPE → model IDs | bilingual local packages prepared; editorial review pending | Mac | Chapter 2 tokenizer-mechanics enrichment complete |
| `X-EMB-001` | X article | How transformer embedding tables are actually trained | ready for Mac drafting | Mac | Chapter 2 and embedding labs complete |
| `X-ATTN-KV-001` | X article | Why LLMs cache K and V—but not Q | concept approved; portable outline ready, drafting waits for canonical Chapter 4 | Mac | Day 4 attention derivation and cached/uncached verification |
| `X-LOOP-001` | X article | Looped Transformers: more effective depth without more stored weights—but not free compute | future creation reminder; drafting waits for Days 6–7 | Mac | Chapter 5 frontier treatment, refreshed primary-source check, and controlled recurrence comparison |
| `ANIM-BPE-001` | Animation | Bytes → characters → Chinese word/phrase tokens | minimal Manim style approved and committed | Mac Studio | Day 2 explanation complete |
| `ANIM-BPE-002` | Animation | Meteor corpus → counted character BPE → vocabulary subset 1–5 → encoding | local 1080p review candidate ready; user visual approval pending | Mac Studio | Verified corpus trace and explicit second-round tie preference |
| `ANIM-EMB-001` | Animation | End-to-end embedding training and tied gradient paths | continuous-animation Mac handoff ready | Mac Studio | Day 2 embedding lab and Day 3 loss derivation |
| `ANIM-CE-001` | Animation | LLM target probability → per-token NLL → masked mean cross-entropy | approved; canonical Day 3 evidence ready for Mac production review | Mac Studio | Chapter 3, target alignment, and PyTorch verification complete |
| `ANIM-NTP-001` | Animation | One-hot next-token supervision → `p-q` gradient → distribution learning across examples | approved; verified 70/30 trajectory ready for Mac production review | Mac Studio | Gradient verification and controlled target-frequency experiment complete |
| `ANIM-LOGLOSS-001` | Animation | Why `-log p_target`: additive sequence surprise, confident-error gradients, and proper probability reporting | approved; canonical derivations and controlled evidence ready for Mac review | Mac Studio | Chapter 3 chain rule, gradient, and expected-scoring treatment complete |
| `ANIM-ATTN-001` | Animation | How loss trains attention routing and value content | approved; complete Mac task packet ready, production waits for Day 4 verification | Mac Studio | Chapter 4 synthesis plus forward, gradient, mask, and autograd evidence |

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

### Task packet: `X-ATTN-KV-001`

**Working title:** Why Do LLMs Cache K and V—but Not Q?

**Hook:** The name “KV cache” looks like arbitrary engineering jargon until the
roles of queries, keys, and values are understood. Then it becomes a compact
description of which attention states future tokens still need.

**Learning promise:** A reader should be able to connect the attention equation
to real autoregressive inference: explain what Q, K, and V do; why the same token
can produce different projected states in different contexts; why causal
attention makes an exact prefix's past states immutable; why past keys and values
remain useful while past queries do not; and why cache retention belongs to the
runtime rather than the mathematical definition of a Transformer.

**Narrative spine:**

1. Open with the learner's inference: if Q finds sources and K/V describe those
   sources, perhaps this explains the term “KV cache.”
2. Derive routing versus payload from
   $q_i=x_iW_Q$, $k_j=x_jW_K$, $v_j=x_jW_V$, and
   $o_i=\sum_j a_{ij}v_j$.
3. Prevent the central misconception by contrasting the same token in financial
   and river contexts: a cache stores context-specific per-position states, not
   one universal pair per vocabulary token.
4. Use the causal boundary to prove that appending a future token cannot change
   earlier per-layer keys and values.
5. Animate or diagram prefill followed by token-by-token decoding: each new query
   reads the stored prefix, its new key/value pair is appended, and the query is
   then no longer needed.
6. Separate architecture from systems implementation: attention always computes
   K/V, while a library or serving engine chooses whether and how to retain them.
7. Explain the compute-memory trade: caching avoids redundant prefix computation
   but consumes memory that grows with retained positions, layers, KV heads, and
   head width.
8. Close with exact-prefix reuse, request completion and release, and forward
   links to GQA, cache quantization, offloading, paged allocation, and eviction.

**Required precision:** Do not describe the cache as a token dictionary or claim
that identical tokens share K/V across arbitrary sequences. Say that caches are
per layer and per exact sequence state. Distinguish ordinary request-local reuse
from serving-level exact-prefix caching. State that retaining K/V is optional and
should preserve model outputs; it improves inference efficiency, not model
knowledge. Distinguish logical cache release from a GPU allocator returning
reserved pages to the operating system. Avoid saying queries are “never” cached
in every specialized implementation; explain why standard autoregressive
attention does not need past queries.

**Source material:**
`learning_artifacts/day-04-attention-and-causal-information-boundary/queries-keys-values-and-retrieval.md`;
`learning_artifacts/day-04-attention-and-causal-information-boundary/why-cache-keys-and-values.md`;
the future canonical Chapter 4 and attention notebook; official Hugging Face,
vLLM, TensorRT-LLM, and `llama.cpp` cache documentation; `CAND-ANIM-009`.

**Expected output:** An English X Article draft with a 2000×800 cover and one
continuous mechanism visual or a small sequence of diagrams. Produce on the Mac
after the canonical Chapter 4 claims and executable evidence stabilize. Consider
a Chinese adaptation only after the English version passes technical review.

**Mac handoff:** Pull the latest `origin/main`, create a dedicated content branch,
record its base commit, and use the task packet plus Chapter 4 as canonical
sources. Do not render animation assets on the DGX Spark.

**Acceptance checks:** A reader can explain (1) Q/K routing versus V payload; (2)
why an unchanged causal prefix yields reusable per-layer K/V; (3) why different
contexts normally require different caches; (4) why past Q is unnecessary for a
future query; (5) when a request cache is released; and (6) which claims concern
Transformer mathematics versus runtime policy. Cached and uncached outputs must
be verified equivalent within the declared numerical tolerance before the
article calls that behavior demonstrated.

### Task packet: `X-LOOP-001`

**Working title:** Looped Transformers: More Depth Without More Weights—but Not
Free Compute

**Hook:** Parameter count is often treated as if it were the whole size of a
model. Reusing one Transformer stack reveals three separate quantities: stored
weights, effective block applications, and computation per token.

**Learning promise:** A reader should be able to explain fixed stack reuse,
variable recurrent depth, and adaptive token-level recursion; distinguish each
from visible chain-of-thought; and evaluate parameter-efficiency claims without
mistaking them for compute, latency, memory, or quality claims.

**Narrative spine:** Start with an ordinary untied decoder, replace repeated
physical stacks with one shared recurrent stack, and track parameters and block
applications separately. Introduce the recurrent state equation and effective
depth accounting. Then compare fixed loops, test-time-variable recurrence, and
router-controlled per-token depth. Close with the Day 7 controlled comparison
and a claim ledger separating published evidence from interpretation and
unverified vendor reporting.

**Required precision:** Do not call a two-pass shared stack a model with twice
the stored size; do not describe the extra depth as free; do not equate shared
and independently parameterized layers; do not generalize one paper's optimal
recurrence count; and do not claim that latent recurrence necessarily hides or
suppresses textual chain-of-thought. The reported Astra architecture remains
unverified until primary architecture evidence appears.

**Source material:** `ARCH-LOOP-001`,
`learning_artifacts/day-04-attention-and-causal-information-boundary/future-recurrent-depth-and-looped-transformers.md`,
Geiping et al.'s [recurrent-depth paper](https://arxiv.org/abs/2502.05171),
Bae et al.'s [*Mixture-of-Recursions*](https://arxiv.org/abs/2507.10524),
the [Nanbeige4.2 technical report](https://arxiv.org/abs/2607.22083), the future
Chapter 5 section, the Day 7 comparison, and `CAND-ANIM-011`. Treat
*Mixture-of-Recursions* as the primary source for learned token-level routing,
not as a synonym for every fixed looped Transformer.

**Expected output:** An English X Article draft, 2000×800 cover, and a compact
mechanism diagram or approved animation excerpt. A Chinese adaptation may follow
technical review. Drafting and media production belong on the Mac Studio.

**Activation reminder:** Revisit at the end of Day 7, after refreshing primary
sources and recording the controlled experiment. Article creation is scheduled;
publication still requires the usual technical and editorial review.

**Acceptance checks:** Every statement is tagged mentally or explicitly as an
accounting identity, local measurement, primary-paper result, interpretation, or
unverified report. Parameter, FLOP, latency, memory, and quality comparisons use
clearly named contracts. A reader can explain why recurrence adds computation
without adding a new copy of the recurrent weights.

### Task packet: `ANIM-ATTN-001`

**Working title:** How Next-Token Loss Teaches Attention Where to Read and What
to Carry

**Learning objective:** Make scaled causal attention and its credit assignment
visible as one continuous computation. A viewer should distinguish the
query/key routing path from the value/content path and understand that both are
trained end to end by downstream next-token loss without a separate target
attention map.

**Approval and ownership:** Dongxi approved this animation during Day 4 after the
two backward branches were derived. All design, production, and rendering belong
on the Mac Studio. The DGX Spark supplies the canonical derivation, executable
verification, experiment evidence, task packet, and later content review.

**Animation form:** One continuous mechanism-first animation. Preserve the
identity and color of every token position, projection, matrix, and gradient
branch. Forward computation moves consistently toward the loss; backward credit
moves in the reverse direction. Avoid a slide sequence that redraws Q, K, V, or
the attention matrix as unrelated objects.

**Canonical forward spine:**

1. Begin with a short sequence represented as rows of $X$; retain position
   identity throughout.
2. Split each row through learned projections into $Q=XW_Q$, $K=XW_K$, and
   $V=XW_V$. Visually separate “routing request,” “routing address,” and
   “message content” without assigning literal linguistic features.
3. Let query rows meet key columns to form $QK^\top$. Maintain row-as-receiver
   and column-as-source orientation.
4. Divide scores by $\sqrt{d_k}$ while stabilizing their spread. Add the causal
   mask before normalization so forbidden future cells contribute neither
   numerator nor denominator.
5. Transform each allowed score row through softmax into $A$. Confirm
   nonnegative weights, allowed-row sum one, and exactly or numerically zero
   forbidden weights.
6. Use the weights as visible transport amounts carrying value-vector components
   into $O=AV$. The result must appear as a newly constructed representation,
   not a selected token or copied embedding.
7. Compress later model computation into a clearly labeled downstream
   next-token loss $L$ without implying a separate attention-supervision label.

**Canonical backward spine:**

8. Reverse motion from $G_O=\partial L/\partial O$ and split visibly at $O=AV$.
9. Send the value/content branch through
   $G_V=A^\top G_O$ and $G_{W_V}=X^\top G_V$. Its visual question is: “What
   should each retrieved source transmit?”
10. Send the routing branch through $G_A=G_OV^\top$, row-wise softmax, and the
    masked scaled scores. Then split it into
    $G_Q=G_RK/\sqrt{d_k}$ and
    $G_K=G_R^\top Q/\sqrt{d_k}$ before reaching $W_Q$ and $W_K$. Its visual
    question is: “Which sources should this receiver favor?”
11. Recombine the three contributions where they reach $X$ and the earlier
    network. Keep forbidden future edges at zero routing weight and zero routing
    gradient for the isolated query.
12. Optionally end with two brief controlled contrasts, only if legible: detach
    $A$ to freeze routing while value content still learns; detach $V$ to block
    the value branch while routing may still receive a signal through the fixed
    values.

**Required precision:** Distinguish gradient sign from optimizer update
direction. Do not imply that a high attention weight is a causal explanation or
that one head has a uniquely readable linguistic role. State that the isolated
single-head derivation omits multi-head output projection, residual pathways,
and later layers. Mask before softmax; do not let a forbidden score enter the
denominator. Identify $R=QK^\top/\sqrt{d_k}+M$ consistently so the scale appears
exactly once in the query/key derivatives. Do not claim that attention receives
its own correct-map labels.

**Source material:**
`learning_artifacts/day-04-attention-and-causal-information-boundary/queries-keys-values-and-retrieval.md`;
`learning_artifacts/day-04-attention-and-causal-information-boundary/dot-products-as-learned-compatibility.md`;
`learning_artifacts/day-04-attention-and-causal-information-boundary/why-scale-dot-products.md`;
`learning_artifacts/day-04-attention-and-causal-information-boundary/causal-mask-before-softmax.md`;
`learning_artifacts/day-04-attention-and-causal-information-boundary/attention-output-as-value-mixture.md`;
`learning_artifacts/day-04-attention-and-causal-information-boundary/attention-weights-are-not-explanations.md`;
`learning_artifacts/day-04-attention-and-causal-information-boundary/how-loss-trains-qkv.md`;
the future canonical Chapter 4, worked solutions, Day 4 notebook, reusable
source, tests, and experiment report.

**Evidence dependency:** Do not freeze displayed numerical values or begin final
rendering until the manual forward pass agrees with a trusted PyTorch reference,
analytical gradients agree with autograd and finite differences under declared
tolerances, forbidden future edges are verified zero, and detach experiments
confirm the two branches. Chapter 4 must be synthesized before final editorial
review.

**Expected outputs:** Editable Manim source, 16:9 H.264 MP4, lightweight GIF
preview, one Chapter 4 still, exact render command, Python/Manim revision
manifest, and a mapping from displayed numbers to committed verification
evidence. Follow `visuals/animations/STYLE_GUIDE.md`.

**Mac handoff:** Use branch `visuals/manim-attention-gradients`. Pull the latest
`origin/main` containing this packet, record the exact base commit, and limit the
production branch to animation source, rendered previews, metadata, and render
instructions. Do not rewrite canonical Chapter 4 prose from the rendering
branch.

**Acceptance checks:** Tensor orientations and shapes remain unambiguous; each
attention row sums to one over allowed sources; forbidden cells never influence
normalization; value mixing remains distinct from routing; every displayed
gradient matches committed evidence; detaching one path produces the claimed
gradient boundary; the final loss is the downstream next-token objective; the
animation never presents attention weights as a complete explanation; all text
and matrix labels remain readable at phone scale.

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

### Task packet: `ANIM-BPE-002`

- User approval: direct request to create this animation on 2026-09-03.
- Base commit: `d91f5eb466990db724437733031bed5c67cf96b1`.
- Branch: `codex/visuals/meteor-bpe`; owner: local Mac Studio.
- Objective: make BPE pair counting, vocabulary growth with retained
  constituents, arbitrary Token IDs, and subsequent encoding visible.
- Inputs: the five user-supplied strings, each once; Chapter 2; `tiny_bpe.py`;
  the approved `STYLE_GUIDE.md` and `manim_style.py` primitives.
- Allowed files: the meteor animation project, its reusable trace module and
  tests, focused Day 2 artifact, and production indexes. Preserve the active
  Day 3 learning state and the earlier BPE render.
- Project: `visuals/animations/projects/meteor-bpe/` (brief, full trace, scene,
  render script, review HTML, media, and metadata).
- Evidence: unique round-1 maximum `(流, 星)` at 2; all 15 round-2 candidates
  at 1; the declared teaching tie preference selects `(流星, 雨)`. Corpus token
  counts 22 → 20 → 19, 17 original characters plus two merges, eight exact
  encoding round trips, and nine passing focused BPE tests.
- Precision: character-level BPE; fixed input frequencies; preference applies
  only to tied maxima; displayed 1–5 are a subset of 19 illustrative IDs;
  original vocabulary entries survive; encoding does not add merges; no claim
  about semantic understanding follows from this tokenizer demonstration.
- Expected outputs: 1080p H.264 MP4, 960×540 GIF, full/half PNG stills, contact
  sheet, source/asset hashes, dependency identity, and a local review page.
- Acceptance: values and rewrites match the trace, ties disclosed on screen,
  normal kerning and Songti glyphs, continuous object identity, no clipped
  frames, and legibility at 50% scale. User visual approval precedes final
  reader-facing integration; source integration was approved on 2026-09-04.
- Return state: local review candidate ready, 40.23 seconds at 1920×1080/30 fps;
  nine focused tests pass. Twelve storyboard checkpoints, merge transitions,
  full/half stills, and source/media hashes checked. Review via the project's
  `review.html`; await user visual approval. No media published or existing
  asset replaced. Source integration approved on 2026-09-04, excluding MP4s.
  MP4s remain local and ignored; the GIF, stills, trace, and render command are
  portable. Candidate ID is `CAND-ANIM-010`: the unpublished local `008` was
  renumbered to preserve Spark's already committed attention candidate.

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

**Dependency status:** Satisfied on 2026-09-03. Chapter 3 now derives softmax,
negative log-likelihood, cross-entropy, causal shifting, label alignment, and the
masked mean; the current animation remains an intentionally limited preview until
the Mac Studio produces the canonical version.

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

**Dependency status:** Satisfied on 2026-09-03 by Chapter 3 and
`experiments/reports/2026-09-03-next-token-distribution.md`. Production remains
an explicit Mac Studio task.

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

**Dependency status:** The canonical Day 3 derivations and controlled
distribution-learning evidence are complete as of 2026-09-03. The Mac Studio
must still map every displayed comparison, including the `1-p_target` alternative,
to verified numbers during production.

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
