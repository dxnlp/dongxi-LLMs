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

- 2026-09-14 course synthesis: Chapter 6 now integrates the completed TinyStories
  run in sections6.14–6.20, with18 worked solutions and a standard-library
  evidence-reading lab. Portable source-hashed metrics and fixed-grid samples:
  `experiments/reports/2026-09-14-tinystories-learning-result.json`.
  Main lesson: completed execution, improved prediction and reliable coherence
  are separate claims. A dedicated Day9 analysis notebook and controlled trained
  comparison remain pending. Dashboard/playground stopped at user request;
  port8765 verified closed. Do not restart services or training automatically.

- 2026-09-14: prefer light, minimal dashboard interfaces: plain headings,
  restrained typography, compact controls and readable charts. Avoid decorative
  slogans, oversized hero text, numbered cards and excessive boxes. Rich
  learning visuals should explain the mechanism, not add interface decoration.

-2026-09-14: user requested loading the trained model in the dashboard for custom
  prompts. Added **Prompt DongxiGPT**, with final update14000 loaded on Spark,
  temperature/seed/new-token controls and explicit generation bounds. It is a
  story completion model, not a chat-tuned assistant. User prompts are not saved;
  weights/logs are unchanged. Active inference service is
  `dongxigpt-playground-20260914.service`; stop it before starting new training.
  `docs/TRAINING_MONITOR.md` documents inference versus log-only modes.

- Day9 learning launch,2026-09-13: user said "let's run the training". Full-data
  preparation plus guarded background launch are authorized and started. Freeze
 14,000 updates,batch16,200 warmup,4-hour cap and checkpoint interval400; inspect
  `outputs/day09-learning-launch/status.json` before making a status claim.
  Initial launcher state was waiting for full data, not yet training. Canonical
  record: `experiments/reports/2026-09-13-tinystories-learning-launch.md`.
  No overnight extension, parallel training campaign or public publishing.
  Start confirmed at20:56:39 UTC after95 tests:1,792,647 deduplicated training
  stories prepared;21,990 validation stories held out. Initial waiting state is
  superseded. Do not infer successful completion from service startup.

- Day9 monitoring,2026-09-13: user accepted a local read-only dashboard. Built
  the training observatory around existing JSON/JSONL evidence: loss, gradients,
  LR, useful throughput, memory reserve, activation RMS and checkpoint story
  comparisons. No GPU job controls, remote publication or new training run.
  `docs/TRAINING_MONITOR.md` has Spark startup/Mac SSH-viewing instructions and
  telemetry limitations. Live-looking timestamps are not process-health proof.

- Day9 pipeline implementation,2026-09-13: user explicitly requested build and
  verification. Separate SDPA trainer/data pipeline now passes82 CPU tests and
  three full-model BF16 Spark updates, with bitwise-identical saved-state replay.
  Prepared only1,024 training/128 validation stories; no multi-hour learning run.
  Canonical report: `experiments/reports/2026-09-13-tinystories-pipeline-smoke.md`;
  commands: `docs/TINYSTORIES_PIPELINE.md`. This supersedes earlier "no GPU run"
  planning statements below without implying Day9 completion or story competence.

- Day9 budget discussion,2026-09-13: learner accepted the tokenizer/model
  baseline and asked for a hardware-informed runtime recommendation. Proposed
  15–30-minute smoke/profile after preparation, then a4-hour learning cap;
  overnight8–12-hour follow-up only after evidence review. This is an agent
  recommendation, not approved execution or a prediction of time to coherence.
  Host snapshot and FP4-vs-BF16 distinction are recorded in the Day9 design note.

- Latest Day9 decision,2026-09-13: user selected TinyStories and asked to settle
  tokenizer/model configuration. Working baseline: reuse GPT-2 byte-BPE only,
  randomly initialize our12-layer512-wide modern decoder,8 MHA heads,1536-wide
  SwiGLU,1024-token context, tied embedding/output;66.64M parameters verified
  by shape-only counting. Canonical design and pending execution gates:
  `learning_artifacts/day-09-pretraining-run-and-diagnosis/tokenizer-and-model-baseline.md`.
  Corpus alternatives remain historical; no bulk data download or GPU run yet.

- Day 9 dataset follow-up, 2026-09-13: learner wants to explore alternatives,
  not automatically adopt TinyStories. Shortlist TinyStories and SimpleStories;
  Cosmopedia stories and human WritingPrompts are potential later contrasts.
  Preserve the English-story objective. Comparison rationale and primary sources
  live in the Day 9 artifact; no final selection, download or run approval.

- 2026-09-13: move forward assuming current chapters are covered for planning;
  do not treat this premise as independently tested mastery. For Day 9 the
  learner chose the easier goal of watching DongxiGPT learn coherent English
  stories. TinyStories is the agent's proposed dataset, not a completed data
  selection or approved run. Current discussion remains on Spark; source and
  next decisions: `learning_artifacts/day-09-pretraining-run-and-diagnosis/README.md`.

- Latest request, 2026-09-09: create the complete Day 8 notebooks and coherent
  chapter directly. The isolated one-step quadratic SGD slider was described as
  boring. Prefer meaningful LLM-system dilemmas and interventions tied to the
  decoder/data/recipe; rich visuals should explain a mechanism, not merely add
  controls. Prepared content is not demonstrated mastery. Day 8 companion path:
  `notebooks/day-08/README.md`; canonical Chapter 6 and twelve worked solutions
  are linked there. Current continuation remains on Spark.

- Correction on 2026-09-09: learner rejected the stale Day 6 handoff. Intended
  continuation is Day 8, not the earliest unfinished lesson. After reporting
  Mac dependency issues, explicitly chose to continue on Spark. "Interactive"
  means rich adjustable visuals **directly in the app conversation**, not only
  notebooks or an external learning website. Use this surface for explanations;
  do not require local PyTorch/Jupyter for a browser-side toy demonstration.
  Keep earlier gaps as optional review, never silently rewind the course.

- On 2026-09-09 explicitly requested live, richly visual learning beyond notebooks:
  linked architecture views, mathematics and controls, single-step experiments,
  paired runs, deliberate failures, and question-driven exploration. Keep the
  coherent chapter and worked notebooks as companions. First proposed pilot is
  the Day 8 Optimizer Playground; it has not been built or verified on Mac.
- Mac Studio is the default learning/CPU/visual/media environment; Spark handles
  approved heavy GPU tasks. Use `Switch to Mac` / `Switch to Spark` to save the
  handoff and commit/push scoped work before departure. On `Continue on Mac` /
  `Continue on Spark`, verify actual execution, safely sync and reread state,
  then resume. Never confuse a Mac client viewing Spark with Mac execution.
  Remind the learner at workload boundaries without automatically switching.
  Standing contract: `docs/LEARNING_WORKFLOW.md`; current packet:
  `docs/handoffs/CURRENT.md` (supersedes `LEARN-MAC-001`: learner returned to
  Day 8 on Spark after reported Mac dependency friction).

- On 2026-09-08 reported mathematical rendering errors on GitHub, explicitly
  including Chapters 4 and 5. Check all book chapters and solutions, not just
  notebook previews. Use dollar delimiters and portable upright function names;
  follow `docs/MATH_FORMATTING.md` and run the book-math regression check.
  Formatting maintenance does not change learning progress or notebook results.

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
- On 2026-09-07 explicitly requested explanatory visuals because text/code-only
  notebooks felt boring. Include data-backed diagrams, heatmaps, vector/gradient
  views, and learning curves next to the mechanism, with concise reading guides
  and runnable plot code. Also requested architecture visuals: include a whole-
  model map with the current component highlighted and a detailed component
  close-up showing operations, branches, and tensor shapes. Chapter 5 now has
  52 saved previews across twelve notebooks, including the Day 7 defense;
  see `docs/NOTEBOOK_VISUALS.md`. Static notebook plots stay
  distinct from approval-gated animation production on Mac Studio.
- Wants every book chapter to contain several focused notebook sessions for its
  important LLM mathematics, architecture, optimization, data, or evaluation
  mechanisms. Use the course-wide mechanism → perturbation/failure →
  integration/evidence pathway in `docs/NOTEBOOK_CURRICULUM.md`; refine and build
  the sessions as each chapter becomes active.
- When requesting a complete chapter, wants all required topics developed for
  a new reader, independently of live learning progress. On 2026-09-05 deferred
  Day 4 notebook practice while requesting the complete chapter and companions.
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
| 5 | Decoder-only Transformer | in progress | `learning_artifacts/day-05-decoder-only-transformer/README.md` |
| 6 | Modern architecture design | material ready; guided study pending | `learning_artifacts/day-06-modern-architecture/README.md` |
| 7 | Architecture synthesis | notebooks ready; guided defense pending | `learning_artifacts/day-07-architecture-synthesis/README.md` |
| 8 | Pretraining recipe and interactive-learning design | Chapter 6 Day 8 foundation + 3 worked notebooks prepared; guided study pending on Spark | `learning_artifacts/day-08-pretraining-data-and-recipe/README.md` |
| 9 | DongxiGPT English-story pretraining | Objective chosen; candidate data and bounded run planning, no GPU experiment yet | `learning_artifacts/day-09-pretraining-run-and-diagnosis/README.md` |

At the start of each new day, create its directory and index. During the lesson,
update the relevant focused topic whenever the learner states a prediction,
demonstrates understanding, encounters a correction, or identifies an open edge.

Day 4 has a complete chapter, twelve worked solutions, three reference notebooks,
and verified forward/gradient/scaling/cache evidence. See
`learning_artifacts/day-04-attention-and-causal-information-boundary/chapter-coverage-and-evidence.md`.
Live notebook practice is deferred. On 2026-09-06 the learner returned to Day 4
conceptual study, reaching backward credit into the Q/K/V projection matrices,
then requested a layer-by-layer Chapter 5 hands-on pathway. Resume from
`notebooks/day-05/README.md`: seven baseline notebooks, three modern sessions
across Days 6–7, and optional recurrence. The learner then requested building
all notebooks before studying them one by one. All eleven are now created and
reference-verified (93 code cells, 38 repository tests). Residuals, normalization,
and MLPs have separate baseline sessions with adjacent solutions. Companion lab:
`book/labs/05-building-a-modern-decoder.md`; report:
`experiments/reports/2026-09-06-decoder-notebooks.md`. Those counts describe the
initial reference revision; the visual revision has 152 code cells, 48 figures,
and 47 tests. Live discussion has since reached Notebook 07 (2026-09-07).

At the learner's request, the Day 5 foundation is now a coherent narrative in
`book/chapters/05-building-a-modern-decoder.md`, with twelve worked conceptual
answers in `book/solutions/05-decoder-notebook-solutions.md`. Extend this same
chapter with Days 6–7 material rather than creating disconnected daily chapters.
Preserve the accessible examples: head-specific gradient paths under a shared
loss; residual skip versus recoverable backup; per-token feature centering and
scaling; contextual inputs to a positionwise MLP; many token losses before one
optimizer step. Focused discussion record:
`learning_artifacts/day-05-decoder-only-transformer/normalization-mlp-and-batched-learning.md`.
Next: review the foundation/Notebook 07 evidence boundary, then transition to Day
6 when the learner is ready. Do not infer mastery from explanations or agent
notebook execution. No animation production or new publication was authorized.

Day 6 preparation (2026-09-07): at the learner's direct request, extended the
same chapter with sections 5.11–5.22 and worked answers 13–24. Includes RMSNorm,
SwiGLU and its product gradients, RoPE and cache offsets, GQA, the lab's explicitly
defined Q/K norm, parameter/cache/FLOP accounting, pinned Qwen shapes, three
accounting-only larger candidates, and a sourced recurrent-depth introduction.
The existing three notebooks passed another source-preserving readiness run
(43 code cells, 13 figures). No duplicate notebooks were created. Begin guided
Day 6 study at RMSNorm/SwiGLU; material readiness is not learner completion.
The Day 7 architecture defense and any budget-controlled trained comparison
remain pending. Both requested recurrent-depth papers are cited in section 5.20;
X-LOOP-001 and CAND-ANIM-011 remain the canonical public-content reminders.

Day 7 notebook preparation (2026-09-07): added the missing core defense
`notebooks/day-07/02_architecture_defense.ipynb`, retaining the existing optional
`01_recurrent_depth.ipynb`. Study defense first, recurrence second; filenames
preserve old links. Includes actual shape traces, budgets, loss gradients,
controlled wrong-offset/time-mixing faults, and a comparison proposal. Twelve
notebooks now pass (168 code cells, 52 figures); 53 tests pass. This does not
execute the proposed trained comparison or complete learner mastery. Continue
guided Day 6 first unless the learner requests the Day 7 review directly.

Day 7 chapter synthesis (2026-09-07): at the learner's request, completed the
Days 5–7 narrative arc in Chapter 5 through section 5.27, with 30 worked answers.
The new conclusion explains actual tensor boundaries, mean-loss backward
connectivity versus optimizer updates, independent finite/causal/cache checks,
the GQA parameter/cache ledger, and distinct parameter/compute/time comparison
contracts. It reuses the verified Day 7 fixture and two saved figures; no new
training result is claimed. The defense rubric is choice → mechanism → shapes →
evidence → trade-off → failure risk → next experiment. Guided study and independent
defense remain pending. The portable record is
`learning_artifacts/day-07-architecture-synthesis/README.md`; reuse existing
animation candidates, with production still approval-gated on Mac Studio.

## Scheduled frontier modules

2026-09-10 user-led addition: [DeepSeek V4.1 causal encoder-decoder and KV flow](learning_artifacts/day-06-modern-architecture/deepseek-v41-causal-encoder-decoder.md).
Distinguish encoder-derived global KV, CSA2 cross-layer sharing/index selection,
per-layer main Q/local KV, and approximate bounded replay. Primary report and
pinned reference source inspected; inline schematic prepared, learner prediction
and numerical verification pending. No model run, media production, or article
approved by this discussion. Resume this requested topic in this conversation;
the separate Day 8 Spark plan and earlier review backlog remain recorded.

| ID | Topic | Planned placement | Status | Durable source |
|---|---|---|---|---|
| `ARCH-LOOP-001` | Recurrent depth and looped Transformers: fixed stack reuse, variable recurrence, adaptive token-level routing, and latent versus token-space computation | Chapter 5; Days 6–7 | conceptual prose ready in section 5.20; Day 7 comparison pending; vendor attribution unverified | `learning_artifacts/day-04-attention-and-causal-information-boundary/future-recurrent-depth-and-looped-transformers.md` |

This queue preserves worthwhile, time-sensitive architecture topics without
expanding the active learning day. Recheck primary sources when the module is
taught; keep model-vendor rumors out of the factual architecture narrative until
they are corroborated.

## Public-content production queue

| ID | Type | Topic | Status | Preferred machine | Dependency |
|---|---|---|---|---|---|
| `X-BPE-001` | X article | What is a token? Unicode → BPE → model IDs | bilingual local packages prepared; editorial review pending | Mac | Chapter 2 tokenizer-mechanics enrichment complete |
| `X-EMB-001` | X article | How transformer embedding tables are actually trained | ready for Mac drafting | Mac | Chapter 2 and embedding labs complete |
| `X-ATTN-KV-001` | X article | Why LLMs cache K and V—but not Q | bilingual local draft and five figures ready for review | Mac | `publications/x-articles/x-attn-kv-001/`; Chapter 4 toy evidence and bounded DeepSeek case study; no serving benchmark claimed |
| `X-LOOP-001` | X article | Looped Transformers: more effective depth without more stored weights—but not free compute | conceptual source ready; drafting waits for Day 7 evidence | Mac | Chapter 5 section 5.20 and refreshed primary abstracts ready; controlled recurrence comparison pending |
| `X-INFER-001` | X article | LLM inference: latency, memory and scheduling | Chinese local draft and seven static figures ready for review | Mac | Chapters 3–5 plus primary-source systems map; no serving benchmark, new video or X upload |
| `X-PD-001` | X article | Prefill and Decode | independent Chinese local draft and seven GIFs with static fallbacks ready for review | Mac | Chapter 4 §§4.10–4.11, Chapter 5 §5.18; phase mechanisms first, short third-party Spark/Mac case; ANIM-PD-001 rendered |
| `ANIM-PD-001` | Animation | Prefill and Decode: seven article mechanisms | seven GIFs and 1080p MP4s rendered; article integrated for local review | Mac Studio | Numerical mechanism checks, event/media checks and direct frame inspection; browser playback not verified |
| `ANIM-BPE-001` | Animation | Bytes → characters → Chinese word/phrase tokens | minimal Manim style approved and committed | Mac Studio | Day 2 explanation complete |
| `ANIM-LOOP-001` | Animation | Fixed recurrence, repeated stack, unfolded shared applications | Chinese introduction and three English 1080p clips ready for visual review | Mac Studio | `visuals/animations/projects/looped-transformer/english/review.html` |
| `ANIM-BPE-002` | Animation | Meteor corpus → counted character BPE → vocabulary subset 1–5 → encoding | local 1080p review candidate ready; user visual approval pending | Mac Studio | Verified corpus trace and explicit second-round tie preference |
| `ANIM-EMB-001` | Animation | End-to-end embedding training and tied gradient paths | continuous-animation Mac handoff ready | Mac Studio | Day 2 embedding lab and Day 3 loss derivation |
| `ANIM-CE-001` | Animation | LLM target probability → per-token NLL → masked mean cross-entropy | approved; canonical Day 3 evidence ready for Mac production review | Mac Studio | Chapter 3, target alignment, and PyTorch verification complete |
| `ANIM-NTP-001` | Animation | One-hot next-token supervision → `p-q` gradient → distribution learning across examples | approved; verified 70/30 trajectory ready for Mac production review | Mac Studio | Gradient verification and controlled target-frequency experiment complete |
| `ANIM-LOGLOSS-001` | Animation | Why `-log p_target`: additive sequence surprise, confident-error gradients, and proper probability reporting | approved; canonical derivations and controlled evidence ready for Mac review | Mac Studio | Chapter 3 chain rule, gradient, and expected-scoring treatment complete |
| `ANIM-ATTN-001` | Animation | How loss trains attention routing and value content | approved; canonical material and numerical evidence ready for Mac production | Mac Studio | Chapter 4, forward/gradient checks, finite differences, and mask/detach evidence complete |

## Production-system tasks

### Task packet: `ANIM-PD-002` — document example, two kinds of waiting

- User approval, 2026-09-15: standalone adaptation of the lifecycle clip, using
  readable document content and output phrases. Do not modify the article.
- Host/base: Mac production, dirty main at ca2cfe1; preserve work and skip pull.
- Revision approval: initial document-processing focus rejected; user asks to
  return to original lifecycle layout and explain TTFT and ITL explicitly.
- Follow-up: timing revision approved; separate project-document facts and
  User query into two clearly labeled input cards, preserving the timeline.
- Objective: request sent → first output arrival (TTFT), then consecutive
  output arrivals (individual ITLs), using readable word cards on one timeline.
- Inputs: ANIM-PD-001 lifecycle, canonical style and authored project brief.
- Allowed outputs: `visuals/animations/projects/two-kinds-of-waiting/`, scoped
  production records. Existing article and its seven clips hash-protected.
- Precision: each word card stands in for one token arrival; no literal
  tokenizer segmentation or measured times. TTFT includes possible serving and
  transport overhead in addition to prefill; all timings are schematic.
- Acceptance: inspect preview/contact sheet and full/half poster; event order,
  immutable protected files, dimensions/codec/GIF diversity and hash checks.
- Deliverables: standalone 1080p MP4, 960px GIF, editable scene/example/exporter,
  local player, metadata and limitations. No X upload, commit or push.
- Return: revised standalone arrival-timeline animation; source and verification
  results live in the project README and metadata. Initial 25.13-second version
  superseded, not accepted. Article and original animation files hash-protected.

### Task packet: `ANIM-PD-001` — seven Prefill and Decode article loops

- Approval: learner explicitly starts animation production after static review,
  2026-09-14. This supersedes the earlier defer-all-animation entries for this
  package only. Verified Mac `mac.lan`, arm64; dirty `main` at `ca2cfe1`, preserved
  existing work and skipped pull. No Spark jobs, commit, push or X action.
- Objective: animate the seven reviewed mechanisms with continuous object
  identity, minimal natural-kerning labels and white canvas; static cover retained.
- Inputs: `X-PD-001` manuscript, seven stills and `visual-plan.md`, Chapter 4
  causal invariance, article numerical fixture and canonical Manim style.
- Allowed files: new `visuals/animations/projects/prefill-decode-article/`,
  `publications/x-articles/x-prefill-decode-001/`, scoped production records.
  Existing KV/inference article and earlier animation projects remain unchanged.
- Outputs: seven 1080p H.264 MP4s, seven 960×540 looping GIFs, posters/contact
  sheets, editable scenes/exporter, per-clip event traces and hash metadata;
  article integration with static fallback, pause control and reduced motion.
- Precision: input-derived K/V precede independent attention outputs; per-layer
  cache; selecting y1 leaves N entries, feeding y1 produces N+1; generation
  loops retain old entries; workload lines and clock lengths are schematic;
  A/B request state stays separate; layerwise transfer leaves a visible tail.
- Acceptance: numerical cached/full, chunked/full and reverse-row equivalence;
  event-order, immutable old-cache geometry, bounds, codec/dimensions, GIF loop,
  frame diversity/size, output/source hashes and all seven figure slots. Inspect
  preview contact sheets then full/half-size stills before delivery.
- Return status: seven loops rendered and integrated for local review, 2026-09-14.
  GIFs are 960×540, 15 fps, individually below 5 MB; MP4s are 1080p H.264 at
  30 fps, retained locally and Git-ignored. Previews and final rendered frames
  inspected; event/media/hash checks pass. Static-toggle and reduced-motion
  controls have DOM-stub tests. See project `metadata.json` and `QA.md`.
  No claim of browser HTML visual QA under the existing local-URL policy
  restriction. User retains publication control.

### Task packet: `X-PD-001` — Prefill and Decode

- X return, 2026-09-14: user-designated draft `2099600211739803648` populated,
  not published. Post-reload audit: 55 body blocks, 6 headings, 7 animations with
  matching captions, 1 cover, zero placeholders, exact text/slots. First clip
  playback sampled successfully. See package `X_DRAFT_TRANSFER.md`; earlier
  local-only entries below predate this authorized transfer.
- Cover return, 2026-09-14: user requested the KV Cache cover style. New original
  built-in-generated black/white stippled cover selected at exact 2000×800;
  prompt/source retained in the package's `cover-concepts/prefill-decode-v2.md`.
  Original white cover, article body and all seven animations preserved.
- Latest production return: learner approved all seven loops as ANIM-PD-001;
  rendered GIFs now replace seven inline stills, with static switches and locally
  retained MP4s. Cover remains static. This supersedes the earlier deferrals
  below; no X transfer/publication or Git integration was performed.
- Earlier editorial direction: accurate but approachable, short example-first
  paragraphs like the earlier KV article. Revised all six sections around a
  meeting/report request and long-speech contrast, plus concrete scheduling A/B.
  Updated two stills; preserved parallel dependencies and representation term.
  At that stage, animation was deferred, with local source/transfer updates only.
- Latest clarification: revised prefill's parallel-position explanation and
  added a dependency still. Position 2 reads input-derived K1/V1, not O1;
  layer-to-layer dependency remains. Seven-image plan; all animation deferred.
- Latest scope, 2026-09-14: learner requested Decode illustration, then clarified
  static images now and all animations together later. Added a feedback-loop
  still in section 3 and refreshed six-image local transfer plan. No animation
  scene or render was started. Representation terminology edits are retained.
- Approval: learner selected a standalone mechanism-first article and requested
  production, 2026-09-13; return 2026-09-14. Local `main` base `ca2cfe1` was
  already dirty with the inference overview; preserved it and skipped pull.
- Objective: explain known-position prefill, first-token selection, cached
  autoregressive continuation and workload-dependent resource behavior. Separate
  scheduling from worker placement. Spark/Mac is a short supporting example.
- Inputs: Chapters 3–5; primary references in the new package `source-map.md`.
- Scope: `publications/x-articles/x-prefill-decode-001/`, article index, focused
  Day 4 record and production trackers. No existing article edits, cloud transfer,
  benchmark, video rendering, installation, Git commit or push. Day 8 unchanged.
- Precision: y1 selection leaves prompt KV length unchanged; processing y1
  appends its per-layer KV. Causal visibility differs from parallel execution.
  Compute/bandwidth tendencies depend on workload. Compatible state must cross
  workers; communication has a tail. EXO case is third-party, not local evidence
  or a general turnkey support claim.
- Return: six-section Chinese manuscript, five original 1600×900 static figures,
  single-line 2000×800 cover, local HTML, editable renderer/checker, source map,
  motion plan and verified five-image X preparation package. Numerical toy checks
  pass for cached/full equality, chunk offset and a broken noncausal control.
  Figure contact sheet and detailed first-token/handoff/shapes views inspected.
  Prior browser URL-policy block respected; browser/mobile HTML QA not claimed.
- Next: learner reviews the animated article and requests refinements. Motion
  extends `CAND-ANIM-009` under completed production packet `ANIM-PD-001`.

### Task packet: `X-INFER-001` — inference beyond token generation

- Approval: user, 2026-09-13; expanded from a request-lifecycle primer to a
  systems overview, then explicitly requested production. Mac `mac.lan` ARM64;
  clean `main` fast-forward sync at base `ca2cfe1`.
- Objective: connect latency, memory and scheduling through one project-document
  example. Generation is brief; cover TTFT/ITL/E2E/throughput, continuous batching,
  paged KV, prefix reuse, chunked prefill, P/D separation, speculation, quantization
  and model sharding versus replicas.
- Inputs: Chapters 3–5, existing KV article, current primary docs and original
  speculation/PagedAttention papers mapped in package `source-map.md`.
- Scope: `publications/x-articles/x-inference-001/`, article index, focused
  conceptual record and production trackers. Do not modify the existing KV
  article/cloud draft. No GPU lab, deployment, X upload, publication or Git write.
- Book placement: bounded public extension of Chapter 4 §4.11 and Chapter 5
  §5.18; full serving curriculum remains deferred beyond v0.1. Day 8 unchanged.
- Precision: client and server clocks differ; chunking preserves context; P/D
  separation is not a blanket throughput improvement; exact speculation requires
  correct rejection/correction; lower bits and more GPUs do not ensure speedup.
  Distinguish weights, KV, buffers, payload arithmetic and measured allocation.
- Deliverables/return: Chinese draft with 8 sections, 68 clean body blocks,
  7 original static PNGs, 2000×800 concept cover, local HTML, source map, editable
  build/check scripts and skill-generated seven-image transfer plan. Structural,
  hash, image and payload checks pass. Contact sheet and three detailed figures
  inspected. Browser file URL blocked by security policy; no workaround attempted,
  desktop/mobile HTML visual QA not claimed.
- Status: local draft ready for learner review. Proposed motion is recorded in
  `visual-plan.md`, not rendered. Next: review wording/figure pacing and select
  motion scope. Detailed animation production requires its own approved packet.


2026-09-11 Git checkpoint: learner approved committing the accumulated local
looped-Transformer and DeepSeek animation packages and synchronizing the current
branch with `origin/main`. This supersedes the earlier uncommitted-status notes
below; those notes remain production history. MP4s remain ignored. DeepSeek QA
passes; older looped-Transformer render manifests retain their historical shared
style hashes (now different from the current style files), with matching media.
No public upload or push is part of this checkpoint.

### Approved production: ANIM-CED-001 and ANIM-KV-002

- Approval: 2026-09-10, user explicitly requested implementation and rendering
  of both `projects/deepseek-ced/DESIGN.md` storyboards.
- Base: branch `codex/visuals/looped-transformer`; HEAD recorded in the render
  manifest. Existing dirty work is preserved; no Git integration authorized.
- Owner: verified Darwin arm64 local Mac; no Spark or model-scale job.
- Inputs: approved style/primitives, pinned DeepSeek report/model/kernel/config,
  and `learning_artifacts/day-06-modern-architecture/deepseek-v41-causal-encoder-decoder.md`.
- Allowed changes: new `visuals/animations/projects/deepseek-ced/` project and
  scoped index/proposal/progress entries. Do not change other rendered projects.
- ANIM-CED-001: distinguish layer-local global-KV source, CED encoder-derived
  global KV, then CSA2 shared storage; retain distinct Q/local KV at each layer.
- ANIM-KV-002: contrast separate K/V with directly produced shared KV; show the
  same rows used for scores and weighted accumulation, then distinct head reads.
- Precision: independently implemented tiny causal NumPy fixture; exact toy
  values, no DeepSeek activation/quality/latency claim. Shared KV is not fused
  independent K and V, nor weight sharing. Omitted sink/RoPE/quantization and
  fixed selected pool are explicitly documented; bounded replay is out of scope.
- Predeclared checks: shared global storage identity and unchanged reads;
  distinct layer states; causal prefix invariance; normalized attention weights;
  shared-buffer arithmetic equals equal-valued separate copies; changing V
  breaks equivalence; opposite queries favor opposite fixed rows.
- Expected return: reproducible source/trace, two English 1080p H.264 videos,
  GIFs/stills/contact sheets, series and standalone players, manifests and
  numerical/geometry/browser QA. MP4s remain ignored. Rendering approval does
  not authorize publication, commit, push, or course mastery completion.

- Return, 2026-09-10: both English films are implemented in
  `visuals/animations/projects/deepseek-ced/`; approximately 50.3s and 56.3s.
  Nine predeclared NumPy checks passed. Local 1080p30 H.264 MP4s, GIFs,
  full/half stills, checkpoint sheets, timelines, source/media manifests and
  series/individual players are present. Decimal cells were widened and moving
  contributions use unlabeled transient copies to avoid numeric overlap.
  See `metadata.json` and `qa.json` for exact render and browser evidence.
  Other animation assets were hash-protected. Ready for learner review;
  uncommitted, unpublished, MP4s ignored, no Spark run.

- Review refinement, 2026-09-10: learner requested the full architecture term
  and fewer words. Title now reads “Causal Encoder–Decoder”; both films remove
  sentence-style bottom captions and use short operation/tensor labels. The
  head-count explanation moves to expandable page notes. Re-rendered in place;
  current durations supersede the initial return and live in metadata/QA.
  This preference is also recorded in the animation style guide.

- Typography refinement, 2026-09-10: learner reported uneven character gaps.
  The tiny Pango SVG layout rounded glyph advances. These two films now opt
  into 16x whole-string shaping followed by uniform vector downscaling; Arial,
  scene wording and motion remain unchanged. `check_typography.py` verifies
  six representative strings against a higher-resolution layout and provides
  a visual comparison. Original Figure 3 posters are preserved. Other films
  retain their previous layout; the reusable helper is opt-in.

2026-09-10 animation design request: two English CED films are storyboarded in
[`visuals/animations/projects/deepseek-ced/DESIGN.md`](visuals/animations/projects/deepseek-ced/DESIGN.md).
Scope: encoder-derived decoder global KV, followed by one shared KV representation
serving score and value roles. Reuses CAND-ANIM-009/017. Distinguish CED from CSA2
sharing and avoid a false K/V-array fusion. Subsequently approved and rendered as
ANIM-CED-001/ANIM-KV-002 above. Next: learner reviews the two videos; no Spark run.

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

### Task packet: `ANIM-KV-006` — one decode step, replacing article code

- Approval: user, 2026-09-12, requested animation instead of the dry code section.
- Base: `d1374d1cbef9a149d19d625ebe61b59399bf7db5`, branch
  `codex/content/x-attn-kv-guide`; dirty article work preserved, no pull attempted.
- Mac-only scope: new `projects/kv-decode-step/`, bilingual article and trackers.
  Inputs: Chapter 4 §4.10, existing `attend_new` code and approved visual style.
- Mechanism: only new hidden state projected; new K/V appended before Q reads
  past/self; scaled matching → softmax → weighted V sum. Old cache stays fixed;
  Q/weights/output are transient, cache survives for subsequent calls.
- Outputs: GIF, local 1080p MP4, source/render script, timeline and media checks,
  PNG/contact sheets. Replace code with new animation slot 4, renumber later
  slots; shorten both languages. Preserve executable example separately.
- Acceptance: event order, old-cache geometry, self inclusion and a transparent
  numerical fixture; existing loops unchanged; seven inline assets/five GIFs,
  no code block or links, bilingual playback/mobile/reduced-motion verification.
- Limits: one head/layer, toy vectors rather than measured model activations;
  omitted positional operations, padding, multi-head/output projection and model
  remainder. No model logits or training/latency claim. No X transfer or Git write.
- Return: rendered and locally verified, 2026-09-12. GIF: 960×540, 8.60 seconds,
  744,617 bytes; MP4: 1920×1080 H.264, 8.63 seconds, Git-ignored. Preview contact
  sheet and final still/mobile frame inspected. Timeline, toy arithmetic, cache
  geometry and unchanged-earlier-project checks pass. Code is preserved in
  `kv-decode-step/example.py`; six-step cached/full attention equivalence passes.
  Both articles are shorter and code-free; seven inline assets include five GIFs.
  Playback, pause, reduced-motion, mobile and ordered-image checks pass locally.
  Awaiting learner review. No X transfer, publication, commit or push.

### Task packet: `ANIM-KV-005` — cache memory growth

- Approval: user, 2026-09-12, after reviewing the proposed two comparisons.
- Base: `d1374d1cbef9a149d19d625ebe61b59399bf7db5` on
  `codex/content/x-attn-kv-guide`; preserve existing uncommitted work.
- Owner: Mac Studio. Inputs: Chapter 5 cache accounting and the article's
  calculated memory chart; shared animation style and existing export pipeline.
- Mechanism: separate K/V bytes = 2 × B × L × T × H_KV × d × b.
  Fixed B=1, L=24, d=64, b=2. T: 4096→8192 at H_KV=8 gives 192→384 MiB;
  separate configuration comparison at T=4096, H_KV: 8→4 gives 192→96 MiB.
- Allowed files: new `visuals/animations/projects/kv-memory-growth/`, bilingual
  article package and production trackers. No changes to existing rendered loops.
- Outputs: editable scene, reproducible render script, 1080p MP4, 960×540 GIF,
  PNG fallback/contact sheets, timeline/checks/hashes, article slot 5 replacement.
- Precision: grouped position glyphs, separate K/V; tensor payload only, not
  measured allocation. Fewer heads is a configuration comparison, not runtime
  deletion of heads. Reset between comparisons is editorial, not eviction.
- Acceptance: exact arithmetic and bar/grid ratios; inspect motion and half-size
  still; verify all four GIFs, mobile layout, reduced motion, image-slot order;
  protect existing media hashes. No X upload, publication, commit or push.
- Return: rendered and locally verified, 2026-09-12. GIF: 960×540, 145 frames,
  9.67 seconds, 1,468,490 bytes; MP4: 1920×1080 H.264, 9.63 seconds, Git-ignored.
  Preview motion and final half-size still inspected; arithmetic, ratio,
  geometry, bounds and media checks pass. All earlier loops remain unchanged.
  Both article reviews now contain four GIFs and two static equation cards;
  playback, pause/resume, reduced motion, mobile layout and upload-order QA pass.
  No X transfer, publication, commit or push. Awaiting learner review.

### Task packets: `ANIM-KV-003` / `ANIM-KV-004` — remaining article loops

- Approval: user, 2026-09-12, following approval of the prefill/decode GIF.
- Base/branch: `d1374d1cbef9a149d19d625ebe61b59399bf7db5`,
  `codex/content/x-attn-kv-guide`; preserve all existing uncommitted work.
- Owner: Mac Studio. No Spark model run, public transfer, commit or push.
- Inputs: Chapter 4 §4.10 and `why-cache-keys-and-values.md`; Chapter 5 and
  `day-06-modern-architecture/deepseek-v41-causal-encoder-decoder.md`; original
  article diagrams 2/5, existing CED sources, and shared animation style.
- `ANIM-KV-003`: juxtapose append/reuse with edit/recompute. Fixed model settings;
  preserve earlier cached objects exactly; editing position 2 conservatively
  invalidates the dependent suffix, without claiming every numeric value changes.
- `ANIM-KV-004`: encoder output generates one decoder global KV bank; several
  decoder layers reuse it with distinct Q/local KV. One selected global vector
  participates in Q matching and weighted accumulation. Keep these three
  mechanisms distinct; no claim that all caches are shared or all entries read.
- Allowed files: new `visuals/animations/projects/kv-article-loops/`, article
  package, animation index/proposals, `PROGRESS.md` and this production queue.
- Outputs: two editable scenes; preview contact sheets; 1080p MP4, 960×540 GIF,
  PNG fallback per loop; timestamps, versions, hashes, event/geometry checks;
  replacement of article slots 3 and 6 in both language versions.
- Acceptance: inspect previews and final still/motion frames; explicit event and
  object-identity assertions; GIF looping/size/frame-diversity checks; verify all
  three article GIF controls and reduced-motion fallbacks plus ordered upload
  plan. Keep the first approved loop byte-identical and all equations static.
- Return (2026-09-12): both loops rendered and integrated into both languages.
  Append/edit: 7.67 seconds, 1,074,254-byte GIF; global sharing: 7.87 seconds,
  1,039,763-byte GIF. Both GIFs are 960×540, with 1080p MP4 and PNG fallback.
  Event/geometry/media checks pass; the entire original prefill/decode project
  remains byte-identical. Bilingual browser QA passes for all three GIFs:
  playback, pause/resume, reduced motion, mobile layout and six upload slots.
  Schematics only; no invented model outputs or benchmarks. Awaiting learner
  review; no X transfer, publication, commit or push. IDs preserve the existing
  `ANIM-KV-002` DeepSeek shared-KV production packet.

### Task packet: `ANIM-KV-001` — article prefill/decode loop

- Approval: user, 2026-09-12; first loop only, derived from `CAND-ANIM-009`.
- Base: `d1374d1cbef9a149d19d625ebe61b59399bf7db5`, branch
  `codex/content/x-attn-kv-guide`; existing article edits are preserved.
- Host: local Mac Studio (verified Darwin arm64). No Spark or model job needed.
- Objective: prefill stores prompt K/V; a selected token is fed back, its K/V
  append before self-attention, and its new query reuses unchanged past K/V.
- Inputs: Chapter 4 §4.10, `why-cache-keys-and-values.md`, current bilingual
  `X-ATTN-KV-001`, and `visuals/animations/STYLE_GUIDE.md` / `manim_style.py`.
- Allowed scope: `visuals/animations/projects/kv-prefill-decode/`, animation
  proposal/index, article package, `PROGRESS.md`, and this task packet.
- Outputs: editable Manim scene/render script, timeline, full contact sheet,
  1920×1080 MP4, 960×540 looping GIF, still PNG, verified media hashes, and
  GIF-enabled bilingual article reviews with PNG/reduced-motion fallback.
- Precision: illustrative token pieces and tensor glyphs; one layer shown;
  remaining model/prediction path explicitly labeled. No invented attention
  weights, trained activations or performance claims. Prompt prefill is parallel;
  final prompt query illustrated, other prompt queries omitted. Cache never
  includes a selected token before that token is processed. Loop reset is editing,
  not cache eviction. English natural kerning; no production labels in animation.
- Acceptance: inspect key frames before final export; unchanged cached geometry;
  event-order assertions; complete-sequence visual QA at full/half size; GIF loop,
  dimensions/duration/hash and MP4 codec checks; article rendering and image-slot
  tests. No X editor transfer, publishing, commit or push authorized.
- Return evidence: rendered on Mac; 8.93-second 1920×1080 H.264 MP4 and
  8.94-second 960×540 GIF (1,154,160 bytes, infinite loop, 134 frames).
  Full contact sheet and half-size still visually reviewed. Event/cache-order
  and unchanged-past-geometry assertions pass. Both local article reviews use
  the GIF in slot 1; browser checks pass for visible motion, pause/resume,
  reduced-motion fallback and six correctly ordered media slots. Source/render
  hashes and limitations are in the project manifest. MP4 remains ignored.
- Status: rendered and locally verified, awaiting learner review. No X transfer,
  publication, commit or push; only this first loop was produced.

### Approved animation: `ANIM-KV-007` — one document, three requests

- Approval: user, 2026-09-13, following the A/B/C social-post storyboard.
- State: rendered, awaiting learner review on verified ARM64 Mac `mac.lan`; branch
  `codex/content/x-attn-kv-guide`, base `d1374d1`. Existing dirty work preserved.
- Objective: distinguish prefill computation, retained KV states and exact-prefix
  reuse; requesting an edit does not edit the input document.
- Inputs: Chapter 4 §4.10, `why-cache-keys-and-values.md`, approved social copy,
  `STYLE_GUIDE.md`, shared `manim_style.py`, existing append/edit scene grammar.
- Allowed outputs: `visuals/animations/projects/kv-prefix-abc/`, animation indexes,
  this packet, focused learning note and progress return. Existing article and
  cloud draft are out of edit scope. No model experiment or Spark job.
- Story: A computes original document and instruction KV; B retains document KV
  while computing a different instruction; C actually edits a document segment,
  retains only earlier shared-prefix KV and recomputes the related suffix.
- Precision: use illustrative token groups, not measured tokenization or model
  activations. Same execution settings and cache availability; document suffix
  recomputation is conservative, not a claim every tensor element changes.
  Show instruction computation using cached prefix; generation is out of scope.
- Deliverables: ~20-second 1080p H.264 MP4, 960×540 looping GIF, full/half still,
  complete contact sheet, local player, editable source/command and hash metadata.
- Acceptance: B's document and KV geometry/content invariant; C changes input
  before invalidating the related suffix; same textual tail is recomputed;
  all current instructions get new states. Natural Arial/Songti, white canvas,
  semantic colors, no production labels, no overlap/clipping; media checks pass.
- Return: 18.97-second 1920×1080 H.264 MP4 (1,727,240 bytes), 960×540
  infinite-loop GIF (2,137,794 bytes), full/half stills, contact sheet and local
  player in `visuals/animations/projects/kv-prefix-abc/`. Fixture, event-order,
  invariant geometry and media checks pass. Full sequence and full/half stills
  visually inspected; browser playback is not claimed. Source/output hashes and
  limitations are in `metadata.json`, visual QA in `qa.json`. MP4 stays ignored.
- Next: learner reviews the animation. Article and X draft unchanged; no
  publication, commit, push, GPU experiment or change to Day 8 learning status.

### Task packet: `X-ATTN-KV-001`

Latest cloud return, 2026-09-12: user manually cleared X draft
`2098863329817051136`; verified empty and uploaded the latest Chinese body.
Preserved its existing title and cover (cloud title differs from local).
Post-reload source equality, 82 text blocks, seven exact media slots, zero
placeholders and all five GIF playback checks pass. X displays 523 words.
Evidence: article package `zh/x-transfer-2026-09-12.md`. Draft only, not published;
no commit/push. This supersedes the earlier local-only status below.

Chinese editorial return, 2026-09-12: learner approved a project-document story,
question-led progression, concrete examples and shorter paragraphs. Local `zh/`
manuscript, HTML review and seven-slot transfer plan are revised; English and
media remain unchanged. Prior Chinese source is retained under `zh/revisions/`.
Reuse assumptions and Bounded Replay uncertainty stay adjacent to the claims.
Follow-up editorial cut: removed the Chinese closing advanced-details and
companion-video sections; added brief answers to the four closing questions.
Seven inline visuals and all source video files remain available.
Further approved clarification distinguishes prefill, KV cache and prefix
caching in the shared-document section, using vLLM's official repeated-document
workload (source rechecked 2026-09-12, mapped separately). New-question prefill
and answer decode remain necessary; no benchmark or new animation is implied.
Earlier Chinese wording was successfully saved to X draft `2098863329817051136`
with reload/slot/playback verification. This revised local wording is NOT synced
to X. Static QA is recorded in `zh/editorial-qa.json`; fresh browser preview was
blocked by URL policy. Next: learner reviews local wording/layout. No publish,
commit, push, new render or change to Day 8 learning status.

Cover concept return, 2026-09-12: user requested a 5:2 editorial cover inspired
by a supplied black-and-white stippled reference, with “KV Cache” as the sole text.
Built-in image generation produced a new layered-information/reuse metaphor;
exact 2000×800 export is `assets/cover-kv-cache-v2.png` in the article package.
Both local previews select this version; previous `assets/cover.png` is preserved.
Original generation and exact prompt are in `cover-concepts/`. Browser/dimension
checks pass. This cover experiment does not change the white animation style,
imply a factual architecture diagram, or establish an engagement improvement.
Awaiting review; no X transfer, publication, commit or push.

**Production return — 2026-09-11:** user approved drafting the refined outline.
Local English manuscript (~1,850 words after the self-contained revision) and Chinese adaptation with retained
English technical terms are prepared in `publications/x-articles/x-attn-kv-001/`.
The package contains a shared 2000×800 cover, five original inline diagrams,
bilingual visual HTML reviews, clean body-only transfer artifacts, image plans,
claim/source map, build script and browser QA. Existing DeepSeek videos are
optional local companions, not assumed X embeds. Exact cache reuse anchors the
article; Causal Encoder–Decoder, cross-layer sharing and shared representation
form the closing case study, with Bounded Replay explicitly approximate.
Recorded toy outputs are attributed, not rerun; payloads and projection counts
are recalculated. Chinese text avoids the prohibited contrast construction.
At the user's request, both article bodies now contain no links. Five inline
figures, a tested NumPy cache update, explicit projection counts and a paired
request example explain the mechanism directly. Source URLs remain in the
separate provenance map. The new one-head example is independently checked
against full causal attention; it does not reproduce the recorded toy logits.
Math-format revision (2026-09-12): three typeset equation cards now accompany the
five diagrams in both languages. Their formula sources, render version and eight
ordered upload slots are retained; browser QA covers desktop/mobile equations.
Inline tensor shapes remain code notation. Article bodies remain link-free.
Later editorial cut (2026-09-12): user removed the small-experiment/stale-cache
section. Both versions now have six inline images (four diagrams and two equations).
The original experiment evidence/assets remain in the repository, outside the article.
No X editor interaction, publication, commit or push. Next: learner reviews
the English argument, Chinese wording and figures before transfer preparation.

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

**Dependency update — 2026-09-05:** Chapter 4 and its solutions are complete.
`experiments/reports/2026-09-05-attention-gradients-cache.md` verifies the toy
cache example. Use the first-layer versus deeper-layer distinction explicitly;
no speedup benchmark is implied. The packet is ready for Mac drafting.

### Task packet: `X-LOOP-001`

Related production update, 2026-09-09: the learner approved the fixed-recurrence
animation recommended alongside this article. See `ANIM-LOOP-001` below. This
does not start article drafting or the trained Day 7 comparison.

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

### Task packet: `ANIM-LOOP-001`

English-series expansion approved on 2026-09-09 after comparison with Sebastian
Raschka's article: create three separate videos under this packet—one-block
introduction, three distinct blocks repeated twice, and an unfolded six-application
view paired by shared weights. Keep the Chinese original. Sources/outputs live
in `projects/looped-transformer/english/`; preserve the same style and checks.
Predeclare the new CPU check: three distinct parameter sets total 6,480 values;
two stack passes execute six blocks; six equal-valued independent copies use
12,960 values and agree forward at initialization; each shared gradient equals
the sum of its two application-copy gradients. No adaptive router or trained
quality comparison is included. English learner-facing pages omit production labels.

- Approval: 2026-09-09; user accepted the proposed fixed-recurrence animation.
- Objective: preserve one block and one parameter set while successive hidden
  states pass through it three times. Separate stored parameters, block
  applications, and analytical matrix-operation work.
- Source: Chapter 5 section 5.20, `CAND-ANIM-011`, Day 7 recurrence notebook,
  and the canonical `src/dongxi_llms/decoder_lab.py` implementation.
- Owner: verified local Mac Studio; branch `codex/visuals/looped-transformer`;
  starting revision is recorded in the project README and render manifest.
- Allowed files: `visuals/animations/projects/looped-transformer/`, shared style
  instructions, animation index/proposal queue and this production ledger.
- Storyboard: input state → one causal block → changed state → return through
  the same block twice → final state. Parameter count stays fixed, application
  count and modeled matrix arithmetic increase. Heatmap colors show signed
  actual activations using a single fixed scale, not reasoning quality.
- Precision: use the existing CPU float64 block and notebook seed. Count block
  parameters only; arithmetic excludes embedding/head, normalization, activation,
  and softmax. No latency, memory, quality improvement, adaptive routing, or
  vendor-architecture claims. No training or GPU jobs.
- Predictions before execution: all three uses share parameter identities;
  stored values stay unchanged; states differ; equal-valued independent copies
  agree forward; shared gradients equal the sum of copy gradients; analytical
  matrix-operation count grows linearly for fixed shapes.
- Outputs: editable Excalidraw geometry and Manim scene, exact trace, local
  1080p MP4, GIF, stills, HTML player, reproducible scripts and hashes.
- Acceptance: numerical identities, fixed heatmap scale, source/asset integrity,
  full/half still and transition inspection, real browser playback. White canvas,
  Arial/Songti, minimal concept text; production labels stay in internal files.
- Return, 2026-09-09: `visuals/animations/projects/looped-transformer/review.html`
  contains a 41.33s 1920×1080/30fps H.264 animation. Editable Excalidraw assets,
  scene, trace, GIF, full/half stills, contact sheet, and reproduction/QA manifests
  are retained alongside it. MP4 is local and ignored.
- Measured evidence: 2,160 block parameters through all three applications;
  53,760 dense matrix FLOPs per application (whole batch), 161,280 for three;
  equal-valued independent-copy forward discrepancy 0; gradient-sum discrepancy
  1.11e-16. All eight numerical checks pass. Counts exclude non-matmul work and
  embedding/head; there is no trained quality or runtime claim.
- Acceptance evidence: six deterministically reproduced sketch files, matching
  source/output hashes, eight scene checkpoints, full/half stills and nine
  transition samples inspected, real Chrome playback/seek and mobile overflow
  checks passed. Final media/page contain no production credits or task labels.
- Exact next action: learner watches the local animation and discusses refinements.
  Source is uncommitted on `codex/visuals/looped-transformer` from
  `8e39e98cd45e024416041aed3015495f02d6a240`; article and adaptive routing remain queued.

English-series return: `projects/looped-transformer/english/review.html` under
`visuals/animations/` selects three videos: single-block recurrence (~41s), a
three-block stack repeated twice (~36s), and unfolded shared applications (~29s).
Each has an individual player, GIF, stills, source and manifest. Nine additional
CPU checks verify three distinct blocks / six applications: 6,480 shared versus
12,960 independent-copy parameters, identical initial forward outputs, and
paired-gradient discrepancy 2.78e-17. Original Chinese artifacts are hash-checked
as unchanged. Player selection, playback/seek, individual pages, English-only
content, responsive layout and manifest checks passed. Review the series before
any further expansion; no article drafting, adaptive routing or publication.

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

**Forward storyboard refinement — 2026-09-06:** The learner requested a Q/K/V
workflow after distinguishing values from attention scores. Show X splitting
into three learned projections, Q/K joining to produce scaled masked scores,
row-wise softmax producing A, and the separate V branch meeting A at O=AV.
Keep labels for scores, weights, and content visible at their respective stages.
Include a controlled contrast with fixed Q/K and changed V: weights unchanged,
output changed. A complementary fixed-V routing change can follow. Use verified
numbers for any new displayed examples. This extends the existing approved
concept; production remains on the Mac Studio.

**Canonical backward spine:**

Review refinement (2026-09-06): for one receiver show the scalar identity
`dL/ds_j = a_j * g dot (v_j - o)`, where s is already scaled, g is the
downstream output gradient, and o is the current value mixture. Contrast a
useful change of message with identical values that make routing ineffective.
The identity was checked against the existing float64 fixture; verify displayed
numbers during Mac production. This extends `ANIM-ATTN-001`'s routing branch.

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
`book/chapters/04-attention-and-the-causal-information-boundary.md`, its worked
solutions, all three Day 4 notebooks, reusable source and tests, and both
`experiments/reports/2026-09-05-*` attention reports.

**Evidence dependency:** Do not freeze displayed numerical values or begin final
rendering until the manual forward pass agrees with a trusted PyTorch reference,
analytical gradients agree with autograd and finite differences under declared
tolerances, forbidden future edges are verified zero, and detach experiments
confirm the two branches. Chapter 4 must be synthesized before final editorial
review.

**Dependency update — 2026-09-05:** All preceding evidence requirements are
satisfied by the complete chapter and the two attention reports. Mac production
can use the verified fixtures; preserve their dtype, shapes, and evidence
boundaries when choosing displayed values.

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
