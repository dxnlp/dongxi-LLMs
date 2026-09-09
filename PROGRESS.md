# Progress and Handoff Ledger

This file is the operational source of truth for resuming work. Update it at the end of every learning day and whenever responsibility passes to another person or agent.

## Current position

- Active release: `v0.1`
- Active day: Day 8 pretraining data and recipe; complete requested materials prepared ahead of guided study
- Status: in progress
- Current focus: Chapter 6's complete Day 8 foundation, three visual worked
  notebooks, twelve conceptual solutions, and a bounded pretraining specification.
  Chapter 5 still covers Days 5–7; its learner review backlog is unchanged.
- Next operational action: remain on verified Spark execution at the learner's
  request after Mac dependency friction. No Mac installation is required for
  browser-side interactive teaching. See `docs/handoffs/CURRENT.md`.
- Next learning action: **Day 8 on Spark**, starting with how actual documents
  become supervised token windows and a correctly weighted update (Notebook 1).
  Discuss a concrete data/recipe dilemma using the prepared figures and code;
  notebooks remain optional companions to live discussion. The learner found
  the isolated quadratic SGD slider boring; do not default back to it.
  Earlier Day 6 resume instructions were an agent handoff mistake, not learner
  intent. Preserve unfinished review without forcing a return to Day 6.
  Material was requested ahead of study: do not infer exercises or mastery.
  Day 5 mastery and Chapter 4 deferred practice remain open. Chapter 6 Day 8
  verification: 3 notebook passes, 29 code cells, 10 figures, 71 repository tests;
  report: `experiments/reports/2026-09-09-day8-material-verification.md`.
  Historical Chapter 5 visual suite: 12 notebook passes, 168 code cells, 52 figures;
  report:
  `experiments/reports/2026-09-07-day7-architecture-defense.md`. Day 6 prose
  checks: `experiments/reports/2026-09-07-day6-chapter-verification.md`.
  Day 7 now has a core defense notebook plus optional recurrence. The learner's
  defense and any specified trained comparison remain pending.
- Last updated: 2026-09-09

## Four-week tracker

Status values: `pending`, `in progress`, `complete`, `blocked`.

| Day | Topic | Status | Primary evidence |
|---:|---|---|---|
| 1 | Laboratory and reproducibility | complete | Chapter: `book/chapters/01-evidence-before-optimization.md`; evidence: `experiments/reports/2026-08-29-qwen3-0.6b-sft-smoke.md` |
| 2 | Tokenization and embeddings | complete | Chapter: `book/chapters/02-text-tokens-and-embeddings.md`; reports: `experiments/reports/2026-08-30-qwen3-multilingual-tokenization.md`, `experiments/reports/2026-08-31-tokenizer-mechanics.md`, `experiments/reports/2026-08-31-embedding-gradient-paths.md`, `experiments/reports/2026-08-31-qwen3-embedding-inspection.md` |
| 3 | Probabilities and next-token loss | complete | Chapter: `book/chapters/03-learning-the-next-token.md`; three interactive notebooks; report: `experiments/reports/2026-09-03-next-token-distribution.md` |
| 4 | Attention from first principles | in progress | Complete Chapter 4, 12 solutions, three notebooks, two reports; hands-on practice deferred, mastery not fully assessed |
| 5 | Decoder-only Transformer | in progress | Day 5 chapter foundation and 12 worked answers; expanded Chapter 5 suite has twelve notebooks, 52 figures, 53 tests; learner mastery not yet assessed |
| 6 | Modern architecture design | in progress | Chapter 5 sections 5.11–5.22, worked answers 13–24, three ready notebooks; guided study pending |
| 7 | Architecture synthesis | in progress | Chapter 5 sections 5.23–5.27 and answers 25–30; core defense notebook and optional recurrence ready; guided defense and trained comparison pending |
| 8 | Pretraining data and recipe | in progress | Complete Day 8 Chapter 6 foundation, three visual worked notebooks, twelve solutions and bounded specification prepared; learner study pending on Spark |
| 9 | Pretraining run and diagnosis | pending | — |
| 10 | Evaluation before training | pending | — |
| 11 | Instruction-data engineering | pending | — |
| 12 | SFT mechanics | pending | — |
| 13 | Qwen3-0.6B full SFT | pending | — |
| 14 | SFT recipe defense | pending | — |
| 15 | Preference data and Bradley–Terry | pending | — |
| 16 | Reward models | pending | — |
| 17 | DPO derivation and implementation | pending | — |
| 18 | DPO experiment | pending | — |
| 19 | Language generation as a policy | pending | — |
| 20 | Baselines, RLOO, and PPO | pending | — |
| 21 | Policy-gradient synthesis | pending | — |
| 22 | GRPO derivation | pending | — |
| 23 | Qwen3-0.6B GRPO/RLVR | pending | — |
| 24 | Reward hacking and instability | pending | — |
| 25 | Rollout systems and monitoring | pending | — |
| 26 | Distillation and inference scaling | pending | — |
| 27 | Capstone evaluation | pending | — |
| 28 | Technical defense and release | pending | — |

## Durable decisions

| ID | Decision | Reason |
|---|---|---|
| D001 | English is the canonical course language. | Precise technical communication and global accessibility. |
| D002 | Chinese is a selective social/localization layer. | A large portion of the existing audience is Chinese-speaking without requiring two canonical courses. |
| D003 | Swedish work lives in `Dongxi_LLMs_Swedish`. | It is a distinct applied research journey toward Swedish reasoning capability. |
| D004 | The first learner is a capable Python user with deep-learning foundations. | This matches Dongxi and the intended public audience. |
| D005 | SFT and RL receive the greatest depth; pretraining remains concise but empirical. | The objective is modern post-training mastery. |
| D006 | Qwen3 is the central model family. | It connects readable architecture code, 0.6B reasoning experiments, and 1.7B post-training. |
| D007 | Qwen3-0.6B is the glass-box model; Qwen3-1.7B is the flagship validation model. | Fast iteration and meaningful validation need different scales. |
| D008 | Evaluation is frozen before a training run. | Training reward or loss alone cannot establish capability improvement. |
| D009 | The 28-day deliverable is a public beta, not a claim of completed lifelong mastery. | Four weeks provides urgency while preserving technical honesty. |
| D010 | `dgx-spark-dongxi` remains the independent platform layer. | Hardware compatibility and course content should evolve independently. |
| D011 | Course material is developed book-first rather than as chronological daily notes. | Each learning day must strengthen a coherent reader-facing narrative; logs and reports provide evidence but do not replace structured chapters. |
| D012 | Maintain `LEARNING_MEMORY.md` as the cross-session and cross-machine artifact index and production ledger. | Topic artifacts, public-content ideas, and portable task packets must remain discoverable beyond a chat transcript without turning the book into daily notes. |
| D013 | Create and actively update `learning_artifacts/day-NN-topic/` during each lesson. | Deep discussions should be durable and reusable by topic and day, while `LEARNING_MEMORY.md` remains a compact index and production queue. |
| D014 | Use a two-way, approval-gated animation proposal loop during learning and course development. | Strong mechanism animations should be surfaced when motion adds explanatory value, while explicit learner approval and the four-animation `v0.1` target prevent silent scope growth. |
| D015 | Use interactive mechanism notebooks for important mathematical ideas. | The learner understands mathematics best by predicting, implementing, perturbing, and interpreting mechanisms in code while retaining deep conceptual discussion. |
| D016 | Integrate emerging architectures through time-stamped frontier modules with explicit evidence tiers. | The course should remain current without presenting papers, implementation reports, secondary reporting, and vendor rumors as equally established facts. |
| D017 | Give every chapter a multi-session interactive notebook pathway. | Important LLM mathematics and architecture should be understood by inspecting, implementing, breaking, and integrating mechanisms rather than reading prose alone. |
| D018 | Mac Studio hosts live learning and media; Spark hosts GPU-dependent work. | A lesson can use both without changing its coherent book placement. |
| D019 | Use explicit departure/arrival phrases and a Git-backed handoff. | Verify execution host, save/push scoped work before moving, sync on arrival, and preserve dirty work. |
| D020 | Add live, linked visual experiments beyond notebooks. | Retain actual calculations, saved controls, evidence limits, and deep discussion; proposed tools are not completed lessons. |

## End-of-day update template

Copy this block below the daily log heading after each session:

```markdown
### Day NN — YYYY-MM-DD

- Status: complete | partial | blocked
- Questions investigated:
- Derivations completed:
- Code or content produced:
- Experiments executed:
- Evidence and results:
- Failures or surprises:
- Claims not yet validated:
- Decisions made:
- Book-facing contribution:
- Public artifacts produced:
- Exact next action:
```

## Daily log

### Day 01 — 2026-08-29

- Status: complete
- Questions investigated: What identifies an experiment? What do smoke tests prove? How should unified memory be monitored? How do observations differ from interpretations?
- Derivations completed: decomposed training memory into fixed and shape-dependent contributors; distinguished overlapping system, cgroup, and CUDA accounting domains.
- Code or content produced: book architecture; Chapter 1; laboratory appendix; exercise solutions; course charter; mastery rubric; repository structure; experiment specification/report templates; environment-manifest collector; two environment manifests; a completed smoke-test specification; and its report.
- Experiments executed: native PyTorch GPU-stack verification; pinned open-instruct GPU-stack verification; three-step Qwen3-0.6B BF16 full-SFT smoke profile.
- Evidence and results: smoke test passed with status 0; three finite losses (`0.360384`, `0.264207`, `0.357005`); 118.84 GiB starting and 109.74 GiB minimum `MemAvailable`; 4.03 GiB cgroup peak; 6.42 GiB CUDA-visible peak; zero cgroup memory events; model saved. See `experiments/reports/2026-08-29-qwen3-0.6b-sft-smoke.md`.
- Failures or surprises: loss was non-monotonic, which did not violate the specification; PyTorch emitted an `sm_121` capability-range warning despite the validated operations passing; the compatibility SFT trainer is deprecated upstream.
- Claims not yet validated: capability improvement, long-run convergence and stability, exact cross-host reproduction, larger batch/sequence safety, and recipe optimality.
- Decisions made: use immutable revision identities and lock hashes; retain all three non-additive DGX Spark memory views; require precommitted success and failure criteria; keep raw platform outputs separate from durable course reports.
- Book-facing contribution: `book/chapters/01-evidence-before-optimization.md`, supported by `book/appendices/a-laboratory-setup.md` and `book/solutions/01-evidence-before-optimization.md`.
- Public artifacts produced: Chapter 1 draft, Appendix A draft, and Chapter 1 exercises with solutions; not yet released externally.
- Exact next action: begin Day 2 by predicting token-count differences for fixed English, Chinese, and Swedish strings before inspecting Qwen3 tokenization.

### Day 02 — 2026-08-30 to 2026-08-31

- Status: complete
- Questions investigated: How do bytes, characters, BPE pieces, IDs, and embeddings differ? How are BPE vocabularies trained and frozen? Why does multilingual token efficiency differ? Where does contextual meaning emerge? How does next-token loss train repeated and tied embedding rows? How do visibility and supervision masks differ?
- Derivations completed: byte-level fallback and merge progression; `[B,T] → [B,T,d]` lookup shapes; repeated-row gradient accumulation; tied output gradient `(p_i-1[i=y])h`; causal contextualization boundary; prompt-gradient flow under response-only loss; distinction between tokenizer entry count and model row count.
- Code or content produced: multilingual tokenizer lab, transparent embedding-gradient lab, pinned Qwen3 embedding-interface inspector, four unit tests, Chapter 2, ten exercises with worked solutions, topic-organized learning artifacts, and portable article/animation packets.
- Experiments executed: pinned Qwen3 English/Chinese/Swedish tokenizer comparison; repeated lookup and tied/untied gradient-path verification; response-only masking verification; pinned Qwen3 tokenizer/embedding boundary and runtime tying inspection.
- Evidence and results: exact tokenizer round trips with 9 Chinese, 11 English, and 20 Swedish tokens in the fixed example; repeated row gradient doubled exactly; untied nonzero input rows `[1,3]` versus all six tied rows; masked prompt rows received nonzero gradients; Qwen3 input/output matrices verified as the same `[151936,1024]` parameter, with 267 model rows beyond the tokenizer's 151,669 entries.
- Failures or surprises: the predicted token-efficiency order was reversed; GPT-SW3 remained gated despite authentication; byte-level vocabulary pieces required source-offset spans for readable reporting; serialized input/output tensor names did not imply untied runtime parameters; the model/tokenizer row mismatch was larger than expected and its rationale remains unverified.
- Claims not yet validated: general language-level tokenizer efficiency, causal attribution to the tokenizer-training corpus, semantic quality from token count, exact Qwen3 training gradient magnitudes, the design reason or performance effect of 267 extra rows, and masking behavior of every trainer API.
- Decisions made: distinguish tokenizer size `V_t` from model vocabulary dimension `V_m`; say “direct lookup-path gradient” when output tying may add other paths; treat loss masking, detaching, and freezing as separate mechanisms; keep the full logits/softmax/loss derivation in Chapter 3.
- Book-facing contribution: `book/chapters/02-text-tokens-and-embeddings.md` and `book/solutions/02-text-tokens-and-embeddings.md`, supported by three experiment reports and reusable source modules.
- Public artifacts produced: BPE and cross-entropy preview animations; `X-BPE-001` and `X-EMB-001` are ready for Mac drafting; final embedding and cross-entropy animations retain their Day 3 dependency.
- Exact next action: begin Day 3 with a mechanism-level discussion of softmax as normalized competition, then derive cross-entropy, perplexity, causal shifting, and a tiny next-token model.

### Day 02 enrichment — 2026-08-31, local Mac

- Status: complete with one retained failed criterion.
- Questions investigated: How can BPE pair counts and tie handling be made
  executable? How do grapheme clusters differ from code points and bytes? What
  do normalization, leading spaces, and chat-template packaging change before
  model computation?
- Predictions recorded before execution: three BPE merges and counts; NFC/NFD
  unit counts; one family-emoji grapheme; leading-space token-identity change;
  chat packaging larger than raw text; exact decode of both normalization forms.
- Code or content produced: `src/dongxi_llms/tiny_bpe.py`,
  `src/dongxi_llms/tokenizer_mechanics_lab.py`, six focused new unit tests, a
  specification, raw JSON output, report, new learning artifact, expanded Chapter
  2 treatment, and an eleventh exercise with solution.
- Experiment executed: CPU-only pinned Qwen3-0.6B tokenizer mechanics on the Mac;
  no model weights or DGX Spark resources used.
- Evidence and results: BPE selected `h+u`, `hu+g`, `hug+s` with counts 10, 10,
  3; NFC/NFD `café` used 4/5 code points but four graphemes each; the family emoji
  used seven code points, 25 bytes, and one grapheme; `token` and ` token` mapped
  to different IDs; raw `Hello` used one token and the chat form nine.
- Failure or surprise: NFD exact source round trip failed because decode returned
  NFC; normalized offsets also omitted the original combining mark. The first BPE
  maximum was tied and required the predeclared tie policy.
- Interpretation boundary: the trace establishes generic BPE mechanics, not
  Qwen's training history; the tokenizer examples do not generalize to other
  revisions, languages, or chat templates.
- Book-facing contribution: Chapter 2 now includes graphemes, concrete
  preprocessing, tested BPE training, tokenizer-family scope, and chat packaging.
  Chapter 1 receives only a short evidence-discipline bridge; tokenizer mechanics
  remain in Chapter 2.
- Public-content contribution: the evidence gaps for expanded `X-BPE-001` are
  closed; the canonical English X Article can now be drafted for review.
- Exact next action: retain the course transition to Day 3; the independent Mac
  content lane may draft `X-BPE-001` from the approved outline and evidence.

### Content production — `X-BPE-001` — 2026-08-31, local Mac

- Status: bilingual local packages prepared; editorial review pending.
- Content produced: an approximately 1,500-word English X Article draft,
  claim-to-evidence map, canonical bilingual NLP terminology map, separate
  cover, five inline figures, editable visual builder, body-only HTML and
  Markdown transfer files, a responsive browser-review HTML with all six media
  assets in reading order, and a generated bottom-to-top image insertion plan
  under `publications/x-articles/x-bpe-001/`.
- Chinese adaptation: `publications/x-articles/x-bpe-001/zh/` contains a natural
  Chinese rewrite with the same evidence boundaries, a localized 2000×800 cover,
  five localized inline figures, source map, responsive review HTML, body-only
  transfer files, and a validated image insertion plan. The final language scan
  found no `不是……而是……` construction or variant.
- Evidence boundary: the article preserves the pinned Qwen3 observations, the
  failed NFD exact-source round trip, the measured BPE tie, and the distinction
  among byte coverage, compression, context occupancy, and understanding.
- Visual system: white canvas, Arial normal-weight English, Songti SC Chinese,
  semantic course palette, minimal text, and geometric alignment. The approved
  Manim BPE still is reused as the coverage/compression figure.
- Terminology revision: both versions now use the standard Tokenization stages—
  normalization, pre-tokenization, tokenization model, post-processing, and Token
  ID output. Chat Template serialization is shown before tokenization. The prose
  and figure labels use NLP terms such as subword Token, byte Token, vocabulary,
  Merge Rank, input representation, and Embedding lookup instead of
  software-architecture metaphors.
- Validation: both Python builders compile; all five placeholders map to existing
  image files; the clean HTML and Markdown bodies omit the title and image
  placeholders; the generated insertion plan reports zero missing images.
- Publication boundary: nothing was transferred to X and nothing was published.
- Exact next action: review both local browser versions, with particular attention
  to title, opening, section balance, and the five localized figures; revise
  locally before any X editor transfer.

### Course production system — `ANIM-SYSTEM-001` — 2026-09-01

- Status: complete.
- Mechanism added: `visuals/animations/PROPOSALS.md` now accepts both user- and
  agent-suggested animation candidates, records their evidence and dependencies,
  and defines states from suggestion through approval, production, review, and
  completion.
- Approval boundary: agents should proactively surface strong animation
  opportunities, but no candidate becomes a production task without explicit
  learner approval. Publication remains a separate decision.
- Workflow integration: the daily workflow and roadmap now include a bounded
  animation-opportunity check; approved concepts receive complete `ANIM-*` task
  packets in `LEARNING_MEMORY.md`.
- Scope boundary: the `v0.1` target remains four signature animations. Extra
  ideas may be recorded without silently expanding the active learning day.
- First candidate: `CAND-ANIM-001` proposes a continuous Day 3 sequence in which
  input tokens produce causal logits and probabilities while each next-token
  target aligns with the preceding position to produce per-position loss. Joint
  review will decide whether it expands `ANIM-CE-001` or becomes a separate
  animation.
- Exact next action: begin Day 3 normally and use the new opportunity check after
  the causal-shifting and cross-entropy mechanisms are derived.

### Day 03 — 2026-09-01 to 2026-09-03

- Status: complete.
- Questions investigated: How does one contextual hidden state become a
  vocabulary-wide logit vector? Why are logits relative? How do stable softmax,
  NLL, one-hot cross-entropy, and perplexity connect? How does $p-q$ route credit
  through the output head and transformer? How do repeated one-hot outcomes
  teach uncertainty? How do causal shifting, teacher forcing, attention masks,
  and loss masks define different boundaries?
- Derivations completed: stable softmax and log-sum-exp; sequence likelihood to
  additive NLL; one-hot cross-entropy identity; exact
  $\partial L/\partial z=p-q$; output-head gradients; expected gradient $p-r$;
  $H(q,p)=H(q)+D_{KL}(q\|p)$; geometric-mean perplexity; tokenizer-dependent
  perplexity; causal target shift and valid-token masked mean.
- Code or content produced: reusable two-logit PyTorch lab, three unit tests,
  three first-class Day 3 notebooks with adjacent solutions, four focused
  learning artifacts, Chapter 3 with twelve exercises, complete worked
  solutions, and Chapter 2 enrichment on BPE atoms, categorical IDs, and the
  vocabulary/sequence trade-off.
- Experiments executed: precommitted full-batch two-logit SGD experiment with 70
  class-0 and 30 class-1 targets; all three notebook reference paths were
  validated with the registered DGX Spark kernel.
- Evidence and results: 7 of 7 experiment criteria passed; probabilities moved
  from `[0.5,0.5]` to `[0.69999999,0.30000004]`; final cross-entropy
  `0.61086428` matched empirical entropy `0.61086434`; maximum recorded autograd
  versus $p-r$ error was `5.96e-08`; all 13 repository tests passed.
- Failures or surprises: no experiment criterion failed. A stale Codex browser
  proxy briefly hid a still-running Jupyter server; all persistent notebook
  servers and kernels were later terminated at the learner's request.
- Claims not yet validated: recovery of the true human-language distribution;
  generalization from the toy context; calibration under dataset or decoding
  shift; capability, truthfulness, or generation improvement from lower loss;
  cross-tokenizer perplexity ranking; hardware performance of large vocabulary
  heads.
- Decisions made: treat notebook lessons as first-class course material while
  keeping reusable logic in importable modules; preserve token IDs as categorical
  addresses; distinguish intrinsic entropy from model mismatch; require an exact
  tokenizer and evaluation contract for perplexity comparisons.
- Book-facing contribution: `book/chapters/03-learning-the-next-token.md` and
  `book/solutions/03-learning-the-next-token.md`, plus targeted Chapter 2 and
  Chapter 2 solution enrichment.
- Public artifacts produced: the approved `ANIM-CE-001`, `ANIM-NTP-001`, and
  `ANIM-LOGLOSS-001` packets now have canonical Chapter 3 source material and
  verified numerical evidence; production remains on the Mac Studio.
- Exact next action: begin Day 4 with attention from first principles, preserving
  the distinction between forward causal visibility and backward credit flow.

### Content production — `ANIM-BPE-002` — 2026-09-03, local Mac

- Status: local review candidate ready; user visual approval pending. This
  independent content lane does not alter the learning position on Spark.
- Request: animate the user's five Chinese strings into `流`, `星`, `雨`,
  `流星`, and `流星雨`, then show the illustrative IDs 1–5 and encoding.
- Produced: `visuals/animations/projects/meteor-bpe/` contains the pre-execution
  specification, full corpus trace, Manim source, renderer, review HTML, 40.23s
  1080p/30fps MP4, GIF, full/half PNG stills, contact sheet, and hashed metadata.
  Reusable computation lives in `src/dongxi_llms/meteor_bpe.py` with five new
  regression tests and a focused Day 2 learning artifact.
- Evidence: unchanged corpus frequencies; unique first pair `(流, 星)` at 2;
  second pair `(流星, 雨)` selected by a declared preference among 15 tied
  maxima at 1; corpus counts 22 → 20 → 19; 19 vocabulary entries retained;
  eight exact encoding round trips. All nine focused BPE tests pass.
- Visual checks: twelve checkpoints, merge transitions, and full/half final
  stills reviewed; glyphs translate intact while token frames fuse. White
  canvas, Songti SC Chinese, Arial English/numerals, minimal labels, and
  geometry-based alignment follow the approved style.
- Scope: character-level BPE and teaching IDs; this animation demonstrates
  tokenizer mechanics, not neural-network learning or a limit on semantic
  understanding. No Spark computation, publication, or replacement of an
  existing animation. Produced on `codex/visuals/meteor-bpe`; source integration
  approved on 2026-09-04, with MP4s kept local and ignored.
- Book placement: Chapter 2 supporting example; trace and conceptual boundaries
  indexed under Day 2. Integrate into the reader-facing chapter only after
  visual approval; no change to the four signature-animation release target.
- Exact content-lane next action: review `projects/meteor-bpe/review.html` under
  `visuals/animations/`, focusing on pacing, pair counting, and the switch from
  tokenizer training to encoding.

### Cross-machine synchronization — 2026-09-04, local Mac

- User approved committing the meteor-animation work and syncing directly with
  `main`, excluding MP4 uploads. New MP4s are ignored; existing tracked media
  are not removed. GIF, stills, source, tests, trace, and reproduction metadata
  remain portable; render locally to restore the review page's video playback.
- Reconciled the local proposal-ID collision: the unpublished meteor candidate
  moves from `CAND-ANIM-008` to `CAND-ANIM-010`; Spark's committed attention
  candidate retains `008`. `ANIM-BPE-002` is unchanged.
- Preserve the remote Day 4 learning state and newer task packets. This sync
  does not finalize visual approval or authorize public publication.

### Day 04 — 2026-09-05 — First attention notebook ready

- Prepared `notebooks/day-04/01_causal_attention_forward.ipynb` with adjacent
  runnable solutions and optional learner implementations.
- Verified 13 code cells in a fresh kernel and three focused regression tests.
  Report: `experiments/reports/2026-09-05-causal-attention-forward.md`.
- Learner has not yet completed the notebook checkpoints. Begin with the
  shared-prefix prediction, then Q/K/V shapes and mask placement.
- Remaining: gradient session, cache session, and Chapter 4 synthesis with
  worked solutions. This session's evidence will be integrated at that synthesis.

### Day 04 — 2026-09-05 — Complete chapter and companion materials

- User requested full chapter coverage, including topics beyond live learning
  progress, and deferred interactive notebook practice.
- Produced complete Chapter 4, twelve worked solutions, notebook sessions 2–3,
  reusable gradients/scaling/cache examples, and a specification/report pair.
- Coverage map: `learning_artifacts/day-04-attention-and-causal-information-boundary/chapter-coverage-and-evidence.md`.
- Verification: 26 repository tests pass; both new notebooks validate and execute
  all 12 code cells each in fresh kernels. Earlier session 1 reference already
  verified; locally edited learner notebooks are preserved.
- Evidence: chain-rule/autograd error 2.22e-16; finite-difference error 1.83e-10;
  verified mask/detach boundaries; paired scaling simulation; cached/full-prefix
  agreement and stale-prefix counterexample. See the linked reports for limits.
- Book and content sources are ready. No learner mastery, training run, serving
  benchmark, animation render, or article publication is claimed.
- Next action: Day 5 conceptual lesson on multi-head attention and residual flow.

## Learning memory and production queue

### Chapter 5 notebook production — 2026-09-06

- User requested all sessions be built before guided study, including the
  optional recurrence lesson, then requested committing and pushing the result.
- Delivered: 11 notebooks, adjacent solutions, transparent decoder source,
  12 new tests, execution runner, specification/report, companion lab, solution
  guide, and updated curriculum/artifact/animation indexes.
- Verification: 93 code cells executed in 11 fresh kernels; 38 repository tests
  pass. Fixed CPU seed-505 training: loss 2.774792432785034 to
  0.0007856183219701052 after 160 AdamW steps; training accuracy 1.0.
- Evidence boundaries: memorization only, no GPU or serving benchmark, no Mac
  execution, no trained recurrence comparison, no completed learner practice.
- Day 3/4 learner notebook edits remain untouched and excluded from the commit.
- Next: notebook 01's first conceptual prediction and lookup exercise. Full
  Chapter 5 narrative synthesis remains scheduled for Day 7.

Detailed learner understanding is maintained by day and topic under
`learning_artifacts/`. Its index, the X article backlog, animation storyboards,
and cross-machine task packets are maintained in `LEARNING_MEMORY.md`.

## Recent material synthesis

### Day 5 foundation synthesis — 2026-09-07

- User requested the Day 5 contribution now, retaining Days 6–7 in Chapter 5.
- Delivered: `book/chapters/05-building-a-modern-decoder.md`, twelve worked
  answers in the existing solution guide, linked notebook/figure pathway, and
  updated topic records, book indexes, and existing animation candidates.
- Validation: chapter's runnable 160-step code excerpt reproduces final loss
  0.0007856183219701052 and accuracy 1.0 under the existing reference contract;
  LayerNorm and distinct-head gradient examples checked; 92 local links checked;
  47 tests pass in 3.572 seconds; validation commands and diff check exit 0.
- Boundaries: documentation synthesis and reproduction of a fixed toy example,
  not a new model comparison, learner mastery, or completed modern architecture.
  No notebooks were edited by this task; active saved/unsaved learner work stays
  separate. No animation rendered; no commit or push performed for this synthesis.
- Next: review this foundation/Notebook 07's evidence limits, then proceed to
  Day 6's isolated modern-variant comparisons when ready.

### Day 6 narrative preparation — 2026-09-07

- User requested ready notebooks and then coherent chapter material ahead of
  guided study. Reused the three already-committed notebooks; no duplicates.
- Extended Chapter 5 through section 5.22; now 24 worked answers. Covers modern
  mechanisms, exact lab conventions, design accounting, pinned Qwen shapes,
  larger calculated candidates, and both requested recurrent-depth papers.
- Verification: new chapter snippet and numerical examples pass; 47 tests pass;
  95 local links and seven heading anchors checked. Three-session readiness:
  43 code cells, 13 figures, source-preserving fresh kernels. Full evidence in
  `experiments/reports/2026-09-07-day6-chapter-verification.md`.
- No notebook edits, large-model allocation/training, animation render, or
  public publication. Day 5/6 mastery and Day 7 defense remain unclaimed.
- Next: guided Day 6 notebook 01, starting with the norm contrast.

### Day 7 chapter synthesis — 2026-09-07

- User requested a coherent chapter update after preparing the core Day 7
  notebook. Extended the existing Chapter 5, not a competing daily chapter.
- Sections 5.23–5.27 complete the narrative through architecture defense;
  exercises and worked answers now total 30. Integrated two verified figures
  and the existing report's tensor, gradient, cost, and controlled-fault evidence.
- New prose distinguishes finite outputs from causal/cache correctness,
  backward connectivity from learning updates, and parameter budgets from
  compute/time contracts. It closes with a concrete defense and a pretraining
  transition. No trained recurrence result or preferred large model is claimed.
- Chapter navigation and portable learning memory updated. Existing notebook
  content and all earlier uncommitted work preserved; no publication or render.
- Verification: mean-loss derivative, gradient connectivity, unchanged weights,
  trace shapes, paired budgets, and three diagnostic signatures rechecked on
  the existing CPU float64 fixture. All 93 local links and six heading anchors
  pass; 27 chapter sections and 30 answers are sequential; twelve notebook hashes
  are unchanged. Details in the Day 7 learning artifact.
- Next: guided Day 6 notebook 01 unless the learner chooses Day 7 review first.
  Material completion does not close outstanding practice or defense requirements.

## Latest maintenance — GitHub math, 2026-09-08

- Learner reported forbidden `operatorname` macros and requested a check across
  all chapters, explicitly reiterating Chapters 4 and 5. Repaired all five
  chapters and affected solutions/lab: 54 macro substitutions, 78 display and
  seven inline delimiter conversions, and one mismatched delimiter repair.
- All 13 book Markdown files pass source checks; all 560 extracted expressions
  render without errors in the local MathJax check. All 59 tests pass. Added a
  dependency-free regression check and portable authoring guidance at
  `docs/MATH_FORMATTING.md`. Live GitHub rendering has not yet been rechecked.
- No notebook/code-model edits, training, or change to learner mastery. Next
  learning action remains Day 6 notebook 01, unless the learner chooses otherwise.

## Latest learning design — 2026-09-09

- Learner requested a more interactive course, agreed on Mac/Spark responsibilities,
  and explicitly requested trigger phrases, pre-departure commit/push, arrival
  sync, reminders, and portable progress. Recorded `docs/LEARNING_WORKFLOW.md`
  and `docs/handoffs/CURRENT.md`; updated agent rules, memory, and reader entry.
- First proposed pilot: Day 8 Optimizer Playground on Mac, with actual gradient
  and update arrows, controlled comparisons, and step/reset/play controls. No
  implementation, Mac environment validation, GPU experiment, or animation
  production occurred. Existing notebook and model code remains unchanged.
- Day 8 is orientation/planning only. Days 4–7 deferred learning requirements
  remain open. Next is the learner-triggered Mac departure, then arrival
  verification and the recorded learning continuation.

## Departure to Mac — 2026-09-09

- Learner invoked `Switch to Mac`. Verified source `spark-aa66`, Linux/aarch64,
  intended repository and `main`; fetched remote was aligned before committing.
- Refreshed `LEARN-MAC-001` with actual departure and next-arrival instructions.
  Point-in-time process checks found no matching course Python/Jupyter/torchrun
  jobs for `dongxi`; NVIDIA compute-process listing was empty. No jobs modified.
- Publishing the scoped workflow/memory records is the departure step; receipt
  of the pushed commit and local execution must still be verified on Mac.
- Arrival phrase: `Continue on Mac`. Preserve all existing learning gaps and
  proposed-versus-built distinctions. This is not a completed lesson.

## Return to Spark and direct visual learning — 2026-09-09

- Learner reported Mac dependency friction and requested continuation on Spark.
  Verified `spark-aa66`; clean fast-forward sync reported already up to date.
- Corrected stale Day 6 resumption across current progress, handoff, workflow,
  and agent memory. Intended lesson is Day 8; earlier gaps remain review items.
- Clarification: rich visuals directly inside the conversation, not only
  notebooks or a standalone website. Built the first small SGD step-size visual
  outside the repo, with adjustable learning rate and discussion action.
- Browser checks passed numerical updates at three rates, light/dark themes,
  and 736px/360px layouts. This is not the full optimizer lab, model training,
  or learner mastery. Notebook/model source and course environment unchanged.
- Next: discuss the learner's chosen step size and proceed through Day 8.

## Complete Day 8 material build — 2026-09-09

- User found the isolated SGD slider boring and requested complete notebooks
  and a coherent Day 8 chapter. Created Chapter 6's full Day 8 foundation,
  three worked visual notebooks, twelve conceptual solutions, companion lab and
  bounded recipe. This is content readiness, not completed guided study.
- Reused the existing modern decoder; added authored byte-text data, correctly
  weighted accumulation, explicit AdamW/schedule/clipping probes, validation and
  complete/incomplete checkpoint controls. No earlier learner notebook changed.
- Verified on Spark CPU: 3 fresh-kernel passes, 29 code cells, 10 PNGs, 71 tests;
  605 book formulas passed source and local MathJax checks, 161 local targets
  resolved. Evidence: `experiments/reports/2026-09-09-day8-material-verification.md`.
- Complete recovery matched exactly in the fixture; missing optimizer/cursor
  controls differed. The missing-cursor control even had slightly lower tiny-
  holdout loss, illustrating why a nicer metric cannot establish faithful replay.
- Extended existing training/optimizer animation candidates and captured
  CAND-ANIM-019 for process-state recovery. Production remains approval-gated on
  Mac Studio. No animation render, GPU campaign, persistent server, commit or
  push was performed by this content request.
- Next: Day 8 on Spark, document windows and the valid-token objective with
  meaningful recipe dilemmas. Earlier study gaps remain optional review.

## Open questions

- Which exact dataset and capability mix should anchor the general-assistant SFT branch?
- Which verifiable task mixture should anchor the GRPO branch beyond initial arithmetic smoke tests?
- What parameter count and corpus budget should the first `DongxiGPT` pretraining run use after Spark profiling?
- Which evaluation examples can remain private or held out to reduce contamination?
