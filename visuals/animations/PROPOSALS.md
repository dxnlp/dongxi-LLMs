# Animation Proposal Inbox

This file is the shared inbox for animation ideas suggested during learning and
course development. Suggestions may come from Dongxi or from a contributing
agent. A suggestion is not approval to produce media.

Approved animation tasks move to `LEARNING_MEMORY.md`, where they receive a
stable `ANIM-*` task ID, a complete task packet, dependencies, acceptance checks,
and machine ownership. This inbox retains deferred and rejected ideas so the
same proposal is not repeatedly rediscovered.

## Candidate queue

### Day 8 canonical material — 2026-09-09

Source: agent automatic mathematics/opportunity check during the learner-requested
Chapter 6 and notebook build. Reference: `book/chapters/06-pretraining-as-a-controlled-system.md`
and `notebooks/day-08/`. Static reference figures are not produced animation.

- Extend **CAND-ANIM-003 / ANIM-NTP-001**: follow document bytes into shifted
  windows; fade ignored targets; accumulate summed NLL divided by the same total
  N across microbatches. Canonical relation: gradient(mean over valid targets)
  = sum of microbatch summed-loss gradients / N. Contrast a one-target tail with
  sixteen targets and visibly expose why equal microbatch weights are wrong.
  Notebook 1 verifies the correct and broken gradients with the actual decoder.
- Extend **CAND-ANIM-018 / ANIM-EMB-001**: show actual decoder gradient coordinates
  entering AdamW's m and v history, bias correction, adaptive update and separate
  decay. Pair with the update-clock warmup/cosine schedule and a clip-after-sum
  intervention. Canonical formulas are Chapter 6 sections 6.7–6.9. Preserve the
  distinction between gradient norm, optimizer update and observed next loss.
  Do not replay the isolated quadratic as the default: learner found it boring.
  Notebook 2 verifies the recurrence and clipping, not optimizer superiority.
- Precision/memory extension to the same stability segment: contrast range and
  resolution with the actual FP16/BF16 casts, then distinguish 16P persistent
  bytes from total memory. CPU cast evidence is not measured GPU BF16 training.

### CAND-ANIM-019 — A checkpoint freezes a process, not only weights

- Source: agent automatic transition/state check, Day 8 material creation.
- State: candidate only; production not approved; Mac Studio production lane.
- Mechanism: at update12 freeze weights, Adam moments, shuffle cursor/RNG,
  update/token clocks and recipe identity. Branch into uninterrupted continuation,
  complete restoration, missing moments, and missing cursor through update24.
- Motion: align the first two trajectories; show the same next data but changed
  update in the missing-moments branch; show changed next data in the cursor
  branch. Keep “file loads,” “loss is finite,” and “trajectory replays” distinct.
- Evidence: Notebook 3 runs the exact small decoder branches; full local CPU
  replay matches, both omissions differ. Use report values for labels. Never
  imply bitwise Mac/Spark equivalence or distributed recovery coverage.
- Dependency: approve storyboard, use canonical Chapter 6/solutions and the
  verified notebook manifest, then create a complete portable ANIM task packet.
  No rendering or public publication authorized by this capture.

### Interactive-learning design — 2026-09-09

Follow-up: first one-step SGD visual now demonstrated directly in conversation
on Spark (not a produced film). It reuses CAND-ANIM-018's mechanism. Full
momentum/AdamW interaction and animation production remain separate pending work;
the learner explicitly chose direct in-app interaction over notebook-only study.

Source: user request for real-time visual lessons and Mac/Spark workflow.
The proposed Day 8 Optimizer Playground reuses CAND-ANIM-018 and ANIM-EMB-001's
gradient/update teaching motivation. Preserve the distinction between a gradient
arrow and an optimizer update, matched initial conditions, and extra history
state for momentum/AdamW. This is a proposed interactive tool, not new measured
evidence or approval for a produced animation. The workflow and portable state
requirements are in `docs/LEARNING_WORKFLOW.md`; rendering remains on Mac Studio
only after explicit production approval. No duplicate animation candidate added.

### Day 7 chapter synthesis — 2026-09-07

Source: user-requested chapter update; agent automatic mathematics review.
Chapter 5 sections 5.23–5.27 reuse the embedding/CE backward packets for
`dL/dz = (p-q)/12` in the twelve-target mean-loss fixture; CAND-ANIM-014 for the
future-state → time-mean → earlier-state leak; CAND-ANIM-016 for incorrect cache
rotation offsets; and CAND-ANIM-017 for distinct query-count, KV-payload, and
whole-model-budget counters. The fixed-parameter/compute/time comparison in
5.25 strengthens CAND-ANIM-011's accounting caveats. Evidence is the executable
Day 7 teaching audit, not trained comparative quality. Two existing static
figures illustrate the prose. No duplicate animation or rendering task is
created; production still requires explicit approval and runs on Mac Studio.

### Day 7 defense notebook — 2026-09-07

Automatic mechanism check: the new core defense notebook supplies actual module
shape traces, parameter/cache accounting, backward connectivity, and two
controlled failure signatures. Reuse the existing embedding/attention/loss
packets and CAND-ANIM-014/016/017 rather than creating duplicate animations.
CAND-ANIM-011 and X-LOOP-001 can reuse its comparison-design worksheet; that
worksheet is a proposal, not a trained result or rendering approval. All media
production remains approval-gated on Mac Studio. Canonical evidence:
`experiments/reports/2026-09-07-day7-architecture-defense.md`.

### Day 6 narrative readiness — 2026-09-07

Source: agent automatic mathematics check during requested chapter creation.
Canonical material is Chapter 5 sections 5.11–5.22 and worked answers 13–24.
Refine existing proposals rather than creating duplicates:

- CAND-ANIM-014: contrast feature centering with RMS magnitude; show the constant
  vector and common-offset cases, epsilon, and the untouched residual bypass.
- CAND-ANIM-015: follow the two expanded branches through SiLU and multiplication;
  extend backward with dL/dc=delta*g and dL/da=delta*c*SiLU'(a). A zero forward
  gate is not necessarily a dead gradient path. Verify numeric motion before render.
- CAND-ANIM-016: preserve coordinate-pair norms and show R_m^T R_n=R_(n-m),
  then keep the causal mask correct while deliberately restarting decode offsets.
- CAND-ANIM-017: keep Hq attention distributions visible while compact Hkv
  storage shrinks. Distinguish tensor payload, dense FLOPs, and actual latency.
  Add Q/K RMS rescaling as an optional subscene; the lab's learned scales are
  shared across heads, and the original QKNorm paper uses a different formula.
- CAND-ANIM-011 / X-LOOP-001: both requested primary papers are linked in the
  canonical frontier introduction. Fixed sharing, variable depth, token-level
  routing, and explicit KV sharing remain distinct. The trained comparison and
  architecture defense still belong to Day 7; no quality result is inferred.

Static notebook previews and reference calculations are available. All animation
production remains on Mac Studio under the existing approval gates. This
chapter-writing request does not commission rendering, article drafting, or
public publication.

### Day 5 foundation synthesis — 2026-09-07

Source: agent automatic math-opportunity check while writing the requested
chapter foundation. Canonical source is now
`book/chapters/05-building-a-modern-decoder.md`, with worked answers in
`book/solutions/05-decoder-notebook-solutions.md`. Reuse existing candidates:

- CAND-ANIM-012: extend the multi-head scene backward through distinct W_O
  blocks, showing dL/do_r=g W_O,r^T. One shared loss need not send identical
  gradients to each head. Symmetric heads can remain duplicates; diverse
  initialization is not proof of learned semantic specialization. The scalar
  1-versus-2 gradient illustration is algebraic, not a trained-model result.
- CAND-ANIM-013: show the skip carrying X into addition, with only the sum
  passed onward. Preserve the distinction from an untouched recoverable backup.
- CAND-ANIM-014/015: canonical readable normalization and affine-collapse
  explanations are now in sections 5.5–5.6; keep position and feature axes clear.
- ANIM-CE-001 / ANIM-NTP-001: reuse section 5.8's twelve aligned target losses
  combining into one mean and one parameter update. Generation selects new IDs
  sequentially with fixed weights; training does not update after each token.
- CAND-ANIM-018 and ANIM-EMB-001 retain the optimizer and shared-parameter paths.

Existing notebooks verify the baseline mechanisms. This refinement captures
storyboard opportunities, not a new rendering commission. All production stays
on Mac Studio after the applicable explicit approval; no media produced here.

### CAND-ANIM-018 — Same gradient, different learning-rate steps

- Source: agent automatic math trigger following learner's optimizer question,
  2026-09-07; introduced in Chapter 5, deeper treatment in Chapter 6/Days 8–9.
- State: discuss; production not approved; rendering stays on Mac Studio.
- Mathematics: scalar L=(e-1)^2, gradient=2(e-1), SGD e_new=e-eta*gradient.
- Motion: place three identical markers at e=.2 on the same quadratic, expose
  the common gradient -1.6, then step with eta=.01, .1, and 1.5. Show small
  progress, larger progress, and overshoot to e=2.6 with increased loss. Keep
  parameter position, gradient sign, update arrow, and loss height distinct.
- Precision: illustrative quadratic, not an LLM loss landscape or recommended
  learning-rate range. Local descent direction does not guarantee a decrease
  for an arbitrary step. AdamW uses adaptive gradient-history statistics and
  must not be animated as plain raw-gradient SGD.
- Evidence: analytical one-step values .216, .36, 2.6; loss .64 to 2.56 in
  the large-step case. Verify rendered trajectories numerically before production.
- Dependencies: approve storyboard and connect to existing ANIM-EMB-001 optimizer
  segment without silently replacing its SGD sketch with an AdamW claim.

### Chapter 5 notebook evidence update — 2026-09-06

Source: agent automatic mathematics check while fulfilling the learner's request
to build the entire Chapter 5 notebook pathway. All eleven reference notebooks
execute successfully; evidence is in
`experiments/reports/2026-09-06-decoder-notebooks.md`.

- `CAND-ANIM-012`: head splitting, independent/vectorized agreement, head ablation,
  and causal invariance now have executable examples in Day 5 notebook 02.
- `CAND-ANIM-013`: zero-branch identity, direct/branch gradient sum, cancellation,
  and pre/post-norm contrast now have examples in Day 5 notebooks 03–04.
- `CAND-ANIM-011`: optional Day 7 notebook verifies fixed shared-block outputs,
  gradient accumulation, parameter/application accounting, and distinct K/V
  states across applications. This is not the planned trained architecture
  comparison or an adaptive-routing implementation. `X-LOOP-001` remains queued.
- Reuse `ANIM-EMB-001` and `ANIM-ATTN-001` for the embedding and attention paths
  rather than duplicating their existing storyboards.

### CAND-ANIM-014 — Centering, scaling, and the residual bypass

- Source: agent, automatic math trigger, 2026-09-06; Chapter 5.
- State: discuss; production not approved.
- Mathematics: LN(x)=gamma*(x-mean(x))/sqrt(var(x)+eps)+beta;
  RMSNorm(x)=gamma*x/sqrt(mean(x*x)+eps).
- Motion: keep one position's feature vector together; show mean subtraction,
  scale normalization, then learned affine change. Contrast RMS scaling without
  centering. Zoom out to pre/post-norm placement and the residual bypass.
- Precision: feature axis, not time; epsilon prevents exact scale invariance;
  affine outputs need not have zero mean/unit variance. Wrong time-axis
  normalization can leak the future despite a valid attention mask.
- Evidence: Day 5 notebook 04 and Day 6 notebook 01 forward/backward references,
  constant-vector and wrong-axis controls. Canonical source: Chapter 5 lab.
- Dependency: learner approval and final storyboard; production on Mac Studio.

### CAND-ANIM-015 — Nonlinear and gated feature transformations

- Source: agent, automatic math trigger, 2026-09-06; Chapter 5.
- State: discuss; production not approved.
- Mathematics: down(GELU(up(x))) versus down(SiLU(gate(x))*up(x)); removing
  nonlinearity collapses two affine layers, including their combined bias.
- Motion: expand one position's features, transform them, and contract. Reveal
  a second gate branch and elementwise multiplication; zero it and watch the
  bias-free output disappear. Keep token positions fixed throughout.
- Precision: SiLU is not a probability; three matrices alter parameter budgets;
  isolated MLPs do not directly mix tokens but can read contextual input states.
- Evidence: Day 5 notebook 05 and Day 6 notebook 01, affine-collapse and
  positionwise-gradient checks. No trained quality ranking.
- Dependency: learner approval and canonical storyboard; render on Mac Studio.

### CAND-ANIM-016 — Relative angles and a cache position mistake

- Source: agent, automatic math trigger, 2026-09-06; Chapter 5.
- State: discuss; production not approved.
- Mathematics: R(theta)(a,b)=(a*cos(theta)-b*sin(theta),a*sin(theta)+b*cos(theta));
  (R_m q) dot (R_n k) = q dot (R_(n-m) k).
- Motion: rotate coordinate pairs while preserving length; shift both positions
  together and keep their dot product fixed. Retain old rotated keys in a cache;
  contrast correct continuing query positions with a mistaken restart at zero.
- Precision: adjacent-pair toy layout, base 10,000; not every checkpoint layout.
  Fixed-vector geometry does not establish long-context generalization.
- Evidence: Day 6 notebook 02 norm/relative-position/gradcheck tests and full
  decoder cached replay with an intentionally wrong offset.
- Dependency: learner approval; coordinate with the cache candidate; Mac only.

### CAND-ANIM-017 — Many queries, compact shared keys and values

- Source: agent, automatic math trigger, 2026-09-06; Chapter 5.
- State: discuss; production not approved.
- Mathematics: Hq/Hkv query heads share each KV head; logical cache payload
  equals 2*layers*batch*Hkv*length*head_width*bytes_per_element.
- Motion: retain four query distributions while KV stores contract from four
  to two to one head. Track source sharing, gradient accumulation, and compact
  cache bytes independently from query attention arithmetic.
- Precision: grouped sharing is not dropping query heads; logical payload is not
  allocator memory or measured speed. The teaching kernel expands KV temporarily.
- Evidence: Day 6 notebook 03 grouped/reference forward and backward checks,
  MHA/MQA endpoints, compact cache bytes, and analytical parameter agreement.
- Dependency: learner approval; consider extending the existing cache candidate
  rather than commissioning a separate film; production on Mac Studio.

### CAND-ANIM-013 — Attention as an update to the residual stream

- Source: agent, automatic mathematics trigger, 2026-09-06
- Book placement: Chapter 5, residual connections
- State: discuss; production not approved
- Equation: X_next = X + MHA(X), initially omitting norm/dropout; later reveal
  the pre-norm form X + MHA(LN(X)).
- Motion: keep the incoming [T,D] stream visible while a branch computes a
  same-shaped update; add coordinatewise. Set the update to zero and retain
  the unchanged input. Reverse gradients through the direct and learned paths.
- Precision: addition differs from feature concatenation; X is the current
  state, not always the original embedding. The identity path does not guarantee
  information preservation or prevent all gradient cancellation.
- Evidence: forward zero-update, backward split, and cancellation controls are
  verified in `day-05/03_residual_stream.ipynb` and the 2026-09-06 report.
- Source: `learning_artifacts/day-05-decoder-only-transformer/residual-stream-and-attention-updates.md`.
- Production: Mac Studio after explicit approval; consider scope alongside the
  existing multi-head candidate rather than assuming an additional film.

### CAND-ANIM-012 — Several retrieval mixtures for one position

- Source: agent, automatic mathematics trigger; introduced 2026-09-06
- Book placement: Chapter 5, bridge from single-head attention
- State: discuss; no production approval
- Mechanism: O_h=A_h V_h, then O_multi=Concat(O_1,...,O_H) W_O.
- Motion: preserve X and receiver identity while multiple head-specific Q/K/V
  branches create different weight rows and value mixtures; concatenate feature
  segments and project them back to model width, leaving token positions fixed.
- Precision: heads see the allowed prefix rather than disjoint token subsets;
  fixed human-readable roles are not guaranteed; W_O differs from the vocabulary
  head; concatenation is across features and is not averaging.
- Evidence: executable split/merge, ablation, and causal controls are in
  notebook 02 and the 2026-09-06 report; Chapter 5 section 5.3 now supplies
  the canonical narrative. No trained specialization experiment is claimed.
- Source artifact: `learning_artifacts/day-05-decoder-only-transformer/multiple-heads-and-output-projection.md`.
- Next decision: consider after multi-head implementation and verification.
  Production requires explicit approval and belongs on the Mac Studio.

| Candidate ID | Source | Book placement | Mechanism | State | Dependency or next decision |
|---|---|---|---|---|---|
| `CAND-ANIM-010` | User | Chapter 2 supporting example | Five supplied Chinese strings → counted BPE merges `流 + 星`, `流星 + 雨` → retained vocabulary → illustrative IDs 1–5 → encoding | review; promoted into `ANIM-BPE-002` | 40.23-second 1080p Mac Studio render ready on 2026-09-03; round-2 tie disclosed; source integration approved on 2026-09-04; final visual approval pending |
| `CAND-ANIM-001` | Roadmap + agent, automatic math trigger confirmed | Chapter 3 | Input tokens → contextual state `[D]` → dense projection against all `[V,D]` output rows → `[V]` logits → probabilities → selected vocabulary index/token ID → append and repeat, while next-token targets align with preceding positions; optionally reveal the `T`-versus-`V` compute trade-off | discuss | Canonical Chapter 3 treatment is complete; decide on the Mac Studio whether to expand `ANIM-CE-001` or defer this separate animation |
| `CAND-ANIM-002` | User | Chapter 3 | Place negative log-likelihood and cross-entropy in the LLM training context: target probability → per-token NLL → masked aggregation across next-token positions → cross-entropy | approved; promoted into `ANIM-CE-001` | Day 3 derivation and verification complete; produce only on the Mac Studio |
| `CAND-ANIM-003` | User | Chapter 3 | Standard next-token update: predicted distribution `p` versus one-hot target `q` → logit gradient `p-q` → target score rises and non-target scores fall → repeated diverse examples shape a distribution | approved; promoted into `ANIM-NTP-001` | Gradient and tiny-model evidence complete; produce only on the Mac Studio |
| `CAND-ANIM-004` | User | Chapter 3 | Why negative log loss: sequence probability products become additive token surprise; confident errors retain strong gradients; expected log loss rewards matching the full data distribution | approved; promoted into `ANIM-LOGLOSS-001` | Canonical derivations and controlled evidence complete; verify the `1-p_target` comparator during Mac production |
| `CAND-ANIM-005` | Agent, automatic math trigger | Chapter 3 | Mean token cross-entropy → exponentiation → perplexity as an effective equal-choice branching factor, while preserving tokenizer and evaluation-distribution dependence | discuss | After derivation, decide whether to extend `ANIM-CE-001` or create a separate short; production only on Mac Studio after approval |
| `CAND-ANIM-006` | Agent, automatic math trigger from learner question | Chapter 2–3 bridge | Consistently permute token IDs, dataset symbols, embedding rows, and output rows while decoded text behavior remains unchanged; reveal that IDs are categorical addresses rather than numerical linguistic features | discuss | Prefer extending `ANIM-EMB-001` if the permutation can remain concise; otherwise defer beyond v0.1. Production only on Mac Studio after explicit approval |
| `CAND-ANIM-007` | Agent, automatic math trigger from learner question | Chapter 3 decoding bridge | Hold logits fixed while temperature continuously rescales their gaps: low temperature sharpens, high temperature flattens, ranking stays fixed, and exact tied maxima reveal the difference between greedy tie-breaking and sampling | discuss | Verify ratios and limiting behavior in the Day 3 notebook; likely defer or use as a compact decoding short. Production only on Mac Studio after explicit approval |
| `CAND-ANIM-008` | Agent, automatic math trigger; approved by user | Chapter 4 | Input states → learned `Q`, `K`, and `V` projections → scaled query-key score matrix → causal mask → row-wise attention distributions → weighted value retrieval; reverse the loss gradient through a value/content path and a query-key/routing path; contrast broken scaling and masking | approved; promoted into `ANIM-ATTN-001` | Chapter, forward, gradient, finite-difference, mask, and detach evidence verified on 2026-09-05; ready for Mac production |
| `CAND-ANIM-009` | Agent, automatic math trigger from learner inference | Chapter 4–modern decoder bridge | Contrast the same token in two contexts to establish distinct request-local K/V states; show optional runtime retention across prefill and decoding, each transient new query reading the unchanged-prefix cache, and logical cache release at sequence completion | discuss | Toy cache shapes and equivalence verified on 2026-09-05; lifecycle derived, no allocator benchmark; distinguish first-layer and contextual deeper-layer keys; production only on Mac after explicit approval |
| `CAND-ANIM-011` | User reminder + agent automatic math trigger | Chapter 5 frontier section | Reuse one visually identical Transformer stack for recurrent state updates; let stored-parameter, effective-depth, and compute counters diverge; then contrast fixed loops with adaptive token-level exit and visible token-space reasoning | discuss | Revisit for production approval during Days 6–7; verify all accounting in the controlled comparison; production only on the Mac Studio after explicit approval |

Q/K/V workflow refinement (user request, 2026-09-06): extend `CAND-ANIM-008`
and its approved `ANIM-ATTN-001` packet. Keep the V content branch separate
from Q/K scores and softmax weights until O=AV. Include fixed-Q/K, changed-V
and fixed-V, changed-routing contrasts. See the packet for precision and
Mac production requirements; no duplicate candidate is created.

### CAND-ANIM-002 — NLL to cross-entropy in an LLM

- Source: user
- Proposed during: Day 3
- State: approved; promoted into `ANIM-CE-001`
- Learning objective: Show that, for a one-hot next-token target, token-level
  cross-entropy equals the negative log-likelihood of the observed token, and
  that the training loss aggregates those terms only across valid target
  positions.
- Why motion is better than a static figure: The same target token must retain
  its identity while its probability is selected, transformed by `-ln`, repeated
  across causal positions, filtered by the loss mask, and reduced into one scalar.
- Moving objects and stable anchors: Keep the token sequence and target alignment
  stable; move from probability bars to highlighted `p_y`, per-position NLL
  tiles, masked/ignored positions, and the final mean cross-entropy.
- Canonical source material:
  `book/chapters/03-learning-the-next-token.md` and
  `learning_artifacts/day-03-probabilities-and-next-token-loss/`.
- Evidence status: derivation, target alignment, masked mean, and PyTorch
  reference computations are complete.
- Precision risks and required caveats: Do not present NLL and one-hot
  cross-entropy as different numerical objectives; distinguish sequence NLL sum
  from the common mean over valid tokens; make visible that target label index
  `k` pairs with logit index `k-1`; exclude padding or ignored labels; retain
  natural logarithms; do not imply that low loss proves truthfulness.
- Dependencies: Day 3 derivation, causal label alignment, loss-mask denominator,
  and PyTorch agreement are complete.
- Suggested destination: Chapter 3, course site, and the embedding-training
  animation handoff.
- Next decision: Mac Studio production may begin from the committed Chapter 3
  material when the learner starts the animation task.

### CAND-ANIM-003 — Standard next-token training update

- Source: user
- Proposed during: Day 3
- State: approved; promoted into `ANIM-NTP-001`
- Learning objective: Make one-hot next-token supervision and its softmax
  cross-entropy gradient visible: `dL/dz = p-q`. Show why one example rewards
  the observed token and locally suppresses every non-target, including
  alternatives that could be valid in another sample.
- Why motion is better than a static figure: The learner needs to preserve token
  identity while probability bars become gradient bars, logits move in opposite
  directions, and repeated examples with different observed targets accumulate
  into a learned conditional distribution.
- Moving objects and stable anchors: Keep candidate tokens and their colors fixed;
  place `p` beside one-hot `q`; transform them into `p-q`; move the target logit
  upward and non-target logits downward; continue the signal through
  `z = Wh+b`, splitting it into `dW = (p-q)h^T` and `dh = W^T(p-q)`; then replay
  a small controlled stream whose target frequencies are visibly known.
- Canonical source material:
  `learning_artifacts/day-03-probabilities-and-next-token-loss/probability-as-competition-and-surprise.md`;
  `book/chapters/03-learning-the-next-token.md`; and the committed tiny-model
  report.
- Evidence status: analytical gradient, PyTorch agreement, optimizer trajectory,
  and controlled 70/30 learned-frequency convergence are verified and reported.
- Precision risks and required caveats: Show gradient descent direction rather
  than confusing gradient sign with parameter motion; state that one-hot
  supervision does not mark unobserved alternatives as valid; do not imply that
  one update sets the target probability to one; distinguish a single local
  update from the expectation over a representative data distribution; use fixed
  verified numbers in the final render.
- Dependencies: core analytical gradient, PyTorch agreement, optimizer-step
  trajectory, and controlled target-frequency experiment are complete;
  output-head chain-rule verification remains for the expanded version.
- Suggested destination: Chapter 3, course site, and the later embedding-gradient
  animation sequence.
- Next decision: Produce and render only on the Mac Studio; decide there whether
  it is a standalone short animation or a
  companion segment to `ANIM-CE-001`.

### CAND-ANIM-004 — Why negative log loss has three jobs

- Source: user
- Proposed during: Day 3
- State: approved; promoted into `ANIM-LOGLOSS-001`
- Learning objective: Explain why `-log p_target` is structurally suited to
  language modeling rather than merely displaying its curve.
- Why motion is better than a static figure: Three transformations occur over
  time: conditional probability factors combine into a sequence product and
  unfold into additive surprise; competing loss choices produce different
  gradient strength near confident errors; repeated outcomes reveal whether a
  scoring rule recovers a distribution or collapses onto its mode.
- Moving objects and stable anchors: Preserve the same target probabilities and
  candidate colors across three acts. Act 1 moves probability factors from a
  product into additive NLL tiles, then contrasts two equal-sum sequences whose
  surprise is either diffuse or concentrated in one catastrophic token. Act 2
  synchronizes loss and gradient curves for log loss versus `1-p_target`. Act 3
  reveals a hidden population distribution `q` through a finite stream of
  one-hot samples and their evolving empirical frequencies `q_hat`, then moves a
  model distribution `p` toward the expected-loss minimum while distinguishing
  generalization from memorization.
- Canonical source material:
  `learning_artifacts/day-03-probabilities-and-next-token-loss/probability-as-competition-and-surprise.md`;
  `book/chapters/03-learning-the-next-token.md`; and the controlled Day 3 report.
- Evidence status: sequence, logit-gradient, and proper-scoring mechanisms are
  derived; $p-q$ autograd and controlled expected-loss behavior are verified.
  The alternative `1-p_target` softmax-gradient comparison remains a Mac
  production check.
- Precision risks and required caveats: Distinguish loss magnitude from gradient
  through softmax; do not claim `1-p_target` itself has a small value for a
  confident error—its softmax gradient becomes small; distinguish one-hot sample
  targets from the population conditional distribution; show expected loss when
  discussing proper scoring; do not use KL nonnegativity before defining it.
- Dependencies: sequence chain rule, log-product identity, analytical and
  autograd gradients for both losses, and a controlled expected-loss comparison
  under a fixed categorical `q`.
- Suggested destination: Chapter 3 and course site.
- Next decision: Mac Studio chooses one three-act animation or three coordinated
  shorts after the verified Day 3 evidence is committed; every act must remain in
  the approved scope.

### CAND-ANIM-005 — Cross-entropy to perplexity

- Source: agent, automatic mathematics trigger
- Proposed during: Day 3
- State: discuss
- Learning objective: Show why exponentiating mean natural-log token loss returns
  to probability scale and yields an effective branching factor, not a literal
  count of equally likely next tokens.
- Why motion is better than a static figure: A mean surprise value in nats is
  abstract. Motion can transform equal-choice distributions with $k$ candidates
  through `ln(k)` loss and back through `exp` to $k$, then morph to unequal
  distributions with the same perplexity while keeping their different shapes
  visible.
- Moving objects and stable anchors: Keep the tokenizer, valid-position mask, and
  evaluation corpus fixed; move per-token NLL tiles into their mean, exponentiate
  the mean, and compare equal and unequal distributions sharing an effective
  branching factor. Then hold one text probability fixed while splitting one
  token into two, showing total NLL remains fixed as mean token NLL and perplexity
  change solely because the counting unit changed.
- Canonical source material:
  `learning_artifacts/day-03-probabilities-and-next-token-loss/nll-cross-entropy-and-perplexity.md`;
  `book/chapters/03-learning-the-next-token.md`.
- Evidence status: formula, geometric-mean interpretation, and tokenizer-unit
  counterexample are analytically demonstrated; executable verification remains
  pending.
- Precision risks and required caveats: Use natural logs so `PPL=exp(mean NLL)`;
  do not call perplexity the literal number of available tokens except in the
  equal-probability teaching case; compare only under the same tokenizer, target
  mask, and evaluation distribution; distinguish population entropy from finite
  test loss; do not imply that lower perplexity alone proves better capability.
- Dependencies: verified mean-loss calculation, equal-choice example, unequal
  distribution counterexample, and tokenizer-comparison limitation.
- Suggested destination: Chapter 3 and course site.
- Next decision: Discuss after the perplexity lesson; if approved, Mac Studio
  decides whether it is the final act of `ANIM-CE-001` or a separate short.

### CAND-ANIM-011 — Reusing depth without pretending compute is free

- Source: user reminder plus agent automatic mathematics trigger
- Proposed during: Day 4 as a future Chapter 5 module
- State: discuss
- Learning objective: Show how one parameterized block or stack can be applied
  repeatedly, increasing effective depth and compute without duplicating its
  stored weights; distinguish fixed recurrence from adaptive token-level depth
  and from visible chain-of-thought tokens.
- Why motion is better than a static figure: The same weights must retain their
  identity while successive hidden states revisit them. Independent counters
  must make clear that parameter storage can stay fixed while layer applications,
  latency, and transformation depth grow.
- Moving objects and stable anchors: Keep the recurrent block and its parameter
  label fixed. Move $s_0,s_1,\ldots,s_r$ through it under
  $s_j=R(e,s_{j-1};\theta_R)$; update
  $L_{\mathrm{effective}}=L_P+rL_R+L_C$ as $r$ changes while the recurrent
  parameter counter remains fixed. In a second act, let token positions exit at
  different depths under a router, then contrast hidden-state loops with
  separately emitted reasoning tokens.
- Canonical source material:
  `learning_artifacts/day-04-attention-and-causal-information-boundary/future-recurrent-depth-and-looped-transformers.md`;
  [*Mixture-of-Recursions*](https://arxiv.org/abs/2507.10524) for adaptive
  token-level routing; the future Chapter 5 treatment; and the planned Day 7
  comparison.
- Evidence status: primary papers and the architectural accounting identity are
  captured; no local implementation or controlled experiment exists yet.
- Precision risks and required caveats: Do not say that effective depth is model
  parameter size; do not promise linear quality gains; do not call shared and
  untied layers equivalent; do not equate latent recurrence with hidden or
  suppressed chain-of-thought; do not present the reported Astra architecture as
  verified.
- Dependencies: complete the ordinary decoder, implement the optional recurrent
  variant, and verify parameter-, compute-, and wall-clock-accounting examples.
- Suggested destination: Chapter 5 and course site; possible later architecture
  article only after the canonical treatment is stable.
- Next decision: Revisit during Days 6–7. Production requires explicit learner
  approval and remains on the Mac Studio.

## Two-way proposal mechanism

### Agent-suggested

During a lesson or course-development session, the agent should actively suggest
an animation when motion would materially clarify at least one of these:

- a state transition or ordered computation;
- object identity across several representations;
- information, gradient, or mask flow;
- competition among alternatives;
- a failure that emerges over time;
- a relationship that becomes misleading when reduced to a static figure.

The suggestion should be short and concrete: name the learning objective, the
objects that move, the decisive transition, and the current evidence boundary.
Do not propose animation merely to decorate a section.

### Automatic mathematics trigger

Do not wait for Dongxi to request an animation when explicit mathematics is
central to an LLM mechanism. Automatically record a candidate for:

- objectives and losses;
- probability transformations and sampling distributions;
- analytical gradients and credit assignment;
- tensor operations whose shapes or axes change;
- causal, attention, padding, or loss-mask mathematics;
- optimization updates and training dynamics;
- multi-step derivations whose intermediate identities must remain visible.

The trigger applies to important mechanisms, not every decorative equation.
Consolidate equations that form one argument into one animation concept, and
check the queue before adding a new ID. Each automatically captured candidate
must state the canonical equation, why motion helps, current evidence state,
precision risks, and the derivation or experiment required before production.
Use source `agent` unless the learner or roadmap originated the idea.

Automatic candidate capture is not production approval. The candidate remains
`suggested` or `discuss` until Dongxi explicitly approves it, and all production
and rendering remains on the Mac Studio.

### User-suggested

Dongxi can propose an idea at any time with ordinary language, for example:

> Animation idea: show how each next token becomes the training target for the
> preceding position.

The agent records it here with `Source: user`, links it to the relevant chapter
or learning artifact, and identifies any conceptual or empirical dependency.

## States and approval gate

1. `suggested` — captured without evaluation.
2. `discuss` — the mechanism, scope, and learning value need joint review.
3. `approved` — Dongxi has approved production; create or update the `ANIM-*`
   task packet in `LEARNING_MEMORY.md`.
4. `producing` — editable source and review media are being created.
5. `review` — content accuracy and visual quality are ready for inspection.
6. `done` — approved source, render command, media, metadata, and integration
   location are preserved.
7. `deferred` or `rejected` — retain the reason and any reconsideration trigger.

No candidate moves from `discuss` to `approved` without Dongxi's explicit
decision. Approval of the concept does not approve publication.

## Proposal template

```markdown
### CAND-ANIM-NNN — short mechanism name

- Source: user | agent | roadmap
- Proposed during: Day NN / chapter / article
- State: suggested | discuss | approved | producing | review | done | deferred | rejected
- Learning objective:
- Why motion is better than a static figure:
- Moving objects and stable anchors:
- Canonical source material:
- Evidence status: conceptual | executable teaching example | measured result
- Precision risks and required caveats:
- Dependencies:
- Suggested destination: book | lab | X article | course site
- Next decision:
```

## Scope discipline

The `v0.1` target remains four excellent signature animations. Additional ideas
may become short supporting animations or move to the `v0.2` backlog. Recording
an idea is cheap; production competes for course-development time and requires an
explicit priority decision.
