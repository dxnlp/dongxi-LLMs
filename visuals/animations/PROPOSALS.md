# Animation Proposal Inbox

This file is the shared inbox for animation ideas suggested during learning and
course development. Suggestions may come from Dongxi or from a contributing
agent. A suggestion is not approval to produce media.

Approved animation tasks move to `LEARNING_MEMORY.md`, where they receive a
stable `ANIM-*` task ID, a complete task packet, dependencies, acceptance checks,
and machine ownership. This inbox retains deferred and rejected ideas so the
same proposal is not repeatedly rediscovered.

## Candidate queue

| Candidate ID | Source | Book placement | Mechanism | State | Dependency or next decision |
|---|---|---|---|---|---|
| `CAND-ANIM-001` | Roadmap + agent, automatic math trigger confirmed | Chapter 3 | Input tokens → contextual state `[D]` → dense projection against all `[V,D]` output rows → `[V]` logits → probabilities, while next-token targets align with preceding positions; optionally reveal the `T`-versus-`V` compute trade-off | discuss | Concept and large-vocabulary cost now introduced; after manual/PyTorch verification decide whether this should expand `ANIM-CE-001` or become a separate Mac Studio animation |
| `CAND-ANIM-002` | User | Chapter 3 | Place negative log-likelihood and cross-entropy in the LLM training context: target probability → per-token NLL → masked aggregation across next-token positions → cross-entropy | approved; promoted into `ANIM-CE-001` | Complete the Day 3 derivation and produce only on the Mac Studio |
| `CAND-ANIM-003` | User | Chapter 3 | Standard next-token update: predicted distribution `p` versus one-hot target `q` → logit gradient `p-q` → target score rises and non-target scores fall → repeated diverse examples shape a distribution | approved; promoted into `ANIM-NTP-001` | Complete gradient verification and tiny-model evidence; produce only on the Mac Studio |
| `CAND-ANIM-004` | User | Chapter 3 | Why negative log loss: sequence probability products become additive token surprise; confident errors retain strong gradients; expected log loss rewards matching the full data distribution | approved; promoted into `ANIM-LOGLOSS-001` | Verify all three mechanisms manually and in code; produce only on the Mac Studio |
| `CAND-ANIM-005` | Agent, automatic math trigger | Chapter 3 | Mean token cross-entropy → exponentiation → perplexity as an effective equal-choice branching factor, while preserving tokenizer and evaluation-distribution dependence | discuss | After derivation, decide whether to extend `ANIM-CE-001` or create a separate short; production only on Mac Studio after approval |

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
  `learning_artifacts/day-03-probabilities-and-next-token-loss/`; planned Chapter
  3 derivation and manual/PyTorch verification.
- Evidence status: concept introduced; executable evidence pending.
- Precision risks and required caveats: Do not present NLL and one-hot
  cross-entropy as different numerical objectives; distinguish sequence NLL sum
  from the common mean over valid tokens; exclude padding or ignored labels;
  retain natural logarithms; do not imply that low loss proves truthfulness.
- Dependencies: Day 3 derivation, causal label alignment, loss-mask denominator,
  and PyTorch agreement.
- Suggested destination: Chapter 3, course site, and the embedding-training
  animation handoff.
- Next decision: Mac Studio production begins only after the canonical Day 3
  derivation and evidence are committed.

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
  upward and non-target logits downward; then replay a small controlled stream
  whose target frequencies are visibly known.
- Canonical source material:
  `learning_artifacts/day-03-probabilities-and-next-token-loss/probability-as-competition-and-surprise.md`;
  planned Chapter 3 gradient derivation and tiny-model report.
- Evidence status: gradient mechanism introduced; manual/PyTorch verification and
  learned-frequency experiment pending.
- Precision risks and required caveats: Show gradient descent direction rather
  than confusing gradient sign with parameter motion; state that one-hot
  supervision does not mark unobserved alternatives as valid; do not imply that
  one update sets the target probability to one; distinguish a single local
  update from the expectation over a representative data distribution; use fixed
  verified numbers in the final render.
- Dependencies: analytical gradient, PyTorch agreement, optimizer-step check, and
  a tiny controlled target-frequency experiment.
- Suggested destination: Chapter 3, course site, and the later embedding-gradient
  animation sequence.
- Next decision: Produce and render only on the Mac Studio after Day 3 evidence is
  committed; decide there whether it is a standalone short animation or a
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
  product into additive NLL tiles. Act 2 synchronizes loss and gradient curves
  for log loss versus `1-p_target`. Act 3 fixes an empirical target distribution
  `q` and moves a model distribution `p` toward the expected-loss minimum.
- Canonical source material:
  `learning_artifacts/day-03-probabilities-and-next-token-loss/probability-as-competition-and-surprise.md`;
  planned Chapter 3 derivation and verification report.
- Evidence status: all three mechanisms introduced; numerical, gradient, and
  proper-scoring verification pending.
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
  branching factor.
- Canonical source material:
  `learning_artifacts/day-03-probabilities-and-next-token-loss/nll-cross-entropy-and-perplexity.md`;
  planned Chapter 3 derivation and verification.
- Evidence status: formula introduced; interpretation and executable verification
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
