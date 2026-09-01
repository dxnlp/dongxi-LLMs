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
| `CAND-ANIM-001` | Roadmap + agent | Chapter 3 | Input tokens → causal logits → probabilities, while each next-token target aligns with the preceding position → per-position loss | discuss | Decide during Day 3 whether this should expand `ANIM-CE-001` or become a separate animation |
| `CAND-ANIM-002` | User | Chapter 3 | Place negative log-likelihood and cross-entropy in the LLM training context: target probability → per-token NLL → masked aggregation across next-token positions → cross-entropy | approved; promoted into `ANIM-CE-001` | Complete the Day 3 derivation and produce only on the Mac Studio |

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
