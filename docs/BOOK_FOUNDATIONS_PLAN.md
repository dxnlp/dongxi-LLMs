# Book Foundations, Depth and Reader Experience Plan

Goal: `BOOK-FOUNDATIONS-2026-10`. Status: complete.
Authorized 2026-10-10: improve this aspect across all fifteen chapters and
complete the goal autonomously. Base: `b722d59cfc115ceca20d5e15b566d972d389737b`,
branch `main`. The original proposal is preserved in the
[baseline record](../experiments/reports/2026-10-10-book-foundations-pass/run-01/original-proposal.md).

## Purpose and boundaries

Make the book teach its argument: a reader should understand why an operation
is needed, follow its computation, predict a controlled change and interpret
what the result establishes. The previous editorial pass made the draft cleaner;
this pass develops explanatory depth, narrative continuity and curiosity.

Retain the fifteen chapters, twenty-eight days, all existing exercise IDs,
measured results, failed experiments and learner Day 9. SFT and reinforcement
learning receive the greatest depth, following decision D005. This authorizes
course development and bounded local numerical/reference verification, not new
model-scale campaigns, publication, hosted CI, animation production or claims
of learner mastery. Spark is the actual execution host; CPU reference checks
use a new task-local environment with the existing teaching lock, never the
shared GPU environment.

## 1. Diagnose the real gap before adding material

The draft often presents a rule, equation and limitation before developing the
reader's concrete problem. Several explanations are present but compressed,
buried or disconnected from their notebook examples. Classify each intervention:

| Classification | Action |
|---|---|
| Missing concept or prerequisite | Teach it at first necessary use |
| Compressed reasoning | Expand the exact missing step or causal explanation |
| Poor placement or disconnected example | Move, integrate or promote existing material |
| Adequately covered | Reuse and link; do not create another primer |

The original claim of almost no small numbers was too broad. Chapter 3 already
contains numerical softmax, output gradients and perplexity; Chapter 4 contains
complete attention arithmetic; Chapter 5 contains normalization examples;
Chapter 12 already teaches baselines, critics, returns and bootstrapping.
The chapter-specific review ledger records what actually changes and why.

Original size diagnostics counted raw Markdown whitespace tokens, including
code and tables: 78,721 across the chapters. Baselines total 10,814 lines and
original proposed chapter targets sum to 12,810, correcting the proposal's
11.9k-to-14.4k totals. These are inventory measures, not quality gates.
Counting the phrase “for example” does not count worked examples, and a fenced
text transcript is not executable code. No length, figure or caveat quotas apply.

## 2. The reader's learning arc

Use this arc where it helps the mechanism, adapting its pacing to the chapter:

**Concrete problem → reader prediction → worked mechanism → controlled change
→ explanation → scoped evidence → next question.**

Open with a difficulty or surprise, then develop it through the mathematics.
A prediction concerns behavior, information flow, gradient direction or a design
trade-off. Give its reference reasoning immediately afterward. Keep the chapter
readable without requiring the reader to open a notebook for a basic definition;
notebooks let the reader inspect and change the mechanism.

Use diagrams, small tables and short code listings when their representation
answers the current question. Choose calculations that expose an important
failure or distinction. Every new numerical claim must be independently
recomputed or tied to an existing runnable fixture; record the source and
verification. Figures retain a source, regeneration command, informative alt
text and adjacent reading guide. Promote adequate saved companion figures before
creating new assets. Static structure diagrams may be repository-native SVG or
Mermaid; measured charts retain their numerical provenance.

The original five-part explanatory unit is an editing aid, not a required
paragraph template. Rewrite, reorder or merge explanatory passages when that
creates a better progression. Preserve meaningful derivations and original
questions, answer mappings and empirical boundaries. Distinct mathematical
assumptions belong beside the equation they qualify. Remove repetitive boundary
prose; do not impose one caveat per section or a percentage ceiling.

## 3. Connected examples and prerequisite flow

Three strands connect the narrative:

1. **Text to learning:** follow a short, explicitly illustrative sequence
   through token IDs, embeddings, contextual states, vocabulary probabilities,
   loss and a parameter update in Chapters 2–6.
2. **Language to an assistant:** follow desired output through serialization,
   masks, stopping, SFT and preferences in Chapters 7–11. The retained
   `item100` Base/full/LoRA responses give a concrete interface failure.
3. **Judgment to policy improvement:** follow a small response distribution
   through reward, score-function gradients, baselines, reuse, group-relative
   updates, failure diagnosis and teacher transfer in Chapters 10–15.

These strands are teaching continuity, not a fabricated checkpoint ancestry.
Show the generic post-training choices separately from the measured campaign's
actual roots. A reference is commonly an SFT policy, but the acquired Instruct
RLVR root must retain its own identity.

Introduce probability, log loss, entropy and basic KL in Chapter 3 before their
dependent arguments. Chapter 11 applies KL to current/reference policies;
Chapter 12 introduces sequential-decision vocabulary as each term is needed.
Later chapters briefly reactivate prerequisites rather than forcing backward
or forward glossary searches. Extend the notation guide with units, shapes and
scoped aliases; local reuse of a symbol is allowed after its meaning is defined.

## 4. All-chapter implementation targets

| Ch | Driving question and intended improvement | Existing material to reuse |
|---:|---|---|
| 1 | What would a successful run actually establish? Develop a claim/evidence decision and a sized memory ledger. | Smoke receipts, checkpoint-interface and experiment-design companions |
| 2 | How can categorical IDs become trainable representations? Connect a short sequence, repeated lookup gradients and vocabulary cost. | Tokenizer/embedding fixtures, existing BPE and lookup explanations |
| 3 | How can positive loss coexist with a zero expected gradient? Connect probability competition, a worked target, the update and entropy/KL. | Three-logit example, 70/30 population, output gradients, perplexity contrast |
| 4 | How does a position retrieve useful past information without seeing its answer? Connect routing, weighted values, scaling and a broken mask. | Complete three-position arithmetic, gradient and cache notebooks |
| 5 | Why does composing these operations produce a decoder? Strengthen residual, normalization and MLP causal explanations and connect the whole forward trace. | Existing whole-model maps, numeric norm examples, Day 5 assembly, Qwen cost formula |
| 6 | What can a falling training curve conceal? Develop optimizer intuition, regularization and a worked diagnosis of loss versus generated stories. | Adam sign reversal, schedule, retained TinyStories evidence and Day 8/9 plots |
| 7 | How do we decide whether behavior improved? Follow responses through correctness, stopping, paired comparison and uncertainty. | Pass@k calculation, paired-bootstrap companion, frozen evaluation cases |
| 8 | How does desired behavior become a training sequence? Follow messages, target ownership, loss/visibility boundaries and learned stopping. | Symbolic transcript arrays, packing heatmap, 10/90 token mixture |
| 9 | Why can the right answer prefix still fail? Follow a masked answer through loss/gradient, then interpret full/LoRA outputs and update-space constraints. | Canonical SFT loop, existing LoRA initialization/merge checks, exact retained responses |
| 10 | What can a comparison tell us that a demonstration cannot? Work one preference pair through score margin, loss and a shortcut failure. | Bradley–Terry fixtures, text reward lab and adversarial/judge cases |
| 11 | Can pair odds improve while both answers become less likely? Expand the exponential-tilt bridge and follow three answers through the reference-relative objective. | Stationarity notebook, existing gradient and likelihood-shifting explanation |
| 12 | How does a final reward change the next-token policy? Sustain one small reward/gradient example through baselines, sampling reuse and clipping; connect to sequential credit and critics. | Enumerated policy lab, probability accounting, categorical comparison and GAE companions |
| 13 | Why can all-wrong groups give zero learning signal? Follow one mixed and one constant group through reward, centering, scaling and token loss. | Population/sample-std example, weighting and controlled-objective notebooks, actual G8 group |
| 14 | Which mechanism caused the apparent improvement to fail? Build a diagnostic decision from reward shortcuts, missing exploration and sampler/objective mismatch. | Exploitable reward, exact KL-gradient controls and retained negative native outcomes |
| 15 | What does the student actually learn from a teacher? Connect candidate selection, hard/soft targets, temperature and prefix distribution to a defensible comparison. | Existing distillation derivatives, response-level cases, genealogy and final evidence tables |

## 5. Technical corrections required in the new explanations

- $e^{-L_{\mathrm{mean}}}$ is geometric-mean observed-target probability under
  the stated mask/aggregation, not each token's probability.
- Shared logit shifts leave normalized softmax invariant; exponentiation itself
  is not shift invariant. Vocabulary mass concentration requires a declared
  distribution and threshold, not an unsupported general claim.
- Finite one-hot queries/keys yield soft attention. An exact hard lookup appears
  as a unique-maximum limit with growing score scale. Independent coordinate
  assumptions explain score variance, not fixed score values.
- Memory costs depend on dtype, gradient storage, optimizer moments and actual
  master-copy policy. Parameter shares depend on the architecture: the retained
  Qwen configuration's MLP is about 44% of total unique parameters, not a universal
  two thirds. Scope normalization, initialization, decay and hyperparameter choices.
- Adam bias correction addresses zero-initialization bias; it is not a complete
  justification of warmup. The current teaching fixture decays all trainable
  parameters; other grouping conventions are separate recipe choices.
- DPO likelihoods are bounded; separable pair margins can diverge even with a
  frozen reference. Shared-parameter updates do not guarantee that chosen absolute
  likelihood rises or rejected likelihood falls. Explain reference-relative odds.
- A critic is not necessarily another policy-sized model. An iid group mean can
  estimate prompt-level behavior-policy value while self-inclusion still changes
  the gradient estimator. Random standardization introduces further dependence.
- $k_3$ is nonnegative and, with the declared support/sampling conditions, unbiased
  for the KL value. It does not universally reduce variance. Differentiable
  sampled estimates retain their explicit behavior/detachment contract.
- Soft targets can express conditional alternatives, uncertainty and teacher bias;
  semantic similarity is not guaranteed. Omit an underspecified information-bit
  comparison. Forward-cover/reverse-mode-seek illustrations need a constrained
  student family; unrestricted categorical objectives both minimize at equality.
- Paired evaluation uses aligned disagreements; overlapping separate Wilson
  intervals do not replace that analysis. Keep terminated and collector-capped
  trajectories distinct in returns and GAE.

## 6. Implementation and acceptance

### A. Baseline and teaching approach

Preserve the original proposal, baseline file identities and historical evidence.
Review existing sources, then develop Chapter 3's early learning example,
Chapters 8–9's interface story and Chapter 12's reward-to-update progression.
Use their approach for the remaining chapters, with independent content review.
These are implementation prototypes, not new empirical training experiments.

### B. Complete all fifteen chapters

Record each chapter's driving question, diagnosed gaps, reused source, worked
example, controlled change, visual reading guide when relevant and transition.
Prioritize post-training depth. Integrate voice improvements during drafting.
Update shared notation and preface to explain the connected reading route.
Existing numbered exercises and their solutions remain mapped and valid; new
predictions receive adjacent reference reasoning.

### C. Independent reader and technical review

For every chapter verify that a reader with the preface prerequisites can:

- explain why the central operation is needed;
- trace the worked computation with stated symbols, units and shapes;
- predict and explain one controlled change;
- distinguish the mechanism result from the empirical conclusion;
- follow the transition and use the existing companion to explore further.

A content-review ledger binds these judgments to final source hashes. It records
agent review, not observed learner understanding. Correct mathematical examples
independently; source-derived numbers retain their exact evidence identity.

### D. Local verification and durable closure

Run existing book math/prose, route/link and notebook-contract checks. Preserve
all notebook code, learner cells, metadata and outputs byte-for-byte; this pass
uses existing cells and standalone assets, so no notebook augmentation contract
is needed. Do not change old checker baselines or historical receipts.

Freeze lesson sources, then run the established isolated CPU verification panel
including all seventy-six fresh notebook references, with offline/model-free
settings and the declared 25 GiB host reserve. It supplies actual execution
receipts for this revision. Numerical content review and readable local math/
figure inspection are distinct from lint and execution. Bind any new figure
source/output separately if it falls outside the runner's inventory.

Write a new foundations report with all-chapter acceptance, actual checks,
source identities, preservation comparisons and limitations. Update `BOOK.md`,
`PROGRESS.md`, `LEARNING_MEMORY.md` and existing animation candidates as needed;
keep new mathematical candidates proposed, with Mac production dependency.
Mark the goal complete only after every chapter and required check is accepted.

## 7. Completed implementation — 2026-10-10

All fifteen chapters and the shared reader guides are revised and independently
accepted. The [foundations report](../experiments/reports/2026-10-10-book-foundations-pass.md)
links author ledgers, numerical replay, exact-source peer review and the
[final isolated CPU receipt](../experiments/reports/2026-10-10-book-foundations-pass/run-01/final-cpu/cpu-verification.json).

- Connected problems, worked mechanisms, controlled changes and adjacent
  reference reasoning are accepted across all fifteen chapters.
- Five new standalone listings and three numerical verification groups pass;
  sixteen existing figures are promoted with sources and reading guides.
- All 1,558 CPU tests pass. Book mathematics covers 1,829 expressions with
  zero issues; prose, routes and notebook contracts pass.
- All 76 fresh notebook references pass: 539 source code cells, 535 executed
  reference cells, 210 emitted images and four preserved unfinished learner cells.
- The 5,893 protected baseline files remain unchanged. All 297 original
  exercise/solution IDs remain mapped; source-bound review and final preservation
  checks are retained in the report's new evidence directory.

This closes the book-production goal. Learner Day 9, empirical model outcomes,
the fifteen-chapter/twenty-eight-day route and media-production assignments
retain their own evidence and status.
