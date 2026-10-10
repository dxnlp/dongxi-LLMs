# Book foundations and reader experience — 2026-10-10

Goal: `BOOK-FOUNDATIONS-2026-10`. Status: complete.

The learner authorized improving explanatory depth and reading experience
across all fifteen chapters and pursuing the goal to completion without routine
confirmation questions. The [revised plan](../../docs/BOOK_FOUNDATIONS_PLAN.md)
uses concrete questions, connected worked examples, first-use prerequisites and
controlled changes. The original proposal is preserved separately.

## Baseline and execution scope

Base commit: `b722d59cfc115ceca20d5e15b566d972d389737b`, branch `main`.
The initial tree contained only the untracked foundations plan. Fetch confirmed
zero divergence; that local work was preserved. The
[baseline](2026-10-10-book-foundations-pass/run-01/baseline.json) binds 5,893
protected notebook/code/asset/experiment files and 54 book Markdown files.
Historical evidence and earlier verification receipts remain evidence for their
own revisions.

Actual host: `spark-aa66`, Linux aarch64, NVIDIA GB10 visible. A new goal-local
CPU environment was created with the existing lock; the
[environment record](2026-10-10-book-foundations-pass/run-01/environment.json)
retains its interpreter and temporary-prefix kernel. Shared GPU dependencies
were unchanged. This is course development and local model-free verification,
not a new GPU campaign, Mac verification, hosted CI or public publication.

## What changed

The review found compressed or disconnected explanations as well as missing
prerequisites. It replaced the original length and caveat quotas with a reader
contract: understand the problem, trace its mechanism, predict a controlled
change and identify what the evidence supports. Existing companions supplied
the mechanisms and figures; this pass connects them in the chapter prose.

Three illustrative strands run through the book: text becomes a learned
prediction; an answer becomes a supervised interface and a preference; a
judgment becomes a policy update and a teacher signal. These are teaching
connections, separate from the actual campaign's checkpoint ancestry.

| Chapter | Concrete development |
|---:|---|
| 1 | Claim/evidence decisions, floating-point reduction order and explicitly scoped optimizer memory ledgers |
| 2 | A repeated `the` token follows lookup, shared gradients and a visible SGD update |
| 3 | A next-token classifier follows probabilities, weight gradients and the improved target probability; entropy and KL explain the irreducible loss |
| 4 | Soft retrieval, score scaling and routing gradients lead to a deliberately broken causal mask |
| 5 | Residual and normalization interventions lead to a complete decoder trace and the pinned Qwen configuration's parameter budget |
| 6 | Optimizer history, warmup and distinct regularization mechanisms lead to diagnosis of the retained story outputs |
| 7 | Aligned successes and failures lead to paired uncertainty and a slice-weighting reversal |
| 8 | One answer follows serialization, token ownership, loss masking, attention visibility and learned stopping |
| 9 | Exact Base/full/LoRA responses motivate a masked answer loss, END removal and LoRA's constrained update space |
| 10 | One comparison follows its score margin, gradient, soft preference and length shortcut |
| 11 | The reward-tilted optimum leads to a three-answer counterexample separating pair odds from absolute likelihood |
| 12 | One reward distribution follows score gradients, baseline noise, leave-one-out rewards, sample reuse, clipping and sequential credit |
| 13 | Mixed and constant groups expose self-inclusion, normalization, token weighting and missing learning signal |
| 14 | A reward shortcut, entropy, exploration and three KL-gradient contracts form a mechanism-based diagnosis |
| 15 | Selected text and soft teacher targets lead to temperature gradients, wrong-prefix teaching and the tail-KL decomposition |

The preface, reading guide, notation and mathematical appendix now support
these connections and define prerequisites near their first necessary use.
The DPO solution guide expands the existing counterexample without changing
question numbering. Mathematical animation opportunities extend existing
proposals; they remain unrendered, with their Mac production dependency.

## Numerical and visual review

New arithmetic is checked against independent formulas and existing canonical
APIs/autograd in the author ledgers and a
[fresh root replay](2026-10-10-book-foundations-pass/run-01/numerical-replay.json).
The [five new standalone Python listings](2026-10-10-book-foundations-pass/run-01/listing-and-figure-review.json)
were also executed verbatim. Qwen accounting uses a cached pinned JSON config
and safetensors header; it does not load a new model. Runtime tying refers to
the existing same-revision inspection, not a conclusion from the file header.

Sixteen existing companion figures are promoted into the argument with reading
guides and regeneration sources. The
[visual review](2026-10-10-book-foundations-pass/run-01/visual-review.json)
records saved-figure inspection and seven locally typeset representative
equations. Local MathText inspection is separate from Markdown source lint;
no GitHub/browser render or MathJax execution is claimed.

The first root numerical replay compared the expanded operations script with
the earlier version-one receipt. Both executed programs exited successfully;
the comparison failed on newly added result fields, with no numeric
disagreement found. The
[failed attempt](2026-10-10-book-foundations-pass/run-01/numerical-replay-attempt-01.json)
and its logs are preserved. Replaying the synchronized final script against
the version-two receipt passed all three calculation groups.

## Acceptance and final verification

All fifteen chapters and the shared guides passed independent read-only peer
review. The receipts bind exact final source hashes:
[Chapters 1–5 and shared guides](2026-10-10-book-foundations-pass/run-01/operations-peer-review.json),
[Chapters 6–7 and 12–14](2026-10-10-book-foundations-pass/run-01/depth-peer-review.json),
and [Chapters 8–11, 15 and solution 11](2026-10-10-book-foundations-pass/run-01/routes-peer-review.json).
The [root reader review](2026-10-10-book-foundations-pass/run-01/root-reader-review.json)
records the chapter-level reason for acceptance. Review also corrected the
logit-versus-normalized-probability prediction, first-use terminology, one
counterfactual reward description and the appendix's derivative/norm conditions.

The [final frozen-source CPU panel](2026-10-10-book-foundations-pass/run-01/final-cpu/cpu-verification.json)
passed all six commands on Spark:

| Check | Actual result |
|---|---|
| CPU test suite | 1,558 tests passed |
| Book mathematics | 54 files, 1,829 expressions, zero issues |
| Strict narrative prose | Passed; no checker exemptions added or changed |
| Course routing | 15 chapters, 28 days, 76 registered notebooks; zero problems |
| Existing notebook contracts | 76 registered references reviewed; zero problems |
| Fresh notebook execution | All 76 passed; 539 source cells, 535 executed references, 210 images, four unfinished learner cells preserved |

The [notebook execution manifest](2026-10-10-book-foundations-pass/run-01/final-cpu/notebooks/manifest.json)
retains each actual output and kernel identity. Minimum observed available
host memory was 116.933 GiB, above the declared 25 GiB reserve. The runner
binds 241 Python source files and 136 lesson inputs; none changed during the
run. The unfinished scaffolds are skipped only in execution copies; canonical
notebook bytes, metadata, completed learner work and saved outputs are preserved.

The [integration audit](2026-10-10-book-foundations-pass/run-01/integration-check.json)
checks local file/fragment navigation, all 333 original numbered chapter lines,
297 original solution IDs and the 5,893 protected baseline identities. All
fifteen chapter sources changed; original exercises and answer mappings remain
valid. The [final acceptance consumer](2026-10-10-book-foundations-pass/run-01/acceptance-check.py)
also checks final author/peer hashes against current sources, the original
proposal, freeze identities and actual execution receipts. Its `completion.json`
binds new calculation/review/preview code and outputs outside the CPU runner's
inventory, along with final plan/tracker/report identities.

The book-production goal is complete. Reader review records agent judgments
and does not claim observed reader engagement or learner mastery. Learner Day 9,
empirical training outcomes and the fifteen-chapter/twenty-eight-day route remain
fixed. No GPU campaign, Mac verification, hosted CI, animation/video production
or public publication occurred.
