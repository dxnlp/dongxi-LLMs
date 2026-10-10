# Book editorial implementation — 2026-10-10

Status: verified; all Phases 0–5 complete. Goal: `BOOK-EDITORIAL-2026-10`.

The [revised editorial plan](../../docs/BOOK_EDITORIAL_PLAN.md) is implemented.
The fifteen chapters, sixteen labs, fifteen solution guides, shared notation,
reproduction appendix and all seventy-six notebook routes now form the reviewed
reader pathway. Learner Day 9 and all historical model evidence remain unchanged.
The learner requested autonomous completion on Spark without routine review questions.

## Source identity and execution

Base commit: `7f5175ba03ed11ff778a067748af7b6e5ded4f58`, branch `main`. Initial local work was the
untracked plan; fetch confirmed no divergence and that work was preserved.
This is a local edited working tree, not a claim of a new commit or publication.
The [final CPU manifest](2026-10-10-book-editorial-pass/run-01/final-cpu/cpu-verification.json) binds exact
SHA-256 identities for 241 executable sources,
136 lesson inputs, the existing lock and manifest.
No executable source or lesson input changed during or after that run.
Final tracking/report updates are separate from the frozen lesson inputs.
The [completion receipt](2026-10-10-book-editorial-pass/run-01/completion.json) binds the final review receipts.

Actual host: `spark-aa66`, Linux aarch64, visible NVIDIA GB10. A new temporary CPU
environment resolved the existing `uv.lock`; the shared GPU environment was unchanged.
Interpreter: `/tmp/dongxi-book-editorial.fBhWhc/venv/bin/python`; kernel: `dongxi-book-editorial`.
CPU PyTorch: `2.14.1+cpu`. All fresh kernels matched the declared
environment prefix and had CUDA unavailable. Available host memory was sampled
before each notebook; the minimum observed was 116.914 GiB
against the declared 25 GiB reserve. This is sampled evidence, not continuous telemetry.
See the [environment record](2026-10-10-book-editorial-pass/run-01/environment.json) and fresh kernel identities.

The actual invocation was:

```bash
JUPYTER_PATH=/tmp/dongxi-book-editorial.fBhWhc/kernel/share/jupyter \
  /tmp/dongxi-book-editorial.fBhWhc/venv/bin/python scripts/run_cpu_verification.py \
  --kernel dongxi-book-editorial \
  --output experiments/reports/2026-10-10-book-editorial-pass/run-01/final-cpu \
  --full-notebooks --command-timeout 1800 --host-reserve-gib 25
```

## Phase acceptance

| Phase | Status | Final evidence |
|---|---|---|
| 0 — Guardrails | Verified | Frozen scanner v1.1 baseline, exact exemptions and repair map; ten focused prose tests |
| 1 — Narrative extraction | Verified | All fifteen chapters/solutions independently reviewed; relocation and exercise maps; mechanisms, failures and unrun boundaries retained |
| 2 — Evidence and depth | Verified | Canonical SFT/PPO/RLOO code, retained figures/raw answers, calculated attention trace, complete first G8 group and separate checkpoint ancestry |
| 3 — Lab routes | Verified | Sixteen labs route all seventy-six notebooks with prediction, checkpoint, intervention, boundary and deliverable; four focused runbooks |
| 4 — Coherence | Verified | Binding symbols/shapes; all seventy-six notebook contracts; 338 prompts paired with 289 existing references and adjacent explanations |
| 5 — Final verification | Verified | Complete current-source suite, all seventy-six fresh notebooks, preservation checks and manual reader/equation/figure/notebook review |

## Final local verification

All six commands returned exit 0 without a timeout:

| Command | Exit | Seconds | Evidence |
|---|---:|---:|---|
| tests | 0 | 216.671 | [Log](2026-10-10-book-editorial-pass/run-01/final-cpu/tests.log) |
| math | 0 | 0.071 | [Log](2026-10-10-book-editorial-pass/run-01/final-cpu/math.log) |
| prose | 0 | 0.161 | [Log](2026-10-10-book-editorial-pass/run-01/final-cpu/prose.log) |
| routes | 0 | 0.077 | [Log](2026-10-10-book-editorial-pass/run-01/final-cpu/routes.log) |
| notebook-contracts | 0 | 0.343 | [Log](2026-10-10-book-editorial-pass/run-01/final-cpu/notebook-contracts.log) |
| notebooks | 0 | 165.356 | [Log](2026-10-10-book-editorial-pass/run-01/final-cpu/notebooks.log) |

The unit suite passed **1,558 tests**, including the twenty-six focused
editorial checker tests. The [fresh notebook manifest](2026-10-10-book-editorial-pass/run-01/final-cpu/notebooks/manifest.json)
retains **76 successful notebooks, 539 source code cells,
535 executed cells, 210 images and
4 preserved unfinished learner cells**.
Only the explicit incomplete learner scaffolds were skipped in execution copies;
source notebooks were not overwritten with these fresh outputs. There were no
notebook execution failures. The earlier October 5 receipts remain evidence for
their original revision and were not reused as this pass.

## Before and after

The frozen v1.1 scanner covers fifty-three files in the five declared book areas.
Counts describe its matcher and scope, not an unrestricted semantic quality score.

| Measure | Before | Final |
|---|---:|---:|
| F2 mechanical joins: matches / affected lines | 1,111 / 759 | 0 / 0 |
| F1 narrative candidates: matches / affected lines | 42 / 41 | 0 / 0 unresolved |
| Exact contextual narrative exemptions | 0 | 2 on 2 lines |
| Notebooks with Prediction heading | 10 / 76 | 76 / 76 |
| Notebooks with Checkpoint heading | 7 / 76 | 76 / 76 |
| Notebooks with Reference solution heading | 29 / 76 | 76 / 76 |

Existing plain-text predictions and solutions often predated the heading pass.
Heading coverage alone does not establish teaching completeness. The final
[review ledger](2026-10-10-book-editorial-pass/run-01/notebook-contract-review.json) classifies all 829 original
Markdown cells, records 338 prompt pairings and binds current prompt, marker,
explanation and complete code hashes. The [final static check](2026-10-10-book-editorial-pass/run-01/notebook-contract-check-v2.json)
verifies declared adjacency and preservation; semantic sufficiency is the
recorded content review, not something inferred from headings.

The [mechanical repair map](2026-10-10-book-editorial-pass/run-01/mechanical-repairs.json) records 1,064 spaces
and forty-seven exact literal-ID/CLI formatting changes across thirty-eight files.
Code, math and link targets were excluded from automatic spacing changes.
The two retained dates in Chapter 5 describe pinned architecture/frontier
provenance, with exact contextual reasons in `scripts/book_prose_exemptions.json`.
No blanket chapter exemption was used.

All lab bodies are 24–34 lines, with no line-limit exceptions. Total line counts
include the collapsed command footer; these are measurements rather than quotas.

| Lab file | Original total | Final total | Final body |
|---|---:|---:|---:|
| 01-evidence-before-optimization.md | 43 | 41 | 26 |
| 02-text-tokens-and-embeddings.md | 34 | 40 | 25 |
| 03-learning-the-next-token.md | 33 | 40 | 25 |
| 04-attention-and-the-causal-information-boundary.md | 34 | 40 | 25 |
| 05-building-a-modern-decoder.md | 188 | 49 | 34 |
| 06-reading-a-pretraining-run.md | 255 | 45 | 30 |
| 07-evaluation-is-a-contract.md | 254 | 42 | 27 |
| 08-instruction-data-as-an-interface.md | 56 | 41 | 26 |
| 09-supervised-fine-tuning.md | 253 | 41 | 26 |
| 10-preferences-and-reward-models.md | 82 | 42 | 27 |
| 11-direct-preference-optimization.md | 299 | 41 | 26 |
| 12-language-generation-as-a-policy.md | 55 | 43 | 28 |
| 13-group-relative-policy-optimization.md | 238 | 43 | 28 |
| 14-when-optimization-goes-wrong.md | 153 | 41 | 26 |
| 15-distill-evaluate-and-defend.md | 86 | 44 | 29 |
| 06-pretraining-as-a-controlled-system.md | 44 | 39 | 24 |

## Reader pathway and evidence review

Operational recovery, journaling, supervisor and exact-command detail now routes
through Appendix D D.6 and the four focused runbooks. Chapters retain the mechanism,
conceptual question, useful example and primary evidence. The final
[all-chapter independent review](2026-10-10-book-editorial-pass/run-01/independent-phase1-phase2-review.json)
and [integration review](2026-10-10-book-editorial-pass/run-01/integration-review.json) cover all fifteen chapters,
original question/answer mappings and 741 local book/runbook targets.
All 297 original explicitly numbered solution headings are retained across all
fifteen guides; moved operational clauses retain their original exercise IDs.
The notebook navigation review resolves 313 local targets. All reviewed targets
resolve, including heading fragments where applicable.

Relocation detail is retained in
[foundations map](2026-10-10-book-editorial-pass/run-01/foundations-relocation-map.md),
[depth map](2026-10-10-book-editorial-pass/run-01/depth-relocation-map.md),
[operations map](2026-10-10-book-editorial-pass/run-01/relocations-operations.json),
[root map](2026-10-10-book-editorial-pass/run-01/root-relocations-and-edits.json) and
[lab/runbook map](2026-10-10-book-editorial-pass/run-01/phase-3-relocations.json).
Mixed questions in Chapters 7, 13 and 14 retain their conceptual parts at the
origin and their receipt-specific extensions in the matching guide.
Chapter 5 question 33 was already conceptual in the baseline; it remains exact,
while loader/digest detail moved from the body. This avoids manufacturing a split.

New reader predictions precede result reveals and do not claim historical
preregistration. Chapter 9 includes canonical SFT/accumulation code and the first
aligned raw Base/full/LoRA answers with stop status. Chapter 12 preserves all
three seeds, matched rollout exposure and 160 versus 480 gradient passes.
Chapter 13 shows the actual first G8 group, all eight responses, zero task rewards
and advantages, 76 valid actions, nonzero stored gradient and changed policy identity.
The [G8 projection](2026-10-10-book-editorial-pass/run-01/g8-first-group-projection.json) binds the unchanged raw record.
Checkpoint genealogy preserves separate story, Base and Instruct roots, failed
G4 supervision, separate export closure and incomplete later story checkpoints.

The [attention calculation](2026-10-10-book-editorial-pass/run-01/attention-forward-calculation.json) and its
canonical CPU reference agree at the declared tolerance and retain the causal
value intervention. This is worked arithmetic, not a trained-model result.
Manual review includes the local [equation proof](2026-10-10-book-editorial-pass/run-01/equation-preview.png),
existing architecture/objective figures and
[fresh notebook figure previews](2026-10-10-book-editorial-pass/run-01/manual-notebook-visual-review.json).
It checks labels, tensor layout, prediction-before-reveal and readable mechanism
explanations. This is representative local review, not GitHub/browser-rendering
evidence, an animation render or observation of learner understanding.

## Preservation, failures and scope exceptions

All 5,056 protected tracked reports, specifications,
archives and model-card evidence files retain their original Git blob identity.
Canonical model modules and original tests remain unchanged; the CPU orchestrator
is the sole changed original Python source. New code is limited to two narrow
editorial checkers and their focused tests.

The seventy-six registered notebooks retain all 539 original complete code objects,
including IDs with missing keys, source/order, metadata, counts and outputs.
Ten ignored checkpoint copies are outside the teaching route and remain entirely
unchanged. Together the baseline covers eighty-six notebook files and 644 code cells.
Independent Git object comparison and the immutable full-file fallback both pass.

The first final static CLI rejected the ledger's added role/note fields because it
compared whole dictionaries instead of protected identity fields. Its
[failed receipt](2026-10-10-book-editorial-pass/run-01/notebook-contract-check.json) is retained. The corrected
checker validates the protected fields and every role/note, with the separate
version 2 pass and sixteen checker tests. Content review also resolved mismatched
prompt/reference placement and the teacher-order/student-initialization explanation;
see the [independent late review](2026-10-10-book-editorial-pass/run-01/independent-late-notebook-review.json)
and [semantic correction record](2026-10-10-book-editorial-pass/run-01/notebook-semantic-corrections.json).
No failure is presented as a successful experiment.

Optional new Markdown exercise variants remain labeled unexecuted. Fresh execution
verifies the preserved complete reference code, not those proposed variants or
learner mastery. Four unfinished learner cells are preserved rather than completed
on the learner's behalf. No new model campaign, GPU job, shared-environment change,
Mac execution, hosted CI, animation production or external publication occurred.
Existing animation candidates received a mechanism extension only.

All six editorial phases are complete. The learner remains at Day 9; the next
learning action is the requested Spark story lesson using the retained evidence.
