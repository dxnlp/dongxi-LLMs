# Independent campaign-assembler and remaining-criteria review

Date: 2026-10-05. Result: **PASS at the source and authored CPU-control boundary**. No native campaign assembler invocation, model/export rehash, GPU job, full suite, network or Git action was performed by this reviewer.

## Frozen candidate and actual controls

| File | SHA-256 before and after controls |
| --- | --- |
| `scripts/assemble_native_campaign_evidence.py` | `e9262b247415cc5a55d030038ad92588caf9aa640f42ee23d0e85d89819cad0a` |
| `tests/test_native_campaign_evidence.py` | `f8dfce45b24d375e9704e3a68f49ec1649e8a84f3737c31cea3bc86eed50e371` |

```bash
env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dongxi/dgx-spark-dongxi/.venv/bin/python \
  -m unittest discover -s tests -p test_native_campaign_evidence.py -v
```

Working directory: `/home/dongxi/dongxi_ai/Dongxi_LLMs`. Actual result: **33 tests passed, exit 0, unittest elapsed 0.259 seconds**. These are small explicitly authored temporary CPU fixtures, not actual native campaign outcomes.

## Reviewed evidence semantics

- The immutable original 45-row archive is checked against its existing exact logical digest, row order, optional IDs, original budgets, dependencies, interventions and criteria. Only declared documentation/execution-scope differences are retained separately.
- DPO failed runs `01` and `02` stay in the closed catalog as unselected historical observations. Their actual native supervision, failed/unknown native exits, cleanup errors, spending and orphan directory inventories remain retainable. An outer adapter exit is not substituted for an unobserved native exit.
- DPO `03` is the predeclared selected replay source, not a fabricated pass. Missing run-03 evidence remains missing; historical replay observations cannot populate its smoke/recovery stage observations. The original pilot remains `run-01`.
- Retained `snapshot_hash_workers`, CPU settings and original producer source bindings are execution provenance, separately labeled from current-source bindings. They are not a scientific intervention, ledger-identity replacement or quota change.
- The actual source-bound story rating join binds publication inputs, packet/codebook, both supplied AI-rating documents and the completed offline report receipt. Authored or unjoined ratings are not promoted to scientific slices.
- First-400 story evidence remains explicitly bounded; later checkpoints and full-14,000-horizon completion remain missing. Common preference/reasoning evidence is not pooled into a universal capability percentage.
- The assembler records actual inventories/genealogy and cost boundaries without declaring package completion. Missing evidence and directory layouts are checked again at closure; existing source/result artifacts are not overwritten.

No blocking inconsistency was found in this candidate extension at the declared scope. Actual assembly must occur after owned jobs stop and must pass its own immutable closure checks; this review is not that receipt.

## Original remaining-package checklist

The `acceptance` and `dependencies` arrays for DXI-03, DXI-17 and DXI-18 were compared directly with `experiments/reports/2026-10-05-native-base-profile/original-acceptance-before-integration.json`: all six comparisons are exactly equal. The following checklist does not edit or strengthen those criteria.

| Package | Remaining evidence/integration at review time |
| --- | --- |
| DXI-03 | Finish the already declared, branch-gated preference/reasoning actual comparisons and assemble current 45-row observations, checkpoint identities, genealogy, exits, failures and costs. Preserve the accepted bounded first-400 story comparison and actual 192-record AI-rating result without inferring the unfinished 14,000-update schedule, absent later checkpoints or human consensus. |
| DXI-17 | After final integration, refresh current-source Linux tests, math/navigation/schema checks and the explicit all-76 CPU reference route with pre/post source hashes and retained failures. Label Linux actual versus Mac unverified, and CI source/local checks versus hosted execution separately. Neither successful Mac execution nor hosted CI success is an added literal original-criterion requirement. |
| DXI-18 | Bring final actual preference/reasoning branches into Chapters 11/13/15 and their adjacent solutions/labs, cards and evidence map; then independently defend actual genealogy/capability/cost/regression tables, unresolved scope and dependency states. Preserve 15 chapters/28 days, learner Day 9, negative results, tested live-visual source versus unobserved host/learner use, and separately approved Mac media production. |

Some reader-facing status summaries still describe earlier snapshots as current: `docs/COURSE_IMPROVEMENT_PLAN.md` lines 374–381 says actual story continuations/reviews remain missing and refers to a Mac dependency; `docs/COURSE_EVIDENCE_MAP.md` lines 143–148 says real story reviews remain empty; DXI-03's long `verified_scope` history still includes the superseded Mac/hosted-before-pilot interpretation. `docs/UPGRADE_ACCEPTANCE_REVIEW.md` likewise opens with short-run/current-snapshot language before later historical sections. Retain the historical receipts, but add an explicit current measured-snapshot prefix during final integration rather than allowing these older summaries to contradict accepted first-400/rating evidence or the original-text platform clarification. No trackers or course files were edited by this reviewer.

These packages remain unfinished pending their actual remaining work and final independent review. Native failures, unexecuted stages, unverified platforms and learner mastery cannot be replaced by this source review.
