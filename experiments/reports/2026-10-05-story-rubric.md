# Blinded story packets and offline ratings

The missing story-rating consumer is now implemented. It prepares anonymous
reading packets, preserves separate rater decisions/disagreements and computes
paired source-opening summaries only when the declared coverage is complete.
It never generates stories or assigns semantic scores. All real campaign
reviews remain absent; the fixtures are explicitly authored instrument controls.

The [specification](../specs/2026-10-05-story-rubric-evaluation.md) and
[fixture guide](../../fixtures/story-rubric/README.md) explain the frozen
contract, raw record schema, blinding limits and comparison policy. The
[module](../../src/dongxi_llms/story_rubric.py) and
[CLI](../../scripts/evaluate_story_ratings.py) use only standard-library
helpers; neither imports model libraries or executes untrusted story text.

## Actual empty-rating control

Root ran both commands from the repository, each with actual exit0:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/evaluate_story_ratings.py prepare --contract fixtures/story-rubric/contract.json --checkpoints fixtures/story-rubric/checkpoints.json --records fixtures/story-rubric/authored-records.jsonl --raters fixtures/story-rubric/raters.json --output experiments/reports/2026-10-05-story-rubric/packet-01
PYTHONDONTWRITEBYTECODE=1 python3 scripts/evaluate_story_ratings.py evaluate --bundle experiments/reports/2026-10-05-story-rubric/packet-01 --output experiments/reports/2026-10-05-story-rubric/empty-ratings-01 --compare authored-baseline-14000 authored-candidate-14000
```

The [packet receipt](2026-10-05-story-rubric/packet-01/receipt.json) binds
source/input bytes and four authored continuation records. Both rating
templates stay empty. The
[actual report](2026-10-05-story-rubric/empty-ratings-01/report.json) correctly
returns `awaiting-ratings`:96 expected cells across two checkpoints, no rated
cells and no paired score or interval. Missing reviews are not0 or assumed
neutrality. Ninety-two absent generation cells in this fixture are distinct
from the actual campaign's44 unrun stage rows.

The original logical contract digest remains
`96155a17e1b1cfe065f005ae61b2c640b5b172509eba7162404bd7737119e659`.
The module SHA-256 is
`cc0b081a1b13bab33787dc21d6efcbbaab6e1fbb596d4a810ac9a10afe989669`;
the CLI SHA-256 is
`9bff293251d58f27a7119ebef5a9e6de5447db32eaf9bcba5589b512e7661c64`.
Their source identities are in both receipts. CLI timing was not separately
measured; no inference latency or training cost is invented from these commands.

## Controls and independent review

Root's fresh focused command passes all28 tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -p test_story_rubric.py -q
```

Controls cover complete text/shuffle/privacy, exact whole-contract binding,
missing/abstained ratings, invalid range/type/IDs/hashes, duplicate submissions,
declared adjudication, fixed comparisons, token/EOS/context geometry, extra
failed attempts and known/unknown additive costs. Resource peaks are rejected
as additive costs. Extra attempts remain charged but cannot replace a primary
attempt. Existing directories refuse overwriting, including empty directories.

A complete authored numeric control reproduces paired bootstrap summaries
with twelve source-opening groups. All four recipes travel together; separate
recipe strata and five rubric dimensions remain visible. Constant assigned
score differences yield an intentionally degenerate[1,1] interval in the
numeric control, not a measured model gain or population uncertainty result.
Incomplete coverage cannot quietly switch to an easier partial-panel interval.

A separate agent performed read-only code review and additional in-memory
grouped-bootstrap verification, then found no remaining material defect. Its
then-current suite had27 tests; the final sealed suite and root replay have28.
This is independent AI implementation review, not two human story ratings.

The first fixture-preparation attempt failed when JavaScript serialization
changed exact producer floating-point values from `1.0` to `1`, invalidating
the logical contract digest. Exact producer serialization restored the original
unchanged contract. The agent's retained context reports
`ValueError: logical_contract_sha256 mismatch`; the full first command, exit
and traceback are unavailable after compaction and remain unknown. This is a
reported instrument/fixture collection failure, not a fabricated raw receipt
or model/GPU failure. Later valid commands do not erase that history.

## Course integration and limits

The [integrated Linux CPU receipt](2026-10-05-story-rubric-cpu/run-01/cpu-verification.json)
now passes1,162 tests, book mathematics/navigation checks and five selected
fresh notebook references. Tests take193.560s internally,196.458740s at the
command boundary. All198 executable source hashes stay unchanged during the
command panel. Known sparse/quantization warning messages remain in the test
log; they are not silently suppressed or changed into failures.

The [notebook manifest](2026-10-05-story-rubric-cpu/run-01/notebooks/manifest.json)
records39 source code cells,35 executed references,13 PNG outputs and four
preserved/skipped learner scaffolds. Five execution-only identity preambles
bring the actual kernel-cell count to40. This is a fresh five-notebook panel,
not another all76 execution or a Mac pass. The prior full76 receipt remains
historical; separate byte comparison verifies its source notebooks unchanged.

The CPU receipt SHA-256 is
`1f2619bbf0df31c156d27236f48b1d8aeda3ba6a974a95a060be7831feca4b3a`;
the notebook manifest digest is
`0497d515957e575bc937a901803c65e55244eb0f98313edfa9ae65c30fbe0d34`.

Chapter7section7.17, exercise/solution23 and Lab7 now connect raw responses to
anonymous display, independent declarations, disagreements, fixed comparisons
and paired uncertainty. The existing Day10 uncertainty notebook supplies the
mathematical reference; no new mandatory chapter, day or notebook is added.
Chapter6 links the [actual pinned-corpus audit](2026-10-05-story-panel-audit.md).
Chapter15 uses [actual model-defense tables](2026-10-05-current-model-defense.md),
not synthetic genealogy as a claimed training outcome.

Independent implementation can proceed while actual Mac execution is
unavailable. The [Mac packet](../../docs/handoffs/MAC_CPU_VERIFICATION.md)
preserves original DXI-17criterion3 and corrects the excessive mandatory-hosted-
success interpretation of criterion5. Original eighteen acceptance/dependency
arrays,15/18 completion and learnerDay9 are unchanged. New animation concepts
are remembered only as uncommissioned Mac proposals.
