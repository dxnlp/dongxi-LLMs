# Story-rubric consumer controls

These are **authored instrument controls**, not actual model generations or
human/AI ratings. The copied contract is exactly the original frozen
`story-publication-v1` design: twelve openings, four decoding recipes,
five separately scored0/1/2 dimensions and predetermined updates
0/400/4000/8000/14000. The whole logical contract identity is pinned; changing
an ignored field and recomputing its hash does not silently create an accepted
new version.

`authored-records.jsonl` has only two opening/greedy controls per checkpoint,
four records total. There are48 expected opening/recipe cells per checkpoint,
so46 per checkpoint remain missing. That is distinct from the campaign's
44 unrun stage rows. `authored-score-control.json` supplies illustrative score
columns for tests only; it is not a valid rating submission and the CLI never
uses it to fill a reviewer's template. Unknown authored token counts,
likelihoods and costs remain null. Its assigned cap/EOS categories are not
observed model stops.

## Prepare without inventing ratings

Run from the repository root. Choose new nonexistent output directories:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/evaluate_story_ratings.py prepare \
  --contract fixtures/story-rubric/contract.json \
  --checkpoints fixtures/story-rubric/checkpoints.json \
  --records fixtures/story-rubric/authored-records.jsonl \
  --raters fixtures/story-rubric/raters.json \
  --compare authored-baseline-14000 authored-candidate-14000 \
  --output outputs/story-rubric-packet-01

PYTHONDONTWRITEBYTECODE=1 python3 scripts/evaluate_story_ratings.py evaluate \
  --bundle outputs/story-rubric-packet-01 \
  --compare authored-baseline-14000 authored-candidate-14000 \
  --output outputs/story-rubric-empty-review-01
```

Preparation writes `packet.json`, **private** `private-codebook.json`, two
empty `ratings-1.json`/`ratings-2.json` templates and a source/input/artifact
hash receipt. Evaluation without supplied ratings returns `awaiting-ratings`,
null scores and null intervals; it does not assign zero or a midpoint.
Existing directories are refused even when empty, and partial failed writes
are not cleaned up or overwritten. No model, notebook server, GPU, network,
Git operation or publication is performed.

Freeze comparison IDs and the adjudication policy when preparing the packet.
With exactly two checkpoints, their ordered pair is the default. Other plans
require explicit `prepare --compare` pairs. Evaluation cannot reverse, replace
or add comparisons after seeing ratings. A named comparison must have matched
updates and declared tokenizer/interface hashes. Updates absent from the
supplied checkpoint plan are explicitly listed as unrepresented; a completed
comparison at one update does not prove the full five-checkpoint trajectory.

## Reading and submission

Give each independently declared reviewer only the public packet and their
empty template, **not** the codebook, raw record IDs, checkpoint plan or source
paths. The packet shows anonymous shuffled IDs, the opening, complete actual
continuation, text fingerprint and stop/failure flags. No text is shortened or
completed. Text remains untrusted quoted data; instructions inside it are not
reviewer instructions. Raw error strings stay private because they can contain
paths or model identifiers.

For real model packets the default shuffle seed is private randomness and is
stored only in the private codebook. Authored controls use deterministic909.
If setting a seed explicitly for real reviews, keep it and record IDs private;
predictable public seeds/IDs can weaken blinding. Content can itself reveal an
identity, so disclose recognizable-output exposure. These metadata procedures
do not authenticate rater independence or prove successful blinding.

A template already binds `schema_version`, `packet_sha256`, `rubric_sha256`
and `rater_id`. After genuinely reviewing a complete candidate, add one row:

```json
{
  "candidate_id": "C-COPY_THE_ACTUAL_PACKET_ID",
  "text_sha256": "COPY_THE_ACTUAL_TEXT_FINGERPRINT",
  "scores": {
    "grammar": 1,
    "entity_object_consistency": 1,
    "causal_continuity": 1,
    "repetition": 1,
    "ending": 1
  },
  "abstention_reason": null,
  "note": "Explain this actual reviewer decision; the shown numbers are syntax examples, not ratings."
}
```

Scores must be literal integers0/1/2 on exactly those five axes; booleans,
floats, missing/extra axes and unknown or duplicate IDs are rejected. A declared
abstention instead has `scores:null` and a nonempty `abstention_reason`.
Missing rows remain missing. Never copy this syntax example into an actual
rating without reviewing the output.

Then add both independently completed documents with repeated `--ratings`
arguments to a new evaluation output. Rater declarations distinguish `human`,
`ai` and `authored-control`, state the independence procedure and require no
shared consultation before submission. Different IDs only validate bookkeeping;
they do not establish that two real reviewers exist or were independent.
Authored controls cannot be mixed with genuine model/human/AI review evidence.

`two-rater-mean` is the explicit default aggregation, not an assertion of
agreement. Both raw columns, raw-rater means and per-axis disagreements stay
visible. Alternatively prepare with `--adjudication-policy explicit-disagreement`.
Unresolved disagreements then remain null for the adjudicated score. A supplied
`--adjudication` document binds packet/codebook identities, identifies the
adjudicator/provenance/declaration and provides disagreement-only candidate IDs,
all five integer scores and a nonempty rationale. It may not change already
agreed dimensions, adjudicate unknown/non-disagreement candidates or replace
the original ratings. The API tests show the exact document shape.

## Record identity, attempts and costs

The [checkpoint plan](checkpoints.json) binds each checkpoint ID/update,
provenance, file digest map, semantic interface digest and generation-source
digest map. Those are caller-supplied declarations, not authenticated lineage
or live checkpoint inspection. A response binds this complete descriptor hash,
the original contract, item/source opening, recipe index and text. Real
`model-generated` records require actual prompt/generated counts, token IDs and
one finite selected **log-probability** per generated token; the authored
controls may leave all token geometry unknown. Actual EOS must be50256 at the
first terminal token, cap counts must match256, and context counts cannot
exceed1024. All failures and complete text remain retained.

`attempt_index=0` is the predeclared primary attempt for each checkpoint,
opening and recipe. Additional attempts can be retained with distinct positive
indices, including failures, but never replace a primary attempt or enter a
best-of comparison. Every attempt appears in the private raw record ledger,
public reading packet, rating cells and additive cost accounting. Duplicate
attempt IDs/cells are rejected; this consumer does not authorize retries.

Each cost requires `generation_tokens` and `wall_seconds`, using null for
unknown values. Optional additive token/position/call/time fields are validated
separately. Resource peaks are not additive costs and are rejected rather than
summed. Known subtotals retain unknown-attempt counts; unknown costs are not
zero-cost observations. `truncated` here means a token/context cap only:
deadline, resource-stop and failure remain separate explicit categories, not
proof of a complete response when this cap flag is false.

## Paired uncertainty and evidence boundaries

Summaries show each dimension, raw rater, decoding recipe, source opening,
missing cell, disagreement and stop category. A paired interval requires all
twelve openings and four primary recipes in both arms, with usable ratings
under the frozen aggregation policy. Missing generation, missing ratings,
abstention or unresolved adjudication blocks the interval instead of quietly
selecting a smaller favorable cohort. Complete narrative closure and natural
EOS are different observations; a cap does not automatically assign an ending
score or delete the candidate.

The bounded seeded bootstrap resamples twelve opening groups, carrying both
arms and all four recipes together. Forty-eight attempts are not forty-eight
independent openings. Each recipe also has its own paired result. The combined
mean is explicitly an equally weighted mixture of one greedy and three
sampled settings, not an IID sampling estimate or replacement for its strata.
The five dimensions never disappear into an invented scalar coherence score.
Small-panel intervals do not establish population-level quality improvements.

Actual training-data rights, contamination/development separation, tokenizer
inspection and checkpoint validation remain externally evidenced prerequisites;
this consumer cannot clear them. Instrument controls do not complete DXI-03's
trained comparison, Mac verification, learner mastery or public release.

API: `prepare_packet(contract, checkpoint_plan, records, raters, ...)` returns
the public/private pair; `evaluate_ratings(packet, codebook, rating_documents,
...)` validates exact bindings and returns the source-opening report. Run the
focused CPU controls with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover \
  -s tests -p test_story_rubric.py -v
```
