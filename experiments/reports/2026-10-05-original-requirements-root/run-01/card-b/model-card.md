# Saved response evaluation model card

This card replays the supplied response ledger under a frozen evaluation contract. It does not run generation, establish training lineage or authorize a release. The results describe retained attempts on these items, not general model capability.

## Evaluation scope

The suite contains 15 items in 14 source groups. Replay retained all 30 responses, including 1 error rows and 1 unsupported answers.

Split labels and task definitions are preserved in model-card.json. They do not prove that evaluation data was untouched during training or recipe selection. Gold references and executable rubric rules are grading inputs only; this exporter performs no response selection.

## Results on retained attempts

Rates use every retained response, including failed and capped attempts. Missing suite items are not silently inserted into the denominator. Task, source-group and split slices, status counts, missing items and every raw response/error remain in replay.json.

| Checkpoint record ID | Attempts | Accuracy | Task success | Format valid | Natural stop | Truncated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| authored-baseline | 15 | 0.466667 | 0.400000 | 0.866667 | 0.866667 | 0.066667 |
| authored-candidate | 15 | 0.866667 | 0.866667 | 0.933333 | 1.000000 | 0.000000 |

## Paired comparisons

authored-candidate minus authored-baseline: task-success difference 0.466667, percentile interval [0.125000, 0.800000], 15 aligned item/sample pairs in 14 source groups (2000 draws, seed 1010).

The source-group bootstrap preserves within-group dependence but a small panel yields fragile descriptive uncertainty. It is not evidence of population-wide improvement.

## Decoding and grading

The frozen contract preserves template, thinking mode, decoding, stopping, output cap, parser and rubric versions. The complete settings, references and format/rubric policies are in model-card.json. Configured device and dtype are declarations, not exporter measurements. Unsupported mathematics stays unsupported; the exporter delegates grading to the existing bounded evaluator.

## Identity and resource evidence

Record ID authored-baseline: unverified or partially unavailable; 15 rows lack a matching supplied run identity.

- generation&#95;tokens: retained known total 0; 0 known and 15 unknown rows.
- scoring&#95;tokens: retained known total 0; 0 known and 15 unknown rows.
- wall&#95;seconds: retained known total 0; 0 known and 15 unknown rows.

Record ID authored-candidate: unverified or partially unavailable; 15 rows lack a matching supplied run identity.

- generation&#95;tokens: retained known total 0; 0 known and 15 unknown rows.
- scoring&#95;tokens: retained known total 0; 0 known and 15 unknown rows.
- wall&#95;seconds: retained known total 0; 0 known and 15 unknown rows.

Supplied run identities are checked for internal hash, input-map, settings and interface consistency. Their device, environment and checkpoint-file digests are recorded metadata, not a fresh weight reload, authenticated signature or hardware benchmark. Unknown costs are not zero measurements. Generated tokens, completed and attempted forward positions and wall time retain their separate units; they are not FLOPs, provider bills or whole-job physical resources.

## Unavailable evidence

Model architecture, training history, verified checkpoint genealogy, independent human behavioral review, broad capability/safety and execution/publication/deployment approval are unavailable from the consumed evidence. No pretrained evaluation, independent behavioral review or training claim is inferred. Authored fixture labels and real local-adapter observations must not be conflated.

## Reproduction evidence

model-card.json contains exact input-byte/source digests when exported from files, the frozen contract, all item/rubric definitions, resource completeness and paired settings. replay.json retains the full graded ledger. Neither artifact modifies its source evidence.

Card identity: a028027b9e0110e5dc29b4430b37b05bd52797d60a5d6ad6ee9a05fe3d6ef9da. Frozen contract: a4b305398408b67b12547fa9b644ab4016f6d93294fb6021086e4729198d71fa. Replay identity: 032b51f7be3028a5249274cda2d5746d92b3372110d47363cfdfa695d801fc41.
