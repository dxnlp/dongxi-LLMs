# Saved response evaluation model card

This card replays the supplied response ledger under a frozen evaluation contract. It does not run generation, establish training lineage or authorize a release. The results describe retained attempts on these items, not general model capability.

## Evaluation scope

The suite contains 15 items in 14 source groups. Replay retained all 30 responses, including 0 error rows and 20 unsupported answers.

Split labels and task definitions are preserved in model-card.json. They do not prove that evaluation data was untouched during training or recipe selection. Gold references and executable rubric rules are grading inputs only; this exporter performs no response selection.

## Results on retained attempts

Rates use every retained response, including failed and capped attempts. Missing suite items are not silently inserted into the denominator. Task, source-group and split slices, status counts, missing items and every raw response/error remain in replay.json.

| Checkpoint record ID | Attempts | Accuracy | Task success | Format valid | Natural stop | Truncated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| local-hf-sha256:1e68a13306bb879c26730ff8217a6b85cff26ece4d071373f55c02375bccad73 | 15 | 0.066667 | 0.066667 | 0.800000 | 0.000000 | 1.000000 |
| local-hf-sha256:b39c8fc43535a8b4fa6fa129bb6faad1494a1e3ffba8f980e57859079d7e3aa3 | 15 | 0.000000 | 0.000000 | 0.800000 | 0.000000 | 1.000000 |

## Paired comparisons

local-hf-sha256:1e68a13306bb879c26730ff8217a6b85cff26ece4d071373f55c02375bccad73 minus local-hf-sha256:b39c8fc43535a8b4fa6fa129bb6faad1494a1e3ffba8f980e57859079d7e3aa3: task-success difference 0.066667, percentile interval [0.000000, 0.214286], 15 aligned item/sample pairs in 14 source groups (2000 draws, seed 1010).

The source-group bootstrap preserves within-group dependence but a small panel yields fragile descriptive uncertainty. It is not evidence of population-wide improvement.

## Decoding and grading

The frozen contract preserves template, thinking mode, decoding, stopping, output cap, parser and rubric versions. The complete settings, references and format/rubric policies are in model-card.json. Configured device and dtype are declarations, not exporter measurements. Unsupported mathematics stays unsupported; the exporter delegates grading to the existing bounded evaluator.

## Identity and resource evidence

Record ID local-hf-sha256:1e68a13306bb879c26730ff8217a6b85cff26ece4d071373f55c02375bccad73: recorded internally consistent metadata; 0 rows lack a matching supplied run identity.

- attempted&#95;forward&#95;calls: retained known total 960; 15 known and 0 unknown rows.
- attempted&#95;forward&#95;tokens: retained known total 47328; 15 known and 0 unknown rows.
- decode&#95;seconds: retained known total 11.532747279910836; 15 known and 0 unknown rows.
- forward&#95;calls: retained known total 960; 15 known and 0 unknown rows.
- forward&#95;seconds: retained known total 12.358089348941576; 15 known and 0 unknown rows.
- generation&#95;tokens: retained known total 960; 15 known and 0 unknown rows.
- model&#95;forward&#95;tokens: retained known total 47328; 15 known and 0 unknown rows.
- prefill&#95;seconds: retained known total 0.8253420690307394; 15 known and 0 unknown rows.
- scoring&#95;tokens: retained known total 0; 15 known and 0 unknown rows.
- wall&#95;seconds: retained known total 19.90877994900802; 15 known and 0 unknown rows.

Record ID local-hf-sha256:b39c8fc43535a8b4fa6fa129bb6faad1494a1e3ffba8f980e57859079d7e3aa3: recorded internally consistent metadata; 0 rows lack a matching supplied run identity.

- attempted&#95;forward&#95;calls: retained known total 960; 15 known and 0 unknown rows.
- attempted&#95;forward&#95;tokens: retained known total 47328; 15 known and 0 unknown rows.
- decode&#95;seconds: retained known total 12.234563129371963; 15 known and 0 unknown rows.
- forward&#95;calls: retained known total 960; 15 known and 0 unknown rows.
- forward&#95;seconds: retained known total 13.166631693311501; 15 known and 0 unknown rows.
- generation&#95;tokens: retained known total 960; 15 known and 0 unknown rows.
- model&#95;forward&#95;tokens: retained known total 47328; 15 known and 0 unknown rows.
- prefill&#95;seconds: retained known total 0.9320685639395379; 15 known and 0 unknown rows.
- scoring&#95;tokens: retained known total 0; 15 known and 0 unknown rows.
- wall&#95;seconds: retained known total 21.36955504893558; 15 known and 0 unknown rows.

Supplied run identities are checked for internal hash, input-map, settings and interface consistency. Their device, environment and checkpoint-file digests are recorded metadata, not a fresh weight reload, authenticated signature or hardware benchmark. Unknown costs are not zero measurements. Generated tokens, completed and attempted forward positions and wall time retain their separate units; they are not FLOPs, provider bills or whole-job physical resources.

## Unavailable evidence

Model architecture, training history, verified checkpoint genealogy, independent human behavioral review, broad capability/safety and execution/publication/deployment approval are unavailable from the consumed evidence. No pretrained evaluation, independent behavioral review or training claim is inferred. Authored fixture labels and real local-adapter observations must not be conflated.

## Reproduction evidence

model-card.json contains exact input-byte/source digests when exported from files, the frozen contract, all item/rubric definitions, resource completeness and paired settings. replay.json retains the full graded ledger. Neither artifact modifies its source evidence.

Card identity: 557de73ef1237b01916f8e47fec072878f05f8f16a1ef38490a4fcc5acd3b8e4. Frozen contract: 099520a69562abbe93bf45982c844f4658125e483331727d7b2f0afc4cbcdec7. Replay identity: 1a43abbec69866e16aeadadc8dadd0cc58bb816e1e0d76b94517357288aafe23.
