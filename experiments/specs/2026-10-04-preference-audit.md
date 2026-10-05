# Preference collection and judge audit experiment

This bounded offline experiment tests the collection and audit instrument
before a real judge or reward model is used. Original text comparisons, authored
reference labels and deterministic simulated judges are distinct provenance
categories. No model is loaded, no API is called and no private labels are
collected. All operations are CPU, mostly standard-library arithmetic.

## Frozen inputs and predictions

Use `fixtures/preference-audit/pairs.json`, its exact SHA256, the versioned
answer-fidelity rubric and strict JSON verdict parser. Eight base pairs cover
six source groups; three turns share one source. Assign source and task family
to one split before generating variants. Derive verbosity and injection variants
of the canonical left answer, changing content IDs while preserving authored
factual labels under the rubric. Keep all variant text and reference labels.

For each of the resulting twenty-four pairs, collect two repeats in both blind
orders from five explicit simulated judges: content-rule, first-slot,
longer-answer, injection-sensitive and repeat-unstable. Judge-facing objects
contain rubric, prompt and A/B texts, but no checkpoint identities or labels.
Add three malformed raw verdict controls and one simulated transport failure.

Predictions: first-slot selection changes the canonical winner after a swap;
longer-answer selection changes after redundant expansion; the injection rule
changes when a wrong left answer carries the untrusted instruction; unstable
repeats disagree; malformed output and transport failure remain invalid rather
than being converted to ties. The keyword rule is a positive parser/audit control,
not an empirical claim of intelligent judging.

## Measurements and acceptance

Retain all 484 raw observations, source/checkpoint/judge/rubric/settings identity,
display order, exact presentation, selected candidate ID, parser version, reason
and failure stage. Report full outcome counts, decisive agreement and its
denominator, all-outcome agreement, source-balanced agreement, paired order
consistency, paired-repeat disagreement, first-position choice and matched
perturbation changes. Null means an unavailable denominator, not zero failure.

Reject source/family split collisions, cross-source multi-turn siblings,
inconsistent candidate IDs, duplicate verdict IDs and unknown judge identities.
Audit pair weighting: all variants and siblings of one source sum to weight one.
Keep ordinal ratings/rankings conversions distinct from inferred preference
strength; ties, missing ratings and invalid schemas are separate outcomes.

Independent tests exercise wrong mappings, same-count split failures, strict
parsing, repeat/order controls, rating/ranking tie conventions, source weighting,
and retained failure rows. Execute the new Day15 notebook in a fresh
`dgx-spark-native` CPU kernel, regenerate figures and inspect them. Record
actual interpreter/platform, durations, checks and source hashes in the report.

## Evidence boundary

This measures controlled simulated vulnerabilities and interface correctness.
It cannot establish human agreement, AI judge quality, a real injection attack
success rate, prompt-population confidence intervals or reward-model capability.
Live human collection or a model judge needs separate authority, privacy/content
terms, held-out review and its own collection protocol. Prepared material does
not advance the learner's Day9 position.
