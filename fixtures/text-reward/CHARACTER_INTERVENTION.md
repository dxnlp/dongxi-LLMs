# Character reward fixture and export

The character arm uses the original `records.json` unchanged. Its alphabet is
fixed independently of labels, with explicit printable-ASCII rejection before
casefolding. The [premeasurement specification](../../experiments/specs/2026-10-04-text-reward-vocabulary-intervention.md)
fixes seeds 1611–1613, budgets and input-disjointness gates. All labels remain
course-authored, not human/AI preference observations.

`frozen-char-preference-seed1611.json` is a complete numeric frozen reward,
not a text-generation model. Load it with `load_char_reward`, supplying an
expected interface/file hash. Its raw scores are detached; high reward does
not establish factual quality. Unsupported characters and overlength input
raise explicit errors.

The [measured report](../../experiments/reports/2026-10-04-text-reward-character.md)
retains negative heldout preference/process outcomes despite fitted training
labels. The word-arm fixture, frozen seed 1601 export and reports remain
historical evidence. New actor/reward comparisons must name their own separate
inputs, splits, protocol and frozen-export identities.
