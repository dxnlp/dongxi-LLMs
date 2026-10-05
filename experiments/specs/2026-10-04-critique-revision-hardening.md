# Critique-stop contract hardening: premeasurement follow-up

This follow-up is frozen before executing its campaign. The original
[specification](2026-10-04-critique-revision.md), protocol, seeds, two-round cap,
mode rules, acceptance rule, candidate pools, authored cases and gold boundary
are unchanged. The original campaign and first passed acceptance remain as
historical evidence, with their historical source hashes.

Independent review found that a returned known action-plus-EOS path marked
`max_tokens` could pass critique validation and trigger a revision. The usable
critique contract must require natural `eos` stopping and `truncated=false`, not
merely an EOS-looking token suffix. Capped/error/malformed critiques must be
retained with their actual output count and failure reason; revision is skipped.
The error-retention branch must also initialize its diagnostic value before a
failing deepcopy/JSON conversion, so such a failure does not erase the round.

Add independent capped-critique and failing-deepcopy regressions before the
follow-up campaign. Execute the same frozen panels into a new run directory and
compare all continuations, decisions, grades, errors and budget counts against
the original campaign. Only measured wall times and source/run identities may
differ. A passing normal-path replay does not retroactively establish that the
old instrument handled capped critiques correctly.

The notebook should load the new current-source campaign; keep both earlier
fresh-kernel manifests and the initial acceptance. Collect a new acceptance
report with current source identities, fresh kernel/figures and an explicit
historical comparison. No numerical recipe tuning, model/API call, acquisition,
GPU operation or new neural self-critique is authorized by this follow-up.
