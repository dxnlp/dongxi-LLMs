# Original preference audit fixture

These eight comparison texts and their reference labels were written for this
course. They are not imported from RLHF Book, a benchmark, a label contractor,
or a language model. Three turns share one story source; six independent source
groups cover train, calibration and test. The split is an identity microscope,
not a statistically adequate reward-model dataset.

Reference labels are **authored**, specified independently of the simulated
judge rules. They are not independent human feedback. The module preserves that
distinction and also supports explicit human and AI identities for future
separately authorized collection. Toy judges are deterministic simulations.
The content-rule judge uses an exposed finite keyword list; high agreement is
built into this fixture, not evidence of a general-purpose evaluator.

The module creates left-answer verbosity and untrusted-instruction variants
without changing the factual label under the declared rubric. Variant IDs and
exact texts are retained in the report. A future live study must review its
variants independently; do not silently reuse reference labels when a change
alters the substantive meaning or intended style criterion.

Run from the repository root without downloads or credentials:

```bash
PYTHONPATH=src python -m dongxi_llms.preference_audit --fixture fixtures/preference-audit/pairs.json
```

Use `--output NEW.json` only for a new evidence file; existing reports cannot
be overwritten by the command. Every raw verdict, invalid parse and simulated
transport failure is retained. The original fixture is immutable input.
