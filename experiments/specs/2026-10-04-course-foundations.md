# Foundation and saved-run reference verification

Mode: bounded CPU teaching verification. Written before execution on2026-10-04.
No pretrained weights or data downloads are required.

Hypotheses: canonical identities are invariant to JSON key ordering but change
with declared controls; a full byte alphabet round-trips unseen Unicode;
frequent adjacent byte sequences can merge; document-isolated causal attention
blocks cross-document interventions; saved-run accounting reproduces the
completed TinyStories report.

Controls: CPU, seed909, fixed original teaching corpora and historical saved
JSON. Notebook references import reusable functions and retain deliberate failed
conditions as labeled controls. Assertions and numerical comparisons determine
mechanism acceptance; plots explain those same tensors.

Run all9 new sessions in Days1,2,9 through fresh kernels with
`scripts/verify_course_notebooks.py --days 1 2 9 --export-figures`.
Per-cell timeout240s; sampled available memory before each session must exceed
25GiB. Credentials, GPU allocations and model downloads are excluded.

Stop on reference exceptions, broken round trips, violated invariants or a
host reserve breach. Preserve source hashes, outputs, generated figures and
failed paths. Numerical/code readiness does not establish learner mastery,
production tokenizer equivalence or a new storytelling improvement.
