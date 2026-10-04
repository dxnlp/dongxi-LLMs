# Authored Chapter 11 Location-Extraction Fixture

Eight training pairs, four validation pairs, four independent prompt/expected
evaluation records. Original repository-authored English examples. This tiny
fixture permits a bounded DPO pipeline smoke once a compatible Chapter9 SFT
checkpoint exists. It is not a human preference dataset or a broad benchmark.

The intended rubric is concise, correct extraction of the recorded location.
Wrong locations are original distractors; every chosen response is verified
against its prompt. Source groups, names, objects, and exact locations are
disjoint between splits. Similar prompt templates are intentional task structure
and prevent claiming broad linguistic generalization. Training on this narrow
distribution can create regressions outside it; retain the untouched SFT model.

The evaluation strings are a strict answer-only contract. Alternative wording
can be valid in ordinary conversation but fails this declared metric. Preserve
all generated strings so a later rubric audit can distinguish extraction errors
from formatting failures. Do not tune the recipe repeatedly against these four
examples and continue calling them a pristine final test.

Use the [lab guide](../../book/labs/11-direct-preference-optimization.md) and
the optional [Spark runner](../../scripts/run_chapter11_spark_dpo.py). No weights
or model-scale results are included in this fixture.
