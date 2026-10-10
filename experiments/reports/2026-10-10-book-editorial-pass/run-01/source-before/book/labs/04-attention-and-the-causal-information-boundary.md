# Lab 4 — Attention and the Causal Information Boundary

Machine: CPU on Mac or Spark, offline. Use the declared Torch/Matplotlib course
kernel. Read [the chapter](../chapters/04-attention-and-the-causal-information-boundary.md) and attempt
[the worked exercises](../solutions/04-attention-and-the-causal-information-boundary.md) alongside this route.

The [Day4 route](../../notebooks/day-04/README.md) traces Q/K/V, causal gradients
and cached equivalence. Predict whether replacing the final token changes an
earlier representation, then inspect scores, masks, attention probabilities and
value mixtures. Routing probabilities are not value content.

Compare correct and deliberately broken masks. Trace gradients into Q/K/V and
distinguish routing from content paths. Increasing key dimension changes raw
score spread; scaling stabilizes it under the stated assumptions, not universally.

Compare full-prefix and cached decoding with the same unchanged prefix.
Plot cache growth and numerical error. A causal earlier state does not depend on
future arrivals; different prompt contexts do not share a cache just because
they contain the same token.

Deliver a causal dependency sketch and one failure that invalidates the cache
equivalence claim. This is not an inference-throughput benchmark.
Implementation: [causal_attention_lab.py](../../src/dongxi_llms/causal_attention_lab.py)
and [attention_evidence.py](../../src/dongxi_llms/attention_evidence.py).
Checks: `PYTHONPATH=src python -m unittest discover -s tests -p test_causal_attention_lab.py`.

Fresh reference execution from the repository root:

```bash
python scripts/verify_course_notebooks.py --days 4 --kernel dgx-spark-native --export-figures
```

A passing reference verifies declared mechanism properties, not broad model
capability or independently assessed learner mastery.
