# Lab 2 — Text, Tokens, and Embeddings

Machine: CPU on Mac or Spark, offline. Use the declared Torch/Matplotlib course
kernel. Read [the chapter](../chapters/02-text-tokens-and-embeddings.md) and attempt
[the worked exercises](../solutions/02-text-tokens-and-embeddings.md) alongside this route.

The [Day2 route](../../notebooks/day-02/README.md) follows bytes, merges, IDs and
embedding paths. Predict how an English-only full-alphabet byte BPE encodes
`数`, then inspect the actual bytes. Round-trip unseen multilingual text. Change
corpus or merge budget and distinguish representability from compression.

Compare vocabulary width and sequence length: the output projection depends on
vocabulary size, while attention depends on sequence positions. More merges
do not by themselves guarantee better language modeling. This educational BPE
works over whole documents, unlike production regex presegmentation.

Trace repeated embedding lookup and tied-output gradients. Predict which rows
receive gradients on each path before viewing the plot. IDs are categorical
addresses; numeric ID distance does not define linguistic similarity.

Deliver text→bytes/merges→IDs→embeddings→contextual states→logits, plus an
intervention distinguishing tokenizer behavior from model learning.
Implementation: [embedding_gradient_lab.py](../../src/dongxi_llms/embedding_gradient_lab.py)
and [course_foundations.py](../../src/dongxi_llms/course_foundations.py).
Checks: `PYTHONPATH=src python -m unittest discover -s tests -p test_embedding_gradient_lab.py`.

Fresh reference execution from the repository root:

```bash
python scripts/verify_course_notebooks.py --days 2 --kernel dgx-spark-native --export-figures
```

A passing reference verifies declared mechanism properties, not broad model
capability or independently assessed learner mastery.
