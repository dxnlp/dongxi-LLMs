# Lab 2 — Text, Tokens and Embeddings

Machine: isolated CPU course kernel on Mac or Spark, offline.
Expected time: 25–45 minutes per notebook; longer routes span several sessions (planning estimate).
Prerequisites: Chapter 1 evidence identity; bytes, indexing and basic gradients.
Deliverable: A text→bytes→IDs→states→logits map plus one controlled tokenizer intervention.

Read the [chapter](../chapters/02-text-tokens-and-embeddings.md) and use the
[worked solutions](../solutions/02-text-tokens-and-embeddings.md) after attempting each exercise.
Write each prediction before executing the reference. Numerical checkpoints use
the notebook’s declared fixture, seed, dtype and tolerance; changing inputs may
change the result. References and interpretations remain adjacent to the attempt.

| Notebook | Question / prediction before reveal | Checkpoint on the declared fixture | One intervention | Evidence boundary |
|---|---|---|---|---|
| [Byte coverage and merges](../../notebooks/day-02/01_byte_bpe_training_and_encoding.ipynb) | Predict: Will English-only training make 数 unknown? | Its toy byte IDs are [230,149,176]; decoding restores 数. | Separate 数据 and 库 into different documents. | Coverage is distinct from compression and understanding. |
| [Vocabulary and sequence cost](../../notebooks/day-02/02_vocabulary_and_sequence_cost.ipynb) | Predict: Must fewer tokens reduce every computation term? | Round trips persist; table size grows with vocabulary while attention depends on sequence length. | Train on English only and reuse the mixed text. | Structural counts are not measured GPU cost. |
| [Embedding gradients](../../notebooks/day-02/03_embedding_gradient_paths.ipynb) | Predict: Must a prompt row have zero gradient when its direct label is ignored? | Repeated IDs accumulate into one row; both inspected prompt-row norms are positive. | Detach the prompt representation in a copied mechanism. | Local sensitivity is not semantic importance. |

The pinned Qwen token counts 9 < 11 < 20 in the [tokenization report](../../experiments/reports/2026-08-30-qwen3-multilingual-tokenization.md) are a separate report-reading checkpoint. They are not outputs of the educational byte-BPE notebooks.

Retain a short explanation of the intervention, observed change and claim it
does not establish. The deliverable should connect these explanations into one
defensible argument, with source/report links rather than an execution-only checklist.

<details>
<summary>Fresh CPU verification and operational reference</summary>

From the repository root, use the isolated environment/kernel described in
[Appendix D](../appendices/d-reproduction-and-environments.md):

```bash
.venv-course/bin/python scripts/verify_course_notebooks.py --days 2 --kernel dongxi-course --expected-prefix .venv-course
```

The verifier retains executed copies and identities in a new directory. Source
notebooks and learner attempts remain intact. Exact runner/replay/export commands
and their evidence boundaries are in the [runbook](../../docs/runbooks/evaluation_tools.md).

</details>
