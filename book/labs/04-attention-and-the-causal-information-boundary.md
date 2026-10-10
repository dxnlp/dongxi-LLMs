# Lab 4 — Attention and the Causal Information Boundary

Machine: isolated CPU course kernel on Mac or Spark, offline.
Expected time: 25–45 minutes per notebook; longer routes span several sessions (planning estimate).
Prerequisites: Chapters 2–3; matrix multiplication, softmax and chain rule.
Deliverable: A causal dependency diagram and a cache failure explained by its changed context.

Read the [chapter](../chapters/04-attention-and-the-causal-information-boundary.md) and use the
[worked solutions](../solutions/04-attention-and-the-causal-information-boundary.md) after attempting each exercise.
Write each prediction before executing the reference. Numerical checkpoints use
the notebook’s declared fixture, seed, dtype and tolerance; changing inputs may
change the result. References and interpretations remain adjacent to the attempt.

| Notebook | Question / prediction before reveal | Checkpoint on the declared fixture | One intervention | Evidence boundary |
|---|---|---|---|---|
| [Causal retrieval](../../notebooks/day-04/01_causal_attention_forward.ipynb) | Predict: Will a changed final token alter earlier outputs? | Rows sum to one; forbidden weights are zero; earlier causal outputs agree within the declared tolerance. | Use the unmasked or late-mask variant. | Attention weights alone do not explain the whole output. |
| [Routing and content gradients](../../notebooks/day-04/02_attention_gradients_and_failures.ipynb) | Predict: Can routing and value paths receive different credit? | Manual/autograd errors are below 1e-12; forbidden-score gradient is zero. | Detach Q/K routing or value content separately. | The IID scaling control is not a statement about learned representations. |
| [KV-cache identity](../../notebooks/day-04/03_kv_cache_equivalence.ipynb) | Predict: Will a stale prefix preserve next-token logits? | Correct replay agrees within 1e-12; stale-prefix error is about 1.80691; final logical payload is 768 bytes. | Change an earlier prefix while reusing its cache. | Logical work and payload are not latency or peak memory. |

Use the [attention/cache report](../../experiments/reports/2026-09-05-attention-gradients-cache.md) for the fixed stale-prefix case. In every figure distinguish forbidden edges, allowed near-zero weights and measured numerical error.

Retain a short explanation of the intervention, observed change and claim it
does not establish. The deliverable should connect these explanations into one
defensible argument, with source/report links rather than an execution-only checklist.

<details>
<summary>Fresh CPU verification and operational reference</summary>

From the repository root, use the isolated environment/kernel described in
[Appendix D](../appendices/d-reproduction-and-environments.md):

```bash
.venv-course/bin/python scripts/verify_course_notebooks.py --days 4 --kernel dongxi-course --expected-prefix .venv-course
```

The verifier retains executed copies and identities in a new directory. Source
notebooks and learner attempts remain intact. Exact runner/replay/export commands
and their evidence boundaries are in the [runbook](../../docs/runbooks/evaluation_tools.md).

</details>
