# Day 8 — Pretraining data and recipe

Companions to [Chapter 6: Pretraining as a Controlled System](../../book/chapters/06-pretraining-as-a-controlled-system.md).
The complete Day 8 subject pathway is prepared; guided learning and the Day 9
GPU run remain separate. Current learning destination is Spark, not Day 6 on Mac.

| Session | Question | Hands-on intervention |
|---|---|---|
| [1. Data, batches and token budgets](01_data_batches_and_token_budget.ipynb) | What exactly contributes to one update? | Change windows; inject split overlap; compare correct accumulation against wrong mean-of-means |
| [2. AdamW, schedules and stability](02_adamw_schedule_and_stability.ipynb) | How does a gradient become a change in weights? | Trace moments; alter gradients; inspect schedule clock, clipping, dtype range and memory ledger |
| [3. Validation and checkpoint recovery](03_validation_and_checkpoint_recovery.ipynb) | Is this a valid, resumable experiment? | Train the tiny decoder; change validation grouping; omit optimizer or data state when resuming |

Each session includes prediction prompts, adjacent runnable solutions, explained
figures, deliberate failures and evidence limits. Ten saved reference PNGs are
visible without starting a kernel; plotting cells regenerate from current values.
Static previews are not live controls and do not update when code is edited.
No notebook server is started by creating or verifying these materials.

The fixture is ten original in-repository sentences and a transparent byte
tokenizer (258 IDs). It reuses the Chapter 5 modern decoder, CPU only, with short
24-update probes. No external data, weights, credentials or GPU downloads.
Use an existing Python environment with PyTorch, Jupyter/ipykernel, nbclient,
nbformat and [the course plotting dependencies](../requirements-visuals.txt).
Notebooks discover the repository root and use portable `python3` metadata;
select the actual course kernel on your machine. Mac execution has not been
verified by the Spark checks, and no Mac dependency installation is implied.

Verification from the repository root on the current Spark:

```bash
/home/dongxi/dgx-spark-dongxi/.venv/bin/python scripts/verify_pretraining_notebooks.py --kernel dgx-spark-native --figures
```

This executes fresh temporary copies and updates generated PNG previews only.
It does not overwrite source notebooks or older learner notebooks. Substitute
your verified local interpreter/kernel on another machine.

Continue with the [conceptual solutions](../../book/solutions/06-pretraining-as-a-controlled-system.md),
[companion lab guide](../../book/labs/06-pretraining-as-a-controlled-system.md),
and [bounded experiment specification](../../experiments/specs/2026-09-09-day8-bounded-pretraining.md).
Live discussion can use these experiments without requiring a notebook session.
