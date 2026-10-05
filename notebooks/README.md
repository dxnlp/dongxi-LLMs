# Notebooks

Notebooks narrate derivations, experiments, and visual analysis. Reusable logic
belongs in `src/dongxi_llms/` and should be imported here.

## Interaction contract

Mathematical mechanism notebooks use this learning cycle:

1. begin with a deep question about the mechanism;
2. record a prediction before execution;
3. implement the smallest transparent version;
4. perturb an assumption or run a deliberately broken variant;
5. distinguish observation from interpretation;
6. state what the result proves and what it does not prove.

The learner and course guide work through these notebooks cell by cell. They are
not passive demonstrations, answer dumps, or collections of decontextualized
calculation questions. Notebook outputs are exploratory until their important
claims are reproduced by reusable source code, tests, and an experiment report.

Each exercise or conceptual checkpoint is followed by a clearly labeled
reference solution and explanation. Attempt the prompt before revealing or
running the reference cell. The reference is deliberately adjacent: the learner
should spend time reasoning about the mechanism, not searching elsewhere for
routine syntax or an unstated canonical answer.

Every book chapter has a planned notebook pathway, normally divided into a
mechanism microscope, a deliberate perturbation or failure, and an integration
or evidence session. The course-wide map and activation rules are in
[`../docs/NOTEBOOK_CURRICULUM.md`](../docs/NOTEBOOK_CURRICULUM.md). Create these
sessions without empty placeholders. The 2026-10-04request authorizes creating
the complete pathway before live learning.

## Sessions

Chapter 5 now includes 52 explanatory figures across twelve notebooks, including
the core Day 7 architecture defense. Saved previews are visible before execution;
runnable cells redraw model maps/close-ups and plot live tensors. The plotting dependency
is pinned in [`requirements-visuals.txt`](requirements-visuals.txt). Design and
regeneration instructions: [`Notebook visuals`](../docs/NOTEBOOK_VISUALS.md).

- [`day-03/`](day-03/) — logits, probability, next-token loss, causal alignment,
  gradients, and a tiny learned distribution
- [`day-04/`](day-04/) — scaled causal attention, gradient and failure diagnosis,
  and KV-cache equivalence
- [`day-05/`](day-05/) — seven ready baseline layer/integration notebooks;
  includes the complete Chapter 5 pathway index
- [`day-06/`](day-06/) — three ready modern-architecture notebooks
- [`day-07/`](day-07/) — core architecture-defense notebook, then optional recurrent depth
- [`day-08/`](day-08/) — three pretraining-recipe notebooks: data/budgets,
  AdamW/stability, validation/recovery; 10 explanatory reference figures

## Complete daily session index — 2026-10-04

The current course has76 accepted focused notebooks across all28 learning days.
Fifteen reference-audit extensions add checkpoint-interface protection (Day1),
mathematical grading/response replay (Day10), preference/judge auditing (Day15),
text reward/outcome/process labels and vocabulary controls (Day16), DPO retention
(Day18), behavior-probability/support and learned critic/reward accounting
(Day20), matched objective gradients/filtering (Day22), and sampled sequence
learnability controls (Day23), actual candidate selection (Day10), teacher-data
and matched rejection SFT (Day11), ragged cache/recovery (Day25), and actual
response-level distillation, critique/revision and student-prefix/teacher-context
distillation (Day26).
The [routing registry](../docs/course_manifest.json) classifies60core sessions,
one optional recurrence session and fifteen accepted extensions. It records location and
dependencies, not mastery or fulfilled experiment criteria. All76 pass fresh
CPU execution in two current batches:530 reference cells and206 figures, with
four preserved unfinished learner cells skipped. The
[current checkpoint](../experiments/reports/2026-10-04-course-upgrade-checkpoint.md)
is separate from the historical61-session build; no new pretrained
model result or learner mastery is implied.
The user explicitly requested the full build ahead of guided study.
Material readiness never records an exercise as independently mastered.
For a clean locked CPU route and retained acceptance-panel evidence, use
[AppendixD](../book/appendices/d-reproduction-and-environments.md). Direct
notebook verification also supports `--notebooks` and `--lane`; unknown or
empty selections fail explicitly and old output directories are preserved.

| Day | Chapter | Notebook route |
|---:|---:|---|
| 1 | 1 | [Laboratory and evidence](day-01/README.md) |
| 2 | 2 | [Tokenization and embeddings](day-02/README.md) |
| 3 | 3 | [Next-token learning](day-03/README.md) |
| 4 | 4 | [Causal attention](day-04/README.md) |
| 5 | 5 | [Decoder components](day-05/README.md) |
| 6 | 5 | [Modern architecture](day-06/README.md) |
| 7 | 5 | [Architecture defense](day-07/README.md) |
| 8 | 6 | [Pretraining system](day-08/README.md) |
| 9 | 6 | [Run diagnosis](day-09/README.md) |
| 10 | 7 | [Evaluation](day-10/README.md) |
| 11 | 8 | [Instruction data](day-11/README.md) |
| 12 | 9 | [SFT mathematics](day-12/README.md) |
| 13 | 9 | [Base-to-assistant experiment](day-13/README.md) |
| 14 | 9 | [SFT recipe defense](day-14/README.md) |
| 15 | 10 | [Pairwise preference](day-15/README.md) |
| 16 | 10 | [Reward models](day-16/README.md) |
| 17 | 11 | [DPO derivation](day-17/README.md) |
| 18 | 11 | [DPO experiment](day-18/README.md) |
| 19 | 12 | [Policy gradients](day-19/README.md) |
| 20 | 12 | [Baselines and PPO](day-20/README.md) |
| 21 | 12 | [Algorithm defense](day-21/README.md) |
| 22 | 13 | [GRPO derivation](day-22/README.md) |
| 23 | 13 | [RLVR experiment](day-23/README.md) |
| 24 | 14 | [Optimization failures](day-24/README.md) |
| 25 | 14 | [Rollout systems](day-25/README.md) |
| 26 | 15 | [Distillation and selection](day-26/README.md) |
| 27 | 15 | [Capstone evaluation](day-27/README.md) |
| 28 | 15 | [Defense and release](day-28/README.md) |

## Run and reproduce

Use [the environment appendix](../book/appendices/d-reproduction-and-environments.md)
to create a portable kernel from [course requirements](requirements-course.txt).
No environment installation is required merely to read saved reference figures.

From the repository root, execute all notebook references without changing
learner cells:

```bash
python scripts/verify_course_notebooks.py --kernel dongxi-course --export-figures
```

On the verified Spark environment the kernel is `dgx-spark-native`.
The verifier uses fresh CPU kernels, offline model access and an execution-only
inline plotting preamble. It saves outputs and source hashes in a new run
directory. Exported PNGs are reference previews, not measured GPU/model-scale
results. Prediction and plot cells can be modified during live study; keep the
saved figure label visible until it is regenerated for the modified inputs.

Explicitly tagged, still-unfinished learner scaffolds containing `...` are
skipped in reference verification; their adjacent complete solutions execute.
The manifest names every skipped scaffold. Completed attempts are not skipped,
and learner source code is never filled in or overwritten by the verifier.
