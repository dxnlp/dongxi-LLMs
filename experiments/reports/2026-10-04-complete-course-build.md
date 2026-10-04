# Complete Course Material Build — 2026-10-04

Outcome: the full **28-day, 15-chapter teaching draft is created and reference-
verified**. The build strengthens existing chapters, creates Chapters 7–15, fills
early notebook gaps and integrates the future sequence. The learner remains on
Day 9; material readiness is not an assessment of completed study.

Start with the [reader's guide](../../book/README.md),
[day-by-day sequence](../../docs/COURSE_SEQUENCE.md) and
[design blueprint](../../docs/COURSE_BLUEPRINT.md). The
[experiment matrix](../../docs/EXPERIMENT_MATRIX.md) connects questions, runnable
routes and evidence limits. [Raw verification JSON](2026-10-04-complete-course-build.json)
records245 source hashes, environment versions, per-notebook identities, figures,
timers, failures and check commands.

## Material inventory

| Asset | Count |
|---|---:|
| Coherent chapters | 15 |
| Chapter words, whitespace count | 51,433 |
| Worked-solution guides | 15 |
| Laboratory guides, including the extra saved-run case | 16 |
| Front-matter sections | 3 |
| Appendices | 4 |
| Daily notebook indexes | 28 |
| Focused notebook sessions | 61 |
| Source code cells | 419 |
| Executed reference cells | 415 |
| Explicit unfinished learner scaffolds retained/skipped | 4 |
| Generated image outputs | 131 |

New sessions are not empty placeholders. They contain an LLM-relevant question,
a prediction, adjacent complete code/explanation, a controlled intervention,
plots and an evidence boundary. Reusable computations are importable modules.
Saved reference PNGs remain visible without starting a notebook server.
They do not update when a learner changes an input; rerun the plot cell for the
new experiment. Existing learner source cells were preserved.

[Portable storyboards](../../visuals/animations/COURSE_STORYBOARDS.md) capture
important mathematics across the course, extending rather than duplicating the
existing BPE/embedding/attention/architecture/pretraining media. New candidates
are precise briefs, not approved rendered films. Production stays on Mac.

## Fresh-kernel verification

Canonical command:

```bash
/home/dongxi/dgx-spark-dongxi/.venv/bin/python scripts/verify_course_notebooks.py --export-figures
```

All 61 references passed in 117.1785 seconds, with no failed notebooks and at least
one explanatory image per session. The kernel was `dgx-spark-native`, CPU-only,
one thread for each numerical library and offline model access. An execution-only inline plotting
preamble ensures figures are captured; source notebooks are not replaced by
executed copies. Four tagged, still-unfinished Day 3exercise scaffolds were skipped
at source indexes 6, 17, 24, 31. Their adjacent full solutions executed. Filled learner
attempts are not skipped; no solution was inserted into a student scaffold.

The minimum observed host `MemAvailable` **before sessions** was 117.4038 GiB,
above the 25 GiB guard. This is not a continuous minimum or proof of no transient
spikes. Executed copies reside temporarily under
`/tmp/dongxi-course-check-o6lwbp2g/`; durable source, generated figures, compact
measured reports and identities are in the repository. The JSON identifies local
diagnostic paths explicitly, not as a portable required dependency.

## Independent checks and review

- 146 regression tests passed in 3.768 seconds, including invariant, finite-difference,
 masking, reference-freeze, checkpoint, verifier and failure-retention cases.
 Expected rejected-run/overwrite messages in tests are negative controls, not
 launched training jobs.
- 54 book Markdown files and 1,053 math expressions passed the source hazard check.
 Dollar delimiters and portable upright names follow the remembered GitHub rules.
 Node/MathJax was unavailable on this host, so a fresh local SVG-render pass
 and live GitHub/browser rendering are **not** claimed.
- Local navigation/coverage audit passed for the book, global routes, daily
 indexes, notebook markdown and image references. `git diff --check` passed.
- All three optional pretrained runner help paths execute without loading
 weights. CPU and file-only tests validate shared mechanisms and interface guards,
 not model-scale CUDA compatibility.
- Root inspected six contact sheets covering 69 new/extended figures, and full-size
 BPE/uncertainty/rare-regression plots after the final changes. Agents inspected
 representative figures in their own sections. This is static plot QA, not
 browser-layout or produced-animation verification.
- Independent section review checked preference/policy mathematics,
 GRPO/distillation graph boundaries and SFT→DPO→RLVR checkpoint interfaces.

## Measured experiment evidence

The [foundation report](2026-10-04-course-foundations.md) covers canonical identities,
Unicode byte round trips, merge behavior, gradient paths, document isolation and
saved-run accounting. The completed TinyStories Spark result is preserved, not
rerun or reinterpreted as reliable storytelling.

[Evaluation/SFT](2026-10-04-evaluation-and-sft-course.md) includes actual full/LoRA
training on a four-request symbolic fixture. Full tuning reaches NLL 0.000555 and
seen sequence exact match 1.00; the declared constrained adapter reaches
NLL 2.802457 and exact match 0.75. Neither is broad assistant evidence.

[Preferences/policy](2026-10-04-preference-policy-cpu.md) retains reward shortcuts,
exact estimator checks and an actual negative tiny-decoder DPO outcome:
training margins improve while independent desired-answer probability falls.
The matched chosen-SFT control and three policy seeds are preserved.

[GRPO/failure/distillation](2026-10-04-grpo-diagnostics-distillation.md) includes
real tiny autoregressive rollouts, group-size costs, verifier edge cases, reward
hacking/repair and teacher-to-student optimization. Both GRPO arms retain
held-out 0/4 success; that negative result is not hidden behind training reward.

Spark SFT, DPO and RLVR protocols have concrete entry points, original data,
frozen small diagnostic panels, parent/template hashes, completion masks,
stop IDs, resource caps, checkpoint output and preserved failure records.
**No new Qwen model-scale experiment was executed.** A smoke arithmetic panel
is not broad reasoning evaluation; prepared capstone rows remain unexecuted
until their own evidence exists.

## Errors found and repaired

1. Old Day 3unfinished scaffolds interrupted reference execution. Explicit tagging
   and a tested verification-only skip preserve the exercises and execute answers.
2. An early headless plotting pass computed correctly but captured no figures.
   The verifier now uses an execution-only inline plotting preamble.
3. Wilson interval endpoint roundoff could produce a tiny negative Matplotlib
   error-bar width. Plot widths are clipped at zero without changing the interval.
4. The intended rare-regression fixture initially improved both slices. The
   corrected declared fixture gives A/B overall 62.5%/70%, common 66.7%/86.7% and
   rare 50%/20%; an invariant test now enforces aggregate-up/slice-down behavior.
   The paired mean difference 0.075 has bootstrap interval [−0.125, 0.275], not proof
   of population superiority. The original mistake is retained in its report.
5. HF checkpoint integration initially lacked a full policy export and used
   incompatible template-file assumptions. Full/adapter outputs are explicit;
   DPO reads current Jinja or legacy template storage and checks parent identity.
6. Cross-review tightened whole-run load/evaluation resource clocks, chat-versus-raw
   contracts and stopping behavior. SFT periodically saves atomic recovery state;
   RLVR journals baseline rows/completed updates before later guard failures.
   DPO/RLVR recovery files do not imply an unimplemented exact-resume CLI.

Preliminary failures are retained in earlier diagnostic/report records; the final
manifest identifies the corrected current sources rather than overwriting history.

## Reproduction and remaining boundaries

Use [AppendixD](../../book/appendices/d-reproduction-and-environments.md) for
portable setup and verification commands. Actual execution here used
Python 3.12.14, Torch 2.13.0+cu130, NumPy 2.5.2, Matplotlib 3.10.8,
nbclient 0.11.0 and nbformat 5.11.1 on Spark Linux/aarch64. The Torch CUDA build does not
mean these teaching references ran on GPU. No installation or model download
occurred, and Mac execution is not claimed without its own environment check.

The base revision is `c0c6c9d6988ed167267d423712e0df1c5efee09d`; scoped build
changes are local and uncommitted. No push, external publication, persistent
server or new animation render occurred.

Material creation is complete. Learner practice, larger trained comparisons,
the owner's repository license choice and public release remain distinct.
Read the [release checklist](../../docs/RELEASE_CHECKLIST.md) before publishing;
do not turn unexecuted model branches or missing permission into fabricated
results. The exact learning continuation remains Day 9diagnosis and a frozen
story-quality contract.
