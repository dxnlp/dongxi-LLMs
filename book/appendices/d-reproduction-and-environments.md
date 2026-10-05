# Appendix D — Reproduction Commands and Environments

The book has a portable CPU route and a platform-specific Spark route. The
verified environment for the 2026-10-04 material build is recorded in its
verification report. A future installation should capture its own versions;
do not infer Mac binary compatibility from a Linux execution result.

## D.1 CPU learning environment

Use a separate Python 3.12 environment for course teaching. It avoids the Mac's
system Python 3.9 and keeps installations separate from the Spark platform lock.
The canonical teaching route now uses `pyproject.toml` plus `uv.lock` instead of
the older range-only requirements file. Linux resolves Torch from an explicit
CPU index; Mac resolves its CPU wheel from PyPI. This follows the
[official uv PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/).
It deliberately does not inherit or replace Spark's GPU lock.

```bash
UV_PROJECT_ENVIRONMENT=.venv-course uv sync --locked --extra course --python 3.12 --no-python-downloads
.venv-course/bin/python -m ipykernel install --prefix .course-kernel --name dongxi-course --display-name "Python (Dongxi CPU course)"
```

Install Python3.12 and uv first; these commands intentionally do not download
another interpreter. Set `JUPYTER_PATH` to the absolute
`.course-kernel/share/jupyter` directory so Jupyter discovers this project-local
kernel. A temporary-prefix kernel does not change the user's existing kernel.
Use a new environment path: exact synchronization may remove packages from its
target environment, so never point it at the shared Spark environment.
`--locked` rejects stale project metadata instead of silently changing the lock;
`--frozen` would omit that freshness check. See
[uv's locking and syncing contract](https://docs.astral.sh/uv/concepts/projects/sync/).
The older `notebooks/requirements-course.txt` remains an explicitly unlocked
compatibility route, not reproduction evidence. The
[IPython kernel guide](https://ipython.readthedocs.io/en/9.9.0/install/kernel_install.html)
explains why a virtual environment must be registered separately with Jupyter.

The `course` extra also includes Transformers/tokenizers/PEFT: CPU tests construct
tiny random local models and verify their serialization/adapter interfaces. They
do not acquire pretrained weights. Model-scale runners still require a
separately profiled and authorized environment. The foundation evidence lab
also has a standard-library-only route.

## D.2 Execute without a notebook server

From the repository root:

```bash
PYTHONPATH=src .venv-course/bin/python -m unittest discover -s tests
.venv-course/bin/python scripts/check_book_math.py
.venv-course/bin/python scripts/check_course_integrity.py
.venv-course/bin/python scripts/verify_course_notebooks.py --kernel dongxi-course
```

To execute only a focused route, add `--days 10 11`. The verifier starts a fresh
kernel for each source notebook, applies CPU/offline settings, retains executed
copies and source hashes in a new output directory, and reports failed paths.
`--export-figures` additionally regenerates reference PNG files. It preserves
source notebook cells and learner edits. The
[NBClient documentation](https://nbclient.readthedocs.io/en/latest/client.html)
describes the execution, working-directory and timeout interfaces used here.

The [course manifest](../../docs/course_manifest.json) declares each chapter,
daily index, notebook lane and dependency. `--notebooks` selects explicit
registered paths; `--lane extension` selects extensions without treating them
as additional days. Unknown selectors fail instead of returning an apparently
successful empty run. Integrity checks reject missing/empty artifacts,
day/chapter disagreement, unregistered notebooks and dependency cycles.

For the five-notebook acceptance panel plus all tests and source checks:

```bash
.venv-course/bin/python scripts/run_cpu_verification.py --kernel dongxi-course --output outputs/cpu-check-01
```

Choose a new empty output directory for every invocation. The orchestrator keeps
each command's actual exit, timeout, log hash and source identity, and continues
the independent checks after a failed test. Fresh notebooks record the actual
kernel interpreter/packages, not only the orchestrator's Python, and the
acceptance runner requires its kernel environment prefix to match. Two virtual
environments can link to the same Python binary without sharing dependencies.
Partial outputs
and errors remain visible. Timeouts stop that invocation's process group and
its detected descendant processes, including separately launched kernel
sessions; unrelated notebook servers are not targets.

`--full-notebooks --command-timeout 1800` explicitly selects the full registered
inventory; it is not implied by a passed small panel. Spark's default pre-notebook
reserve stays25GiB. A small hosted CPU runner can explicitly use
`--host-reserve-gib 2` for these tiny references; that is not permission to
weaken the25GiB model-scale Spark safety rule. On Mac the Linux `MemAvailable`
probe is unavailable and is recorded as unknown, not as a passed measurement.

The matched-control CLI-success integration is deliberately scoped to Linux
with an observed 25 GiB reserve. On Mac or an ineligible small hosted VM, that
one test records a skip; its scientific/helper controls remain enabled. Return
the skip and its reason with the actual platform receipt rather than presenting
it as a passed model-runner check. The orchestrator's scope describes only its
recorded invocation platform, not reproduction on another machine.

The separate tiny story replay test now explicitly selects a portable teaching
policy: one labelled `psutil` available-memory sample with a2GiB threshold.
It keeps both seeds and both checkpointing modes, including each real fresh
2-to-5 continuation and retained failed spending. Unknown, malformed or low
samples refuse the child; there is no blanket Mac skip. The helper's default
Linux25GiB collector and the production runner's guard are unchanged. Its
[37-test Linux CPU verification](../../experiments/reports/2026-10-05-native-profile-inspection/portable-story-verification.json)
does not prove a Mac or hosted run, nor a continuous memory minimum.

On Spark, substitute the established interpreter
`/home/dongxi/dgx-spark-dongxi/.venv/bin/python` and kernel `dgx-spark-native`.
Do not copy that virtual environment onto Mac. A successful verifier run is
material readiness evidence; learner practice and model-scale outcomes remain
separate records.

## D.2.1 Local verification and platform evidence

The course uses local verification. The learner requested removal of the
GitHub Actions workflow on2026-10-05; do not recreate hosted CI without a new
explicit request. The local verifier, dependency lock and notebook route remain
available. Earlier workflow definitions and source hashes describe the tested
historical revision, not a requirement to keep hosted automation enabled.

A Linux CPU installation proves only the reported Linux panel; a separate Mac
run would still not transfer live kernels or checkpoints to the learner's Mac
Studio. The
[reproduction specification](../../experiments/specs/2026-10-04-course-reproduction.md)
fixes these scope distinctions before measurement.

Original DXI-17 criterion3 requires separate platform evidence, not a successful
Mac run. Criterion5 requires distinguishing workflow source and locally tested
commands from hosted execution, not a compulsory hosted success. The
[original-text review](../../experiments/reports/2026-10-05-platform-scope-reading.md)
preserves earlier, stronger Mac/hosted verification requests as parallel work;
it does not convert portable source checks into another platform's pass.

| Platform or workflow | Actual evidence | Unobserved boundary |
| --- | --- | --- |
| Spark/Linux ARM64 | Final clean locked CPU panel:1,533 tests; all76 fresh references,539 code cells/535 executed/210 images/four preserved unfinished skips | CPU/Linux evidence is not a Mac, hosted, inline-host learner or model-quality pass |
| Mac | Portable route and return-evidence packet | Not executed/unverified in this task |
| Hosted CI | Workflow removed at the learner's request; earlier source evidence is historical | Not required; do not recreate without an explicit request |

The final [current-source verification report](../../experiments/reports/2026-10-05-final-course-verification.md),
[CPU test log](../../experiments/reports/2026-10-05-final-goal-cpu/run-01/tests.log)
and [full76 fresh-kernel manifest](../../experiments/reports/2026-10-05-final-goal-notebooks/run-01/manifest.json)
record the goal-closure snapshot before CI removal. Full76 execution and the final test/
five-selected-reference orchestrator were separate runs on the same frozen
sources, not two all76 executions. The [signed original-criteria review](../../experiments/reports/2026-10-05-final-original-criteria-signoff.md)
closes18/18 bounded packages against86 unchanged criteria/dependencies, not
another platform's execution or learner mastery. The model-free
[archive closing receipt](../../experiments/reports/native-campaign-evidence-20261005-run-01/closing-bindings.json)
retains45 original stage rows/74 model-directed terminal receipts and failures. Its source
and measurement JSON does not transfer weights, raw corpus, virtual environments
or live kernels. Preclosure15/18 documents stay immutable; later status projection
is separate from scientific archive identity.

The [earlier complete Linux panel](../../experiments/reports/2026-10-05-current-cpu-closure.md)
remains evidence for its own sealed revision. Its historical stronger
Mac/hosted-gate wording does not override the original criteria or establish
current-source verification after later executable edits. No host, package or
learner-completion status is changed by the commands in this appendix.
Before implementing a reproduction gap, check current upstream fixes and the
installed stack, reuse adequate existing tools and retire resolved/irrelevant
work. Historical memory-hang warnings do not authorize new monitoring machinery;
the standard finite-resource bounds above remain.

## D.3 Existing measured GPU run

The TinyStories [pipeline guide](../../docs/TINYSTORIES_PIPELINE.md) and
[completed report](../../experiments/reports/2026-09-14-tinystories-learning-result.md)
describe the actual first learning run. It used pinned data/tokenizer revisions
and source hashes, with fresh model weights, a14,000-update horizon and four-hour
cap. Analyze its portable JSON in the Day 9 notebooks without its raw checkpoint.

Reproduction of a model-scale run starts with a resource profile, not an
unqualified command copied into a fresh terminal. Preserve25 GiB host reserve,
inspect concurrent model processes, and use the platform's bounded execution
mechanism. Actual GPU learning outcomes require their own reports.

## D.4 Post-training experiment route

Chapters9,11 and13 supply SFT, DPO and RLVR protocols. Their CPU integrations
test their objectives and graph boundaries. Optional model runners require
explicit model/checkpoint/tokenizer revisions, local data contracts, evaluation
panels and resource bounds. Read their corresponding lab guides before invoking
them. A config for an unexecuted pretrained comparison is not a measured model.

Record the parent checkpoint and objective. Full SFT and LoRA have different
trainable parameter counts; DPO needs a frozen reference; policy optimization
needs a named rollout policy and correct old/current/reference roles. Retain
CPU and CUDA RNG state, optimizer history, data position and schedule where
the declared resume contract requires them. Never trust arbitrary checkpoint
pickle files from an unverified source.

The [earlier CPU source checkpoint](../../experiments/reports/2026-10-05-rlvr-io-readiness.md)
passes998tests and six selected fresh references, not all76 on another host.
Native SFT/DPO/RLVR recovery uses separate model/semantic and shared-I/O
journals with independently retained receipts before payload inspection.
The same physical journal retains later failed spending; copying receipts is
not cross-machine budget verification. Read the exact CLI/lab inputs before
running: incomplete caps are not inferred, and source readiness does not
authorize pretrained loading, GPU work or private physical-resource changes.

## D.5 Return evidence and machine switches

For every run, return a concise manifest: source identity, execution host,
interpreter/packages, configuration, data/model identities, actual exit status,
runtime, target/update counts, resource measurements and report path. Avoid
environment dumps that include credentials. Commit small evidence and source;
keep weights and raw dataset caches under ignored output paths.

The machine-switch phrases and [handoff](../../docs/handoffs/CURRENT.md) preserve
the learning position. Repository synchronization does not synchronize a live
kernel, running process, credential or local checkpoint directory. A final
release additionally needs a scoped publication decision and its own review.

For the parallel, still-unexecuted Mac verification, use the
[return-evidence packet](../../docs/handoffs/MAC_CPU_VERIFICATION.md). A listed
Mac checkout is not command access or a platform pass. Earlier guidance treated
actual supported Mac execution as mandatory; that stronger interpretation is
historical and superseded by the original criterion3 evidence-separation rule
above. Mac remains not executed/unverified and hosted CI remains not executed.
Neither is an added successful-run prerequisite for the authorized Spark
campaign or original upgrade signoff. Independent in-scope work can continue
without another routine approval question; final current-source Linux
verification and the dependent campaign defense still require actual evidence.
