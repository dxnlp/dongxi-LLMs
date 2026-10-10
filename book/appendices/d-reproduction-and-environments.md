# Appendix D — Reproduction Commands and Environments

The book has a portable CPU route and a platform-specific Spark route. The
verified environment for the 2026-10-04 material build is recorded in its
verification report. A future installation should capture its own versions;
do not infer Mac binary compatibility from a Linux execution result.
The subsequent [editorial verification record](../../experiments/reports/2026-10-10-book-editorial-pass.md)
binds its own revised lessons and commands separately from the earlier receipts.

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

Install Python 3.12 and uv first; these commands intentionally do not download
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
.venv-course/bin/python scripts/check_book_prose.py --strict-narrative
.venv-course/bin/python scripts/check_book_math.py
.venv-course/bin/python scripts/check_course_integrity.py
.venv-course/bin/python scripts/check_notebook_contracts.py
.venv-course/bin/python scripts/verify_course_notebooks.py --kernel dongxi-course
```

To execute only a focused route, add `--days 10 11`. The verifier starts a fresh
kernel for each source notebook, applies CPU/offline settings, retains executed
copies and source hashes in a new output directory, and reports failed paths.
`--export-figures` additionally regenerates reference PNG files. It preserves
source notebook cells and learner edits. The
[NBClient documentation](https://nbclient.readthedocs.io/en/latest/client.html)
describes the execution, working-directory and timeout interfaces used here.

The notebook contract check binds each reviewed challenge to its prediction,
checkpoint, adjacent runnable reference and explanatory Markdown. It separately
checks preserved code objects, outputs, IDs and notebook metadata against the
editorial baseline. Its visible-heading rules and explicit pairing ledger
support the recorded content review; they cannot prove that an explanation is
correct or that an arbitrary implicit question has been discovered. Numerical
claims still need fresh execution and reader review.

The prose checker separates number/word spacing from narrative review. Spacing
checks cover chapters, laboratories, solutions, front matter and appendices.
Operational identifiers and session language are reviewed in conceptual chapters;
reproduction details remain legitimate in this appendix and the runbooks. Exact
contextual exemptions live in `scripts/book_prose_exemptions.json` with reasons.
The strict command checks resolved findings; `--json` exposes counts and source
locations for an editorial audit. It preserves literal code, math and link targets.

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
reserve stays 25GiB. A small hosted CPU runner can explicitly use
`--host-reserve-gib 2` for these tiny references; that is not permission to
weaken the 25GiB model-scale Spark safety rule. On Mac the Linux `MemAvailable`
probe is unavailable and is recorded as unknown, not as a passed measurement.

The matched-control CLI-success integration is deliberately scoped to Linux
with an observed 25 GiB reserve. On Mac or an ineligible small hosted VM, that
one test records a skip; its scientific/helper controls remain enabled. Return
the skip and its reason with the actual platform receipt rather than presenting
it as a passed model-runner check. The orchestrator's scope describes only its
recorded invocation platform, not reproduction on another machine.

The separate tiny story replay test now explicitly selects a portable teaching
policy: one labelled `psutil` available-memory sample with a 2GiB threshold.
It keeps both seeds and both checkpointing modes, including each real fresh
2-to-5 continuation and retained failed spending. Unknown, malformed or low
samples refuse the child; there is no blanket Mac skip. The helper's default
Linux 25GiB collector and the production runner's guard are unchanged. Its
[37-test Linux CPU verification](../../experiments/reports/2026-10-05-native-profile-inspection/portable-story-verification.json)
does not prove a Mac or hosted run, nor a continuous memory minimum.

On Spark, use this same isolated CPU teaching route for course verification.
The platform GPU interpreter and kernel belong to separately declared native
workloads; do not synchronize the course lock into that shared environment.
A successful verifier run is
material readiness evidence; learner practice and model-scale outcomes remain
separate records.

## D.2.1 Local verification and platform evidence

The course uses local verification. The learner requested removal of the
GitHub Actions workflow on 2026-10-05; do not recreate hosted CI without a new
explicit request. The local verifier, dependency lock and notebook route remain
available. Earlier workflow definitions and source hashes describe the tested
historical revision, not a requirement to keep hosted automation enabled.

A Linux CPU installation proves only the reported Linux panel; a separate Mac
run would still not transfer live kernels or checkpoints to the learner's Mac
Studio. The
[reproduction specification](../../experiments/specs/2026-10-04-course-reproduction.md)
fixes these scope distinctions before measurement.

Original DXI-17 criterion 3 requires separate platform evidence, not a successful
Mac run. Criterion 5 requires distinguishing workflow source and locally tested
commands from hosted execution, not a compulsory hosted success. The
[original-text review](../../experiments/reports/2026-10-05-platform-scope-reading.md)
preserves earlier, stronger Mac/hosted verification requests as parallel work;
it does not convert portable source checks into another platform's pass.

| Platform or workflow | Actual evidence | Unobserved boundary |
| --- | --- | --- |
| Spark/Linux ARM64 | Earlier 2026-10-05 closure: 1,533 tests; all 76 fresh references, 539 code cells/535 executed/210 images/four preserved unfinished skips | CPU/Linux evidence is not a Mac, hosted, inline-host learner or model-quality pass |
| Mac | Portable route and return-evidence packet | Not executed/unverified in this task |
| Hosted CI | Workflow removed at the learner's request; earlier source evidence is historical | Not required; do not recreate without an explicit request |

The final [current-source verification report](../../experiments/reports/2026-10-05-final-course-verification.md),
[CPU test log](../../experiments/reports/2026-10-05-final-goal-cpu/run-01/tests.log)
and [full 76 fresh-kernel manifest](../../experiments/reports/2026-10-05-final-goal-notebooks/run-01/manifest.json)
record the goal-closure snapshot before CI removal. Full 76 execution and the final test/
five-selected-reference orchestrator were separate runs on the same frozen
sources, not two all 76 executions. The [signed original-criteria review](../../experiments/reports/2026-10-05-final-original-criteria-signoff.md)
closes 18/18 bounded packages against 86 unchanged criteria/dependencies, not
another platform's execution or learner mastery. The model-free
[archive closing receipt](../../experiments/reports/native-campaign-evidence-20261005-run-01/closing-bindings.json)
retains 45 original stage rows/74 model-directed terminal receipts and failures. Its source
and measurement JSON does not transfer weights, raw corpus, virtual environments
or live kernels. Preclosure 15/18 documents stay immutable; later status projection
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
and source hashes, with fresh model weights, a 14,000-update horizon and four-hour
cap. Analyze its portable JSON in the Day 9 notebooks without its raw checkpoint.

Reproduction of a model-scale run starts with a resource profile, not an
unqualified command copied into a fresh terminal. Preserve 25 GiB host reserve,
inspect concurrent model processes, and use the platform's bounded execution
mechanism. Actual GPU learning outcomes require their own reports.

## D.4 Post-training experiment route

Chapters 9,11 and 13 supply SFT, DPO and RLVR protocols. Their CPU integrations
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
passes 998 tests and six selected fresh references, not all 76 on another host.
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
historical and superseded by the original criterion 3 evidence-separation rule
above. Mac remains not executed/unverified and hosted CI remains not executed.
Neither is an added successful-run prerequisite for the authorized Spark
campaign or original upgrade signoff. Independent in-scope work can continue
without another routine approval question; final current-source Linux
verification and the dependent campaign defense still require actual evidence.

## D.6 Checkpoints, recovery and job supervision

A checkpoint is a claim about a boundary in computation. A useful manifest
names the parent weights, tokenizer and template, trainable parameter set,
objective, optimizer/data/RNG state and execution identity. Matching tensor
shapes alone does not preserve token meaning or a continuation. Chapter 1's
eight-ID example illustrates the semantic boundary; Chapters 5, 6, 9, 11 and
13 explain what the additional state means.

This section consolidates reproduction detail extracted from those chapters.
The unchanged reports retain exact manifests, failures, journal fields,
timings and commands; this reference does not replace their empirical record.

### Budget boundaries and spent work

Keep two clocks: accepted logical progress and physical work already spent.
An optimizer-update limit counts committed updates. A physical resource
journal also retains loading, failed attempts, generation, evaluation,
checkpointing and interrupted work. Resume must not refund that spending.
Similarly, a generated-token counter differs from valid training targets,
processed positions and unique source targets. Name each denominator.

Check a proposed update's complete geometry before doing the work. If it
cannot fit the remaining target/update budget, refusal should leave parameters,
optimizer history and data/RNG cursors unchanged. Deadline bounds require
checking later work too: a last evaluation or export can exceed a job cap
even after every optimizer update respected its limit. The
[matched story controls](../../experiments/reports/2026-10-05-native-story-first400-comparison.md)
and [native recovery deadline report](../../experiments/reports/2026-10-05-native-dpo-recovery-deadline.md)
show why accepted updates and accepted supervision are different claims.
Use the reports' actual source/config identities for reproduction.

### Supervised fine-tuning state and adapter exports

Saving a weight tensor is insufficient for exact training replay. Retain Adam
moments, schedule position, data order/cursor, CPU/CUDA RNGs and the committed
boundary under the declared replay contract. A checkpoint save must become
durable before it can be advertised as a recovery point; a file that exists
after a partial write is not automatically a committed snapshot. Payload
format, atomic replacement and digest verification remain implementation
details of the canonical runners, rather than new course infrastructure.

For LoRA, name both frozen base identity and adapter weights/configuration.
Export changes the representation of the update, so compare adapter and merged
paths under a declared dtype and tolerance. The retained comparison found a
BF16 merge mismatch and separately accepted an independently verified FP32
merge. Preserve both outcomes: a successful high-precision export does not
retroactively make the failed lower-precision one pass. See the
[native SFT replay report](../../experiments/reports/2026-10-05-native-sft-replay.md),
[assistant comparison](../../experiments/reports/native-assistant-comparison-20261005-run-02/README.md)
and [SFT runner guide](../../docs/runbooks/sft_spark_runner.md).

### Policy reference and pending rollout identity

DPO's reference is part of its objective. Replacing it with the latest policy
on resume changes the log-ratio baseline; recovering policy weights alone
does not recover the same optimization problem. A matched chosen-SFT control
also needs the same sampled examples, exposure and update geometry. See the
[DPO recovery report](../../experiments/reports/2026-10-05-native-dpo-recovery-deadline.md)
and [runner guide](../../docs/runbooks/dpo_spark_runner.md).

For sampled policy optimization, a collected pending rollout is already an
observation. Retain its tokens, old-policy identity and likelihoods, masks,
stops, rewards and applicable sampler state. Applying that same update before
collecting a new batch preserves the experimental history; resampling after
an interruption replaces it and spends more work. Old-policy likelihoods
determine importance ratios, while the fixed reference determines the KL
anchor. They are distinct even when their weights initially agree.
The [tiny cache/recovery report](../../experiments/reports/2026-10-04-batched-cache-recovery.md)
tests saved-cache replay separately from full-prefix rebuild at a tolerance;
the [native RLVR recovery report](../../experiments/reports/2026-10-05-native-rlvr-recovery-cleanup.md)
is separate model-scale evidence. Use the [RLVR runner guide](../../docs/runbooks/rlvr_runner.md)
for the operational route.

A local digest can detect altered bytes relative to an independently trusted
expected identity. A sidecar supplied by the same untrusted sender does not
authenticate a checkpoint. Data-only tensor loading narrows the interface,
but the bounded trusted local fixtures are not an adversarial sandbox. Never
load arbitrary pickle checkpoints to reproduce an example.

### Supervision and durable evidence

Scientific validity, exit status, cleanup and resource compliance form a
conjunction when the run contract requires all of them. A child exit of zero
does not imply the supervisor accepted the job. An unknown exit remains
unknown; a timeout, failed cleanup gate or missing response stays in the record.
The RLVR pilot evidence retains G4's failed supervision and separate export
closure alongside G8's accepted pilot. Six completed initial conditions and
two failed ones cannot be rewritten as eight completed comparisons.

Keep evaluation observations distinct from missing rows. Preserve the retained
response bytes, stop reason, behavior/target likelihood meanings, grading rule,
coverage and selection rule. The [evaluation guide](../../docs/runbooks/evaluation_tools.md)
explains how to read those artifacts. A recovered job demonstrates continuity
for its tested state and platform, not broader model quality or portability.

Before supervising a new run, follow the existing platform resource policy,
declare finite limits and preserve the required host reserve. Reuse adequate
tools and upstream fixes. This book's editorial verification uses fresh local
CPU references and does not launch a new model-scale campaign.
