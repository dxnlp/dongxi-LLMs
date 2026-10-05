# A clean CPU route without inheriting Spark CUDA

DXI-17 is in progress. The declarative routing contract, CPU dependency lock,
fresh-kernel identity/failure checks and original CI definitions are built.
An actual new Linux/aarch64 environment installs and executes the selected
panel; Mac Studio and hosted CI execution remain pending.

The [predeclared specification](../specs/2026-10-04-course-reproduction.md)
and [actual JSON evidence](2026-10-04-course-reproduction.json) distinguish the
early registry failure from the completed initial notebook panel. No old report
is replaced, no pretrained weights are downloaded, and the shared Spark
environment/kernel is unchanged.

## What was actually checked

A new temporary environment at
`/tmp/dongxi-course-reproduction.ZfVaEu/venv` installed122packages from the
resolved lock using uv0.12.6. Actual Python3.12.14, Torch2.14.1+cpu with
`torch.version.cuda=None`, Transformers5.18.0, tokenizers0.23.2, PEFT0.20.0,
NumPy2.5.3 and Matplotlib3.10.8 differ from the established Spark GPU stack.
The first clean-environment full suite passed277tests in13.895s at that
intermediate source revision. Later-added source and any final integrated
check must be identified separately, not silently included in that count.

Five fresh kernels then passed Day1 checkpoint-interface, Day3 logits/softmax,
Day10 mathematical grading, Day15 simulated judging and Day20 sampling-support
notebooks. They contain39code cells;35complete reference cells executed and
four explicitly unfinished Day3 learner exercises were preserved/skipped.
Thirteen original image outputs were produced. Minimum sampled pre-notebook
`MemAvailable` was118.154GiB; this is not continuous peak-memory measurement.
Each notebook records its actual isolated kernel interpreter/package identity,
source hash, output location and image counts. Source notebooks are untouched.

The earlier integrated attempt failed before commands because a new Day16
notebook was not yet registered during parallel authoring. Its failure and
source/lock identities are retained in JSON. Registration was subsequently
added; it does not itself prove that the new experiment passed its criteria.

## Contracts and failure checks

- [Registry](../../docs/course_manifest.json) and
  [validator](../../src/dongxi_llms/course_manifest.py): exactly15chapters/
  28day routes; core/optional/extension lanes; explicit dependencies; unknown,
  duplicate, unsafe/symlink, mismatched, missing/empty and cyclic routes rejected.
- [Verifier](../../scripts/verify_course_notebooks.py): explicit unknown
  selectors fail; declared routes choose figure destinations. In execution
  copies only, source skip tags cannot hide complete references and tensor
  ellipsis indexing is not misclassified as an unfinished exercise. Actual
  kernel identity and partial failed outputs are retained. Independent review
  found that an output directory without a manifest could still contain old
  notebooks; all nonempty output directories are now rejected. The locked
  acceptance runner also checks the kernel's actual environment prefix, not
  interpreter symlink equality or a display-name assertion.
- [Orchestrator](../../scripts/run_cpu_verification.py): actual command exits,
  timeouts, logs/hashes and changed-source detection. The first cleanup design
  assumed kernels shared the parent process group; inspection of installed
  Jupyter showed separate sessions. Explicit descendant cleanup and an actual
  independent-session timeout regression replaced that assumption.
- [CI definition](../../.github/workflows/course-cpu.yml): read-only permissions,
  immutable action IDs checked against official action repositories, Linux x64
  and Mac ARM64 source lanes, locked CPU installation, temporary-prefix kernel,
  offline model hub execution and retained artifacts even after failure.
  A local YAML/security/lock check is not a hosted run.

Linux's explicit CPU index prevents automatic CUDA-wheel selection. Mac uses
the locked PyPI CPU candidate; an available wheel in a lock is not binary
compatibility evidence for the learner's machine. The lock and commands follow
[uv's PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/)
and [locked synchronization contract](https://docs.astral.sh/uv/concepts/projects/sync/).

## Remaining criteria and interpretation

The [final consolidated verification](2026-10-04-course-reproduction-verification.json)
now passes all291tests (unittest reports7.099s; process wall8.818s),
54book files/1114math expressions and15chapters/15solutions/4appendices/
67registered notebooks with zero source/navigation issues. All five selected
fresh kernels pass35complete reference cells/13image outputs, preserving the
four unfinished Day3exercises. The actual kernel prefix matches the isolated
environment, all four supervised commands exit0 without timeout and no
recorded source changes during the run occur. All source hashes/actual logs,
platform identities and notebook manifests are retained. Earlier attempts
remain separately labeled historical records.

Independent final review passes38focused tests. In addition to output/kernel
and separate-session cleanup protections, it exposed control/NUL-containing
paths, malformed selector types and lost malformed-notebook JSON diagnostics.
These now fail with retained validation problems and have regressions. A
[shared command-identity correction](2026-10-04-command-provenance-hardening.md)
also distinguishes `python -m` launch arguments from Python's rewritten argv.

Separately execute the supported
Python3.12 locked route on the actual Mac, and record failures rather than
inferring a pass from Linux. Hosted CI execution awaits an appropriate Git/
workflow request. The explicit full-notebook command is in
[AppendixD](../../book/appendices/d-reproduction-and-environments.md); the initial
five-notebook panel does not prove a new full67-session pass.

Dependency installation contacts public package registries; model-hub execution
uses offline settings. No model/dataset acquisition, GPU campaign, live GitHub
math-render test, server restart, publication, animation rendering or Git
commit/push is part of this verification. Learner position remains Day9.
