# Mac CPU verification

This packet addresses parallel, still-unexecuted Mac portability verification.
It is not a compulsory Mac-success gate for Spark or DXI-17: the
[literal original-scope review](../../experiments/reports/2026-10-05-platform-scope-reading.md)
corrects that stronger historical interpretation. It does not restart the
learner at an earlier day, launch training or authorize Git publication.
The learner remains on Day9; verification is course production.

## Execution location and source

The app lists a local Mac project at `/Users/yongchao/dongxi-LLMs`. That is a
candidate checkout, not proof that a command has executed on macOS. The current
course task executes on Spark/Linux. No enabled tool in this task can launch a
shell command on that Mac project. Selecting or listing a project does not
produce a platform receipt.

On an actual Mac task, verify the location before doing any work:

```bash
pwd
uname -s
uname -m
git status --short --branch
git rev-parse HEAD
```

Require `Darwin` and `arm64` for the Mac Studio lane. Preserve a dirty checkout;
do not reset, discard or automatically stash it. A clean, properly authorized
checkout may follow the [arrival workflow](../LEARNING_WORKFLOW.md). Spark's
current upgrade changes are local and unpublished: pulling the last remote
commit cannot verify changes that have not been transferred. Record the
tested commit, dirty state, lock digest and executable-source hashes. A receipt
from older source remains useful historical evidence, not a current-source pass.

## Isolated environment and checks

Use the commands in [Appendix D](../../book/appendices/d-reproduction-and-environments.md),
with a new project-local CPU environment and kernel prefix. Do not copy Spark's
virtual environment, use Mac's system Python3.9 or install into a shared runtime.
The locked teaching route requires an available Python3.12 and uv. If either
is absent, record that prerequisite rather than silently downloading an
interpreter or weakening the lock.

Once environment setup is in scope and those prerequisites exist:

```bash
UV_PROJECT_ENVIRONMENT=.venv-course uv sync --locked --extra course --python 3.12 --no-python-downloads
.venv-course/bin/python -m ipykernel install --prefix .course-kernel --name dongxi-course --display-name "Python (Dongxi CPU course)"
```

Set `JUPYTER_PATH` to this checkout's absolute `.course-kernel/share/jupyter`
directory. From the verified Mac checkout, the existing bounded acceptance
command runs tests, source checks and five fresh notebooks:

```bash
JUPYTER_PATH=/Users/yongchao/dongxi-LLMs/.course-kernel/share/jupyter .venv-course/bin/python scripts/run_cpu_verification.py --kernel dongxi-course --host-reserve-gib 2 --command-timeout 600 --output outputs/mac-cpu-check-01
```

Use a new output directory for each attempt; retain failed attempts. Substitute
the actual checked path if the checkout differs. The2GiB value is the explicit
small CPU teaching policy, not a change to Spark's25GiB model-run reserve. The
runner disables CUDA and model-hub access and records the actual kernel prefix.
Linux `MemAvailable` is unavailable on Mac: retain its unknown value and the
portable teaching probe/skip reasons, rather than claiming a continuous minimum.

Full notebook execution is a separate, explicit scope:

```bash
JUPYTER_PATH=/Users/yongchao/dongxi-LLMs/.course-kernel/share/jupyter .venv-course/bin/python scripts/run_cpu_verification.py --kernel dongxi-course --host-reserve-gib 2 --full-notebooks --command-timeout 1800 --output outputs/mac-cpu-full-01
```

The small panel cannot imply that all76 notebooks ran. Do not use
`--export-figures` to overwrite reference assets during this verification.

## Evidence to return

Return the actual `cpu-verification.json`, command logs and notebook execution
manifest, including source hashes, kernel interpreter/packages, Darwin/ARM64,
lock identity, actual exit status, runtime, failures and skips. Check source
hashes before and after; executed copies must not replace learner cells.
Credentials, raw environment dumps, data caches and large checkpoints are not
part of the return packet.

When an actual Mac receipt becomes available, reconcile it with DXI-17's
original five criteria. Criterion3 requires honest separation of actual Linux
and unverified Mac evidence; it does not explicitly require successful execution
on every listed platform. Criterion5 likewise distinguishes source/local checks
from an actual hosted CI run without requiring hosted success. Preserve the
historical stronger Mac and hosted-CI requests as separate unfinished
portability/publication tasks. Neither has run here, and neither is a claimed
pass or a newly added mandatory package-success gate.

Predeclared Spark stages use actual Linux/source verification and their own
scientific, smoke/recovery/interface and resource gates. This packet does not
waive those gates, alter original dependency metadata or choose a new recipe,
sweep or publication action.
