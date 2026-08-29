# Appendix A — Laboratory Setup and Reproducible Runs

This appendix supports Chapter 1's Qwen3 smoke-test case study. It contains the
machine-specific procedure; the conceptual standard for experiments remains in
[Chapter 1](../chapters/01-evidence-before-optimization.md).

The commands target the NVIDIA DGX Spark used for this project. Treat the
[DGX Spark platform repository](https://github.com/dxnlp/dgx-spark-dongxi) as the
current source for platform compatibility, locked environments, memory safeguards,
and troubleshooting. Do not assume these commands or versions apply unchanged to
another machine.

## A.1 What belongs where

The project separates four kinds of artifact:

| Artifact | Purpose | Location |
|---|---|---|
| Book prose | Teach a durable argument | `book/` |
| Experiment specification | Freeze the intended protocol before execution | `experiments/specs/` |
| Experiment report | Preserve observations, interpretations, and limitations | `experiments/reports/` |
| Raw outputs | Retain logs, telemetry, and model artifacts | platform `outputs/` directory |

Machine and Python identities live in `manifests/`. Reusable course code belongs
in `src/dongxi_llms/`; notebooks narrate experiments but should import reusable
logic rather than duplicate it.

## A.2 Safety boundary

DGX Spark uses unified memory. Preserve at least 20–25 GiB for the operating
system, desktop, agents, and interactive work. Large jobs should execute in a
bounded cgroup and should not share the machine with an unnecessary inference
server.

Never discover capacity by intentionally driving the host to system OOM. Begin
with the smallest useful configuration and profile upward.

Before a material run:

1. identify processes you own and any concurrent model servers;
2. record starting `MemAvailable`;
3. confirm the job's cgroup memory and swap limits;
4. sample whole-system and CUDA-visible memory during execution;
5. stop before the system reserve is threatened.

The platform wrappers apply a 100 GiB job limit and disable swap within that job
scope by default. The host's normal swap configuration remains unchanged.

Set explicit repository paths once for the commands below. Replace the example
values when your clones live elsewhere:

```bash
DONGXI_COURSE_REPO=/home/dongxi/dongxi_ai/Dongxi_LLMs
DONGXI_PLATFORM_REPO=/home/dongxi/dgx-spark-dongxi
```

## A.3 Verify the host

From the platform repository:

```bash
cd "$DONGXI_PLATFORM_REPO"
./scripts/check_system.sh
```

Inspect the output rather than treating a zero exit code as the complete result.
The check should identify architecture, DGX OS, GPU and driver, CUDA toolkit,
Python prerequisites, usable and currently available memory, swap state, and tool
availability.

A warning may be acceptable; a failure needs to be understood before training.
Do not suppress warnings merely to make the setup look clean.

## A.4 Recreate the native locked environment

```bash
cd "$DONGXI_PLATFORM_REPO"
./scripts/setup_native.sh
```

The important evidence is not package installation alone. The verifier should
execute representative operations on the GPU, including BF16 scaled dot-product
attention with backward computation, a Triton kernel, and a compiled forward and
backward path.

The native environment is located at:

```text
$DONGXI_PLATFORM_REPO/.venv
```

Keep workload-specific environments isolated. Installing packages manually into
this environment changes its identity and makes the existing lock insufficient to
describe it.

## A.5 Capture the environment manifest

The course repository includes a credential-free collector:

```bash
cd "$DONGXI_COURSE_REPO"

"$DONGXI_PLATFORM_REPO/.venv/bin/python" \
  scripts/capture_environment.py \
  --lock-file "$DONGXI_PLATFORM_REPO/uv.lock" \
  --output manifests/YYYY-MM-DD-dgx-spark-native.json
```

Inspect the result before committing it. It records:

- code commit and dirty-worktree state;
- operating system, kernel, architecture, GPU, driver, and CUDA toolkit;
- total and currently available memory;
- interpreter and course-relevant package versions;
- lock-file path and SHA-256 hash;
- confirmation that hostname, username, and credentials were not collected.

Available memory in the manifest is context at capture time. The experiment report
must separately record starting and minimum available memory during the actual run.

## A.6 Recreate the pinned trainer environment

The Qwen3 SFT profile uses a separate pinned open-instruct checkout and environment:

```bash
cd "$DONGXI_PLATFORM_REPO"
./scripts/setup_open_instruct.sh
```

The setup verifies the expected upstream revision, applies the recorded DGX Spark
compatibility patch, recreates the frozen environment, and runs representative GPU
operations inside the memory-safe scope.

Its environment is separate from the native environment:

```text
$DONGXI_PLATFORM_REPO/.venv-open-instruct
```

Do not combine the two environments simply because both contain PyTorch and
Transformers. Their versions and purposes differ.

Capture a second manifest with the trainer interpreter and trainer lock:

```bash
cd "$DONGXI_COURSE_REPO"

"$DONGXI_PLATFORM_REPO/.venv-open-instruct/bin/python" \
  scripts/capture_environment.py \
  --lock-file "$DONGXI_PLATFORM_REPO/.external/open-instruct/uv.lock" \
  --output manifests/YYYY-MM-DD-dgx-spark-open-instruct.json
```

Record the trainer commit and compatibility-patch hash in the experiment
specification as well as the environment manifest.

## A.7 Write the specification before the run

Copy `experiments/specs/template.yaml` and fill every applicable identity and
configuration field. Resolve model and dataset revisions before execution. Define
success and failure criteria that can be decided from preserved outputs.

For the Chapter 1 case study, the completed contract is:

[`../../experiments/specs/2026-08-29-qwen3-0.6b-sft-smoke.yaml`](../../experiments/specs/2026-08-29-qwen3-0.6b-sft-smoke.yaml)

Review these questions before launching:

- Is the mode declared as smoke, learning, or reference?
- Is the hypothesis falsifiable?
- Are model, data, trainer, patch, and environment identities immutable?
- Are the batch geometry and sequence lengths explicit?
- Is the memory reserve a decision criterion rather than a vague intention?
- Are the claims this run cannot establish written down?
- Are output paths known and writable?

If a material field changes after review, update the planned specification before
execution. If the protocol changes during execution, preserve the deviation in the
report rather than rewriting history.

## A.8 Execute the Qwen3 smoke profile

From the platform repository:

```bash
cd "$DONGXI_PLATFORM_REPO"

MAX_TRAIN_STEPS=3 PROFILE_SAMPLES=32 \
  ./scripts/memory_profile_sft.sh 1 1 1024 Qwen/Qwen3-0.6B
```

The positional values are:

```text
microbatch size = 1
gradient accumulation = 1
sequence length = 1024
model = Qwen/Qwen3-0.6B
```

The wrapper enters the memory-safe scope automatically and writes a timestamped
directory below:

```text
$DONGXI_PLATFORM_REPO/outputs/memory-profiles/sft/
```

Do not infer success from the last visible progress bar. Preserve and inspect the
exit status, training log, memory summary, sampled telemetry, and output model.

## A.9 Inspect the evidence

For a selected run directory:

```bash
run_dir="$DONGXI_PLATFORM_REPO/outputs/memory-profiles/sft/RUN_ID"

sed -n '1,120p' "$run_dir/summary.txt"
rg 'Step: ' "$run_dir/training.log"
find "$run_dir/model" -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort
```

Derive starting and minimum available memory from the sampled system log:

```bash
awk '
  NR == 2 { start = $3 }
  NR > 1 && (minimum == 0 || $3 < minimum) { minimum = $3 }
  END {
    printf "start_available_gib=%.2f\n", start / 1024 / 1024
    printf "minimum_available_gib=%.2f\n", minimum / 1024 / 1024
  }
' "$run_dir/system-memory.tsv"
```

Check model and dataset cache revisions against the planned identities. A Hub
repository name without the resolved revision is not enough for the durable
report.

## A.10 Write the report

Copy `experiments/reports/template.md`. Preserve:

- intended protocol and deviations;
- exact command and exit status;
- all decision measurements;
- representative output locations;
- observations separated from interpretations;
- failed criteria, warnings, and surprises;
- claims not established;
- the decision and exact next experiment.

The completed Chapter 1 report is:

[`../../experiments/reports/2026-08-29-qwen3-0.6b-sft-smoke.md`](../../experiments/reports/2026-08-29-qwen3-0.6b-sft-smoke.md)

Raw model files remain in the gitignored output directory; the small durable report
links the evidence and carries the conclusion.

## A.11 Reproduction checklist

Before declaring the laboratory ready, confirm:

- [ ] Host check inspected, with failures resolved and warnings understood
- [ ] Locked environment recreated without ad hoc package changes
- [ ] Representative GPU forward, backward, and kernel paths executed
- [ ] Machine and Python manifest captured
- [ ] Model, data, trainer, code, patch, and lock revisions recorded
- [ ] Hypothesis and decision criteria written before execution
- [ ] Job launched inside the memory boundary
- [ ] Logs, three memory views, exit status, and outputs preserved
- [ ] Observations separated from interpretations
- [ ] Unsupported claims named explicitly
- [ ] Experiment report and exact next action recorded

Passing this checklist establishes a trustworthy starting point. It does not make
future experiments trustworthy automatically; each material run needs its own
identity, contract, evidence, and bounded conclusion.
