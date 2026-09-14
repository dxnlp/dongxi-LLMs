# Read-only training observatory

## Optional prompt playground — added2026-09-14

At the learner's explicit request, the dashboard now includes **Prompt DongxiGPT**
above the training charts. The opt-in service loads the completed run's final
checkpoint onto Spark for inference only. Default monitoring without the flag
retains the log-only behavior described below.

```bash
PYTHONPATH=src OMP_NUM_THREADS=4 /home/dongxi/dgx-spark-dongxi/.venv/bin/python \
  -m dongxi_llms.training_monitor --runs outputs --port 8765 \
  --playground-run day09-learning-01
```

The running service is `dongxigpt-playground-20260914.service`, replacing the
stopped log-only `dongxigpt-monitor-20260914.service`. It stays on loopback8765
with8G cgroup cap and no cgroup swap. It holds the model on GPU while idle;
before future training, explicitly stop this known inference service to avoid
concurrent GPU workloads. Do not start another monitor on the same port.

The page always identifies the fixed inference run/update separately from the
chart selector. Give a story opening rather than expecting chat-style instruction
following. Temperature0 is greedy; positive temperature samples with a local
seed. Prompts/completions are not saved by the playground. They remain in the
current browser page; no external API is called. Checkpoint files and training
logs remain unchanged.

Bounds: one request at a time,1–256 new tokens, at most767 prompt tokens,
1024 total positions including BOS,60-second generation deadline checked between
tokens,25GiB host available reserve before loading/generation and between tokens.
Oversized/nonfinite/invalid requests are rejected; EOS, token limit, context,
deadline and memory stops are distinguished. Requests need JSON and an explicit
custom header; foreign origins/hosts are rejected and no CORS access is granted.
The server—not a client path—chooses the checkpoint. Tensor-only loading uses
CPU mmap, discards optimizer references, then copies only the model to CUDA.
No optimizer is created, no gradient computation occurs, and no training controls
were added. There is no streaming/cancel endpoint in this first version.

Verification:97 tests passed; a real final-checkpoint request returned114 tokens
and natural EOS in about1.02 seconds. This is an API smoke check, not a latency
benchmark or browser visual/interaction QA. See the playground verification report.

## Default log-only mode

The dashboard reads the story pipeline's existing files. It neither imports
PyTorch nor loads checkpoints, owns training processes, changes recipes or
enforces safety. Closing the page/server does not stop the trainer.

## Start on Spark

From the course repository:

```bash
PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python \
  -m dongxi_llms.training_monitor --runs outputs --port 8765
```

Open `http://127.0.0.1:8765/?run=day09-smoke-v2`. Any direct child directory
containing `run.json` appears in the experiment selector. New observations are
read on the next refresh (every three seconds while the page is visible).
No GPU or external web libraries are required; assets and API are same-origin.
The server binds only127.0.0.1. Do not expose this development monitor publicly.

On Mac, open an SSH tunnel using your actual Spark SSH alias:

```bash
ssh -N -L 8765:127.0.0.1:8765 YOUR_SPARK_SSH_ALIAS
```

Then open `http://127.0.0.1:8765` on Mac. Keep the tunnel and server alive;
training itself stays on Spark. If the local port is occupied, choose another
local port in the tunnel and browser. This is a browser view of Spark logs,
not automatic synchronization of the Python environment or checkpoint files.

## Views and evidence boundaries

- Raw batch/validation NLL, learning rate, pre-clip gradient norms and measured
  valid-target throughput, against optimizer updates or cumulative valid tokens.
- Fixed prompt comparisons across initial/checkpoint/final observations and
  greedy/fixed-seed sampling. Text uses safe text rendering, not HTML execution.
- Valid-position utilization, per-update timings and clipping frequency over
  the displayed records; no inferred wall-clock ETA.
- Sampled host available memory with the reserve threshold. Final CUDA allocated
  and reserved peaks are labeled as peaks, not current GPU telemetry.
- Latest observed per-stage activation RMS, exact-metric table and checkpoint
  file sizes. No checkpoint tensor data is exposed.

The trainer's update timings exclude validation and saving. Separate durations
for those operations are not currently logged, so the dashboard does not invent
them. There is no GPU-utilization polling, process heartbeat or exit-code feed.
Fresh file timestamps are labeled "recent telemetry", not proof a process is
healthy. A final observation means that artifact exists, not that every success
criterion passed. Stale logs, incomplete files, nonfinite metrics and recorded
memory breaches are explicitly flagged. Backend API failures leave a prominent
disconnected warning rather than pretending the old values are live.

Each JSONL read is bounded to the most recent2MiB. The memory display preserves
min/max within bins; exact metrics remain in the trainer's full logs. Up to200
intermediate observations/checkpoints are shown, plus initial/final observations.
This is a lightweight local viewer, not a long-horizon telemetry database.

## Verification

```bash
PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python \
  -m unittest discover -s tests -p 'test_training_monitor.py' -v
```

Tests use synthetic logs and an ephemeral loopback HTTP server: incremental
updates, malformed/partial writes, nonfinite data, stale versus recent status,
reserve warnings, bounded reads, symlink/path traversal rejection, checkpoint
non-deserialization and rejection of POST/foreign-host/origin requests. All
training artifacts remain untouched. Browser visual/interaction QA was not
performed; the page was handed to the app browser for learner review.

Implementation: `src/dongxi_llms/training_monitor.py` and
`visuals/training-monitor/`. The dashboard is local-only; no cloud deployment,
new training run, dependency installation, commit or push is implied.
