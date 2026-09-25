# Learning run01 — launch ledger

User explicitly requested running training on2026-09-13. This ledger records
preflight and launch state, not a completed learning result.

## Confirmed start update

Full preparation completed successfully at20:56:29 UTC. After95 regression
tests passed, the training child started at **2026-09-13T20:56:39 UTC**
(22:56:39 Stockholm). This supersedes the initial waiting state recorded below.
Portable [launch evidence](2026-09-13-tinystories-learning-launch.json) includes
the exact data manifest, profile metrics, final recovery check and environment.

Prepared training corpus:1,792,647 distinct documents,1,793,132 windows,
390,708,926 valid targets. Excluded320,241 within-training exact duplicate
documents and6,601 training documents matching validation. There are485 long
training documents handled with target windows. Raw files were SHA256-verified.
All21,990 validation stories are held out, with512 seeded windows used for
repeated development evaluation. This run's14,000 batches will consume only part
of the shuffled prepared corpus; preparation does not mean all tokens trained.

Canonical [specification](../specs/2026-09-13-tinystories-learning-01.md) and
[frozen launch configuration](../configs/tinystories-learning-01.json).

## Preflight completed

- No competing GPU process at initial inspection;118GiB host available and
  approximately3.5TiB free disk. Existing uncommitted work preserved.
- Batch8 and batch16 each completed40 BF16 full-model updates (exit0). Warm
  valid-token throughput was4,474.65/s and4,498.32/s respectively. Batch16
  selected; its CUDA allocated peak was10.52GiB, host available minimum103.51GiB.
  Sampling and source-length differences make these brief profiles noisy, not
  a claim that batch16 is intrinsically superior.
-92 regression tests passed after data-policy and graceful-stop changes.
- Repeated3-update smoke and saved-state replay under final training sources
  both exited0; model, optimizer, stream/counters and CPU/CUDA RNG states were
  bitwise identical. See `outputs/day09-launch-recovery.json`.
- New installed-environment/lock capture: `outputs/day09-learning-environment.json`.
  Source hashes are frozen in the launch config; no installs or pretrained
  neural weights were introduced.

## Data preparation

Attempt v1 (`dongxigpt-data-20260913.service`) failed with exit1: the official
validation source lacks a delimiter at EOF. No complete manifest or training
result was produced; partial outputs are retained, not consumed.

Attempt v2 (`dongxigpt-data-20260913-v2.service`) uses complete HF downloads with
verified LFS SHA256, then accepts the verified final EOF boundary. The validation
split is complete:21,990 documents,21,994 windows,4,680,541 targets;4 stories need
multiple windows, no normalized exact duplicates. Training preparation was
still in progress when the guarded launcher was started. Final full-data counts
must be read from `data/cache/day09-full-v2/manifest.json`, not inferred from this
partial progress record. Within-split exact duplicates and train/validation
overlaps are excluded and counted; near-duplicate auditing is not performed.

## Background execution

- Launcher service: `dongxigpt-learning-20260913.service`.
- Launch registered:2026-09-13T20:53:27 UTC (22:53:27 Stockholm).
- Initial state: `waiting_for_full_data`—NOT yet training at that timestamp.
- Status: `outputs/day09-learning-launch/status.json`; regression output is
  retained beside it. Journal records stdout/stderr and actual child exit status.
- Data failures, changed sources, failed recovery/regression gates, insufficient
  disk or a competing GPU process prevent the training child from starting.
- Intended training output: `outputs/day09-learning-01/`.
- Frozen recipe:14,000 updates, batch16/accumulation1,200 warmup, lr3e-4 to3e-5,
 512 seeded validation windows, six fixed completions up to256 tokens,
  checkpoint/observation every400 updates. All model weights newly randomized.
- Four-hour training cap,25GiB host reserve,80G cgroup cap with no charged swap;
  no automatic restart, extension or further experiment.

Check status before saying training started or finished:

```bash
systemctl --user show dongxigpt-learning-20260913 -p ActiveState -p ExecMainStatus
journalctl --user -u dongxigpt-learning-20260913 -n 12 --no-pager
```

The read-only dashboard is at `http://127.0.0.1:8765`. Choose
`day09-learning-01` after it appears. A service being active while waiting for data
is not evidence of model updates. After completion, inspect process exit, memory,
metrics, saved checkpoints and all fixed samples before judging success.
