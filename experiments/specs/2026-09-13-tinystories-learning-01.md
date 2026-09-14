# DongxiGPT Stories learning run 01

User authorized running training on2026-09-13. Mode: learning, not benchmark.
This specification precedes full-data preparation and the long run.

## Fixed objective and controls

Train the previously selected66.64M random-initialized decoder on the pinned
original TinyStories corpus/GPT-2 tokenizer. No pretrained neural weights,
architecture sweep, tokenizer retraining, public upload or multi-GPU work.
Hypothesis: held-out next-token loss improves and fixed-prompt completions become
more grammatical/consistent than random initialization. Coherence is not promised
within the budget. Retain bad samples and all fixed evaluation observations.

Data preparation consumes the full original train/validation streams. Normalize
line framing as in the smoke pipeline; preserve story text otherwise. Reserve
validation first, deduplicate normalized exact stories within each split and
exclude training stories matching validation. Record exclusion counts, identities,
length statistics and target counts. Reject unexpected embedded special-token
IDs. Near-duplicate contamination is not claimed to be solved.

Evaluation: seeded fixed512-window development selection from the prepared
validation split, never trained on; repeat it at all observations. This is
development evidence, not a pristine final publication test. Existing three
authored prompts, greedy and fixed-seed temperature.8 generation, up to256 new
tokens each; total prompt+continuation stays inside1024. Check natural EOS versus
truncation. No cherry-picking or changing decoding across checkpoints.

## Safety and gates

- Preserve local dirty work. Record source hashes/environment, not just base SHA.
- Full preparation is bounded to60 minutes with80GiB cgroup memory cap and no
  cgroup swap. No training alongside inference servers; preserve25GiB host reserve.
- Profile fresh model on smoke data at microbatch8 and16, each at most40 updates
  /10 minutes; choose the faster useful-token configuration that passes guards.
  No weight transfer from profile to learning run.
- Rerun regression tests and small recovery checks after any required changes.
- Learning wall-clock cap: **four hours**, including observation/save work.
  A graceful deadline stops new updates, saves the latest completed state and
  records that the schedule may be incomplete. External timeout adds5 minutes
  only for shutdown; no automatic overnight extension.
- Set a fixed update horizon from measured profile with overhead allowance
  before launch. AdamW betas(.9,.95), eps1e-8, matrix decay.1, clip1; LR3e-4
  with warmup and cosine decay to3e-5. Batch/update horizon, warmup and checkpoint
  cadence are recorded in a launch addendum once measured, not guessed as facts.
- BF16 CUDA autocast, FP32 parameters/AdamW, causal SDPA, activation checkpointing.
  Keep the verified separate-document window/padding policy for this baseline.
- Cap training/checkpoint outputs at a preflight budget of100GiB (enough for
  <=100 saved optimizer checkpoints). Fail before launch if insufficient disk.
- Run under a named systemd user service so it survives browser disconnection;
  journal and run artifacts retain status. Dashboard remains read-only.

Acceptance for launch: data manifest complete, finite profile updates and
positive measured throughput, safe memory, tests passing. Acceptance for learning
is evaluated after exit from full metrics/samples, not from process startup.

## Launch addendum after profile

Both40-update profiles exited0. Excluding their first5 updates, batch8 processed
4,474.65 valid targets/s with mean update0.40010s; batch16 processed4,498.32/s
with mean update0.82415s. The throughput difference is under1%, not a meaningful
scaling victory. Batch16 fits safely (peak CUDA allocated11,293,046,784 bytes,
minimum sampled host available103.513GiB) and is selected as the effective batch.
Batch8's corresponding figures were6,113,605,632 bytes and109.876GiB. These are
brief measurements on smoke lengths, not a four-hour speed guarantee.

Freeze batch16, accumulation1,14,000 updates,200 warmup updates, lr3e-4 decaying
to3e-5, seed909. Save/evaluate every400 updates and at completion, up to35
periodic checkpoints. New random weights: do NOT resume any profile checkpoint.
At the measured batch16 time, update work alone projects to3.21 hours; the rest
of the4-hour cap is overhead margin, not promised throughput. Actual processed
tokens and duration are empirical outputs. Fixed512-window validation uses CPU
torch.Generator seed909, independent of training and sampling generators.

Data-preparation attempt v1 exited1 before any training: its strict streaming
reader rejected the official validation file's undelimited last story. Attempt
v2 downloads complete raw files, verifies their pinned HF LFS SHA256 hashes and
only then accepts EOF as the final document boundary. Failed v1 artifacts remain
for diagnosis; no incomplete manifest is allowed through the launch gate.

Raw source hashes: train `c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f`
(1,924,281,556 bytes); validation
`94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4`
(19,447,282 bytes), resolved from HF metadata at the pinned revision.

The named launcher waits for the completed data manifest, checks source hashes,
the final recovery report, free disk, an idle GPU and regression tests, then
starts exactly the frozen command. If preparation or any gate fails it exits
without starting training. Status and actual child exit code are retained under
`outputs/day09-learning-launch/`; training output is
`outputs/day09-learning-01/`. No automated restart/extension is configured.
