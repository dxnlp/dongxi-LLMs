# Collaboration Instructions

These instructions apply to humans and coding agents contributing to `Dongxi_LLMs`.

## Required read order

Before starting work:

1. Read `README.md`.
2. Read the relevant section of `ROADMAP.md`.
3. Read `PROGRESS.md`, especially the current position, durable decisions, latest daily log, and open questions.
4. Inspect existing work before creating a competing artifact.

## Source of truth

- `ROADMAP.md` defines the release plan and day-level outcomes.
- `PROGRESS.md` defines the current state and exact next action.
- Experiment specifications define intended runs.
- Experiment reports and stored outputs define empirical evidence.
- Chat transcripts and agent summaries are not durable project state.

If two artifacts conflict, stop and record the conflict rather than silently choosing one.

## Daily workflow

For each learning day:

1. Restate the questions and expected outcome.
2. Write predictions before running material experiments.
3. Implement the smallest transparent mechanism first.
4. Run a smoke experiment before a longer run.
5. Preserve configurations, environment identity, metrics, and representative outputs.
6. Distinguish observations from interpretations.
7. Update `PROGRESS.md` with evidence and an exact next action.

Do not mark a day complete merely because prose or code exists. The stated evidence of completion must be present.

## Empirical integrity

- Never invent, estimate, or silently extrapolate experiment results.
- Label projected memory, throughput, and quality separately from measurements.
- Record failed and negative-result runs.
- Do not select only favorable checkpoints without documenting the selection rule.
- Training loss and reward are not sufficient evidence of general capability improvement.
- Evaluation data must not be used for training or recipe selection unless explicitly designated as development data.

## Code and content design

- Canonical prose, notation, source code, and comments are English-first.
- Reusable logic belongs in importable Python modules; notebooks narrate and visualize experiments.
- Keep mathematical notation consistent across lessons.
- Define symbols, tensor shapes, gradient boundaries, and optimization direction.
- Prefer small readable implementations before framework integrations.
- Generated figures and animations must retain their source, render command, and learning objective.

## External material and licenses

The sibling repositories are references. Do not copy their prose, book images, or substantial code without checking the applicable license and recording attribution.

In particular, `dgx-spark-dongxi/NOTICE.md` records unresolved licensing for some upstream-derived setup material. Link to that platform project or create an independent implementation rather than copying uncertain material into a public course.

## Long-running jobs

- Use the memory-safety practices documented by `/home/dongxi/dgx-spark-dongxi`.
- Declare smoke, learning, or reference mode.
- Record model and dataset revisions, command/configuration, seed, dtype, batch geometry, sequence lengths, attention backend, starting memory, peak memory, runtime, and software lock.
- Keep at least 20–25 GiB available for the operating system and interactive work unless a newer validated platform policy supersedes this limit.
- Do not run a concurrent large inference server and training job without an explicit profiled reason.

## Scope control

The target is a coherent `v0.1` public beta in 28 learning days. Protect correctness and continuity over breadth. Place valuable but nonessential additions in the `v0.2` backlog rather than silently expanding the active day.
