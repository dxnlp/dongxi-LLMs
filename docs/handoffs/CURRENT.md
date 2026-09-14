# Current learning handoff

- Task ID: `LEARN-SPARK-DAY09-001`.
- Updated:2026-09-14. Current machine: Spark, Linux/aarch64.
- Repository: `dxnlp/dongxi-LLMs`, branch `main`.
- Current continuation: **Continue on Spark** — completed Day9 run interpretation.
- Contract: [learning workflow](../LEARNING_WORKFLOW.md).
- User authorized committing and pushing the course, pipeline, dashboard and
  report work on2026-09-14. Check Git status and remote state on arrival; do not
  infer synchronization from this handoff or reset/stash local work blindly.

## Current evidence

Run `day09-learning-01` completed14,000 updates,48.84M valid targets and3h22m,
with actual child exit0 and final fixed-development NLL1.674315. The
[completed report](../../experiments/reports/2026-09-14-tinystories-learning-result.md)
and its adjacent portable JSON supersede all earlier waiting/preparation states.
Raw logs and final checkpoint remain under `outputs/day09-learning-01/`.
The checkpoint is `update-014000.pt`; final observation is `final.json`.

The local dashboard/playground was stopped on user request on2026-09-14;
port8765 was verified closed and the inference process exited. Do not restart
it or training merely to resume reading. No checkpoint or result was deleted.

Chapter6 now contains the completed case in sections6.14–6.20, eighteen worked
answers, and a [standard-library evidence lab](../../book/labs/06-reading-a-pretraining-run.md).
The existing three Day8 mechanism notebooks remain companions. A dedicated
Day9 analysis notebook is planned, not built or executed.

The main evidence distinction is between completed computation, improved
fixed-development prediction, and reliable coherence. Inspected samples remain
repetitive/inconsistent; there is no systematic story-quality evaluation or
unique established cause for those errors. The trained controlled comparison
required by Day9 is still open. Brief batch-size profiles and sampling probes
do not complete that learning requirement.

## Exact next action

1. Read Chapter6's completed-run case and its worked solutions with the learner.
2. Define the story-evaluation contract: fixed new prompts, entity/event
   consistency, complete outputs, EOS versus caps, and controlled decoding.
3. Propose one bounded training comparison only after defining the intended
   claim. Do not automatically extend this checkpoint, restart training,
   download data, install software, or start a server.
4. Treat previous chapters as covered for planning at the learner's request.
   Preserve earlier practice gaps without rewinding to Day6.
5. Check repository status before a new session. When clean, sync with
   `git pull --ff-only`; when dirty, preserve work and reconcile explicitly.

## Machine and production boundaries

Mac Studio is the default interactive/media lane; Spark owns heavy GPU work.
The learner currently chooses Spark. The new lab needs no PyTorch and can be
read/run on either machine after source sync. Do not equate the local desktop UI
with the execution host or claim a Git sync moves running services.

The learner prefers deep explanations, useful visuals and light minimal
interfaces, not calculation quizzes or decorative dashboard slogans.
Explicit LLM mathematics still triggers animation candidate capture; production
requires approval and remains on Mac Studio. Chapter synthesis extended existing
budget/next-token/temperature candidates, without authorizing film production.

The user requested committing and pushing this synthesis and its supporting
pipeline/dashboard work. Departure on an explicit machine-switch phrase follows
the usual scoped commit/push and
handoff contract. Learner mastery and a completed Day9 defense remain unassessed.
