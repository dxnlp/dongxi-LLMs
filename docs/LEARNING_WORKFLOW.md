# One learning journey, two execution environments

Established at the learner's request on 2026-09-09. This is the course's
standing interaction and machine-switching workflow, not a new chapter or an
assertion that a software interface has already been built.

## Machine responsibilities and reminders

Mac Studio is the default home for live discussion, interactive visual lessons,
small CPU mathematics, chapter/solution editing, articles, and approved animation
production. DGX Spark is the execution home for CUDA-dependent mechanisms,
substantial model training/inference, GPU profiling, and large checkpoints.

Choose by workload, not simply by day number. The same chapter may involve both:

| Course stage | Mac Studio: understand, design, interpret | Spark: measure or execute |
|---|---|---|
| Days 4–7 architecture review | Attention/normalization/gradient interventions, shape maps, design defense | CUDA-specific checks or an explicitly specified larger comparison |
| Day 8 pretraining recipe | Token/batch accounting, optimizer playground, schedule/clipping controls, experiment specification | Bounded hardware-fit/profile check only when separately specified |
| Day 9 pretraining | Predict outcomes, inspect curves/samples, discuss failure evidence | Approved pretraining, checkpointing, throughput/memory measurement |
| Day 10 evaluation | Metric microscopes, frozen evaluation design, error analysis | Model-dependent generation/evaluation batches |
| Days 11–14 instruction data/SFT | Data/label-mask inspection, small gradient examples, recipe defense | Approved model-scale SFT and checkpoint evaluation |
| Days 15–18 preferences/DPO | Probability/loss visualizations, controlled toy objectives | Approved reward-model/DPO training and evaluation |
| Days 19–25 policy optimization | Policy-gradient toys, reward/advantage controls, diagnosis | Approved rollouts, RL training, GPU-system measurements |
| Days 26–28 synthesis | Compare evidence, write/defend chapters, prepare approved media | Remaining explicitly budgeted model comparisons |

Remind the learner at a workload boundary, not every turn. Before a lesson
needs CUDA, heavy inference, or a long run, explain why Spark is needed and
offer `Switch to Spark`. Before returning to responsive visual exploration,
writing, or animation production, offer `Switch to Mac`. Do not initiate a
machine transfer or an expensive job merely because a reminder is appropriate.

## Four trigger phrases

These are course conversation conventions, not built-in application commands.
Recognize capitalization and clear natural-language equivalents; the word
"Mac" or "Spark" by itself is not authorization to switch.

| Learner says | Meaning | Required response |
|---|---|---|
| **Switch to Mac** | Prepare departure to Mac Studio | Save progress/handoff, validate and commit/push scoped course work, then give `Continue on Mac` |
| **Continue on Mac** | Arrived at Mac Studio | Verify local execution, safely sync repo, reread memory/handoff, resume the recorded next action |
| **Switch to Spark** | Prepare departure to DGX Spark | Save progress/handoff, validate and commit/push scoped course work, then give `Continue on Spark` |
| **Continue on Spark** | Arrived at DGX Spark | Verify Spark execution, safely sync repo, reread memory/handoff, resume the recorded next action |

The learner explicitly requested commit and push **before switching**. The
departure phrases authorize those operations for the scoped course work and
handoff record. They do not authorize staging unrelated files, publishing an
article, rendering a film, starting training, or force-pushing. A planning or
memory-only request is not itself a departure trigger.

## Departure checklist

1. Verify actual execution host, repository, branch, remote, and worktree state.
   A Mac application displaying an SSH task can still execute entirely on Spark.
2. Update `PROGRESS.md`, the active topic artifact, and
   [the current handoff](handoffs/CURRENT.md): what was discussed, demonstrated,
   prepared, and still open; exact next action; destination and return phrase.
   Preserve pending article/animation candidates and their approval states.
3. Record any relevant running job's host, run ID, status, configuration/commit,
   logs, and checkpoint location using safe read-only checks. Never assume
   terminal processes migrate or stop; do not kill or restart them automatically.
   Save actual lesson controls/seed/state if an interactive tool provides them.
4. Review the diff and validate proportionately. Keep credentials, large
   datasets/checkpoints, temporary runtimes, and unrelated edits out of Git.
   Stage exact scoped files and commit. If scope is unclear, ask before staging.
5. Push normally to the intended remote branch. If divergence or rejection
   occurs, stop for safe reconciliation; do not force-push, blindly merge, stash,
   reset, or discard local work. Do not call the transfer ready if push failed.
6. Verify the pushed commit and local/remote relationship. Report any remaining
   local-only work explicitly. Give the destination, branch, pushed commit, and
   exact arrival phrase. If there is nothing new, verify synchronization instead
   of creating an empty commit. The handoff's own introducing commit can be
   recovered with Git history; do not create endless self-referential commits.

## Arrival checklist

1. Confirm actual execution host and course checkout before any computation.
   On `Continue on Mac`, a shell still reporting Spark/Linux is a mismatch:
   tell the learner to open or select a local Mac project/task, then repeat the
   phrase. On `Continue on Spark`, verify the intended Spark host. Do not infer
   execution location from the browser URL or where the application window sits.
2. Inspect branch, upstream, and dirty state. With a clean checkout on the
   intended branch, run `git pull --ff-only`. If dirty, diverged, on the wrong
   branch, or unable to reach the remote, preserve state and explain the blocker.
   Do not continue using potentially stale instructions as if synchronization
   succeeded. Missing local checkout: ask for its location or arrange initial
   setup explicitly; never assume the Spark absolute path exists on the Mac.
3. After successful sync, reread `AGENTS.md` and its required project files,
   `PROGRESS.md`, `LEARNING_MEMORY.md`, `docs/handoffs/CURRENT.md`, and the active
   topic. Confirm the expected handoff/commit is present. An old handoff that
   disagrees with newer progress must be reconciled, not silently followed.
4. Confirm the local runtime and relevant assets. Keep platform environments
   separate: portable CPU teaching code is not a claim that CUDA, Triton,
   kernels, dependency locks, or bitwise results transfer to macOS. Recreate
   needed environments from platform-appropriate specifications; never copy a
   virtual environment or credentials between machines.
5. Give a short recap: current lesson, what is actually complete, open question,
   relevant jobs, and the next interaction. Resume that action; do not restart
   the course or silently mark deferred practice complete.

Git transfers the course's durable files, not live Python variables, browser
tabs, local-only credentials, or a shared running process. A conversation can
be continued or handed off through available application features, but this
protocol must also work in a fresh conversation reading the repository.

## Interactive lesson contract

Use **question → prediction → intervention → visible evidence → explanation →
durable record**. Favor profound conceptual questions over arithmetic quizzes.

- Keep a whole-model map visible while opening component-level details.
- Link equation symbols, tensor shapes, plots, and controls to the same actual
  computation. Label a schematic, synthetic landscape, saved result, and live
  model measurement distinctly.
- Offer play/pause, one step, reset, and paired runs from the same initial state.
  A rewind must restore the necessary optimizer/RNG state or replay reproducibly;
  it cannot merely move a plotted marker backward.
- Use deliberate failures: future-information leaks, wrong cache positions,
  mismatched loss masks, unstable updates, or contaminated evaluation.
- Let the learner's question choose the next intervention. Provide worked
  explanations adjacent to exercises; retain notebooks as implementation access
  and the chapter as the coherent narrative.
- Keep small interventions local and usable while Spark is disconnected. A
  future Spark dashboard starts read-only; job launch/stop controls require
  explicit authority and safety checks, not a slider side effect.

For implemented tools, persist only meaningful small state: lesson/component ID,
seed, initial conditions, control values, optimizer settings/state or replay
recipe, step index, code revision, prediction, interpretation, and linked run ID.
Saving state is a requirement for future tools, not a claim of existing widget
state in the present session.

## First proposed interactive tool

**Optimizer Playground — Mac Studio, Day 8 / Chapter 6.** Compare SGD, momentum,
and AdamW using a clearly labeled small loss landscape. Connect the gradient
arrow to the actual update arrow, adjustable learning rate, trajectory, and loss.
Allow single steps, reset, and paired initial conditions; later connect the
mechanism to a tiny decoder. Explain when apparent divergence or oscillation
comes from the objective, step size, or accumulated optimizer state.

Status: design proposed and remembered; not implemented or validated on the Mac.
Begin with one complete pilot, not a simultaneous rewrite of every notebook.
Library/UI selection and implementation remain the next explicitly requested
build task. Animation candidates reuse existing optimizer/gradient topics;
interactive controls do not automatically approve produced animation media.

The default learning continuation is still the recorded Day 6 norm comparison,
with Day 4/5 practice and Day 7 defense open. Day 8's outline and interactive
pilot may be explored when the learner chooses; neither completes those earlier
requirements. See the live handoff for the current destination and next step.
