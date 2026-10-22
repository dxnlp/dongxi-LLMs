# One learning journey on the DGX Spark

Established at the learner's request on 2026-09-09 and simplified on 2026-10-22: the
**DGX Spark is the single working machine** for live discussion, interactive lessons,
notebooks, experiments, chapter and article writing, and animation production. The
earlier Mac Studio / Spark split, with its `Switch to Mac` / `Continue on Mac` /
`Switch to Spark` / `Continue on Spark` handoff phrases, is retired. Historical
records that mention it keep their original wording.

Rich visuals should appear directly in the conversation; browser-side toy
interactions don't need a notebook kernel. New notebooks are GPU first (see the
`execution` types in `docs/course_manifest.json`). Prefer concrete LLM-system
interventions over generic controls.

## Starting a session

1. Confirm the repository, branch and worktree state (`git status --short --branch`).
2. With a clean checkout, run `git pull --ff-only`. Work may have been pushed from
   another session. If the tree is dirty, history has diverged or the pull fails,
   preserve the local work and explain the blocker; never reset, stash or discard
   it to synchronize.
3. Reread `AGENTS.md` and its required files, `PROGRESS.md`, `LEARNING_MEMORY.md`, the
   active topic, and `docs/handoffs/CURRENT.md` when resuming an unfinished task.
   Reconcile an old handoff that disagrees with newer progress; don't silently follow it.
4. Check the runtime the lesson needs: the `Python (DGX Spark Native)` kernel, CUDA,
   and any locally cached models. Keep environments separate; never copy a virtual
   environment or credentials.
5. Give a short recap of the current lesson, what is complete, open questions and
   running jobs, then resume the recorded next action.

## Ending a session or handing off a task

1. Update `PROGRESS.md` and the active topic artifact, and, for unfinished work,
   [the current handoff](handoffs/CURRENT.md): what was discussed, prepared and
   still open, the exact next action, and pending article/animation approvals.
2. Record any running job's run ID, status, configuration/commit, logs and
   checkpoint location using read-only checks. Don't kill or restart jobs
   automatically.
3. Commit and push when the learner asks. Stage exact scoped files and keep
   credentials, large datasets/checkpoints and temporary runtimes out of Git. On
   divergence or rejection, stop for safe reconciliation; never force-push.

Git carries the durable files, not live Python state, browser tabs or running
processes. This protocol must also work in a fresh conversation that only reads
the repository.

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
- Keep small interventions lightweight and responsive. A future Spark dashboard
  starts read-only; job launch/stop controls require explicit authority and
  safety checks, not a slider side effect.

For implemented tools, persist only meaningful small state: lesson/component ID,
seed, initial conditions, control values, optimizer settings/state or replay
recipe, step index, code revision, prediction, interpretation, and linked run ID.
Saving state is a requirement for future tools, not a claim of existing widget
state in the present session.

## First proposed interactive tool

**Optimizer Playground — Day 8 / Chapter 6.** Compare SGD, momentum,
and AdamW using a clearly labeled small loss landscape. Connect the gradient
arrow to the actual update arrow, adjustable learning rate, trajectory, and loss.
Allow single steps, reset, and paired initial conditions; later connect the
mechanism to a tiny decoder. Explain when apparent divergence or oscillation
comes from the objective, step size, or accumulated optimizer state.

Status: design proposed and remembered; not implemented or validated.
Begin with one complete pilot, not a simultaneous rewrite of every notebook.
Library/UI selection and implementation remain the next explicitly requested
build task. Animation candidates reuse existing optimizer/gradient topics;
interactive controls do not automatically approve produced animation media.

The intended continuation is **Day 8**, explicitly confirmed by the learner.
The earlier Day 6 default was an agent handoff mistake. Keep Day 4–7 gaps as
review items; return only when needed or requested. Current user intent
overrides stale resume instructions. See the live handoff for the next step.
