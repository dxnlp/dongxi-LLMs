# Collaboration Instructions

These instructions apply to humans and coding agents contributing to `Dongxi_LLMs`.

## Session synchronization

At the start of every new learning session, and whenever the learner returns to
resume work, synchronize before relying on local project state:

1. Confirm the intended repository and branch with `git status --short --branch`.
2. If the working tree is clean, run `git pull --ff-only` before reading trackers
   or making changes; work may have been pushed from another machine.
3. If the tree is dirty, history has diverged, or a fast-forward pull fails, do
   not reset, stash, overwrite, or merge blindly. Preserve the local work, report
   the state, and reconcile it safely before continuing.
4. After a successful pull, reread the required project files because remote
   changes may have updated instructions, progress, learning memory, or task
   packets.

## Required read order

Full-course production request,2026-10-04: prepare all15 chapters and all28 days,
including substantial future prose, visual notebooks, worked solutions, reusable
code and experiments. This explicitly overrides the usual incremental rule that
future notebooks wait for their chapter's live activation. Preserve completed
evidence and learner cells. Run bounded CPU references and label new model-scale
protocols as unexecuted until their own evidence exists. Course readiness does
not move the learner's Day9 position or authorize external publication.

The learner's machine-switching phrases are standing course instructions:
`Switch to Mac`, `Continue on Mac`, `Switch to Spark`, `Continue on Spark`.
Follow `docs/LEARNING_WORKFLOW.md` and `docs/handoffs/CURRENT.md`. Departure
means update the handoff and commit/push scoped course work before telling the
learner to move. Arrival means verify the actual execution host, safely sync,
reread the durable state, and continue. Never silently discard dirty changes,
assume a Mac UI means local execution, or migrate/terminate running jobs.
Remind the learner when the next workload belongs on the other machine; do not
switch or launch expensive work without their instruction. A memory/planning
request alone does not execute departure.
The learner's latest requested lesson takes precedence over older "next action"
entries. Unfinished earlier practice is a review backlog, not an automatic
rewind. Current override (2026-09-13): discuss **Day 9 on Spark**, with the first
DongxiGPT objective of coherent short English stories. Treat current chapters as
covered for planning at the learner's request, not as independently assessed
mastery or completed training evidence. Continue using rich
interactive visuals directly in the conversation. Notebooks are companions,
not the required interface; do not force Mac setup or restart Day 6.

Before starting work:

1. Read `README.md`.
2. Read `BOOK.md` for the reader-facing narrative architecture.
3. Read the relevant section of `ROADMAP.md`.
4. Read `PROGRESS.md`, especially the current position, durable decisions, latest daily log, and open questions.
5. Read `LEARNING_MEMORY.md` for the learning-artifact index, public-content ideas, and cross-machine task packets.
6. Read the active day's index and relevant topics under `learning_artifacts/` for demonstrated understanding and unresolved conceptual edges.
7. Inspect existing work before creating a competing artifact.
8. For machine switches, read the current handoff and workflow; reconcile them
   with the latest progress before resuming. Mac is the default live-learning
   environment; Spark is the GPU execution environment.

## Source of truth

Gap-applicability rule,2026-10-05: before implementing a proposed improvement,
check whether it still exists, matters locally, and is already solved by current
upstream versions or adequate existing code. Reuse resolved fixes instead of
building duplicate infrastructure. In particular, historical DGX unified-memory
hang warnings do not justify a custom monitoring project on the updated local
stack. Keep ordinary bounded runs; prioritize useful course/model results.
See `docs/REFERENCE_REPOSITORY_AUDIT.md` and `LEARNING_MEMORY.md` for evidence.

Spark continuation correction,2026-10-05: the learner challenged Mac verification
as a Spark blocker and then explicitly instructed completion of the goal.
Separate package-completion dependencies from execution prerequisites. Preserve
all original acceptance/dependency arrays as the completion baseline; for Spark
execution, DXI-17 requires actual Linux/source verification only. Actual Mac
portability remains unfinished parallel evidence work, not an additional
successful-execution criterion or prerequisite for Spark profiles, recovery or
pilots. Continue the predeclared bounded campaign
after its own scientific, interface, recovery, resource and authority gates pass.
This supersedes older text coupling all Spark launches to a Mac pass; it does not
waive Mac evidence, authorize unlimited compute/publication or count unrun jobs.

Current goal authorization,2026-10-05: the learner confirmed the proposed next
step aligns with the improvement goal, said to perform it, and requested goal
completion without repeated routine approval questions. The fixed20-update,
900-second native profile is now approved, including its input preparation,
declared limits, model loading and bounded evaluation/training. Continue the
agreed goal's routine implementation, verification and predeclared bounded
experiment progression when prerequisites are met; do not ask the same approval
again. Preserve original acceptance criteria and actual failures. Stop only for
a genuine missing authority, unavailable platform or consequential choice that
cannot be resolved safely within the agreed scope. This is not unlimited
compute, a new model-scale sweep, destructive cleanup, public publication or
authority to invent Mac/hosted evidence. Earlier per-step approval reminders are
historical where superseded by this instruction.

Continuation correction,2026-10-05: a blocked dependency blocks its dependents,
not unrelated approved implementation. Finish independent audits, consumers,
course integration and evidence review before reporting the external boundary;
do not ask whether to continue routine work already included in the goal.
Preserve original acceptance: DXI-17 distinguishes actual Linux execution,
unverified Mac portability and CI source/local/hosted evidence. Its literal
criterion3 does not require a successful Mac run, just as criterion5 does not
require hosted success. Stronger historical requests remain parallel evidence
work; never label an unexecuted Mac run passed.
This correction does not authorize publication, waive dependencies or invent
an unavailable machine's execution.

- `ROADMAP.md` defines the release plan and day-level outcomes.
- `BOOK.md` defines the book structure and the narrative placement of course material.
- `PROGRESS.md` defines the current state and exact next action.
- `learning_artifacts/` defines the durable conceptual record organized by day and topic.
- `LEARNING_MEMORY.md` defines the artifact index, public-content queue, and portable cross-machine task packets.
- Experiment specifications define intended runs.
- `docs/course_manifest.json` defines chapter/day/notebook routing and optional
  extension dependencies; registration is not a readiness or mastery claim.
- `pyproject.toml` plus `uv.lock` define the isolated CPU teaching environment.
  Never synchronize that lock into the shared Spark GPU environment. Use a new
  task/project-local environment and temporary-prefix kernel, retain actual
  platform evidence, and distinguish workflow source from hosted CI execution.
- Experiment reports and stored outputs define empirical evidence.
- `docs/COURSE_IMPROVEMENT_PLAN.md` defines the active eighteen-package
  reference-audit upgrade; `docs/course_improvements.json` records statuses,
  dependencies and evidence. Read both before resuming production work.
  Planned packages are not implementations, model-scale permission or learner
  completion. Preserve the15chapter/28day core and separate enrichment lanes.
- Chat transcripts and agent summaries are not durable project state.

If two artifacts conflict, stop and record the conflict rather than silently choosing one.

## Daily workflow

For each learning day:

1. Create or read the day's directory and index under `learning_artifacts/`.
2. Restate the questions and expected outcome.
3. Write predictions before running material experiments.
4. Implement the smallest transparent mechanism first.
5. Run a smoke experiment before a longer run.
6. Preserve configurations, environment identity, metrics, and representative outputs.
7. Distinguish observations from interpretations.
8. During the lesson, create or update the focused topic artifact after each substantive prediction, correction, demonstrated explanation, surprising observation, or unresolved edge. Do not postpone this until the end of the day.
9. Perform an animation-opportunity check after each substantive mechanism. Any
   explicit mathematics central to an LLM mechanism—especially an objective,
   probability transformation, gradient, tensor operation, masking rule,
   optimization update, or multi-step derivation—automatically triggers candidate
   capture in `visuals/animations/PROPOSALS.md` without waiting for the learner to
   ask. Also capture non-mathematical candidates when motion would materially
   clarify a transition, identity, flow, competition, or time-dependent failure.
   Reuse or expand an existing candidate rather than creating duplicates. Record
   the source (`user`, `agent`, or `roadmap`), canonical equation or mechanism,
   evidence state, and production dependency. Candidate capture does not
   authorize production: actively surface strong candidates to the learner, wait
   for explicit approval, then promote approved concepts to complete `ANIM-*`
   task packets in `LEARNING_MEMORY.md`. All animation production remains on the
   Mac Studio unless the learner explicitly changes that assignment.
10. Create or update the book-facing material to which the day's learning belongs.
11. Update `PROGRESS.md` with evidence and an exact next action.
12. Update `LEARNING_MEMORY.md` when a new topic must be indexed or the session produces a reusable public-content idea or portable task.

When mathematics is central to the mechanism, pair the discussion with an
interactive mechanism notebook when doing so materially improves understanding.
The learner additionally requests live visual interaction beyond notebooks:
follow the linked-equation/control/experiment contract in
`docs/LEARNING_WORKFLOW.md`. Preserve notebooks as implementation companions,
not the only interactive surface. Record proposed tools separately from built,
verified tools and actual learner mastery.
Use the learning cycle `deep question → prediction → small implementation →
perturbation or broken variant → interpretation → evidence boundary`. Guide the
learner through the notebook interactively rather than treating it as a passive
demonstration or a stream of arithmetic quizzes. Keep reusable computations in
`src/dongxi_llms/`; notebooks should narrate, expose intermediate tensors, and
invite controlled modification. Put a clearly labeled runnable reference
solution and mechanism explanation immediately after each learner exercise or
checkpoint. Preserve prediction-before-reveal, but do not force the learner to
search outside the notebook for routine syntax or the canonical reasoning.

Every book chapter must have a planned interactive notebook pathway, normally
two to three focused sessions: a transparent mechanism microscope, a deliberate
perturbation or failure, and an integration/evidence session. At the start of a
chapter, refine its entries in `docs/NOTEBOOK_CURRICULUM.md`, create the active
notebook directory and session index, and link completed notebooks from the
chapter and solutions. The2026-10-04full-course request explicitly authorizes
building all future notebook routes now; avoid empty placeholders. Evidence-oriented chapters may use executable
audits, simulations, or metric explorations instead of forcing an artificial
neural-network mechanism.

Notebook lessons must also explain visually when spatial structure, tensor
layout, comparison, or a trajectory materially clarifies the mechanism. The
learner explicitly requested this on 2026-09-07. Use data-backed Python plots
next to the relevant exercise and solution, with axis/shape labels, a short
reading guide, and runnable regeneration code. Preserve existing learner cells.
For architecture lessons, also provide a whole-model map highlighting the current
component and a detailed branch/operation diagram with tensor shapes. Clearly
separate architecture schematics from measured plots and distinguish baseline
learned position embeddings from modern RoPE inside attention.
Use `docs/NOTEBOOK_VISUALS.md` for reference previews, dependency handling,
evidence boundaries, and visual verification. Static notebook plots do not
authorize animation rendering; Mac Studio remains the animation production lane.

Do not mark a day complete merely because prose or code exists. The stated evidence of completion must be present.

## Book-first course development

`Dongxi_LLMs` is a coherent technical book with executable companion material,
not a chronological collection of daily notes. The 28-day roadmap governs the
learning and production schedule; it does not dictate the final chapter boundaries.

For every learning day:

1. Identify where the material belongs in the book before drafting: front matter,
   conceptual chapter, worked example, lab, appendix, or companion repository.
2. Create or update learner-facing book material as understanding develops. Daily
   logs, experiment specifications, reports, and chat transcripts are source
   evidence; they are not substitutes for a chapter or section.
3. Integrate new material with the existing narrative, notation, prerequisites,
   examples, and forward references. Do not create an isolated “Day NN” note when
   the idea belongs inside an existing chapter.
4. Give conceptual chapters a deliberate learning arc: motivation and questions,
   intuition, precise definitions and derivations, transparent implementation,
   experiment, evidence and limitations, exercises, summary, and connection to
   what follows. Use only the elements that genuinely help that chapter.
5. Keep durable concepts in the main narrative. Put machine-specific setup,
   time-sensitive commands, and operational troubleshooting in appendices or the
   platform repository, and link them from the relevant chapter.
6. Preserve failed experiments and detailed telemetry in reports while bringing
   only the evidence needed for the argument into the book prose.
7. Before marking a day complete, either link its book-facing contribution or
   explicitly record why its evidence will be integrated during a named later
   synthesis day. “No course content” is not the default.

When the learner requests a complete chapter, cover the full assigned outline,
including topics not yet reached in live discussion. Write for a new reader and
include the required derivations, examples, executable companions, exercises,
solutions, and evidence boundaries. Keep learner progress and deferred practice
in the trackers; do not use them to limit the chapter's subject coverage.

Continuously check book-level coherence: chapter order, prerequisite flow,
terminology, notation, repeated explanations, pacing, and whether each experiment
advances the book's central argument.

## Cross-session and cross-machine memory

Do not rely on chat history as the only record of a valuable discussion or task.
Preserve deep conceptual learning in the active topic under `learning_artifacts/`
with the learner's initial model, refined mechanism, concrete examples,
demonstrated understanding, evidence level, important limitations, reuse
opportunities, and remaining gaps. Keep `LEARNING_MEMORY.md` as the compact index
and production queue rather than duplicating every topic there.

Before work is handed to another session or machine, create or update a portable
task packet containing a stable task ID, base commit, branch, learning objective,
inputs, allowed files, expected outputs, precision requirements, acceptance
checks, and return evidence. Git is the synchronization boundary: never assume
that uncommitted files on one machine are visible on another.

Public articles and animations are derived from the canonical book argument.
Record promising ideas immediately, but delay final wording when their underlying
chapter or derivation is not yet stable. On completion, update the packet rather
than leaving status only in a chat transcript.

## Empirical integrity

- Never invent, estimate, or silently extrapolate experiment results.
- Label projected memory, throughput, and quality separately from measurements.
- Record failed and negative-result runs.
- Do not select only favorable checkpoints without documenting the selection rule.
- Training loss and reward are not sufficient evidence of general capability improvement.
- Evaluation data must not be used for training or recipe selection unless explicitly designated as development data.

## Code and content design

- Canonical prose, notation, source code, and comments are English-first.
- Write course prose for a reader following a book, not for the author recalling a work session.
- Reusable logic belongs in importable Python modules; notebooks narrate and visualize experiments.
- Keep mathematical notation consistent across lessons.
- Follow `docs/MATH_FORMATTING.md` for GitHub-compatible book math. Run
  `python3 scripts/check_book_math.py` after equation edits; do not rely on a
  notebook preview as evidence of GitHub Markdown rendering.
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
