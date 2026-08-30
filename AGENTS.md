# Collaboration Instructions

These instructions apply to humans and coding agents contributing to `Dongxi_LLMs`.

## Required read order

Before starting work:

1. Read `README.md`.
2. Read `BOOK.md` for the reader-facing narrative architecture.
3. Read the relevant section of `ROADMAP.md`.
4. Read `PROGRESS.md`, especially the current position, durable decisions, latest daily log, and open questions.
5. Read `LEARNING_MEMORY.md` for the learning-artifact index, public-content ideas, and cross-machine task packets.
6. Read the active day's index and relevant topics under `learning_artifacts/` for demonstrated understanding and unresolved conceptual edges.
7. Inspect existing work before creating a competing artifact.

## Source of truth

- `ROADMAP.md` defines the release plan and day-level outcomes.
- `BOOK.md` defines the book structure and the narrative placement of course material.
- `PROGRESS.md` defines the current state and exact next action.
- `learning_artifacts/` defines the durable conceptual record organized by day and topic.
- `LEARNING_MEMORY.md` defines the artifact index, public-content queue, and portable cross-machine task packets.
- Experiment specifications define intended runs.
- Experiment reports and stored outputs define empirical evidence.
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
9. Create or update the book-facing material to which the day's learning belongs.
10. Update `PROGRESS.md` with evidence and an exact next action.
11. Update `LEARNING_MEMORY.md` when a new topic must be indexed or the session produces a reusable public-content idea or portable task.

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
