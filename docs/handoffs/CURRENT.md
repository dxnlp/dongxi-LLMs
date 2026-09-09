# Current learning handoff

- Task ID: `LEARN-MAC-001`
- Prepared: 2026-09-09.
- State: departure requested with `Switch to Mac` on 2026-09-09; Spark-side
  handoff prepared for publication. Mac arrival and runtime validation pending.
  Verify the published commit on arrival; do not infer that study has resumed.
- Source execution verified: `spark-aa66`, Linux/aarch64.
- Destination: Mac Studio, local execution; local path/runtime not yet verified.
- Repository: `dxnlp/dongxi-LLMs`; intended branch: `main`.
- Base commit before this workflow record: `221caed238e08468c58cf1223822206d2aee63b8`.
- Departure phrase: **Switch to Mac**.
- Arrival phrase: **Continue on Mac**.
- Contract: [learning workflow](../LEARNING_WORKFLOW.md).

## What the next session must know

The learner wants live, richly visual interaction beyond text/code notebooks:
clickable architecture components, linked equations and calculations, adjustable
controls, paired runs, and deliberate failures. Mac is the default learning and
media machine; Spark handles approved GPU work. These choices are recorded;
the interactive textbook and Optimizer Playground have **not** been built.

Days 1–3 are recorded complete. Chapter 4 is written but hands-on practice was
deferred. The live baseline discussion reached Day 5 notebook 07; mastery remains
unassessed. Chapter 5 covers Days 5–7, with 12 verified notebooks and 30 worked
answers. Day 6 guided study and Day 7 independent defense remain open. Day 8
received an outline and a proposed interactive route, not completed instruction.

The latest math-formatting repair is committed in the base above. Five chapters
and affected companions were fixed; 560 expressions rendered in local MathJax,
and 59 tests passed. Keep GitHub-safe math rules in force. These results were on
Spark; they do not establish Mac environment readiness.

## Exact next action on Mac

1. On `Continue on Mac`, follow the arrival checklist: verify local execution,
   synchronize safely, then reread the durable course state.
2. Locate the local checkout/runtime and explain any missing setup. Do not try
   to use `/home/dongxi/dgx-spark-dongxi/.venv` as a Mac environment or silently
   change previously committed notebooks to force a kernel match.
3. Recap this plan. Default lesson continuation is Day 6's LayerNorm/RMSNorm
   contrast using the existing notebook as a companion, not the entire lesson.
   The proposed Day 8 Optimizer Playground is the first new interactive build
   candidate; proceed with it when requested, without marking Day 6/7 complete.
4. For a build request, inspect available Mac tooling, choose the smallest
   appropriate UI, and implement/verify the pilot before expanding to other
   lessons. Reuse the course computations and adjacent worked explanations.

## Task boundaries and acceptance

- Inputs: `AGENTS.md`, `PROGRESS.md`, `LEARNING_MEMORY.md`, the workflow above,
  `docs/NOTEBOOK_CURRICULUM.md`, Day 6/7 artifacts, and
  `learning_artifacts/day-08-pretraining-data-and-recipe/README.md`.
- Current allowed outputs: workflow/memory/progress records. Future implementation
  scope must be established when the learner requests the pilot; this packet is
  not approval for GPU training, renderer production, or public publication.
- Arrival success: expected course commit present, correct execution host,
  platform-appropriate runtime checked, short progress recap, and recorded next
  action resumed without inventing completed work.
- Return evidence: changed paths/commit, what actually ran and on which machine,
  tests and limitations, learner's new explanations/open questions, and next step.
- Running jobs: departure checks on 2026-09-09 found no matching course Jupyter,
  Python, or torchrun processes for user `dongxi`; the NVIDIA compute-process
  query returned no rows (exit 0). This is a point-in-time observation, not a
  guarantee about future jobs or differently named processes. Nothing was
  stopped, restarted, or migrated by the handoff.
- Interactive state: no live playground exists yet; no slider values, runtime
  variables, or browser session are claimed to have been saved.

The departure task commits/pushes this packet and the scoped workflow records,
then reports the verified published commit in the conversation. On arrival,
Git history can identify this packet's commit without embedding its own hash
in itself. No application-level conversation transfer has been performed;
opening a local Mac course task and using the arrival phrase is sufficient.
