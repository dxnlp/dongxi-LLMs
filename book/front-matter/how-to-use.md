# How to Use the Book and Companion Repository

Start with the preface and Chapter 1. Chapters 2–5 develop a common decoder;
Chapters 6–9 use it to explain training, evaluation and SFT. Chapters 10–13 introduce
preferences and policy optimization. Chapters 14–15 diagnose failures and assemble
the final comparison. The [28-day route](../../docs/COURSE_SEQUENCE.md) links
every day's material and practical work.

For a lesson, first read its motivating question. Write a prediction in ordinary
language before executing the reference. A useful prediction says which quantity
will change, why, and what would contradict your explanation. It need not contain
a calculated number. Next run the small implementation, examine the plot, and
change one meaningful assumption. Finish by explaining the discrepancy between
your prediction and the observation.

Each notebook supplies a runnable reference and an explanation adjacent to its
exercise. You can inspect the answer immediately when syntax is holding you up;
independent understanding is tested later by reconstructing the argument or
designing another intervention. Prefer thoughtful questions about information
flow, gradients, controls and competing explanations to calculation speed.

Use a fresh kernel and run from top to bottom when checking readiness. Running
only a later cell can accidentally use stale variables from an earlier trial.
Source notebooks are preserved by the course verifier, which writes executed
copies elsewhere. Saved previews are fixed reference observations; run the plot
cell to regenerate it from your current state. The image's source and evidence
type matter as much as its appearance.

The reusable modules under `src/dongxi_llms/` hold the computations. The notebooks
hold the question, prediction, intervention, visuals and interpretation. Tests
check mathematical or protocol invariants. Experiment specifications freeze the
intended comparison; reports record the actual result, including failures.
Writing a report before measuring the outcome would break this chain.

Three execution modes are available:

| Mode | Purpose | Typical environment |
|---|---|---|
| Mechanism reference | Inspect exact arithmetic, masks, gradients and failures | CPU on Mac or Spark |
| Bounded model experiment | Measure learning and resources under a fixed cap | Spark |
| Saved-evidence analysis | Reproduce a reported comparison without its weights | Either machine |

Use [Appendix D](../appendices/d-reproduction-and-environments.md) for environment and verification commands. The CPU route avoids
Hub downloads. Spark extensions record model revisions, data identity, objective,
batch geometry, stop conditions and memory reserve. Their specifications are
study material even before their executions are approved or measured.

The day trackers distinguish ready material from learner practice. Reading all
solutions is different from independently explaining them. Executing all notebooks
is different from training every Qwen checkpoint proposed by the roadmap. Preserve
these distinctions when assessing yourself or presenting the course to others.

When switching machines, use the established phrases: **Switch to Mac**, then
**Continue on Mac**, or **Switch to Spark**, then **Continue on Spark**. The
departure saves scoped work and the arrival verifies host and repository state.
Git transfers the source and durable records, not an active Python kernel or
an SSH-forwarded browser port. See the [workflow](../../docs/LEARNING_WORKFLOW.md).

At the end of a chapter, use its worked solutions and the mastery rubric to
identify the next question. The final capstone asks for a defended checkpoint
comparison, an explicit genealogy, a model card and a release decision based on
the accumulated evidence. A failed capability claim can still produce a useful,
well-documented experiment and a better next hypothesis.
