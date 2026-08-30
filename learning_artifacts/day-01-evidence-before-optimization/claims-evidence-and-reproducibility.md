# Claims, Evidence, and Reproducible Runs

- Day: 01
- Date opened: 2026-08-29
- Status: understanding demonstrated; experiment verified
- Book destination: `book/chapters/01-evidence-before-optimization.md`
- Related evidence: `experiments/reports/2026-08-29-qwen3-0.6b-sft-smoke.md`
- Related production tasks: none currently queued

## Questions that drove the discussion

- What is an observation, an unsupported claim, or a supported interpretation?
- What information belongs in an experiment specification versus its report?
- Does successful process exit prove that the complete experiment succeeded?

## Learner's initial model

The learner initially treated a successful training process exit as sufficient
support for the run being fully successful. The initial experiment hypothesis
also omitted what a smoke experiment could not prove.

## Refined mental model

An experiment is judged against all precommitted criteria. Exit status, finite
loss, nonzero or finite gradients, and memory safety answer different questions.
A process can exit successfully while the experiment fails because a safety or
scientific criterion was violated.

Specifications preserve intended identity and decision rules before execution.
Reports preserve observed results afterward. Environment identity—including GPU
and driver, Python and package versions, Git commit, and available memory—belongs
with the specification or manifest. Measured exit code, minimum `MemAvailable`,
and observations versus interpretations belong in the report.

## Concrete examples and derivations

The completed smoke experiment preserved three different memory views rather
than adding them as if they measured disjoint memory:

- system `MemAvailable`;
- cgroup memory accounting;
- CUDA-visible allocation.

The run exited with status 0 and produced finite losses, but this supports only
the tested operations and shapes. It does not establish capability improvement,
long-run stability, or recipe optimality.

## Demonstrated understanding

- Correctly classified observations, unsupported claims, and supported
  interpretations in the interactive exercise.
- Correctly identified exit code, measured minimum `MemAvailable`, observations,
  and interpretations as report content.
- Correctly identified GPU/driver, package versions, Git commit, and available
  memory as environment or specification identity.
- Refined the success rule to: even when training exits successfully, the
  experiment is not fully successful if a safety criterion fails.

## Evidence and limitations

The Qwen3-0.6B BF16 smoke run is reproducibly documented in the linked report.
Its result validates the particular tested operations on the recorded GB10
environment; it is not a general claim about every shape, duration, model, or
software revision.

## Open edges

- Apply the same evidence discipline to every later tokenizer, training, and
  evaluation lab.
- Keep capability claims separate from training-health evidence.

## Reuse opportunities

This discussion already anchors Chapter 1 and should become the evidence pattern
used by all later learning artifacts and experiment reports.
