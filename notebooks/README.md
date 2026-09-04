# Notebooks

Notebooks narrate derivations, experiments, and visual analysis. Reusable logic
belongs in `src/dongxi_llms/` and should be imported here.

## Interaction contract

Mathematical mechanism notebooks use this learning cycle:

1. begin with a deep question about the mechanism;
2. record a prediction before execution;
3. implement the smallest transparent version;
4. perturb an assumption or run a deliberately broken variant;
5. distinguish observation from interpretation;
6. state what the result proves and what it does not prove.

The learner and course guide work through these notebooks cell by cell. They are
not passive demonstrations, answer dumps, or collections of decontextualized
calculation questions. Notebook outputs are exploratory until their important
claims are reproduced by reusable source code, tests, and an experiment report.

Each exercise or conceptual checkpoint is followed by a clearly labeled
reference solution and explanation. Attempt the prompt before revealing or
running the reference cell. The reference is deliberately adjacent: the learner
should spend time reasoning about the mechanism, not searching elsewhere for
routine syntax or an unstated canonical answer.

Every book chapter has a planned notebook pathway, normally divided into a
mechanism microscope, a deliberate perturbation or failure, and an integration
or evidence session. The course-wide map and activation rules are in
[`../docs/NOTEBOOK_CURRICULUM.md`](../docs/NOTEBOOK_CURRICULUM.md). Create these
sessions as their chapters become active rather than adding empty placeholders.

## Sessions

- [`day-03/`](day-03/) — logits, probability, next-token loss, causal alignment,
  gradients, and a tiny learned distribution
- [`day-04/`](day-04/) — scaled causal attention, gradient and failure diagnosis,
  and KV-cache equivalence
