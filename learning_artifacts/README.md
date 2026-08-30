# Learning Artifacts

This directory is the durable conceptual record of the interactive learning
journey. Artifacts are organized first by learning day and then by topic. They
preserve deep discussions, corrected mental models, concrete examples, evidence
levels, and open edges before the material is synthesized into the book.

Learning artifacts are neither chat transcripts nor finished chapters:

- chat is transient interaction;
- learning artifacts preserve what changed in the learner's understanding;
- experiment reports preserve empirical evidence;
- book chapters turn validated material into a coherent reader journey;
- `LEARNING_MEMORY.md` indexes these artifacts and manages public-production
  tasks that may move across sessions or machines.

## Directory convention

```text
learning_artifacts/
  day-NN-topic-slug/
    README.md
    focused-topic.md
```

Create the day directory when a learning day begins. Create or update a focused
topic file immediately after a substantive discussion; do not wait for the end
of the day or for the user to request documentation.

## What deserves an artifact update

Update the relevant topic when any of these occurs:

- the learner makes a prediction or states an initial mental model;
- an explanation corrects or materially refines that model;
- the learner explains a mechanism back or completes a calculation;
- an observation surprises or falsifies a prediction;
- an important limitation or unresolved question appears;
- the discussion suggests a book section, exercise, X article, animation, or
  experiment.

Do not record conversational filler. Preserve the conceptual progression and the
learner's own successful explanations where they provide evidence of
understanding.

## Evidence vocabulary

- `introduced`: discussed but not yet explained back or calculated;
- `demonstrated`: explained or calculated correctly in the interactive lesson;
- `observed`: measured interactively but not yet preserved in a reproducible
  report;
- `verified`: supported by preserved executable evidence;
- `open`: a named gap remains.

Never promote `observed` to `verified` without a durable report or equivalent
executable evidence.

## Required topic structure

Start from [`template.md`](template.md). Keep each file focused enough that a
future author or agent can reuse it without reconstructing the original chat.

