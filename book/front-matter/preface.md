# Preface — From API User to Model Developer

A language model can produce a fluent paragraph while failing to track who owns
the object mentioned two sentences earlier. A training run can lower its loss
while teaching an evaluation leak. A reward can rise while the behavior we care
about deteriorates. These tensions are the reason to study model development
as a sequence of arguments supported by computation.

This book begins with the question “what would count as evidence?” It then follows
text through tokenization, embeddings, causal attention and a complete decoder.
We train that decoder, evaluate its behavior, teach it with demonstrations and
preferences, and finally treat generation as a policy that can receive rewards.
The last chapters ask whether the improvements survive independent evaluation,
whether their computational cost is acceptable, and whether another person can
reconstruct the experiment.

The reader is assumed to know Python, arrays, basic derivatives and the broad
idea of neural networks. You need not have trained a language model. The first
examples are small enough to inspect their actual tensors. Later lessons retain
that transparency while introducing the practical constraints of real runs.
The notebooks provide reference solutions next to each question so routine
syntax does not become the main obstacle.

There are two scales of evidence. A tiny categorical policy can establish that
a gradient formula agrees with enumeration. It cannot establish that the same
algorithm improves a pretrained assistant. Conversely, a successful large run
can demonstrate practical behavior without revealing why a particular update
worked. The course connects these scales and labels what each observation supports.

Our first from-scratch model, DongxiGPT, trained on TinyStories. Its fixed
development loss fell substantially, and its samples gained recognizable story
structure. Yet some stories repeated familiar phrases or exchanged speaker roles
without explanation. We retain those outputs because they create a better
learning problem than a gallery of the nicest samples. The completed-run evidence
is used in Chapter 6; subsequent CPU demonstrations are labeled separately from
future pretrained-model experiments.

The learning schedule is28 days; the narrative is15 chapters. Those numbers serve
different purposes. A day organizes practice and a tangible deliverable. A chapter
builds one argument, often synthesizing several days. The material can be studied
more slowly. The schedule is an organizational route, not a promise of mastery
after a fixed number of hours.

Mac Studio is the default place for reading, small experiments, notebook plots
and approved media production. DGX Spark supplies model-scale GPU measurements.
The same durable files travel through Git, while environments, running processes,
credentials and model weights remain machine-specific. This lets a reader examine
a mechanism without maintaining a permanently running GPU service.

The course has succeeded for a reader when they can explain a design, derive
its relevant mathematics, implement it, perturb an assumption, and defend the
result using appropriate evidence. The final defense is a conversation about
decisions and their consequences. Producing attractive metrics alone does not
finish that conversation.
