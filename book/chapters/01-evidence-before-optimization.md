# Chapter 1 — Evidence Before Optimization

Companion: [guided laboratory route](../labs/01-evidence-before-optimization.md).

A training process ends without an error. Its loss is lower than it was one minute
ago. A checkpoint appears on disk. Did the model improve?

Perhaps—but none of those facts answers the question by itself.

Language-model development is unusually good at producing persuasive numbers.
Losses move every step, rewards can rise rapidly, samples are easy to cherry-pick,
and modern frameworks can hide thousands of implementation decisions behind a
short command. The first skill of model development is therefore not optimization.
It is learning how to make a claim that the evidence can actually support.

This chapter establishes that discipline. We will define the identity of an
experiment, separate observations from interpretations, design falsifiable tests,
and use a three-step Qwen3-0.6B fine-tuning run as a worked example. The run is
small on purpose: it lets us examine the logic of an experiment before scale makes
the logic difficult to see.

## 1.1 Learning outcomes

After completing this chapter, you should be able to:

- identify the code, environment, data, model, configuration, seed, and hardware
  needed to describe an experiment;
- explain why a random seed does not guarantee identical results;
- write a hypothesis with observable success and failure conditions;
- design a comparison that changes one meaningful variable at a time;
- distinguish an observation, a supported interpretation, and an unsupported
  claim;
- explain what a smoke test establishes and what it leaves unknown;
- interpret system, cgroup, and CUDA memory measurements on a unified-memory
  machine without adding overlapping values.

**Prerequisites:** ordinary Python and a basic distinction between a model,
a training objective and a measurement. No large-model execution is required
to work through this chapter's evidence questions.

The practical reproduction instructions live in
[Appendix A](../appendices/a-laboratory-setup.md). The complete experiment contract
and evidence remain in the companion repository.

The [four Day 1 notebooks](../../notebooks/day-01/README.md) make the evidence
standard executable: canonical run identity, deliberately failed smoke criteria,
a read of the actual later TinyStories result, and a checkpoint/tokenizer
interface audit. They use CPU computations
and stored reports, with adjacent explanations and useful plots.

## 1.2 An experiment produces measurements, not conclusions

Consider a hypothetical log:

```text
step 1: loss = 2.41
step 2: loss = 2.36
step 3: loss = 2.33
exit status: 0
```

We can state several things about it, but those statements do not have equal
status.

**Observation:** the reported loss decreased from 2.41 to 2.33, and the process
returned exit status 0. These values were directly recorded.

**Supported interpretation:** this particular execution completed three reported
steps. If the trainer's behavior and logs have also been inspected, the result is
evidence that the tested training path is operational.

**Unsupported claim:** the model acquired a generally useful capability. Nothing
in the log measures behavior on a held-out task, and three loss values cannot
establish generalization.

This gives us a basic evidence chain:

```text
configuration → execution → recorded observations → bounded interpretation
```

The last arrow is where many errors occur. An interpretation should not contain
more certainty or scope than the observations can carry.

### Loss is not capability

Training loss answers a narrow question: how well does the current model predict
the selected training tokens under the selected objective? It does not directly
answer whether the model follows instructions better, solves new problems, becomes
more truthful, or retains earlier capabilities.

Even a perfectly measured reduction in training loss can coexist with:

- memorization rather than generalization;
- regression on unrelated tasks;
- worse generation behavior caused by changed formatting or termination;
- contamination between training and evaluation data;
- optimization of a proxy that does not represent the desired capability.

Later chapters will build evaluation contracts for those questions. For now, the
important habit is to name what a metric measures instead of calling it “better.”

## 1.3 The identity of an experiment

“I trained Qwen3 with seed 42” does not identify a reproducible experiment. A
useful experiment identity is a tuple:

$$
\mathcal{E} = (C, L, H, M, \mathcal D, \mathcal K, s),
$$

where:

- $C$ is the code revision, including local patches;
- $L$ is the locked software environment;
- $H$ is the relevant hardware and system identity;
- $M$ is the model and exact model revision;
- $\mathcal D$ is the dataset, revision, split, and selection procedure;
- $\mathcal K$ is the full experiment configuration, separate from model parameters;
- $s$ is the collection of random seeds.

The result should also retain its outputs and measurements. Without them, we know
what was intended but not what occurred.

### Code identity

A branch name such as `main` moves over time. Record an immutable commit hash. If
the working tree contains uncommitted changes, record that fact too: a commit hash
cannot identify modifications that exist only on one machine.

External trainers deserve the same treatment. Record the upstream revision and
the identity of any applied patch. “Latest open-instruct” is not an experiment
identity.

### Environment identity

A package list is useful, but a lock file is stronger because it describes the
dependency solution that should be recreated. Record both the lock file and a
cryptographic hash of its contents. The interpreter, CUDA runtime, and important
framework versions should also appear in the manifest so a reviewer can quickly
see what actually ran.

An import test is not sufficient. A library may import while a real GPU kernel
fails to compile or execute. Environment validation should exercise representative
operations: the attention backend, forward and backward computation, compilation,
and any custom kernels on which the experiment depends.

### Model and data identity

Repository names are mutable pointers. Record immutable model and dataset
revisions, the dataset split, filters, sample count, and ordering or selection
procedure. Two runs with the same model name and seed can still see different
tokens if the dataset changed or preprocessing was reordered.

### Configuration identity

The configuration includes more than the learning rate. At minimum, training work
should record:

- precision and attention backend;
- optimizer and schedule;
- microbatch size and gradient accumulation;
- sequence lengths and packing behavior;
- gradient checkpointing and clipping;
- total steps or token budget;
- random seeds;
- checkpoint source and output policy.

These fields define the computation. Omitting them makes comparisons ambiguous.

## 1.4 Reproducibility is larger than a seed

A pseudorandom seed selects a repeatable stream for a particular random-number
generator. Training involves several possible generators and several other sources
of variation:

- dataset shuffling and worker processes;
- parameter initialization;
- dropout and sampling;
- asynchronous or nondeterministic GPU kernels;
- floating-point reduction order;
- compiler, driver, and framework changes;
- distributed communication order.

Floating-point addition is not associative:

$$
(a+b)+c \neq a+(b+c)
$$

in finite precision. Two correct parallel reduction orders can therefore produce
slightly different values. During many optimization steps, small numerical
differences can lead to visibly different trajectories.

In this book, **repeatability** means rerunning the recorded procedure in the same
controlled environment, while **reproducibility** means independently recreating
the result from its preserved identity and protocol. Terminology varies across
fields, so the operational definition matters more than the label.

The objective is not always bit-for-bit equality. The objective is to know which
variation is acceptable, which quantities should agree, and what evidence would
indicate a genuine implementation difference.

## 1.5 Write the experiment contract before execution

A useful experiment begins as a contract, not a command. It answers five
questions:

1. What question are we investigating?
2. What do we predict, and why?
3. What remains fixed, and what changes?
4. Which measurements decide the outcome?
5. What will the experiment explicitly not establish?

Compare these hypotheses:

> The training run should work.

and:

> Qwen3-0.6B will complete three BF16 full-SFT optimizer steps with SDPA,
> microbatch size 1, and sequence length 1024. Every reported loss will remain
> finite, expected model outputs will be saved, and whole-system available memory
> will remain above 24 GiB.

The second is useful because the result can contradict it. “Work” has been replaced
by observable conditions.

### Conjunctive success criteria

If a hypothesis requires execution **and** numerical validity **and** memory
safety, all three matter. Suppose training exits successfully but available memory
falls below the precommitted safety reserve. The execution claim is supported, but
the experiment fails its overall success criteria.

This prevents a common form of hindsight bias: redefining success around whichever
number looks favorable after the run.

The same principle applies to recovery: native child processes can finish
successfully while a required deadline, final-state comparison or supervised
cleanup condition fails. Preserve the passing observations and the failed
overall result. [Appendix D](../appendices/d-reproduction-and-environments.md#d6-checkpoints-recovery-and-job-supervision)
links the actual recovery cases and their unchanged failure records.

### Controlled comparisons

In a controlled comparison, the independent variable is the factor deliberately
changed, dependent variables are the measured outcomes, and other important
conditions are held fixed.

If batch size 1 is safe, this is an interpretable next comparison:

```text
batch size:      1 → 2
sequence length: 1024
attention:       SDPA
everything else: fixed
```

Changing batch size, sequence length, and attention backend simultaneously might
be useful for testing a complete recipe, but it cannot isolate which change caused
the observed difference.

## 1.6 Smoke tests occupy the first rung of evidence

A smoke test asks whether a narrow path through the system executes. Depending on
the protocol, it may validate:

- model, tokenizer, and dataset loading;
- preprocessing and collation;
- forward computation and loss construction;
- backward computation and optimizer execution;
- metric logging and checkpoint saving;
- basic resource safety for the tested configuration.

It does not normally validate:

- useful capability improvement;
- convergence over a realistic token budget;
- stability over hours or days;
- the largest safe batch or sequence length;
- optimality relative to alternative recipes;
- identical behavior on another system.

A smoke test is valuable precisely because its claim is small. It catches basic
failures cheaply before a long experiment turns them into expensive failures.

We will use three experiment modes throughout the book:

| Mode | Typical duration | Primary question |
|---|---:|---|
| Smoke | 1–15 minutes | Does the intended path execute and record evidence? |
| Learning | 30 minutes–4 hours | Does a visible, interpretable learning signal appear? |
| Reference | 8–24+ hours | Does the conclusion survive a publication-quality comparison? |

Duration is a guide, not a definition. The claim and evidence standard define the
mode.

## 1.7 Memory on a unified-memory system

On a conventional accelerator server, people often speak as though CPU RAM and
GPU VRAM were independent budgets. NVIDIA DGX Spark uses unified memory: CPU
allocations, CUDA allocations, model state, activations, optimizer state, Linux
file cache, and other processes draw from the same physical pool.

A rough decomposition of training pressure is:

$$
M_{\text{train}} \approx
M_{\text{weights}} + M_{\text{gradients}} + M_{\text{optimizer}}
+ M_{\text{activations}} + M_{\text{temporary}} + M_{\text{runtime}}.
$$

Some terms are nearly fixed for a given model; activations and temporary buffers
depend on batch geometry, sequence length, attention implementation, and
checkpointing. This is why multiplying a batch-1 peak by eight is not a reliable
prediction for batch size 8.

We preserve three views:

| View | What it tells us |
|---|---|
| Whole-system `MemAvailable` | Remaining headroom for the entire machine |
| Cgroup `memory.peak` | Peak memory charged to the isolated job scope |
| CUDA-visible process allocation | Allocation attributed through NVIDIA's process accounting |

These accounting domains overlap. They must not be added together. If a cgroup
peak is 7 GiB and CUDA-visible allocation is 6.4 GiB, reporting 13.4 GiB would
double-count some memory rather than reveal a more accurate total.

Capacity is established incrementally. Start with the smallest useful batch and
sequence, change one dimension at a time, monitor the whole-system reserve, and
stop before interactive work or the operating system is placed at risk.

## 1.8 Case study: a three-step Qwen3-0.6B smoke test

We can now examine a real experiment without asking it to prove more than it was
designed to prove.

### Question and prediction

The question was whether the pinned DGX Spark open-instruct stack could complete
an existing Qwen3-0.6B full-SFT smoke path while preserving memory headroom.

Before execution, the experiment predicted three BF16 optimizer steps using SDPA,
microbatch size 1, sequence length 1024, gradient checkpointing, and seed 42. The
success criteria required:

- exit status 0;
- all three optimizer steps;
- finite reported losses;
- saved model outputs;
- at least 24 GiB of whole-system available memory throughout the run.

The contract explicitly excluded claims about general capability improvement,
long-run convergence, larger configurations, cross-host numerical identity, and
recipe optimality.

### Identity and protocol

The run recorded immutable revisions for the Qwen3-0.6B model, the selected
instruction dataset, the trainer, its local compatibility patch, and the
environment lock. It used 32 requested examples and ran inside a 100 GiB cgroup
with swap disabled for that job.

The complete pre-run contract is preserved in the
[experiment specification](../../experiments/specs/2026-08-29-qwen3-0.6b-sft-smoke.yaml).

### Observations

| Measurement | Observed value |
|---|---:|
| Exit status | 0 |
| Completed optimizer steps | 3 of 3 |
| Reported losses | 0.360384, 0.264207, 0.357005 |
| Starting `MemAvailable` | 118.84 GiB |
| Minimum `MemAvailable` | 109.74 GiB |
| Cgroup peak | 4.03 GiB |
| CUDA-visible peak | 6.42 GiB |
| Memory and OOM events | 0 |
| Runtime | 21 seconds |
| Saved weight file | 1,192,135,096 bytes |

The third loss is higher than the second. That is not a failed criterion: the
prediction required finite losses, not monotonic loss. Three different
microbatches can have different difficulty, and three points are far too few for
a convergence claim.

The trainer did not report gradient norms. We therefore do not manufacture a
gradient-norm observation. The completed training path is evidence that backward
and optimizer operations executed, but it is not a recorded measurement of the
gradient distribution.

### Interpretation

All precommitted criteria passed. On the recorded host and stack, this exact
profile loaded the identified model and data, preprocessed the requested examples,
executed three BF16 full-SFT steps through the SDPA path, saved the resulting model,
and preserved the memory reserve.

That is the conclusion. It is intentionally narrower than “the model improved.”
Establishing improvement would require a capability definition, a frozen
evaluation set, an appropriate baseline, and uncertainty-aware comparison. We
will build that machinery later.

The complete measurements, warnings, and output locations are preserved in the
[experiment report](../../experiments/reports/2026-08-29-qwen3-0.6b-sft-smoke.md).

## 1.9 A reusable reasoning pattern

For technical decisions throughout this book, use the following sequence:

> Choice → rationale → evidence → trade-off → failure risk → next experiment

Applied to this case:

- **Choice:** begin with three steps at batch 1 and sequence length 1024.
- **Rationale:** exercise the complete path with low cost and conservative memory.
- **Evidence:** execution, losses, telemetry, and saved outputs were preserved.
- **Trade-off:** the run provides system evidence but almost no capability evidence.
- **Failure risk:** a longer or larger run may expose behavior absent from the smoke
  configuration.
- **Next experiment:** only after defining the relevant question, vary one dimension
  or move to a learning-mode capability experiment.

This pattern keeps model development attached to decisions rather than dashboards.

## 1.10 A checkpoint is more than its tensor shapes

Suppose two tokenizers each contain eight IDs. In one, IDs 2 and 3 mean `cat`
and `dog`; in the other, those meanings are exchanged. Both fit an eight-row
embedding and an eight-logit output. Shape checks pass while the new interface
asks the checkpoint to read and predict different symbols.

This is an interface failure. A checkpoint's learned association with row 2
does not change when a new tokenizer calls that row `dog`. Preserve token
meanings when a supervised checkpoint becomes a parent for another objective.

The vocabulary mapping is necessary but insufficient. Identical mappings can
encode `CAT` differently if normalization changes. Merge ranks, pre-tokenization,
post-processing, decoding, added-token behavior, special IDs, chat templates
and generation stops also belong to the supported semantic interface. Runtime
padding buffers are operational state; the training padding/truncation policy
is a scientific configuration choice.

Keep three questions separate: which upstream revision was requested, which
actual bytes and encoding rules were used, and which computation occurred.
A declared revision cannot authenticate arbitrary local files. A byte or
interface fingerprint cannot establish quality or recover missing provenance.
For an older checkpoint without an interface record, explicit audited adoption
can record its actual saved semantics while leaving its original source unproven.

The [interface notebook](../../notebooks/day-01/04_checkpoint_interface.ipynb)
builds a local eight-ID fixture without pretrained weights or downloads. Compare unchanged
save/reload with swapped meanings and changed normalization before considering
a downstream handoff. [Appendix D](../appendices/d-reproduction-and-environments.md#supervised-fine-tuning-state-and-adapter-exports)
provides manifests, validation controls and adapter-export details; Chapter 9
develops the training and numerical handoff mechanisms.

## 1.11 Exercises

Questions 1–7 develop the core evidence argument. Questions 8–9 are optional
evidence-reading extensions: use section 1.10 and
[Appendix D](../appendices/d-reproduction-and-environments.md#d6-checkpoints-recovery-and-job-supervision)
for their interface/recovery context. Their original numbers remain stable.

1. A run reports losses `1.9`, `1.7`, and `1.8`, then saves a checkpoint. Classify
   each statement as an observation, supported interpretation, or unsupported
   claim:

   a. The third reported loss exceeded the second.
   b. The training path completed its requested steps.
   c. The checkpoint is a better general assistant.
   d. Training is converging.

2. Two researchers use the same model name, learning rate, and seed but obtain
   different losses. List four identity fields you would investigate before
   blaming GPU nondeterminism.

3. Rewrite “I expect training to work” as a falsifiable smoke-test hypothesis. It
   must include execution, numerical validity, output preservation, and a memory
   boundary.

4. A profile reports a 5.8 GiB cgroup peak and a 7.1 GiB CUDA-visible peak. What
   total memory did the job consume? Explain why the two numbers are insufficient
   to answer exactly.

5. A run exits successfully but crosses its precommitted memory reserve. Which
   parts of the hypothesis are supported, and what is the overall result?

6. Design the next controlled memory experiment after a safe run at batch size 1
   and sequence length 1024. State the independent variable, two dependent
   variables, and at least four controlled variables.

7. A model and a new tokenizer have the same vocabulary size, but IDs 2 and 3 have
   exchanged meanings. Why can loading succeed while the handoff is invalid?
   What check catches an unchanged vocabulary with a changed normalizer?

8. What can explicit adoption establish for an older checkpoint lacking an
   interface fingerprint, and what remains unproven? Should an invocation
   timestamp be part of recovery configuration equality?

9. Two children finish a replay experiment with exit 0, but its fresh resume
   hits the external deadline above the memory reserve. Why is the overall
   replay still failed? Is the outer adapter's exit 1 the native child's exit?

Attempt the exercises before reading the
[solutions](../solutions/01-evidence-before-optimization.md).

## 1.12 Chapter summary

- A run produces observations; interpretations must remain bounded by them.
- Experiment identity includes code, environment, hardware, model, data,
  configuration, and seeds.
- A seed controls some randomness but cannot guarantee identical training.
- Hypotheses and success criteria are written before execution.
- Controlled comparisons isolate one meaningful change.
- Smoke tests validate narrow execution paths, not general capability.
- Unified-memory measurements describe overlapping accounting domains and are not
  additive.
- Detailed records belong in experiment reports; the book uses them to support a
  precise argument.
- Checkpoint handoffs preserve token meanings and generation interfaces, not just
  shapes; actual file identity is distinct from declared upstream ancestry.

The next chapter begins at the model's input boundary. Before a language model can
predict a token, text must be divided into tokens and mapped to vectors. We will
compare that boundary across English, Chinese, and Swedish—and predict the outcome
before inspecting it. A multilingual ranking prediction and a later exact Unicode
round-trip criterion will fail. We will keep both failures, because a valid
experiment can improve the model of the system by contradicting the model in our
head.
