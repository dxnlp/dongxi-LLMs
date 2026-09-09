# Chapter 6 — Pretraining as a Controlled System

Chapter 5 assembled a decoder. Token IDs became embeddings; attention mixed
information across allowed positions; residual blocks transformed the states;
the output projection produced vocabulary-sized logits. Cross-entropy supplied
gradients to the trainable parameters. That describes how learning is possible.
It does not yet describe a trustworthy training run.

Imagine two researchers starting with identical weights. They report the same
batch size, learning rate, and number of steps, yet obtain different results.
Neither must be lying. One may count microbatches as steps, another optimizer
updates. Their padding policies may give different numbers of supervised
targets. Their examples may arrive in different orders. One may resume AdamW's
history while the other restores only the weights. The model is only one state
inside a larger process.

This chapter develops that process as an explicit contract: **what data enters,
what objective is computed, how parameters change, what is measured, and what
must survive interruption**. It covers the complete Day 8 foundation of the
Days 8–9 chapter. Day 9's larger controlled run and its empirical diagnosis are
not invented here; they will supply the next layer of evidence.

## 6.1 The questions a recipe must answer

A recipe is not merely a list of hyperparameters. It should let another person
reconstruct both the computation and its interpretation. What does one token
count mean? Which predictions contribute to the loss? When does the optimizer
step? What is held fixed during validation? What exactly is restored after a
failure? Which success conditions are safety checks, and which concern quality?

The [three companion notebooks](../../notebooks/day-08/README.md) follow one
small decoder through this chain:

```text
document identity and split
  → token IDs and shifted windows
  → logits, summed NLL, valid-target count
  → accumulated gradient, clipping, AdamW update
  → fixed validation and recoverable training state
```

The data fixture consists of ten original sentences authored in this repository:
eight training documents and two validation documents. It is deliberately small
enough to inspect. It is not a benchmark or a representative sample of human
language. The modern Chapter 5 decoder remains the model; we change the system
around it rather than hiding the mechanism behind a large training framework.

## 6.2 Data identity is part of the experiment

“English text” is not a reproducible dataset specification. A corpus description
needs its source, revision, permitted use, selection rules, transformations,
deduplication policy, and split membership. A fingerprint identifies particular
content; it does not establish that content's quality or legal suitability.
Filtering changes the distribution being learned, so it belongs in the recipe,
not merely in a preprocessing script nobody records.

Split documents before constructing overlapping examples. If neighboring
windows from the same document enter both training and validation, a validation
score may partly measure recognition of previously exposed text. Document-level
splitting closes that route but does not eliminate duplicated documents,
paraphrases, shared sources, or benchmark contamination. Those require additional
checks appropriate to the actual dataset.

Our fixture rejects repeated document IDs and exact cross-split text matches
after case folding and whitespace normalization. It deliberately does not claim
near-duplicate detection. In a production specification, replace this small
check with an explicit, auditable data pipeline—not a vague statement that the
corpus was “cleaned.”

The distinction from Chapter 3 remains important: a training continuation is an
observed target, not a declaration that every alternative continuation is wrong
in human language. The corpus defines the empirical evidence from which the
model estimates a distribution.

## 6.3 Tokenization fixes the units of the budget

A tokenizer determines both the ID meanings and the number of sequence
positions used to represent text. Vocabulary size and sequence length therefore
affect different costs. A larger output vocabulary increases the projection's
work; a tokenizer that uses more tokens for the same text increases the number
of transformer positions. Neither “more tokens” nor “fewer tokens” is inherently
better without naming the representation and resource budget.

For transparency, this chapter uses a byte tokenizer, not BPE. Each UTF-8 byte
has its own ID from 0 through 255. EOS is 256 and BOS is 257, giving vocabulary
size 258. The Chinese character “数” becomes byte IDs 230, 149, 176. No unknown
character is required, but one character is not one training token. This is the
byte-level foundation discussed in Chapter 2, without learned merges.

For each document we create `[BOS] + bytes + [EOS]`. Inputs omit the final token;
labels omit the first. Thus every byte and the ending EOS is predicted once.
BOS provides context but is not a target. The loss receives logits of shape
$[B,T,V]$ and integer labels of shape $[B,T]$. It does not shift them a second
time. Remember that a vocabulary-sized output is produced at every training
position, even though there is only one observed target ID at that position.

A changed tokenizer is a changed experimental contract even if vocabulary size
stays constant. Loading a same-shaped embedding matrix is not enough when row
17 now names a different token. Resume must preserve ID meanings, not just tensor
dimensions.

## 6.4 Windows, padding, and document boundaries

The fixture divides the already shifted pairs into windows of length 16. A
document tail is right-padded: unused input positions contain EOS, while unused
labels contain the loss function's ignore value, -100. A genuine ending EOS
still contributes to loss. Ignoring every occurrence of EOS would silently
remove an intended prediction.

Each window starts fresh context and fresh position indices. No window crosses
a document. This simple choice loses preceding context at a window boundary;
it is not dense packing and is not presented as the most efficient production
policy. Reducing window length preserves the fixture's total target count while
changing which context some targets can use. Equal token counts can describe
different conditional prediction problems.

Dense packing instead joins short examples into fuller sequences. That can
reduce wasted padding, but requires a declared boundary policy. An EOS token is
a learned symbol, not automatically an attention barrier. If documents must be
independent within a packed sequence, causal attention needs additional
document-boundary restrictions, such as a block-diagonal causal mask. If
cross-document attention is allowed, say so and account for the resulting
training problem.

Do not confuse three mechanisms: the causal mask controls future visibility;
the attention mask controls allowed sources; the loss mask controls which
predictions enter the objective. Right padding is safe for the fixture's valid
prefix positions because causal attention forbids them from looking forward
into it. A zero loss mask alone would not make arbitrary padding invisible.

## 6.5 A run has several clocks

Let a microbatch contain $b$ windows of length $T$. Accumulate gradients over
$A$ microbatches before an optimizer update, using $R$ data-parallel ranks. If
every position is a valid target, one update accounts for

$$
N_{\mathrm{update}}=bTAR.
$$

For $S$ such updates, the target presentation budget is

$$
N_{\mathrm{run}}=SbTAR.
$$

These are exact only under the full-valid-position assumption. Padding, ignored
prompt positions, variable lengths, or a partial accumulation window change the
count. The runtime should count valid labels explicitly. Our 24-update,
single-rank recipe uses $b=1$, $T=16$, and $A=2$: 768 processed positions is a
ceiling on supervised targets, not a promise that all 768 carry loss.

Track optimizer updates, processed positions, valid target presentations, and
unique corpus content separately. Repeating an epoch increases presentations
without creating new source information. Reporting “one million tokens” without
saying which counter is being used makes comparisons ambiguous.

Shuffling is also state. Each window appears once in an epoch, but order matters
because the parameters and AdamW moments change between updates. Our stream
stores a permutation, cursor, epoch counter, and generator state. That is enough
for its single-process sampler; it does not represent a distributed loader's
workers, prefetch queues, or rank-specific state.

## 6.6 Gradient accumulation must preserve the objective

Gradient accumulation is often described as “simulating a larger batch.” The
essential condition is more precise: it must differentiate the same objective
at the same parameters. Let microbatch $j$ contain $n_j$ valid targets, each
with negative log-likelihood $\ell_{jk}$. The desired mean loss is

$$
L=\frac{\sum_j\sum_{k=1}^{n_j}\ell_{jk}}{N},\qquad N=\sum_j n_j.
$$

By linearity of differentiation,

$$
\nabla_\theta L=\sum_j\frac{1}{N}\nabla_\theta\left(\sum_{k=1}^{n_j}\ell_{jk}\right).
$$

Thus each microbatch can backpropagate its summed NLL divided by the same total
$N$. Do not zero the gradients or update the parameters between these backward
calls. Step once after the accumulation window.

The tempting alternative is to average each microbatch's mean. It equals the
desired objective only when their valid target counts are equal. A one-target
tail should not receive the same aggregate influence as a sixteen-target
window. If there are 17 targets total, their correct weights are $1/17$ and
$16/17$, not one half each.

Notebook 1 compares these two implementations against a single full-batch
backward pass through the actual decoder. The correct form agrees to floating
point tolerance; the wrong form does not. No optimization step is needed to
expose the bug. This is a stronger check than hoping training loss will reveal
an incorrect denominator.

```python
optimizer.zero_grad(set_to_none=True)
N = sum(number_of_valid_labels(batch) for batch in microbatches)
for batch in microbatches:
    (summed_next_token_nll(model, batch) / N).backward()
# Check, clip, and step only after this loop.
```

This is schematic code; the runnable implementation is in the notebook and
`pretraining_lab.py`. Dropout, cross-example operations, and floating-point
reduction order can complicate equivalence. Distributed gradient averaging
adds another normalization factor. The single-rank fixture does not validate a
distributed accumulation implementation.

## 6.7 AdamW: the gradient is not the update

The loss gradient describes local sensitivity: how an infinitesimal parameter
change would affect this objective. SGD turns it directly into an update,
$\Delta\theta=-\eta g$. AdamW uses history as well. With elementwise operations,
its first and second moment estimates are

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

The first moment remembers signed gradients. The second remembers squared
magnitudes; it is not a mean-subtracted variance estimate. Starting both at zero
introduces an initial bias, addressed by

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t}.
$$

For the unfused, non-AMSGrad variant used here, the update is

$$
\theta_t=(1-\eta_t\lambda)\theta_{t-1}
-\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
$$

The decay term shrinks the old parameter separately from the gradient moments.
This is why AdamW's decoupled weight decay is not simply “add an L2 gradient
before Adam's adaptive transformation.” The implementation follows the
[PyTorch AdamW update definition](https://docs.pytorch.org/docs/2.13/generated/torch.optim.AdamW.html).

Suppose gradients were positive for several updates and are now slightly
negative. The first moment may remain positive, so the next update can continue
in its previous direction. Likewise, a zero current gradient does not imply
zero movement: past moments or weight decay can still matter. An absent gradient
(`grad=None`) is a distinct implementation case, not interchangeable with an
explicit zero tensor.

Notebook 2 checks a hand-written recurrence against PyTorch using an actual
decoder embedding gradient. The tests also check several sequential updates.
The plotted SGD and AdamW changes use the same numerical learning rate to expose
different mechanisms, not to rank optimizers under a fair tuning budget.

Our tiny fixture applies decay to all trainable parameters and says so in its
contract. Larger recipes often use parameter groups with different decay rules.
Those choices, along with betas and epsilon, must be recorded rather than
inferred from the name “AdamW.”

## 6.8 Learning rate, warmup, and decay

Learning rate controls update scale, but its meaning depends on the optimizer.
A large rate can overshoot a locally useful direction; a small rate can make
progress slow. Neither rate guarantees that a noisy, nonconvex training problem
will improve at every step.

Warmup gradually introduces the intended update scale while parameters and
optimizer statistics are adjusting. It may improve stability; it is not a repair
for leaked labels, invalid data, or nonfinite arithmetic. Later decay reduces
the rate as the allocated optimization budget is consumed.

For one-based update number $u$, warmup length $W$, total updates $S$, peak
$\eta_{\max}$ and floor $\eta_{\min}$, our exact schedule is

$$
\eta_u=\begin{cases}
\eta_{\max}u/W,&1\leq u\leq W,\\
\eta_{\min}+\frac{\eta_{\max}-\eta_{\min}}{2}
\left[1+\cos\left(\pi\frac{u-W}{S-W}\right)\right],&W<u\leq S.
\end{cases}
$$

Here $W=3$, $S=24$, peak 0.01 and floor 0.001. These are bounded teaching values,
not proposed defaults for a larger model. The schedule ticks once per optimizer
update, not once per microbatch. In code the input is the number of already
completed updates; the notebook makes the conversion explicit.

If the effective batch doubles, the same update-based warmup now consumes twice
as many full-valid targets. A token-based schedule would express a different
invariant. The right comparison begins by deciding what should remain equal:
updates, token exposure, compute, wall time, or some combination with explicit
trade-offs.

## 6.9 Clipping and precision: safeguards with boundaries

Global gradient-norm clipping applies one scale to the concatenated gradient:

$$
g'=g\min\left(1,\frac{c}{\lVert g\rVert_2}\right),
$$

with the zero-norm case left unchanged and numerical safeguards in code. When
active, it preserves direction while limiting norm. Clip after accumulation:
clipping is nonlinear, so clipping each contribution separately is generally
different. Contributions 10 and -9 illustrate the issue: clipping their sum to
1 gives 1; clipping each to magnitude 1 before summing gives 0.

Clipping does not turn NaN into trustworthy evidence. Check finite losses and
gradients and specify whether a failed step aborts, retries, or is skipped. Nor
does clipping directly bound AdamW's final parameter-update norm: its adaptive
transformation and decay happen afterward.

Precision introduces another distinction: numerical range versus resolution.
BF16 uses eight exponent bits like FP32 but only seven stored fraction bits;
FP16 has five exponent bits and ten stored fraction bits. BF16 therefore offers
a much wider exponent range than FP16, while resolving fewer nearby values
around a number such as 1. In Notebook 2, 65536 overflows FP16 but remains finite
in BF16; a small increment above 1 survives FP16 but rounds away in BF16.

Mixed precision is a policy across operations and stored tensors, not one label
for the entire run. Autocast can use lower precision for selected operations
while parameters and optimizer state stay FP32. When gradient scaling is used,
unscale before inspecting or clipping gradients. BF16 often does not need loss
scaling for exponent range, but it does not make all training numerically safe.
The notebooks demonstrate casts and FP32 CPU training; they do not verify a
CUDA BF16 training recipe.

## 6.10 Memory and throughput are measured quantities

For $P$ unique parameters with FP32 weights, gradients and two FP32 Adam moments,
a simple persistent tensor ledger is

$$
M_{\mathrm{persistent}}\approx(4+4+8)P=16P\ \text{bytes}.
$$

That estimate excludes activations, attention intermediates, temporary optimizer
buffers, allocator overhead, framework state and other processes. It is not
peak memory. Tied weights count once as parameters; do not double-count an
embedding matrix merely because it also serves the output projection.

Changing microbatch size or sequence length can change activation memory while
leaving this persistent ledger nearly unchanged. Accumulation permits smaller
microbatches but still needs parameter gradients and optimizer state. BF16
activations do not automatically halve all persistent memory categories.

On Spark, distinguish GPU allocated/reserved/peak measurements from host
`MemAvailable` in the unified-memory environment. Retain the platform's
20–25 GiB host reserve and verify it during an approved run. A parameter-count
estimate alone does not establish that a model fits safely.

Measure throughput with a named denominator: processed positions/s and valid
targets/s answer different questions. State whether timing includes compilation,
warmup, validation, checkpointing and data loading. Accelerator timing also
requires appropriate synchronization. The CPU smoke verification is not a
GPU throughput benchmark, so this chapter reports none by inference.

## 6.11 Validation measures a fixed prediction problem

Held-out loss should use a fixed split, tokenizer, alignment, context policy,
mask and reduction. Sum the valid targets' NLL across the corpus, then divide
by their total count. Do not average unequal batch means. For that fixed unit,
perplexity is $\exp(L)$; comparing it across tokenizers without reconciling units
can be misleading.

Evaluation must not update parameters. In PyTorch, `eval()` selects evaluation
behavior for affected layers; `no_grad()` suppresses autograd recording. Neither
is a substitute for the other. Our helper restores the model's prior mode after
evaluation so a measurement does not silently alter subsequent training.

Notebook 3 plots each current training batch's pre-update loss against the
fixed held-out fixture's post-update loss. These are different populations and
measurement times, so their gap is not a pure estimate of generalization error.
Neither curve must decrease monotonically. A tiny finite loss can coexist with
data leakage; a nonzero loss can coexist with successful learning of uncertainty,
as Chapter 3 established.

Validation used repeatedly to select recipes is itself part of development.
Preserve a separate final test contract when making an eventual generalization
claim. Samples add qualitative evidence but do not replace fixed quantitative
evaluation or justify selecting only favorable generations.

## 6.12 Recovery restores a process, not just a matrix

A resumable state includes model parameters and optimizer history, plus the
clocks and data position that determine the next update. The fixture stores the
completed-update count, valid-target counter, shuffle state, CPU random state,
configuration, tokenizer identity, and data fingerprints. Its learning-rate
schedule is a pure function of the saved count and recipe.

Notebook 3 saves a trusted local checkpoint after update 12, then compares
continuations to update 24. Complete restoration reproduces the uninterrupted
parameter state and update history in the tested CPU environment. Omitting AdamW
state keeps the next batches but changes the updates. Omitting the data cursor
changes the next batches. Both incomplete runs can execute successfully, showing
why a readable checkpoint file is not a recovery test.

There is an instructive twist in the [recorded reference run](../../experiments/reports/2026-09-09-day8-material-verification.md):
the missing-cursor branch ends with slightly lower loss on this tiny holdout than
the correctly resumed branch. It still fails the recovery contract. A favorable
metric cannot retroactively make a different experiment the intended one, and
this one tiny comparison does not establish that restarting the data is better.

The basic need to save optimizer state alongside model state is also documented
in [PyTorch's checkpoint tutorial](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
Our checkpoint boundary is **after an optimizer update**. Mid-accumulation
recovery would additionally require partial gradients and the microstep index.
Production systems may need Python, NumPy and CUDA RNG states, rank-specific
samplers, worker/prefetch state, gradient-scaler state and atomic file publication.
The fixture is not a distributed checkpoint implementation.

Exact replay in one software/device configuration does not promise bitwise
equality across Mac and Spark or across library versions. A portable checkpoint
can preserve the intended experiment while arithmetic differs. Define a numerical
tolerance and meaningful continuation checks for that broader setting instead
of silently extending this lab's exact-equality claim.

## 6.13 From understanding to a bounded experiment

The [Day 8 specification](../../experiments/specs/2026-09-09-day8-bounded-pretraining.md)
fixes an executable CPU control and separates it from a later Spark candidate.
It names the data, model, token budget, optimizer, schedule, clipping, precision,
validation, recovery, time boundary and failure criteria. Its tiny run establishes
mechanical evidence; the GPU candidate still needs a real corpus decision,
profiling, safety measurements and explicit execution approval.

Before Day 9, be able to explain why each field is needed. A successful process
exit is necessary evidence of completion, not sufficient evidence that every
scientific and safety criterion passed. Conversely, a deliberate negative
control that exposes an incorrect recipe is a successful teaching experiment,
not a result to hide.

## Exercises

Use explanations and small interventions, not arithmetic speed. Adjacent
notebook references and the [worked solutions](../solutions/06-pretraining-as-a-controlled-system.md)
provide complete reasoning.

1. Why can document-level splitting still leave validation contamination?
2. If shorter windows preserve the target count, what learning condition changes?
3. Explain why EOS, a loss mask, and an attention boundary are not interchangeable.
4. What does $SbTAR$ count when some labels are ignored? What does it not count?
5. Derive the contribution of a short microbatch to a valid-token mean objective.
6. Why can AdamW move a parameter against the newest gradient's suggested direction?
7. What changes when accumulation doubles but the update-based schedule stays fixed?
8. Why clip after accumulation, and why is that not a guarantee against NaNs?
9. Explain how BF16 can avoid FP16 overflow yet lose more detail near 1.
10. Why does a 16-bytes-per-parameter ledger not prove that a run fits on Spark?
11. How would you test that changing validation batch size preserves the metric?
12. Design two separate recovery failures: same data but missing moments; same
    saved weights but missing data position. What observations distinguish them?

## What follows

The decoder defines a family of conditional distributions. The training system
defines how evidence changes that distribution over time. Reliable pretraining
requires both to be explicit. We now have a recipe that can be explained,
perturbed and recovered on a small example. Day 9 will ask what happens when an
approved run actually consumes a larger budget: how loss, examples, gradients,
throughput and memory together support—or limit—our conclusions.
