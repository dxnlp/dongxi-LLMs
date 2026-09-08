# Chapter 5 — Worked Solutions and Notebook Guide

The [companion lab](../labs/05-building-a-modern-decoder.md) provides the
sequential route. Every exercise has its runnable solution and explanation
immediately below it in the notebook. This guide summarizes what a sound
interpretation must preserve; it is not a replacement for running the cells.

The [Chapter 5 narrative](../chapters/05-building-a-modern-decoder.md) now covers
the Day 5 foundation, Day 6 modern mechanisms, and Day 7 architecture synthesis.
Its 30 conceptual exercises are answered below. The table covers the original
eleven notebooks; the core Day 7 defense immediately below it brings the full
pathway to twelve. Study that defense before optional recurrence.

| Session | Core reasoning in the worked solution |
|---|---|
| [1. Lookup and positions](../../notebooks/day-05/01_embeddings_and_positions.ipynb) | Repeated IDs select the same embedding row; different positions add different vectors. Lookup gradients sum over occurrences. Output tying can add gradients to rows absent from the input. Removing position embeddings does not remove the causal graph. |
| [2. Multiple heads](../../notebooks/day-05/02_multi_head_attention.ipynb) | Split and merge preserve time as its own axis. Loop and vectorized results and gradients agree under identical weights. Zeroing a head changes its value contribution, not the definition of causal visibility. |
| [3. Residuals](../../notebooks/day-05/03_residual_stream.ipynb) | For Y=X+XW, the incoming gradient is G+GW transpose in row-vector notation. A zero branch yields identity; a branch equal to -X cancels both state and gradient. |
| [4. LayerNorm](../../notebooks/day-05/04_layernorm_and_placement.ipynb) | Normalize over features with population variance and epsilon, then apply scale/bias. Time-axis statistics can leak future information. Zero branches make a pre-norm block identity, but post-norm still normalizes. |
| [5. MLP](../../notebooks/day-05/05_positionwise_mlp.ipynb) | Without activation, the combined weight is W_down W_up and combined bias is W_down b_up+b_down. Nonlinearity prevents this general collapse. Positionwise computation still acts on contextual inputs. |
| [6. Assembly](../../notebooks/day-05/06_assemble_decoder.ipynb) | Blocks preserve residual width, whereas the vocabulary head outputs one logit per candidate token. Tying is Parameter identity, not copying. A copy oracle fails the next-token objective despite near-zero incorrectly unshifted loss. |
| [7. Learning](../../notebooks/day-05/07_one_batch_learning.ipynb) | The declared batch is consistent and shifted once. Optimization changes parameters and reduces the correct loss. The frozen control stays unchanged. No gold labels are declared for the reversed-context probe, so its outputs cannot be scored as a generalization result. |
| [8. Modern norm/MLP](../../notebooks/day-06/01_rmsnorm_and_swiglu.ipynb) | RMS scaling does not center features. A SiLU gate is not a probability. Three gated projections require a different width for approximately matched parameters; biases must be accounted for. |
| [9. RoPE](../../notebooks/day-06/02_rotary_positions.ipynb) | Rotations preserve coordinate-pair norms; R_m transpose R_n gives R_(n-m). Cached keys keep their original positions. Restarting new query/key offsets breaks equivalence even with a correct causal mask. |
| [10. GQA/costs](../../notebooks/day-06/03_gqa_qknorm_and_costs.ipynb) | Fewer KV heads reduce compact cache payload and KV projection parameters, not the number of query distributions. QK normalization is explicitly specified. Logical bytes and dense arithmetic estimates are not measured latency or total runtime memory. |
| [11. Recurrence](../../notebooks/day-07/01_recurrent_depth.ipynb) | Shared parameters receive the sum of gradients through their uses. Weight sharing preserves stored size while additional applications spend computation. Different application states generally require different K/V; no adaptive router or quality gain is established. |

## Architecture defense

The core [Day 7 notebook](../../notebooks/day-07/02_architecture_defense.ipynb)
now makes this defense executable. Its worked solutions establish:

- Trace: Q output is `[2,6,16]`, compact K projection output `[2,6,8]` before
  splitting, MLP intermediate `[2,6,32]`, and vocabulary logits `[2,6,16]`.
  The two occurrences of width 16 have different roles; hooks are removed.
- Accounting: the modern fixture has 4,960 unique parameters and 3,072 compact
  float64 KV bytes; the dense forward matmul estimate is 125,952, not latency.
- Backward: all parameter tensors have connected finite gradients in the
  fixture, but weights remain unchanged without an optimizer step. Nonzero
  gradients are not a capability or component-importance measurement.
- Diagnosis: the reference passes finite/causal/cache checks. Wrong RoPE
  offsets remain causal but fail replay. Full-time centering can remain finite
  while failing causality and replay. These controlled examples do not imply
  a one-to-one mapping from every real failure signature to a unique cause.
- Comparison: one versus two shared applications can match stored parameters
  while spending different computation. Specify actual training/evaluation and
  measurement budgets before executing the proposal.

Use the modern tiny model after session 10. Explain, without relying on its
class names alone, what each operation does to the token, head, and feature
axes; why it is causal; which parameters it trains; and which invariants survive
its controlled intervention. An answer is strong when it identifies the relevant
test and its evidence boundary, not merely when it repeats a layer name.

Notebook readiness and the recorded reference experiment are complete material
deliverables. The Days 5–7 narrative, including the architecture-defense method,
is written; the learner's independent defense, any trained comparison, and
learner completion remain separate.

## Day 5 foundation — worked conceptual solutions

### 1. Vocabulary scores are not input positions

An output position scores the possible next IDs, so its last dimension equals
the vocabulary size: 16. Six input positions produce logits `[B,6,16]`.
Adding a seventh position produces `[B,7,16]`, not `[B,6,17]`. Increasing the
vocabulary is a different architectural change requiring corresponding table
and output-head capacity. The hidden feature width remains a separate quantity.

### 2. Repeated lookup does not imply repeated contextual states

Two occurrences of ID 3 retrieve exactly the same row `E[3]` from the same
parameter snapshot. Different learned position rows can make their initial
states differ; attention can then gather different prefixes. Neither step
guarantees that every pair of outputs is different, but equality of token IDs
does not require equality of those later states. The lookup path adds the two
occurrences' gradient contributions into the same stored row.

### 3. One objective can send different learning signals

For one position use row-vector notation:
$u=o_1W_{O,1}+o_2W_{O,2}$. If the gradient at $u$ is $g$, then the gradients
at the two head outputs are $gW_{O,1}^\top$ and $gW_{O,2}^\top$. They need
not match. For a minimal algebraic example, let $g=[1,1]$, $W_{O,1}=[1,0]$,
and $W_{O,2}=[0,2]$, with scalar head outputs. The returned gradients are
1 and 2. This demonstrates different paths, not a trained specialization result.

Those gradients continue through each head's own attention activations into
its projection parameters. Heads learn jointly, not as independent predictors.
Exactly identical corresponding Q/K/V parameters and output blocks, with an
otherwise symmetry-preserving optimizer and computation, can keep two heads
identical. Random initialization helps break this symmetry; it does not force
unique linguistic roles or prevent redundancy.

### 4. Heads trade mixture count against feature width

Under the baseline constraint $D=Hd$, width 16 can give four heads of width 4
or eight heads of width 2. The latter provides more separately parameterized
source distributions but fewer value features within each head. It does not
create eight full-width transformers or assign disjoint token ranges. More
heads may help a task, but architecture counts alone do not establish that.

### 5. The sum is not an untouched backup

The output contains $Y=X+F(X)$, not the ordered pair $(X,F(X))$. Different
inputs and updates can produce the same sum, so there is no general recovery
guarantee. A zero branch makes $Y=X$ and preserves the direct gradient. A branch
$F(X)=-X$ makes $Y=0$ and its input derivative zero. Thus a bypass offers an
identity route without enforcing lossless information storage or a nonzero
total gradient.

### 6. Causality extends beyond attention

If normalization computes an earlier position's statistics using all time
positions, changing a future token can change that earlier normalized state.
Attention's mask cannot undo the dependency already introduced. LayerNorm over
features computes each token's statistics from its own state. Provided those
states were causally computed, this local operation preserves the information
boundary. Other full-sequence operations must be checked by the same principle.

### 7. Normalizing the branch differs from normalizing the sum

Pre-norm gives $Y=X+F(\mathrm{LN}(X))$. If the branch is identically zero,
$Y=X$ and its input Jacobian is the identity. Post-norm gives
$Y=\mathrm{LN}(X+F(X))$, becoming $\mathrm{LN}(X)$ under the same
intervention. Its backward signal passes through the LayerNorm Jacobian.
This is not merely a naming difference; the skip contribution is itself
transformed in post-norm. The final normalization after a pre-norm stack remains
a separate operation and is not part of the zero-branch identity assertion.

### 8. Local feature processing can use contextual information

The MLP applies the same parameters independently to every position, but it
receives states already updated by attention. It can therefore transform
contextual features without gathering other tokens itself. Removing its
activation yields the row-vector affine map with weight
$W_{\rm up}W_{\rm down}$ and bias
$b_{\rm up}W_{\rm down}+b_{\rm down}$.

In PyTorch storage convention, this is `down.weight @ up.weight` and
`down.weight @ up.bias + down.bias`. GELU prevents this collapse in general;
it does not change the token axis or guarantee a readable concept per neuron.

### 9. Two projections, and one optional shared parameter

Attention's output projection maps concatenated heads `[B,T,H*d]` back to
`[B,T,D]`. The vocabulary head maps final normalized states `[B,T,D]` to logits
`[B,T,V]`. Under input/output tying, the stored vocabulary-head weight is the
same Parameter object as the embedding table `[V,D]`; the linear operation
uses its transpose. Copying values into another Parameter gives two independent
objects whose updates may diverge. True sharing accumulates classifier and
lookup contributions into one parameter, including classifier gradients on
rows absent from the input lookup.

### 10. Per-position supervision, batched learning

The model produces logits at all 12 positions of the two-sequence batch using
one fixed parameter snapshot. Each aligned label selects a target log-probability;
their negative mean is the scalar loss. Backward accumulates the corresponding
contributions into shared parameters, and one optimizer step changes them.
Causal masking constrains each prediction despite parallel computation.

In ordinary free generation, the next input token is the previously selected
prediction, rather than the supplied training continuation. The model selects
one new ID at a time and usually keeps its parameters fixed. Token-by-token
generation and position-wise training losses describe different operations.

### 11. A derivative describes a local change

For $\ell(e)=(e-1)^2$ at $e=.2$, the derivative is $-1.6$. A positive SGD
learning rate moves right. With $\eta=.1$, $e$ becomes $.36$ and loss becomes
$.4096$, below $.64$. With $\eta=1.5$, $e$ becomes $2.6$ and loss becomes
$2.56$. The initial direction is a descent direction, but the larger step
overshoots. AdamW additionally transforms gradients using optimizer state;
do not describe its exact update as plain raw-gradient SGD.

### 12. Memorization validates a narrow claim

Correctly aligned loss reduction, finite gradients, parameter changes, and a
stable frozen control support that the tested computation can optimize the
declared batch. The unique input IDs make simple successor memorization
possible; this is not a test that requires understanding syntax or context.

Contextual generalization needs held-out examples where token identity alone
is insufficient, such as the same last token with different preceding contexts
and appropriately defined targets. Claims about head roles need interventions
on trained models, defined evaluation metrics, and controls for redundancy or
compensation by other components. Neither different attention maps nor a
changed output after ablation establishes a unique semantic function by itself.
The reversed-context probe cannot supply an accuracy score without declared
gold labels.

## Day 6 modern decoder — worked conceptual solutions

### 13. Removing an offset is different from scaling a vector

LayerNorm subtracts the feature mean. Adding a common constant to the input
does not change its centered vector or variance. RMSNorm does not subtract
that constant: both its numerator and RMS magnitude change, generally producing
a different result. For a positive constant vector, LayerNorm produces zero
before its affine transform, whereas RMSNorm produces values approximately one
before its learned scale. With default affine parameters these remain zero and
approximately one. Arbitrary learned LayerNorm beta or RMS scales change those
final values. Neither method mixes token positions when statistics are over D.

### 14. Epsilon and vector length matter

For positive $a$, normalizing $ax$ with epsilon is equivalent to normalizing
$x$ with epsilon divided by $a^2$. The outputs are approximately equal only
where that denominator change is negligible. Without learned scaling, RMSNorm
gives approximately unit root-mean-square magnitude, not a zero mean. Its L2
length is approximately $\sqrt D$, not one. This distinction becomes important
when interpreting RMS-normalized Q/K scores as though they were cosines.

### 15. A gate is a feature multiplier, not a probability

SiLU is $a\sigma(a)$: it can be negative for negative $a$ and exceed one for
sufficiently positive $a$. Its outputs are not normalized across coordinates.
For $u=\mathrm{SiLU}(a)\odot c$, an arriving gradient $\delta$ gives
$d\mathcal L/da=\delta\odot c\odot\mathrm{SiLU}'(a)$. At $a=0$,
the derivative of SiLU is one half. Thus a zero gate can suppress forward
content without preventing gradient flow into the gate weights. This requires
appropriate nonzero content, inputs, and downstream gradients; it is not a
guarantee for every state or loss. The notebook's zero-output result also
depends on its bias-free projections.

### 16. Count all three projections

At feature widths D and F, the bias-free gated MLP has three matrices totaling
$3DF$, compared with two totaling $2DF$ in a bias-free ordinary MLP. Matching
matrix budgets suggests gated width near two-thirds of the original width.
The notebook's biased GELU MLP at D=16,F=32 has 1072 parameters. SwiGLU at
F=32 has 1536, and F=22 has 1056. A claim about quality must distinguish exact
and approximate parameter matching, and additionally declare training/evaluation
data and compute budgets. Interface-compatible replacements are not automatically
fair quality comparisons.

### 17. Relative position arises inside a dot product

For orthogonal pair rotations, $R_m^\top R_n=R_{n-m}$. Substituting this into
$(R_mq)^\top(R_nk)$ gives $q^\top R_{n-m}k$. A common position shift cancels
for fixed unrotated q and k. A whole language model also changes those vectors
with tokens, boundaries, and prior layers; the identity alone does not make its
outputs invariant to arbitrary sentence shifts. Being able to calculate an
angle at a new index likewise does not establish trained quality beyond the
training context range. Different checkpoints can use different pair layouts,
bases, or scaling rules.

### 18. Causal edges can use incorrect geometry

If the prefix occupies positions 0 and 1, the next Q/K should use position 2.
Restarting them at position 0 creates the wrong angle relative to already
rotated cached keys. The mask may still forbid every future edge, but the
allowed scores differ from the full-pass computation. Correct replay requires
both valid visibility and consistent positions under the same parameters and
prefix. Cached keys keep their original rotations; they are not rotated again
each time they are read.

### 19. Share source representations, not query distributions

GQA shrinks K/V projections from $DH_qd$ each to $DH_{kv}d$ each, and compact
K/V tensors from `[B,Hq,S,d]` to `[B,Hkv,S,d]`. Query tensors still use Hq,
as do attention distributions `[B,Hq,T,S]` and head outputs `[B,Hq,T,d]`.
Different Q vectors can form different weights over shared K and thus different
mixtures of shared V. Backward sums the contributions from all using query
heads into their shared K/V activations and parameters. Explicitly repeating
K/V verifies this arithmetic, but does not establish an efficient memory-access
implementation.

### 20. State the implementation, not just the acronym

The lab computes RMS statistics over each projected Q/K head's d features,
applies learned feature scales, then applies RoPE and retains division by
$\sqrt d$. There are separate Q and K scale vectors of length d per layer,
broadcast over heads and positions, so the extra parameter count is 2d per
layer. Separate statistics do not imply separate learned parameters for every
head. This is not the original L2-normalized, learned-score-scale QKNorm formula.
Learned coordinate scaling can also change direction, and need not commute with
rotation. Preserving the stated operation order is part of reproducing the lab.

### 21. One reduction does not reduce every cost equally

With fixed layer count, batch, source length, head width, and element size,
halving Hkv halves compact KV payload and K/V projection parameters. Q and
output projections, MLPs, embeddings, and norms remain. The dense score/value
matmul term still uses Hq; therefore total parameters and full arithmetic do
not halve. In the recorded fixture, four to two KV heads changes cache payload
6144 to 3072 bytes, parameters 5472 to 4960, and estimated matmul FLOPs
138240 to 125952. Latency, peak allocated memory, and quality need their own
measurements; none is established by those ratios alone.

### 22. Projections connect different feature spaces

Residual width D is the interface between blocks. Concatenated head width Hq*d
is internal to attention. A Q projection can map D to Hq*d and W_O can map
back. The pinned Qwen config sets D=1024 and Hq*d=16*128=2048, giving stored
projection shapes `[2048,1024]` and `[1024,2048]`. The chapter links the exact
configuration revision. This establishes those fields and the resulting shape
calculation, not compatibility with the toy model's RoPE layout, parameter
names, full implementation, or trained checkpoint.

### 23. Parameters are only one memory category

A 100M parameter count gives weight payload only after choosing a storage
dtype or quantization scheme. Training additionally needs gradients, optimizer
states, saved or recomputed activations, and temporary workspaces. Sequence
length, batch size, checkpointing, precision, attention implementation, and
sharding all affect the result. Inference KV payload is a separate function of
layers, retained length, KV heads, head width, batch, and cache dtype. The
larger candidate table is an analytical design exercise, not a successful
allocation or training run. Selecting a candidate requires profiling the actual
recipe and resolving the assumed tokenizer/vocabulary and data budget.

### 24. Shared weights do not imply shared states or free depth

Applying the same block twice stores one set of weights but evaluates two
transformations. Its second input, activations, and generally K/V differ from
the first application's; backward accumulates both uses' parameter gradients.
Two equal-valued independent blocks can initially produce the same outputs
while owning twice the block parameters and receiving separate updates.

A parameter-matched experiment permits additional recurrent computation; a
compute-matched experiment must compensate elsewhere for the additional work.
Neither guarantees equal latency or peak memory. Adaptive early exit and
recursion-specific KV sharing need explicit mechanisms beyond ordinary weight
tying. A fixed two-use notebook does not establish those mechanisms or any
trained quality benefit. Visible reasoning tokens and latent block applications
are separate ways of spending computation, not interchangeable evidence.

## Day 7 architecture defense — worked conceptual solutions

### 25. Equal widths can describe different spaces

Residual states have shape `[B,T,D]`; their last axis contains learned features.
Logits have shape `[B,T,V]`; each entry on the last axis scores a candidate token
ID. Equality of D and V is incidental. Growing V from sixteen to 10,000 changes
the embedding table from `[16,16]` to `[10000,16]` and logits from `[2,6,16]` to
`[2,6,10000]`. The tied head uses that enlarged table. Residual states remain
`[2,6,16]`, and internal attention and MLP dimensions need not change. Adding
vocabulary entries is still a change to model parameters and the tokenizer
contract, not merely renaming the final axis of an existing trained model.

### 26. Backward connectivity is not an optimizer update

A non-None finite gradient on every parameter tensor establishes that the
tested loss has backward paths to those tensors and that the resulting entries
are finite. It does not establish nonzero values in every entry, useful roles,
good conditioning, or generalization. No optimizer step means no learning update
was applied, even though gradients were calculated. The notebook explicitly
checks unchanged parameter values rather than equating backward with training.

With twelve equally weighted token losses, the mean objective gives each logit
the derivative `(p - q) / 12`. The corresponding parameter contributions add
through the chain rule. A sum objective would multiply this batch's gradients
by twelve. That scale change must not be confused with twelve sequential
updates; nor does it imply every optimizer's eventual step scales identically.
Masking or weighting would require the objective's actual denominator instead.

### 27. Legal visibility can coexist with incorrect positions

One plausible cause is restarting RoPE offsets for the suffix after a cached
prefix. No future edge is opened, but new Q/K rotations no longer match the
positions used by full recomputation. Compare cached suffix logits with the
same full-pass positions under fixed weights and evaluation mode, then inspect
the new position indices and already-rotated cached keys. Restore the correct
offset and repeat the test. A recovered match supports this diagnosis in the
controlled fixture. It is not a universal conclusion from the initial signature:
cache concatenation, mask alignment, stale weights, or prefix mismatch can also
cause replay failure. Test more lengths and cache split points before broadening
the correctness claim.

### 28. Causality belongs to the whole computation graph

Subtracting a mean over all sequence positions makes the normalized state at
an early position depend on later input states. The information path is
`future token → future state → time-axis mean → earlier normalized state`.
Masked attention cannot erase a dependency already introduced on its input
path. Per-position feature normalization does not create this particular
cross-time edge. Perturb future tokens while holding the prefix fixed and
compare earlier outputs; the deliberate time-centering case fails this test
even though its logits remain finite. Finiteness, causality, and cache
equivalence are separate properties.

### 29. State the memory category before claiming a reduction

At fixed layer count, batch, source length, head width, and cache element size,
halving KV heads halves compact K/V cache payload and K/V projection parameter
counts. It does not halve Q/output projections, MLPs, embeddings, gradients,
optimizer state, or all activations. Query-head count still determines the
number of attention distributions and the dense score/value arithmetic term.
In the paired teaching configurations, cache payload goes from 6,144 to 3,072
bytes while total unique parameters go from 5,472 to 4,960. Measure actual peak
memory and latency on the intended workload before claiming a runtime benefit;
evaluate trained quality before claiming that the savings preserve capability.

### 30. Fix a question, not an attractive outcome

A valid fixed-parameter question is: “With the same stored block weights in
size, the same training-token budget, and a declared training recipe, does two-use
recurrence achieve lower validation cross-entropy than one use?” Use a declared
initialization/seed policy, identical tokenizer and data splits, a checkpoint
selection rule, and recorded resource measurements. The shared weights are
shared across uses *within* each model; separately trained comparison models
will generally learn different values.

This question allows additional computation. A compute-efficiency study instead
declares a common training-compute budget and how it is measured or estimated,
then makes any resulting token/update differences explicit. Inference recurrence
count and inference costs need a separate evaluation contract. Evidence against
the quality hypothesis could be no repeatable validation improvement under the
declared criterion; evidence against a practical deployment choice could be a
small gain accompanied by an unacceptable measured latency or memory cost.
Nonfinite loss, safety-limit violations, or a failed causal/cache contract should
trigger the predeclared failure rules, not be hidden by favorable samples. No
such trained comparison has been run by creating this material.
