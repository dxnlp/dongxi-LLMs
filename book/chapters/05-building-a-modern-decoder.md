# Chapter 5 — Building a Modern Decoder

## Foundation: from contextual states to a trainable model

Attention explains how one position can retrieve information from its legal
context. It does not, by itself, explain an entire language model. We still
need a representation to carry through the network, transformations that make
use of retrieved information, and a way to connect the result to next-token
learning.

This chapter builds that connection. Its foundation, developed through Day 5,
uses learned absolute positions, ordinary multi-head attention, LayerNorm, and
a GELU feed-forward network. It is a complete small baseline, not yet the full
modern architecture. Days 6–7 will extend this same chapter with RMSNorm,
SwiGLU, RoPE, grouped-query attention, Q/K normalization, cost accounting, and
architecture synthesis. The optional recurrent-depth extension follows those
foundations; it is not required to understand this part.

Prerequisites are the token/embedding distinction from Chapter 2, next-token
cross-entropy from Chapter 3, and causal attention from Chapter 4. By the end
of this foundation, you should be able to follow every tensor boundary, explain
why the block has two different processing branches, and distinguish a working
training mechanism from evidence of language capability.

The seven [Day 5 notebooks](../../notebooks/day-05/README.md) are first-class
companion lessons. Read an explanation, predict the effect of its intervention,
then inspect the runnable solution. The [worked-solution guide](../solutions/05-decoder-notebook-solutions.md)
also answers the conceptual exercises at the end of this chapter.

## 5.1 One continuous computation

The baseline has a simple outer structure:

1. Look up token embeddings and add learned position embeddings.
2. Pass those states through a stack of causal decoder blocks.
3. Normalize the final states.
4. Project each state to one logit per vocabulary entry.

![Whole-model map from token IDs through embeddings, decoder blocks, final normalization, and the vocabulary head.](../../notebooks/figures/chapter-05/day-05-06_assemble_decoder-architecture-map.png)

This is an architecture schematic, not a measurement of activation values or
running time. Its highlighted stack is expanded throughout the following
sections. The figures reused here have runnable drawing code in the companion
notebooks; see the [visual reproduction guide](../../docs/NOTEBOOK_VISUALS.md).

Keep the axes distinct:

| Symbol | Meaning | Baseline value |
|---|---|---:|
| $B$ | Number of sequences in a batch | 2 |
| $T$ | Input positions per sequence | 6 |
| $V$ | Vocabulary size | 16 |
| $D$ | Residual-stream feature width | 16 |
| $H$ | Attention heads | 4 |
| $d$ | Features per head | 4 |
| $F$ | MLP intermediate feature width | 32 |
| $L$ | Number of decoder blocks | 2 |

The small vocabulary is a teaching choice: IDs range from 0 to 15. These
integers have no supplied word mapping and are not intended to tokenize real
language. The vocabulary size is neither the sentence length nor the feature
width, even though two of the numbers happen to equal 16.

The main tensor path is $[B,T]\to[B,T,D]\to[B,T,V]$. Inside a block, feature
width can temporarily change, but each branch must return $[B,T,D]$ so its
update can be added to the stream.

## 5.2 Give each position a starting representation

Let $E\in\mathbb R^{V\times D}$ be the token embedding table and
$P\in\mathbb R^{M\times D}$ the learned position table, where $M$ is the
maximum supported length. For token ID $i_{b,t}$ at position $t$:

$$
X^{(0)}_{b,t}=E[i_{b,t}]+P[t].
$$

The ID selects a row; its numerical magnitude is not a language feature.
Repeated occurrences of the same ID retrieve the same row of $E$. Adding a
different position row can give those occurrences different initial states.
Both vectors have $D$ features, and their addition still has $D$ features.
There is no concatenation here.

![Token and position lookups meet at an addition, producing the first residual state.](../../notebooks/figures/chapter-05/day-05-01_embeddings_and_positions-architecture-detail.png)

The position table is shared across sequences: position 3 uses the same $P[3]$
in each sequence. Token identity and position are distinct inputs to the initial
representation. Subsequent attention makes that representation contextual.

Removing position embeddings does not turn a causal decoder into a model with
no order-related structure at all: different positions still have different
allowed prefixes. It does remove this explicit learned position signal. The
notebook separates that intervention from removing the causal mask.

### How lookup rows learn

An embedding table is a trainable parameter matrix. Backward routes each
occurrence's gradient into the selected row, summing contributions when the
same ID occurs multiple times.

For an intentionally simple objective that sums all looked-up coordinates, each
occurrence contributes a derivative of 1 per selected coordinate. Why 1? Raising
one selected coordinate by $\delta$ raises the sum by $\delta$: the ratio of
change in output to change in that coordinate is $\delta/\delta=1$. Two uses
of the same coordinate contribute 2. In actual next-token training, these
contributions are gradients from the entire downstream network, not constants.

This selected-row statement concerns the lookup path. If the vocabulary output
head shares $E$, another gradient path can reach rows absent from the input.
We return to that connection in Section 5.7.

**Companion:** [Notebook 1 — Embeddings and positions](../../notebooks/day-05/01_embeddings_and_positions.ipynb).
Inspect repeated-ID gradients and change the position signal while leaving
causal visibility intact.

## 5.3 Several contextual views, learned together

For one receiving position, one attention head creates one distribution over
allowed source positions. Its value coordinates all use that distribution to
form a weighted mixture. A wider value vector can carry more features, but it
does not create another independently parameterized source-weight pattern.

Multiple heads make several patterns available. In the prefix “The tired animal
crossed the river because it”, a useful representation at “it” could need
information about the entity and its earlier description. Separate heads could
retrieve those aspects differently. This is a motivation, not a claim that
particular heads reliably own linguistic jobs.

For head $r$, suppressing the batch dimension:

$$
Q_r=XW_{Q,r},\quad K_r=XW_{K,r},\quad V_r=XW_{V,r},
$$

$$
A_r=\operatorname{softmax}_{\text{sources}}
\left(\frac{Q_rK_r^\top}{\sqrt d}+C\right),\qquad O_r=A_rV_r.
$$

Here $C_{t,s}=0$ when $s\le t$ and $-\infty$ otherwise. The symbol $V_r$
means value activations, not the vocabulary size $V$. Softmax acts on the
scores; values bypass softmax and supply the content being mixed.

Each $W_{Q,r},W_{K,r},W_{V,r}$ has mathematical shape $[D,d]$. Implementations
can combine all heads into one larger projection and then reshape its output.
That is a computational organization, not a requirement for separate Python
modules per head.

| Attention boundary | Shape |
|---|---|
| Incoming states | $[B,T,D]$ |
| Projected and split Q, K, V | $[B,H,T,d]$ |
| Source weights | $[B,H,T,T]$ |
| Value mixtures | $[B,H,T,d]$ |
| Concatenated heads | $[B,T,Hd]$ |
| Output after $W_O\in\mathbb R^{Hd\times D}$ | $[B,T,D]$ |

Our baseline chooses $Hd=D$. Heads do not divide the sentence into different
token groups: every head can attend to the legal prefix at every position.
Concatenation joins feature slices, not time positions.

### Why does the same loss not make every head identical?

Heads are separately parameterized but **jointly learned**. They do not each
receive an independent instruction to predict the next token. The loss judges
the final prediction made from their combined contribution.

At one position, partition $W_O$ into row blocks $W_{O,r}$. Writing head outputs
as row vectors $o_r$, their combined update is:

$$
u=\sum_r o_rW_{O,r}.
$$

If $g=\partial\mathcal L/\partial u$ is the arriving row-vector gradient:

$$
\frac{\partial\mathcal L}{\partial o_r}=gW_{O,r}^\top.
$$

The same arriving gradient passes through different output-weight blocks and
then through different attention computations. Thus the same objective can
produce different Q/K/V parameter gradients. Complementary features can improve
the combined prediction; duplicating a feature is not always equally useful.

Random initialization helps break symmetry, but it does not guarantee diverse
roles. If corresponding head parameters and output connections are exactly
identical and the update rule preserves their symmetry, they can remain
duplicates. Conversely, initially different heads can become redundant. The
ordinary objective contains no guarantee of unique specialization.

More heads are therefore not automatically better. At fixed $D=Hd$, increasing
$H$ narrows each head. It changes the balance between the number of mixtures and
the feature capacity of each mixture.

**Companion:** [Notebook 2 — Multi-head attention](../../notebooks/day-05/02_multi_head_attention.ipynb).
Compare looped and vectorized heads under identical weights. Ablate a head and
inspect the resulting update; a changed output alone does not prove a specific
semantic role or a quality improvement.

## 5.4 Carry a state forward and revise it

Without a residual connection, a transformation passes on $Y=F(X)$. Everything
needed downstream must survive through $F$. A residual sublayer instead uses:

$$
Y=X+F(X).
$$

For example, $X=[2,3]$ and $F(X)=[1,-1]$ produce $Y=[3,2]$. The skip path
carries $X$ unchanged into the addition, but the next layer receives the sum.
It does not receive an untouched backup of $X$.

The **residual stream** is this evolving state across many sublayers. At a later
layer, $X$ already contains earlier updates; it is not the original embedding
being reinserted. A useful analogy is revising the current draft rather than
rewriting a document from a blank page. Unlike a tracked document edit, however,
an added vector does not preserve a separately recoverable revision history.

### Why does the bypass help?

First, identity behavior is easy. When $F(X)=0$, the output is exactly $X$.
The learned branch can make a small correction without reconstructing the
incoming representation in its entirety.

Second, backward has a direct route. For a flattened column-vector state
$y=x+f(x)$, write $g=\partial\mathcal L/\partial y$ and let $J_f$ be the
Jacobian of the branch. Then:

$$
\frac{\partial\mathcal L}{\partial x}=g+J_f^\top g.
$$

The first contribution bypasses the learned transformation; the second passes
through it. Earlier layers need not receive their entire learning signal solely
through a long chain of branch transformations. This structural path can make
deep optimization easier, but it does not guarantee well-behaved gradients.

The counterexample matters: if $f(x)=-x$, both the output and its input
derivative are zero. The two paths cancel. Residual connections make identity
available; they do not enforce information preservation or nonzero gradients.

**Companion:** [Notebook 3 — The residual stream](../../notebooks/day-05/03_residual_stream.ipynb).
Inspect $X+XW$, isolate the two gradient contributions, and compare the zero-
branch control with the cancellation case.

## 5.5 Normalize the branch's reading of the stream

As states are transformed and updated, their numerical scale can change.
LayerNorm prepares a consistently scaled view of a token's features. Think of
adjusting a baseline and contrast: we want the next operation to read relative
feature differences without being as sensitive to a common offset or scale.

Take one token's four features, $[10,20,30,40]$. Their mean is 25. Subtracting
it gives $[-15,-5,5,15]$; dividing by the standard deviation gives approximately
$[-1.34,-0.45,0.45,1.34]$. The values now describe how far each feature lies
above or below the token's mean, relative to its feature spread.

For a token vector $x\in\mathbb R^D$:

$$
\mu=\frac1D\sum_i x_i,\qquad
\sigma^2=\frac1D\sum_i(x_i-\mu)^2,
$$

$$
\hat x_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad
\operatorname{LN}(x)_i=\gamma_i\hat x_i+\beta_i.
$$

Use population variance, dividing by $D$, not the sample-variance correction
$D-1$. Epsilon prevents division by zero and bounds the denominator away from
zero. Before the learned affine transform, the variance is
$\sigma^2/(\sigma^2+\epsilon)$: approximately one when the variance is large
relative to epsilon, not identically one. A constant vector becomes zero before
the affine transform and $\beta$ afterward.

The vectors $[10,20,30,40]$ and $[100,200,300,400]$ have the same relative
pattern, so their normalized versions are approximately equal. Centering
removes the common offset exactly; epsilon makes positive-scale invariance
approximate. The normalization does discard some input information. Learned
$\gamma,\beta\in\mathbb R^D$ adjust each feature's scale and offset, but do
not recover the original per-token mean and magnitude. Nor must the final
output after those adjustments have zero mean or unit variance.

These learned parameters are shared across positions. The **statistics** are
computed separately at each position. For $[B,T,D]$, normalize over $D$—not
over tokens or across the batch. Using full-sequence time-axis statistics can
let a future token affect an earlier state, even if attention is correctly
masked. Causality must hold throughout the model, not only inside softmax.

### Placement changes the identity path

A pre-norm sublayer is:

$$
Y=X+F(\operatorname{LN}(X)).
$$

Normalize what the branch reads; add its update to the unnormalized stream.
The bypass has not been replaced by normalized values. With a zero branch,
$Y=X$.

A post-norm sublayer is:

$$
Y=\operatorname{LN}(X+F(X)).
$$

Here normalization acts on the combined result. A zero branch gives
$Y=\operatorname{LN}(X)$, generally not $X$. Backward also passes through that
normalization: the sublayer no longer has the same untouched identity route.
This establishes a structural difference, not universal superiority for every
possible training recipe. Our baseline chooses pre-norm and still applies a
final normalization after the stack.

**Companion:** [Notebook 4 — LayerNorm and placement](../../notebooks/day-05/04_layernorm_and_placement.ipynb).
Implement the statistics explicitly, inspect a wrong-axis intervention, and
compare zero-branch behavior under pre-norm and post-norm.

## 5.6 Compute features from the context already gathered

Attention mixes information between positions. A position-wise feed-forward
network, or MLP, transforms features within each position. These jobs are
complementary: retrieve relevant information, then compute useful features from
the state now available.

For “The cat sleeps”, the same MLP processes the states at “The”, “cat”, and
“sleeps” separately. It does not directly consult another position. Yet its
input at “sleeps” can already contain information gathered from earlier tokens
by attention. Position-wise does not mean context-free.

The baseline MLP expands, applies a nonlinear function, and projects back:

$$
\operatorname{MLP}(X)=\operatorname{GELU}(XW_{\rm up}+b_{\rm up})W_{\rm down}
+b_{\rm down}.
$$

Using row-vector mathematics, $W_{\rm up}:[D,F]$ and $W_{\rm down}:[F,D]$.
The state shape follows $[B,T,16]\to[B,T,32]\to[B,T,16]$. Expansion creates
learned combinations of features, not copies of the input and not new token
positions. GELU acts elementwise; the final projection recombines the resulting
features into a same-width update.

![The MLP expands the feature dimension, applies GELU, and projects back without mixing positions.](../../notebooks/figures/chapter-05/day-05-05_positionwise_mlp-architecture-detail.png)

Why introduce nonlinearity? Without it:

$$
(XW_{\rm up}+b_{\rm up})W_{\rm down}+b_{\rm down}
=X(W_{\rm up}W_{\rm down})
+(b_{\rm up}W_{\rm down}+b_{\rm down}).
$$

This is just one affine transformation. Adding an intermediate width alone
does not escape that family. A nonlinear activation enables input-dependent
responses that cannot generally collapse into one affine map. Informally, a
feature combination can matter differently depending on what else is present.
That does not mean each intermediate neuron has a readable semantic rule.

PyTorch stores `Linear.weight` as `[output_features, input_features]` and
evaluates `X @ weight.T + bias`. Consequently, the equivalent stored weight
in the notebook is `down.weight @ up.weight`. This is the same derivation in a
different storage convention, not reversed mathematical logic.

**Companion:** [Notebook 5 — Position-wise MLP](../../notebooks/day-05/05_positionwise_mlp.ipynb).
Remove the activation and construct the exact affine equivalent. Perturb one
position at the MLP input and check that other positions' MLP outputs stay
unchanged. A whole decoder need not have that invariance because it has attention.

## 5.7 Assemble the block and the vocabulary interface

We can now write one complete pre-norm block:

$$
U=X+\operatorname{MHA}(\operatorname{LN}_1(X)),\qquad
Y=U+\operatorname{MLP}(\operatorname{LN}_2(U)).
$$

![A pre-norm decoder block with separate attention and MLP updates and their residual bypasses.](../../notebooks/figures/chapter-05/day-05-06_assemble_decoder-architecture-detail.png)

The two normalization modules have separate learned parameters. The MLP reads
the state after attention's update. Both additions preserve $[B,T,D]$.
Stacking blocks repeats these roles with separate parameters in each baseline
block. Parameter sharing across depth is a later, explicitly different design.

After the stack, let $h_{b,t}$ be one final normalized state. A vocabulary head
produces:

$$
z_{b,t}=h_{b,t}W_{\rm vocab},\qquad W_{\rm vocab}\in\mathbb R^{D\times V}.
$$

This is one logit for each vocabulary entry, predicting the next token at that
position. Logits are not probabilities; softmax over $V$ turns them into a
distribution. During generation we usually select from the final position's
distribution, append the selected ID, and repeat with fixed parameters.

Do not confuse two output projections. Attention's $W_O$ merges head features
into a $D$-wide stream update. The vocabulary head maps the final $D$-wide state
to $V$ candidate scores. Neither the token ID's value nor the feature index
in a hidden state is itself a probability.

Our default baseline ties input and output weights, so
$W_{\rm vocab}=E^\top$. The same stored parameter serves as a lookup table and
a classifier matrix. Copying equal numbers into a separate parameter is not
tying: separate copies can subsequently receive different gradients and updates.
With true tying, backward adds contributions from both uses into $E$.

### Initialization is part of the baseline

`TinyDecoder` initializes embedding and linear weights from a normal
distribution with mean zero and standard deviation 0.02, zeros linear biases,
and uses LayerNorm's initial scale 1 and offset 0. It ties the vocabulary head
after initialization. This specifies the teaching model, not an optimal rule
for every depth and width. Randomness helps break symmetry; setting all
trainable weights to zero is not a substitute for a controlled initialization.

Before training, verify finite activations and gradients, vocabulary output
shape, true parameter identity under tying, and causal invariance. Changing a
future token must not change an earlier output in a deterministic full forward
pass. Cached prefix replay should agree with the corresponding full pass under
fixed weights and correct position offsets. These are implementation checks,
not performance benchmarks.

The reusable [decoder implementation](../../src/dongxi_llms/decoder_lab.py)
contains these operations without fused attention, dropout, or padding. Its
clarity and tiny CPU fixtures are deliberate; it is not a serving system.

**Companion:** [Notebook 6 — Assemble the decoder](../../notebooks/day-05/06_assemble_decoder.ipynb).
Follow the complete tensor path, compare tying with copying, and test a copy
oracle against correctly shifted labels. Almost zero loss on an incorrect
current-token objective would not validate next-token learning.

## 5.8 Many token losses, one shared parameter update

Does the model learn token by token? Supervision is position-wise, but an
optimizer step normally combines errors from many positions and sequences.

The declared teaching batch begins with two seven-ID sequences:

```text
sequence 1:  1  2  3  4  5  6  7
sequence 2:  8  9 10 11 12 13 14
```

Inputs take the first six IDs, and labels take the last six:

```text
inputs 1:   1  2  3  4  5  6      labels 1:   2  3  4  5  6  7
inputs 2:   8  9 10 11 12 13      labels 2:   9 10 11 12 13 14
```

Every position produces 16 logits. Thus logits have shape $[2,6,16]$, labels
have shape $[2,6]$, and there are 12 supervised positions. For this unpadded
batch, the mean loss is:

$$
\mathcal L=\frac1{BT}\sum_{b,t}
-\log\operatorname{softmax}(z_{b,t})_{y_{b,t}}.
$$

Each prediction can read only its legal prefix, despite processing positions
in parallel within a layer. The observed next token is available to the loss,
not to the earlier position's forward representation. Labels are shifted once;
the loss helper accepts that alignment and does not shift again.

Differentiation distributes over the sum. Each shared parameter receives the
aggregate of its per-position contributions, scaled by the averaging factor.
This is not a sequence of parameter updates after individual tokens: all the
predictions in the forward pass use the same parameter snapshot. Microbatch
gradient accumulation can combine still more contributions before an optimizer
step; it is not needed in this tiny experiment.

### Backward measures sensitivity; the optimizer changes parameters

`backward()` calculates and accumulates gradients. `step()` applies an update.
For plain SGD the sketch is $\theta_{\rm new}=\theta-\eta\nabla_\theta\mathcal L$.
The learning rate $\eta$ scales the step, not the quality of the training data.

An illustrative scalar objective makes the distinction tangible. Let
$\ell(e)=(e-1)^2$ at $e=0.2$. Its derivative is $-1.6$. SGD with $\eta=0.1$
moves to $0.36$, toward the minimum. But $\eta=1.5$ moves to $2.6$, increasing
the loss from $0.64$ to $2.56$. A locally helpful direction does not justify an
arbitrarily large step. These numbers describe a quadratic, not a recommended
LLM learning rate or the geometry of the actual token loss.

The notebook uses AdamW, which tracks gradient history and squared-gradient
history for adaptive updates; it is not literally the SGD formula. Its weight
decay is explicitly zero in this experiment. Chapter 6 will develop optimizer
state, regularization, schedules, and clipping in depth.

The essential training loop reuses the course implementation:

```python
import torch
from dongxi_llms.decoder_lab import TinyDecoder, teaching_batch, next_token_loss

torch.set_num_threads(1)
torch.manual_seed(505)
model = TinyDecoder()              # CPU float32 under the standard default dtype
ids, labels = teaching_batch()     # [2,6], already shifted once
optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)

for _ in range(160):
    optimizer.zero_grad(set_to_none=True)
    logits = model(ids)            # [2,6,16]
    loss = next_token_loss(logits, labels)
    loss.backward()
    optimizer.step()
```

Run it after the notebook's import setup. This excerpt exposes the mechanism;
the canonical `fit_one_batch` also checks finite losses and gradients and
records initial and final measurements. Passing logits directly to the
cross-entropy helper avoids a separate, less stable probability-then-log step.

### What the fixed experiment establishes

The [recorded reference experiment](../../experiments/reports/2026-09-06-decoder-notebooks.md)
uses seed 505, the declared batch, and 160 AdamW steps at learning rate 0.02.
It reports mean cross-entropy decreasing from 2.774792432785034 to
0.0007856183219701052 and training-token accuracy reaching 1.0. The later
[visual verification](../../experiments/reports/2026-09-07-decoder-architecture-visuals.md)
reproduced those values. These are stored measurements, not a new run or a
claim about the learner's notebook output.

![Recorded one-batch loss trajectory and initial versus final output probabilities.](../../notebooks/figures/chapter-05/day-05-07_one_batch_learning-visual-learning.png)

This saved plot belongs to the fixed reference run; rerunning a notebook cell
with different settings does not rewrite it. The training-loss history contains
pre-update measurements, with the separately measured post-final-update loss
appended by the plotting function.

Fitting this deliberately consistent batch shows that the assembled computation
and optimizer can learn its labels. The frozen control provides a useful
contrast: predictions do not improve merely from repeatedly evaluating unchanged
weights. But fitting is not generalization. These sequences have distinct input
IDs and simple successor associations, so memorization does not require rich
contextual reasoning or the useful participation of every attention head.
Even a much simpler model could learn such associations.

The notebook also probes reversed contexts, but defines no gold labels for that
probe. Its outputs are observations, not a scored generalization result. A real
capability claim needs held-out tasks, appropriate controls, and a declared
evaluation contract.

**Companion:** [Notebook 7 — One-batch learning](../../notebooks/day-05/07_one_batch_learning.ipynb).
Inspect the aligned labels, frozen control, parameter changes, and evidence
boundary. Learning a batch is the beginning of a training-system check, not the
end of model evaluation.

## 5.9 Exercises: defend the mechanism

Use explanations and small controlled changes rather than memorizing layer
names. [Worked answers](../solutions/05-decoder-notebook-solutions.md#day-5-foundation--worked-conceptual-solutions)
are provided for every question.

1. The vocabulary has 16 entries and the input has six positions. Why does the
   output have 16 scores per position? Which number changes if you add a token
   position without changing the vocabulary?
2. The same token ID appears twice. Which vectors must match at lookup, and
   which states can differ after positions and attention?
3. All heads minimize the same loss. Why need their gradients and attention
   patterns not match? What fully symmetric setup could keep two heads equal?
4. Does adding more heads at fixed width give every head more information
   capacity? Distinguish source mixtures from features per head.
5. A skip path carries $X$ unchanged. Why does that not make $X$ recoverable from
   the output in general? Give an identity case and a cancellation case.
6. How can normalization over time leak future information even when attention
   has a correct mask? Contrast this with per-token feature normalization.
7. Zero the branch output in pre-norm and post-norm sublayers. Explain their
   different outputs and backward paths.
8. Why is a position-wise MLP not necessarily context-free? Why can two affine
   projections without an activation collapse into one?
9. Distinguish attention's output projection, the vocabulary head, and tying the
   vocabulary head to the embedding table. Why is a copied tensor insufficient?
10. Explain one batch update without saying that parameters change after every
    token. What changes when the same model freely generates text instead?
11. Does the negative gradient guarantee lower loss for every learning rate?
    Use the scalar example to explain local sensitivity versus a finite update.
12. A tiny decoder achieves perfect accuracy on the declared batch. What has
    been established, and what additional evidence would contextual
    generalization or head-specialization claims require?

## 5.10 From a working baseline to modern design choices

The baseline is now a connected argument: embeddings create states; attention
retrieves context through several learned views; MLPs transform available
features; normalization prepares branch inputs; residual paths carry the
evolving state; and the vocabulary head connects that state to next-token loss.
Backpropagation trains the designated parameters jointly through this chain.

The next part will change specific mechanisms while preserving these roles.
RMSNorm changes normalization, SwiGLU changes the MLP, RoPE changes position
handling inside attention, and GQA changes key/value sharing. None should be
introduced as an unexplained replacement acronym. Each needs a controlled
comparison and explicit parameter, compute, and memory accounting.

Those extensions already have [companion notebooks](../labs/05-building-a-modern-decoder.md#change-one-mechanism-at-a-time),
but their full narrative and the architecture defense remain the Days 6–7 work.
This foundation does not claim the final `DongxiGPT` design or the planned
larger configurations are complete. It gives us the baseline from which their
choices can be explained and tested.
