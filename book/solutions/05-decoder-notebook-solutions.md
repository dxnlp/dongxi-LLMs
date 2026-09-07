# Chapter 5 — Worked Solutions and Notebook Guide

The [companion lab](../labs/05-building-a-modern-decoder.md) provides the
sequential route. Every exercise has its runnable solution and explanation
immediately below it in the notebook. This guide summarizes what a sound
interpretation must preserve; it is not a replacement for running the cells.

The [Day 5 foundation of Chapter 5](../chapters/05-building-a-modern-decoder.md)
now supplies the continuous narrative. Its twelve conceptual exercises are
answered below; the table also retains the Days 6–7 notebook pathway.

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

Use the modern tiny model after session 10. Explain, without relying on its
class names alone, what each operation does to the token, head, and feature
axes; why it is causal; which parameters it trains; and which invariants survive
its controlled intervention. An answer is strong when it identifies the relevant
test and its evidence boundary, not merely when it repeats a layer name.

Notebook readiness and the recorded reference experiment are complete material
deliverables. The Day 5 narrative foundation is written; Days 6–7 narrative
extensions, the architecture defense, and learner completion remain separate.

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

Pre-norm gives $Y=X+F(\operatorname{LN}(X))$. If the branch is identically zero,
$Y=X$ and its input Jacobian is the identity. Post-norm gives
$Y=\operatorname{LN}(X+F(X))$, becoming $\operatorname{LN}(X)$ under the same
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
