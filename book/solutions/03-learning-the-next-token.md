# Solutions — Chapter 3: Learning the Next Token

These are reference explanations, not scripts to memorize. A complete answer
connects the mathematics to the computation graph and keeps every empirical
claim within its evaluation contract.

## Exercise 1 — The complete tensor bridge

The shapes are:

| Object | Shape |
|---|---|
| Input token IDs | `[2,5]` |
| Input embeddings | `[2,5,8]` |
| Final hidden states | `[2,5,8]` |
| Logits | `[2,5,10000]` |
| Shifted logits | `[2,4,10000]` |
| Shifted integer labels | `[2,4]` |

Lookup preserves batch and sequence axes and adds the embedding width. The
transformer preserves those outer axes while contextualizing each length-8
state. The output head maps each state from hidden coordinates to 10,000
candidate-token scores.

The label at each supervised position is one integer ID naming the observed
candidate. A cross-entropy implementation can use that ID to select the target
log-probability without materializing a length-10,000 one-hot vector. The model
must score all candidates; the dataset observed only one target at that position.

The final input position has no later token within this sequence, so the usual
shift drops its logit row and drops the first token from the label side:

```python
shift_logits = logits[:, :-1, :]
shift_labels = input_ids[:, 1:]
```

## Exercise 2 — Permutation invariance

To swap IDs 3 and 900 without changing behavior, apply the same permutation to:

- the tokenizer's piece-to-ID and ID-to-piece mappings;
- rows 3 and 900 of the input embedding table;
- rows 3 and 900 of the output matrix, or the corresponding shared rows if
  weights are tied;
- output biases for those candidate IDs;
- every dataset input ID and target label;
- any special-token IDs, masks, or decoding constraints that refer to the
  affected IDs;
- optimizer state associated with the permuted parameter rows when continuing an
  existing training run.

After the coordinated change, each text symbol selects the same numeric vector
and each output coordinate is decoded back to the same symbol as before. All
logits, probabilities, losses, and decoded text can therefore remain unchanged.

If numeric ID distance encoded linguistic similarity, such a permutation would
destroy the model's function. Because the coordinated permutation preserves it,
the printed numbers are categorical addresses. Linguistic relationships arise
from the learned vectors and transformations associated with those addresses.

## Exercise 3 — Stable competition

The first vector is the second plus a shared constant 1000:

\[
[1002,1001,999]=[2,1,-1]+1000.
\]

Softmax cancels that constant:

\[
\frac{e^{z_i+1000}}{\sum_je^{z_j+1000}}
=
\frac{e^{1000}e^{z_i}}{e^{1000}\sum_je^{z_j}}
=
\frac{e^{z_i}}{\sum_je^{z_j}}.
\]

The two vectors therefore produce the same mathematical distribution.

Computing $e^{1002}$ directly can overflow in finite precision. Stable softmax
subtracts $m=\max_i z_i$ first:

\[
[1002,1001,999]-1002=[0,-1,-3].
\]

Every exponential is now at most one. This is not a changed model or an
approximate probability rule; it is an algebraically identical computation with
a safer numeric range. Stable target NLL should likewise use `log_softmax` or
log-sum-exp rather than `log(softmax(...))` after a probability has rounded to
zero.

## Exercise 4 — Confidently wrong

The target is class 1, so:

\[
q=[0,1,0].
\]

Therefore:

\[
p-q=[0.7054,-0.7405,0.0351].
\]

Interpretation by coordinate:

- Class 0 has positive gradient `0.7054`. Gradient descent subtracts it, strongly
  lowering that wrong candidate's logit.
- Class 1 has negative gradient `-0.7405`. Gradient descent subtracts a negative
  number, strongly raising the target logit.
- Class 2 has positive gradient `0.0351`. Its logit is lowered only slightly
  because the model had assigned it little probability.

The gradient sign is not the update direction; the optimizer moves opposite the
gradient under the SGD sketch $z\leftarrow z-\eta\nabla z$.

Class 0 receives more correction because it captured most of the limited
probability budget. Lowering an already unlikely class 2 would release little
mass. Cross-entropy concentrates correction on the beliefs most responsible for
the observed mistake.

## Exercise 5 — From logits into parameters

Let:

\[
z=Wh+b,
\qquad
g=\frac{\partial L}{\partial z}=p-q.
\]

The chain rule gives:

\[
\frac{\partial L}{\partial b}=g,
\qquad
\frac{\partial L}{\partial W}=gh^\top,
\qquad
\frac{\partial L}{\partial h}=W^\top g.
\]

$gh^\top$ is an outer product with shape `[V,D]`. Under a simple gradient step,
the target row is strengthened in the direction of the current hidden state,
while non-target rows are weakened in proportion to their probabilities. The
output head is learning how to read contextual geometry as token evidence.

$W^\top g$ has shape `[D]` and tells the transformer how the state should change
to make the target more competitive. Backpropagation sends this signal through
the layers that created $h$. The transformer is learning to construct better
contextual geometry.

The local logit update is explanatory, but logits are normally disposable
activations. Reusable learning lives in $W$, $b$, transformer parameters, and
possibly a tied embedding table.

## Exercise 6 — Nonzero loss, zero expected gradient

At $p=[0.7,0.3]$, the `dog` one-hot target produces:

\[
g_{dog}=[0.7,0.3]-[1,0]=[-0.3,0.3].
\]

The `cat` target produces:

\[
g_{cat}=[0.7,0.3]-[0,1]=[0.7,-0.7].
\]

Weighting by target frequencies:

\[
\begin{aligned}
\mathbb{E}[g]
&=0.7g_{dog}+0.3g_{cat}\\
&=0.7[-0.3,0.3]+0.3[0.7,-0.7]\\
&=[0,0].
\end{aligned}
\]

Each sample still prefers its observed outcome, but across the distribution the
corrections balance. With small stochastic minibatches, realized gradients can
fluctuate around this zero expectation.

The cross-entropy decomposition is:

\[
H(q,p)=H(q)+D_{KL}(q\|p).
\]

At $p=q$, KL mismatch is zero while $H(q)$ remains positive because both outcomes
can occur. The optimizer has reached the best calibrated prediction available
for that empirical distribution. Positive loss measures irreducible uncertainty,
not necessarily residual model error.

## Exercise 7 — Same NLL, different behavior

Model A's product is:

\[
0.5^4=0.0625.
\]

Model B's product is approximately:

\[
0.99^3\times0.0644\approx0.0625.
\]

The products and therefore total sequence NLLs are equal. This establishes one
narrow claim: both models assign approximately the same probability to this
exact observed sequence.

The token-level structures differ. A spreads moderate uncertainty across every
position. B is nearly certain three times and catastrophically surprised once.
That difference can affect:

- calibration;
- maximum and tail token loss;
- where gradients concentrate;
- sensitivity to a decision at the catastrophic position;
- free-generation trajectories after that decision.

These claims require token-level loss distributions, calibration evaluation,
generated trajectories, or task-specific robustness tests. Total NLL correctly
forgets this information when it reduces many factors to one product.

## Exercise 8 — Tokenizer-dependent perplexity

For tokenizer A:

\[
\operatorname{NLL}_{total}=-\log0.25\approx1.386.
\]

There is one token, so mean NLL is 1.386 and:

\[
\operatorname{PPL}_A=e^{1.386}=4.
\]

For tokenizer B, the complete text probability remains:

\[
0.5\times0.5=0.25,
\]

so total NLL is still 1.386. There are two tokens, however, so mean NLL is
$1.386/2=0.693$ and:

\[
\operatorname{PPL}_B=e^{0.693}=2.
\]

The lower number does not indicate a more probable text. The denominator counts
different units.

A multilingual comparison should freeze the corpora, domains, preprocessing,
special tokens, target masks, and context policy; report tokenizer compression
such as tokens per UTF-8 byte; and normalize total NLL to a common unit such as:

\[
\text{bits per byte}
=
\frac{\operatorname{NLL}_{total}}
{N_{bytes}\ln2}.
\]

It should also compare how much raw text fits inside the fixed token context.
Even then, lower predictive loss does not by itself establish comprehension,
truthfulness, or downstream quality.

## Exercise 9 — Audit causal alignment

The supervised pairs are:

```text
<BOS>                 → I
<BOS> I               → love
<BOS> I love          → dogs
<BOS> I love dogs     → <EOS>
```

The state at position $t$ sees tokens through $x_t$ and predicts $x_{t+1}$. The
final EOS position has no later target in this sequence.

An unshifted objective compares the position-$t$ logits with $x_t$. Because the
current token is already part of the position-$t$ input, a model can learn an
identity-like map from the current embedding to the same token's output row. It
can achieve very low loss on that objective without learning the continuation
relationships listed above.

During generation, the last visible token must predict something unseen. A
current-token copier therefore solves the measured task but not the intended
one. This is label misalignment, distinct from future leakage caused by a broken
causal attention mask.

## Exercise 10 — Three boundaries

1. **Attention visibility:** the response may read the prompt wherever causal and
   padding/document masks permit it. A zero loss label does not make the prompt
   invisible.
2. **Direct loss contribution:** a prompt label of `-100` contributes no local
   cross-entropy term and is excluded from the valid-target denominator.
3. **Gradient flow:** a supervised response state can depend on prompt keys and
   values. Its loss therefore backpropagates into earlier prompt representations,
   input embeddings, and shared transformer parameters.

The activation-level qualifier matters. At layer $\ell$, a response position
reads prompt states from layer $\ell-1$. Those earlier-layer states can lie on the
loss path. The final-layer output at a masked prompt position need not receive a
gradient merely because a later final-layer response output exists; that exact
activation may have no consumer. Explicit detaching, blocked attention, or frozen
parameters creates still different behavior.

Loss masking means “do not grade this target,” not “remove this token from
context” or “prevent every related parameter update.”

## Exercise 11 — Teacher forcing and temperature

Teacher forcing evaluates each prediction under a real prefix from the dataset.
During generation, the model consumes its own previous selections. One imperfect
or unusual selection changes the next context, and later predictions may operate
on prefixes that were rare in training. These trajectory errors can compound
even if shifting and causal masking are correct.

Lower temperature sharpens the current distribution and may reduce the chance of
sampling low-probability candidates. It cannot guarantee a good trajectory:

- the highest-probability candidate can still be wrong;
- positive temperature does not change greedy ranking;
- sharpening can amplify systematic model bias;
- deterministic choices can enter repetition loops;
- if $p$ is calibrated, $\tau<1$ distorts its frequencies away from the data by
  overrepresenting the mode.

Temperature changes the decoder's risk profile. It does not eliminate the
difference between gold training prefixes and model-generated prefixes.

## Exercise 12 — Evidence discipline

A supported claim is:

> In the precommitted two-logit, full-batch PyTorch experiment, repeated 70/30
> one-hot targets produced probabilities approximately `[0.7,0.3]`; autograd
> matched $p-r$, and final mean cross-entropy matched empirical entropy within
> the declared tolerances.

A plausible but unsupported interpretation is:

> Therefore a transformer trained on natural text will recover the true human
> continuation distribution for unseen contexts.

The controlled experiment has no context representation, no unseen examples,
and no distribution shift. It establishes an optimization mechanism, not
generalization or language understanding.

The smallest next experiment could define several related but nonidentical input
contexts with a deliberately shared feature structure, hold out some combinations,
and precommit whether a tiny contextual classifier should recover known target
frequencies on those held-out combinations. The report would need separate
training and held-out losses, calibration error or frequency comparisons, fixed
seeds and splits, and failure criteria. A successful result would support
generalization only within that synthetic family, not natural language broadly.
