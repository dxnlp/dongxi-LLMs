# Chapter 3 — Learning the Next Token

A tokenizer can turn text into IDs, and an embedding table can turn those IDs
into vectors. Neither operation yet answers the question a language model is
trained to solve:

> Given the available prefix, what should come next?

The answer begins as one contextual hidden state, becomes a score for every
candidate token, turns into a probability distribution, and is compared with one
observed target. That comparison produces a scalar loss. Backpropagation then
turns the loss back into corrections for the output head, transformer, and
embeddings.

The complete path is:

```text
token IDs [B,T]
   → embeddings [B,T,D]
   → contextual hidden states [B,T,D]
   → output projection
   → logits [B,T,V]
   → probabilities [B,T,V]
   → shifted targets [B,T-1]
   → masked mean loss
   → gradients
   → parameter updates
```

This chapter follows that path in both directions. Along the way, it resolves
several apparent contradictions:

- one input position produces thousands of output scores but only one selected
  token;
- every training label can be one-hot while the model learns uncertainty;
- a successful optimum can have positive loss but zero expected gradient;
- a model can achieve a very low loss while learning the wrong task;
- a prompt can have no direct loss and still receive gradient influence;
- two models can have equal perplexity while behaving differently.

Chapter 2 established the discrete interface and the distinction between an
input embedding and a contextual hidden state. We now turn that hidden state
into a trainable prediction.

## 3.1 Learning outcomes

After completing this chapter, you should be able to:

- trace tensor shapes from token IDs through logits, labels, and reduced loss;
- explain why one position emits one logit for every model-vocabulary row;
- interpret logits as relative evidence rather than probabilities;
- implement stable softmax and stable target negative log-likelihood;
- explain how greedy decoding, sampling, temperature, top-$k$, and top-$p$
  convert a distribution into one token ID;
- derive one-hot cross-entropy and the exact logit gradient $p-q$;
- route that gradient through the output head into $W$, $b$, and $h$;
- explain how repeated one-hot targets learn a non-one-hot distribution;
- separate irreducible entropy from reducible KL mismatch;
- derive sequence NLL, mean token loss, and perplexity, including their
  evaluation limits;
- align causal logits with next-token labels and identify the unshifted-copying
  failure;
- distinguish attention visibility, target masking, and gradient flow;
- interpret the controlled two-logit experiment without generalizing beyond its
  evidence.

As in Chapters 1 and 2, equations establish mechanisms, experiments establish
bounded observations, and neither alone establishes broad language capability.

## 3.2 One contextual state becomes a vocabulary-wide score vector

Let:

- $B$ be batch size;
- $T$ be sequence length in tokens;
- $D$ be hidden width;
- $V$ be the model's output-vocabulary dimension.

The transformer produces:

\[
H\in\mathbb{R}^{B\times T\times D}.
\]

For one batch item and position, write:

\[
h=H[b,t,:]\in\mathbb{R}^{D}.
\]

The output head contains one row for every candidate token ID:

\[
W_{out}\in\mathbb{R}^{V\times D},
\qquad
b\in\mathbb{R}^{V}.
\]

It computes:

\[
z=W_{out}h+b,
\qquad
z\in\mathbb{R}^{V}.
\]

For candidate ID $i$:

\[
z_i=W_{out,i}\cdot h+b_i.
\]

$z_i$ is a **logit**: a raw compatibility score between this context and token
$i$. It is not constrained to be positive, does not sum to one, and has no
standalone percentage interpretation.

This explains the asymmetry that often surprises new readers:

```text
input at one position:  one token ID
contextual result:      one D-dimensional state
output at that position: V logits, one for every candidate ID
```

If $V=10{,}000$, one position emits 10,000 logits. A sequence emits one such row
at every position, giving `[B,T,V]`. During teacher-forced training, many rows can
be supervised in parallel. During cached generation, the decoder normally uses
only the newest position's `[V]` row to choose the next token.

### The coordinates are token IDs

The coordinate of the probability vector already is the candidate token ID.
Coordinate 42 is not later translated into some unrelated address: under the
paired tokenizer-model contract, index 42 names token ID 42.

The integers themselves have no ordinal meaning. ID 900 is not “more” than ID 9,
and neighboring IDs need not represent related strings. If we consistently
permute:

- tokenizer IDs;
- input embedding rows;
- output rows;
- dataset labels;

the model's function is unchanged. The network learns relationships among token
events through shared parameters and contexts, not arithmetic relationships
among the printed ID numbers.

### The output head is large for a reason

Dense projection across all positions can be written:

\[
Z_{[B,T,V]}=H_{[B,T,D]}W_{out,[V,D]}^\top.
\]

Its arithmetic scales approximately as $BTDV$, and the matrix contains $VD$
weights when untied. Chapter 2's pinned Qwen3-0.6B interface had $D=1024$ and
$V=151{,}936$. Its tied embedding/output matrix contains 155,582,464 values, and
one dense vocabulary projection performs roughly 155.6 million
multiply-accumulates per position. Tying avoids a second matrix of that size; it
does not remove the projection computation.

Vocabulary design therefore creates a systems trade-off. A smaller vocabulary
can leave text split into more tokens, increasing $T$, transformer positions,
autoregressive steps, and KV-cache use. A larger vocabulary increases
embedding/output cost through $V$ and may allocate many rows to rare pieces. A
rough full-sequence decomposition is:

\[
\text{cost}
\approx
c_1LTD^2+c_2LT^2D+c_3TDV,
\]

where $L$ is layer count and the constants hide implementation details. The
equation is a scaling guide, not a latency prediction. Vocabulary size alone
does not determine compression; which languages and domains received capacity
matters.

## 3.3 Softmax turns relative evidence into competition

Softmax converts logits into a categorical distribution:

\[
p_i=\frac{e^{z_i}}{\sum_{j=0}^{V-1}e^{z_j}}.
\]

Now:

\[
p_i>0,
\qquad
\sum_i p_i=1.
\]

Each probability is conditional on the current context and relative to every
other candidate. Increasing one candidate's logit can lower another candidate's
probability even if the second logit does not change, because all candidates
share the denominator.

For logits `[2,1,-1]`, exponentiation produces relative positive weights:

```text
exp(logits) ≈ [7.389, 2.718, 0.368]
```

Normalizing gives approximately:

```text
probabilities ≈ [0.7054, 0.2595, 0.0351]
```

The largest logit becomes the largest probability, but its value `2` never meant
“200%” or any other absolute probability.

### Only differences matter

Adding the same constant $c$ to every logit changes nothing:

\[
\frac{e^{z_i+c}}{\sum_j e^{z_j+c}}
=
\frac{e^c e^{z_i}}{e^c\sum_j e^{z_j}}
=p_i.
\]

Softmax therefore identifies relative gaps, not an absolute logit height. This
shared-shift symmetry will reappear when the logit gradients sum to zero.

### Stable softmax

Directly computing $e^{1002}$ can overflow even though logits
`[1002,1001,999]` describe the same distribution as `[2,1,-1]`. Let
$m=\max_j z_j$ and compute:

\[
p_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}.
\]

The largest shifted logit is zero, so every exponential is at most one. The
subtraction preserves the exact mathematical distribution while avoiding an
unnecessary numerical range.

```python
import math

def stable_softmax(logits):
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    denominator = sum(weights)
    return [weight / denominator for weight in weights]
```

Very unlikely probabilities can still underflow in finite precision. When we
need target log-likelihood, stable `log_softmax` or `logsumexp` avoids computing a
rounded zero and then taking `log(0)`.

## 3.4 From a distribution back to one token ID

Softmax produces a vector, not an emitted word. A decoding rule chooses one
coordinate:

```python
last_logits = logits[:, -1, :]           # [B,V]
probabilities = softmax(last_logits)     # [B,V]
next_id = choose(probabilities)          # [B,1]
```

The selected index is appended to the input, and the process repeats.

### Greedy decoding and ties

Greedy decoding selects:

\[
\operatorname*{argmax}_i z_i
=
\operatorname*{argmax}_i p_i.
\]

If several candidates share the exact maximum, the mathematical argmax is a
set. A concrete implementation needs a policy. Many tensor libraries return the
first maximal index; another decoder could randomize among tied maxima. Near-ties
are different from exact ties and can be sensitive to numerical precision.

### Sampling, temperature, top-$k$, and top-$p$

Sampling draws one categorical index instead of always choosing the maximum.
Temperature rescales logit gaps:

\[
p_i(\tau)=
\frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}},
\qquad \tau>0.
\]

The ratio form reveals the mechanism:

\[
\frac{p_i(\tau)}{p_j(\tau)}
=
\exp\left(\frac{z_i-z_j}{\tau}\right).
\]

- $0<\tau<1$ sharpens the distribution;
- $\tau=1$ preserves the trained distribution;
- $\tau>1$ flattens it;
- $\tau\to0^+$ concentrates mass on maximal logits;
- $\tau\to\infty$ approaches uniformity over a finite vocabulary.

Positive temperature never changes logit ordering, so it cannot change greedy
argmax. Interfaces that accept temperature zero usually implement greedy
decoding as a special case; literal division by zero is undefined.

Top-$k$ keeps a fixed number of highest-ranked candidates. Top-$p$ keeps the
smallest high-probability prefix whose cumulative mass reaches a threshold.
Survivors are renormalized before sampling. Temperature changes probability
ratios and can therefore change the size of a top-$p$ set; positive temperature
does not change the ranking used by top-$k$.

Lower temperature can reduce low-probability excursions, but it does not solve
generation drift. The highest-probability token can still be a bad choice, and
sharpening can amplify a systematic error or repetition loop. If the model is
already calibrated, $\tau<1$ moves its sampling frequencies away from the data
distribution by overrepresenting the mode.

These are decoding transformations. Ordinary maximum-likelihood training uses
the unfiltered model distribution in its cross-entropy objective.

## 3.5 Likelihood becomes additive surprise

For a causal model, the probability of a token sequence follows the chain rule:

\[
P(x_0,\ldots,x_{T-1})
=
\prod_{t=0}^{T-1}P(x_t\mid x_{<t}).
\]

The initial term may be conditioned on a beginning token or another declared
prompt convention. Multiplying many probabilities produces very small numbers
and makes it difficult to see which token caused trouble. Logs turn the product
into a sum:

\[
\log P(x_{0:T-1})
=
\sum_{t=0}^{T-1}\log P(x_t\mid x_{<t}).
\]

Negative log-likelihood, or NLL, is:

\[
\operatorname{NLL}(x_{0:T-1})
=
-\sum_t\log P(x_t\mid x_{<t}).
\]

At one target position $y$:

\[
L=-\log p_y.
\]

This quantity is predictive surprise. A likely observed token contributes little
loss. An unlikely observed token contributes much more. As $p_y\to0$, the loss
diverges, strongly rejecting a model that declares an observed outcome
impossible.

Stable target NLL can be computed directly from logits:

\[
L=-z_y+m+\log\sum_j e^{z_j-m},
\qquad m=\max_jz_j.
\]

This is algebraically identical to $-\log p_y$ but avoids materializing an
underflowed target probability.

## 3.6 Labels, cross-entropy, and the exact logit gradient

At one position, the model output is a length-$V$ logit vector. The ordinary
label is one integer ID:

```text
logits: [V]     one score per candidate
label:  scalar  one observed target ID
```

Framework cross-entropy usually accepts integer labels without constructing a
dense one-hot vector. Conceptually, let $q$ be the target distribution:

\[
H(q,p)=-\sum_iq_i\log p_i.
\]

For a one-hot target $y$:

\[
q_y=1,
\qquad
q_{i\ne y}=0,
\]

so every term except one disappears:

\[
H(q,p)=-\log p_y.
\]

Token-level one-hot cross-entropy and observed-token NLL are numerically the same
quantity. The names emphasize different views: NLL focuses on the observed
sample; cross-entropy compares target and model distributions.

### Deriving $\partial L/\partial z_i=p_i-q_i$

Let:

\[
S=\sum_j e^{z_j},
\qquad
p_k=\frac{e^{z_k}}{S}.
\]

Then:

\[
\log p_k=z_k-\log S.
\]

Substitute into cross-entropy:

\[
\begin{aligned}
L
&=-\sum_kq_k\log p_k\\
&=-\sum_kq_k(z_k-\log S)\\
&=-\sum_kq_kz_k+\log S,
\end{aligned}
\]

because $\sum_kq_k=1$. Differentiate with respect to one logit $z_i$:

\[
\begin{aligned}
\frac{\partial L}{\partial z_i}
&=-q_i+\frac{1}{S}\frac{\partial S}{\partial z_i}\\
&=-q_i+\frac{e^{z_i}}{S}\\
&=p_i-q_i.
\end{aligned}
\]

For a one-hot target:

\[
\frac{\partial L}{\partial z_y}=p_y-1<0,
\qquad
\frac{\partial L}{\partial z_i}=p_i>0\quad(i\ne y).
\]

Gradient descent moves opposite these signs. In a direct-logit sketch:

\[
z_y\leftarrow z_y+\eta(1-p_y),
\qquad
z_i\leftarrow z_i-\eta p_i.
\]

The target rises. Every non-target falls in proportion to the probability it
wrongly captured. A confidently wrong candidate receives more correction than
an already unlikely candidate because probability is a limited budget: removing
mass from the main wrong competitor helps the target most.

The gradient components sum to zero:

\[
\sum_i(p_i-q_i)=1-1=0.
\]

This is the differential form of the shared-logit-shift symmetry. The loss has no
reason to move all logits together because that direction changes no
probability.

### From logit correction to model learning

Logits are intermediate activations, not normally optimizer parameters. For
$z=W_{out}h+b$, write $g=p-q$. The chain rule gives:

\[
\frac{\partial L}{\partial b}=g,
\qquad
\frac{\partial L}{\partial W_{out}}=gh^\top,
\qquad
\frac{\partial L}{\partial h}=W_{out}^\top g.
\]

The output rows learn how to recognize contexts that support their candidate
tokens. The target row moves toward the current state under a simple SGD update;
high-probability wrong rows move away. Meanwhile, $W_{out}^\top g$ sends a
blended error signal into the transformer, asking it to construct a more useful
context representation next time. Backpropagation distributes that signal
through every operation that created $h$.

One example therefore does not store a correction in a disposable logit. It
changes reusable parameters and can affect many other contexts.

## 3.7 How one-hot observations teach a distribution

Suppose indistinguishable instances of one context continue with `dog` 70% of
the time and `cat` 30% of the time. Every training row still provides one
observed target. Let $r=[0.7,0.3]$ be the empirical distribution and $q$ be one
sample's one-hot target.

Because:

\[
\mathbb{E}[q]=r,
\]

the expected logit gradient is:

\[
\mathbb{E}[p-q]=p-r.
\]

The competing updates balance when:

\[
p=r.
\]

At that point a `dog` sample still has gradient:

\[
[0.7,0.3]-[1,0]=[-0.3,0.3],
\]

and a `cat` sample has:

\[
[0.7,0.3]-[0,1]=[0.7,-0.7].
\]

Their frequency-weighted average is zero:

\[
0.7[-0.3,0.3]+0.3[0.7,-0.7]=[0,0].
\]

Individual minibatches can therefore keep producing noisy gradients near the
optimum even though the population gradient vanishes.

### Entropy is not model error

Let $q$ now denote the population target distribution and $p$ the model. Add and
subtract the target's own log term:

\[
\begin{aligned}
H(q,p)
&=-\sum_iq_i\log p_i\\
&=-\sum_iq_i\log q_i
  +\sum_iq_i\log\frac{q_i}{p_i}\\
&=H(q)+D_{KL}(q\|p).
\end{aligned}
\]

$H(q)$ is uncertainty in the data. The model cannot reduce it. The nonnegative
KL term measures mismatch and reaches zero when $p=q$ on the relevant support.
Thus a perfectly calibrated model can have positive cross-entropy. Zero expected
gradient plus positive loss can represent successful learning rather than an
optimizer failure.

### The model sees samples, not $q$

The optimizer never receives the true population distribution directly. For a
context observed $n$ times, it sees counts $n_i$ and an empirical estimate:

\[
\hat q_i=\frac{n_i}{n}.
\]

Exact long contexts rarely repeat. A language model must share parameters and
representations across related contexts so that evidence from many examples
interacts. A sufficiently flexible model can instead memorize unique training
prefixes and drive empirical loss down without reducing population mismatch.
Falling training loss with rising held-out loss is therefore evidence of
overfitting under a fixed evaluation contract.

## 3.8 Controlled experiment: two logits learn 70/30

We tested the distribution-learning claim with the smallest transparent system:

- one abstract context;
- two float32 trainable logits initialized to zero;
- 70 class-0 and 30 class-1 one-hot targets;
- full-batch SGD with learning rate 0.5 for 500 steps;
- mean PyTorch cross-entropy;
- precommitted probability, loss, finiteness, and gradient-agreement criteria.

The complete specification was committed before execution. The measured
trajectory was:

| Step | $p_0$ | $p_1$ | Mean CE | Gradient norm |
|---:|---:|---:|---:|---:|
| 0 | 0.5000000 | 0.5000000 | 0.6931473 | $2.83\times10^{-1}$ |
| 10 | 0.6854001 | 0.3145998 | 0.6113629 | $2.06\times10^{-2}$ |
| 50 | 0.6999989 | 0.3000011 | 0.6108643 | $1.58\times10^{-6}$ |
| 500 | 0.7000000 | 0.3000000 | 0.6108643 | $7.45\times10^{-9}$ |

All seven precommitted criteria passed. PyTorch autograd agreed with $p-r$ at
every recorded checkpoint; the maximum absolute disagreement was
$5.96\times10^{-8}$. Final loss matched empirical entropy within the declared
tolerance.

The final logits were approximately:

\[
[0.42364874,-0.42364898].
\]

Their difference has an exact interpretation:

\[
\frac{p_0}{p_1}=e^{z_0-z_1}
\quad\Longrightarrow\quad
z_0-z_1=\log\frac{0.7}{0.3}\approx0.8472979.
\]

The observed gap was $0.8472977$. The individual logit heights remain arbitrary;
adding a shared constant preserves the distribution.

This experiment demonstrates the optimization mechanism in a controlled
empirical system. It does not establish that a transformer recovers the true
human-language distribution, generalizes to new contexts, understands either
label, or remains calibrated under deployment shift. See the full
[specification](../../experiments/specs/2026-09-03-next-token-distribution.yaml)
and [report](../../experiments/reports/2026-09-03-next-token-distribution.md).

## 3.9 From token surprise to perplexity

For valid aligned targets, a common reduced loss is:

\[
L_{mean}
=
\frac{\sum_{b,t}m_{b,t}\left[-\log p_{b,t}(y_{b,t})\right]}
{\sum_{b,t}m_{b,t}},
\]

where $m_{b,t}=1$ for included targets and zero otherwise. The denominator is
the number of valid targets, not padded tensor capacity.

Perplexity is:

\[
\operatorname{PPL}=e^{L_{mean}}.
\]

Equivalently:

\[
e^{-L_{mean}}
=
\left(\prod_{t=1}^{N}p_t(y_t)\right)^{1/N},
\]

so perplexity is the reciprocal geometric-mean probability assigned to observed
targets. Calling it an “effective branching factor” is exact for an equal-choice
teaching case and only an intuition for unequal real distributions.

### One scalar hides where surprise occurred

Model A can assign `[0.5,0.5,0.5,0.5]` to four observed targets, producing
sequence probability $0.0625$. Model B can assign approximately
`[0.99,0.99,0.99,0.0644]`, producing the same product and total NLL.

The metric correctly says that both assign equal probability to this exact
sequence. It does not say they have equal calibration, robustness, token-level
loss tails, gradient concentration, or free-generation trajectories. Model B
concentrates almost all surprise at one catastrophic point. Token-level traces,
quantiles, calibration tests, and generated behavior answer questions that mean
loss discards.

### Per-token perplexity depends on the tokenizer

Suppose tokenizer A represents a text fragment as one token with probability
$0.25$. Its total NLL and mean token NLL are both $1.386$, so its perplexity is
$4$.

Tokenizer B represents the same fragment as two successive tokens, each assigned
probability $0.5$. The text probability and total NLL remain:

\[
0.5\times0.5=0.25,
\qquad
-\log0.25=1.386.
\]

But mean NLL is now $1.386/2=0.693$, so perplexity is $2$. Nothing became twice
as predictable; the counting unit changed.

Per-token perplexity comparisons therefore require the same tokenizer, target
alignment, loss mask, preprocessing, context policy, and evaluation corpus. For
cross-tokenizer or multilingual comparisons, bits or nats per byte can provide a
more common unit when raw-text normalization is also fixed. A lower perplexity
on one corpus establishes higher geometric-mean probability under that contract,
not general superiority, truthfulness, or downstream usefulness.

## 3.10 Causal shifting: predict what is not yet visible

For tokens:

```text
<BOS>  I  love  dogs  <EOS>
```

the causal training relationships are:

| Logit position sees through | Target |
|---|---|
| `<BOS>` | `I` |
| `<BOS> I` | `love` |
| `<BOS> I love` | `dogs` |
| `<BOS> I love dogs` | `<EOS>` |

The state at position $t$ may use $x_{\le t}$ and predicts $x_{t+1}$:

\[
y_t=x_{t+1},
\qquad 0\le t<T-1.
\]

Explicit alignment is:

```python
shift_logits = logits[:, :-1, :]  # [B,T-1,V]
shift_labels = input_ids[:, 1:]    # [B,T-1]
```

Some causal-LM APIs accept `labels=input_ids` and perform this shift internally.
That convenience is an implementation contract to verify, not a universal rule
to assume.

### The unshifted-copying trap

Comparing `logits[:,t,:]` directly with `input_ids[:,t]` asks the model to recover
a token that is already present at that position. A copy-like model can obtain
near-zero loss while learning no next-token relationship. This is different from
a broken causal attention mask:

- **unshifted labels:** the target is the currently visible token;
- **broken causal mask:** the representation can inspect future tokens.

Either can create deceptively low loss. Compatible tensor shapes prove neither
semantic alignment nor causal validity.

### Teacher forcing and generation drift

Training can process all positions in parallel because the full sequence is
available while the causal mask blocks future visibility. Each position receives
the real preceding tokens. This is teacher forcing.

Generation is different. The model appends its own selected token and consumes
that altered prefix on the next step. One poor or merely unusual choice can move
the trajectory into a context that appeared rarely in training, making later
predictions less reliable. This train-generation context mismatch is often
called exposure bias:

\[
\text{training prefixes}\sim p_{data},
\qquad
\text{generated prefixes}\sim p_{model}.
\]

It can occur even with perfect shifting and causal masking. Lowering temperature
may reduce random low-probability deviations, but it cannot repair a wrong mode
and can make systematic errors more deterministic.

## 3.11 Loss masks grade targets; attention masks control visibility

Three controls answer different questions:

| Mechanism | Question |
|---|---|
| Causal attention mask | May this state read a future token? |
| Padding or document mask | May it read padding or an unrelated document? |
| Loss mask or ignored label | Should this aligned prediction contribute loss? |

EOS is normally a meaningful target: it teaches the model when to stop. PAD is
storage structure and normally should not contribute loss. In PyTorch-style
cross-entropy, ignored labels are often represented by `-100`.

### Mask target tokens, then respect the shift

Consider:

```text
index:   0      1       2       3          4             5      6
token:  BOS   User   capital  France   ASSISTANT       Paris   EOS
label: -100   -100    -100     -100       -100            5      6
```

With the standard internal shift, label index $k$ is predicted by logit index
$k-1$:

```text
logits at position 4 → Paris at label position 5
logits at position 5 → EOS at label position 6
```

The hidden state at `ASSISTANT` predicts `Paris`; the hidden state at `Paris`
already contains `Paris` and predicts `EOS`. Applying a visible-position mask
directly to logits without respecting this target shift can move the supervision
boundary by one token.

### No direct prompt loss does not mean no gradient

Answer-only supervision can let answer positions read the prompt while excluding
prompt targets from the loss. The answer loss still has a path through attention:

```text
answer loss
   → answer hidden state
   → prompt keys and values
   → earlier-layer prompt states
   → prompt embeddings and shared parameters
```

A loss mask removes selected scalar loss terms. It does not detach prompt
computation. The activation-level statement is layer-sensitive: an answer state
at layer $\ell$ reads prompt states from layer $\ell-1$, so earlier prompt states
can receive gradient while the unused final-layer prompt output need not. An
attention block, explicit detach, or frozen parameter creates different
boundaries.

The masked mean must divide by valid targets:

```python
loss = cross_entropy(
    shift_logits.reshape(-1, V),
    shift_labels.reshape(-1),
    ignore_index=-100,
    reduction="mean",
)
```

If two padded sequences contain four and two valid targets, the denominator is
six, not their padded capacity of eight. Otherwise padding changes the gradient
scale despite contributing no evidence.

## 3.12 First-class companion notebooks and evidence

The chapter has three interactive companion lessons. They are course material,
not disposable scratchpads:

| Notebook | Central question | Mechanisms |
|---|---|---|
| [Session 01](../../notebooks/day-03/01_logits_softmax_nll.ipynb) | Why do relative scores determine belief and surprise? | stable softmax, NLL, $p-q$, perplexity |
| [Session 02](../../notebooks/day-03/02_causal_shift_and_masks.ipynb) | How can low loss certify the wrong task? | shifting, copying failure, ignored labels, prompt gradients |
| [Session 03](../../notebooks/day-03/03_distribution_learning.ipynb) | How do one-hot examples become a distribution? | gradient balance, optimizer trajectory, logit-gap equilibrium |

Each notebook uses prediction, implementation, perturbation, interpretation, and
an evidence boundary. Every exercise is followed by a runnable reference solution
and mechanism explanation. Reusable computations remain in importable source:

- [`next_token_distribution_lab.py`](../../src/dongxi_llms/next_token_distribution_lab.py)
- [`test_next_token_distribution_lab.py`](../../tests/test_next_token_distribution_lab.py)

The notebooks teach mechanisms; the specification and report establish the
experiment's empirical identity. Successful execution alone does not broaden the
claims beyond the stated controls.

## 3.13 Common reasoning failures

### “The hidden state is already a probability vector”

$h$ has hidden width $D$ and learned internal coordinates. The output head must
map it to $V$ vocabulary logits before softmax.

### “One input token produces one output score”

One position produces one score for every output candidate. Decoding later
chooses one coordinate.

### “Token IDs are numbers the model learns arithmetic relationships among”

They are categorical addresses. A consistent permutation of IDs and matching
rows preserves behavior.

### “Softmax interprets each logit independently”

All candidates share the denominator. Probabilities express normalized
competition and depend on relative gaps.

### “Temperature changes the greedy answer”

Positive temperature preserves ranking. It changes sampling probabilities, not
argmax.

### “Every wrong token receives equal punishment”

For a non-target token, the logit gradient is $p_i$. A confidently wrong
candidate receives a larger downward correction.

### “One-hot labels force the model to become deterministic”

Across samples, expected gradients recover the empirical conditional
distribution. One-hot is a per-example representation, not necessarily the
population distribution.

### “Positive loss at zero gradient means optimization failed”

At $p=q$, KL mismatch is zero while irreducible entropy can remain positive.

### “Lower perplexity proves a better model”

It establishes higher geometric-mean target probability only under a fixed
tokenizer and evaluation contract. It does not establish general capability.

### “Low training loss proves next-token learning”

Unshifted labels can reward copying, and memorization can reduce empirical loss
without improving held-out behavior.

### “A zero prompt loss mask prevents prompt parameters from changing”

Visible prompt computations can remain ancestors of supervised answer losses.

## 3.14 Exercises

1. **The complete tensor bridge.** For $B=2$, $T=5$, $D=8$, and $V=10{,}000$,
   state the shapes of input IDs, embeddings, final hidden states, logits,
   shifted logits, and shifted integer labels. Explain why the label does not
   need shape `[V]`.

2. **Permutation invariance.** A tokenizer swaps IDs 3 and 900. State every
   model and dataset object that must be permuted to preserve behavior. Explain
   why this proves that numeric ID distance is not linguistic similarity.

3. **Stable competition.** Explain why logits `[1002,1001,999]` and `[2,1,-1]`
   produce the same softmax distribution. Then state why subtracting the maximum
   improves finite-precision behavior without changing the model.

4. **Confidently wrong.** For target class 1 and probabilities
   `[0.7054,0.2595,0.0351]`, interpret every component of $p-q$. Distinguish
   gradient sign from optimizer motion and explain why class 0 receives more
   correction than class 2.

5. **From logits into parameters.** Starting from $z=Wh+b$ and $g=p-q$, give
   the gradients for $W$, $b$, and $h$. Explain what the output head and the
   transformer learn from their respective gradients.

6. **Nonzero loss, zero expected gradient.** For a 70/30 target distribution,
   show how the two one-hot gradients cancel at $p=[0.7,0.3]`. Explain why the
   remaining cross-entropy is successful learning rather than failure.

7. **Same NLL, different behavior.** Compare `[0.5,0.5,0.5,0.5]` with
   `[0.99,0.99,0.99,0.0644]`. State what equal products establish and which
   behavioral claims require additional metrics.

8. **Tokenizer-dependent perplexity.** One tokenizer assigns probability 0.25
   to one token; another assigns 0.5 to each of two tokens representing the same
   text. Compare total NLL and per-token perplexity. Design a more defensible
   multilingual comparison.

9. **Audit causal alignment.** For `<BOS> I love dogs <EOS>`, list every
   supervised `(prefix,target)` pair. Explain how an unshifted copy model can
   achieve low loss while being useless for next-token generation.

10. **Three boundaries.** A user prompt is readable by an assistant response but
    has labels `-100`. Explain attention visibility, direct loss contribution,
    and possible gradient flow separately. Include the layer-level activation
    nuance.

11. **Teacher forcing and temperature.** Explain why correct teacher-forced loss
    can coexist with free-generation drift. Evaluate the claim that lowering
    temperature “solves” the mismatch.

12. **Evidence discipline.** The controlled two-logit experiment passed all
    criteria and learned 70/30. Write one supported claim, one plausible but
    unsupported interpretation, and the smallest next experiment needed to test
    generalization across related contexts.

Worked answers are available in
[the Chapter 3 solutions](../solutions/03-learning-the-next-token.md). The three
companion notebooks provide executable versions of the central mechanisms.

## 3.15 Summary and what follows

A contextual hidden state $h$ is mapped through an output head into one logit per
model-vocabulary row. Softmax turns relative logit gaps into a normalized
distribution, and decoding selects or samples one coordinate as the next token
ID. Stable implementations subtract the maximum or use log-sum-exp without
changing the mathematics.

One-hot cross-entropy equals observed-token NLL, and its logit gradient is
$p-q$. That compact expression explains target reward, probability-weighted
correction of wrong candidates, shared-shift invariance, and gradient flow into
the output head and transformer. Across repeated outcomes, the expected gradient
is $p-r$, so one-hot examples can learn a non-one-hot distribution. Cross-entropy
then separates into irreducible entropy plus reducible KL mismatch.

Sequence likelihood becomes additive surprise under the logarithm. Mean loss and
perplexity are useful only under an explicit tokenizer, alignment, mask, and
evaluation contract. Causal shifting ensures each state predicts a token it has
not yet seen; attention and loss masks define different boundaries; backward
credit can flow through visible prompt computation even when prompt targets are
ignored.

We have treated the transformer as the function that creates $h$. Chapter 4
opens that box. It derives queries, keys, values, scaled dot products, causal
masks, attention distributions, and the gradients that decide which earlier
positions influence the next-token state.
