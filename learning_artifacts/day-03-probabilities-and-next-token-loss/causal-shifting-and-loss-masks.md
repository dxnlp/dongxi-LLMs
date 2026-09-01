# Causal Shifting and Loss Masks

- Day: 03
- Date opened: 2026-09-01
- Status: introduced
- Book destination: Chapter 3 sections on causal label alignment and masked loss
- Related evidence: planned manual tensor trace and PyTorch verification
- Related production tasks: candidate `CAND-ANIM-001`; `ANIM-CE-001`

## Questions that drive the discussion

- Which hidden-state position predicts which target token?
- Why can a causal language model consume `input_ids` and labels with the same
  apparent shape while still training on the next token?
- What happens if logits and labels are compared without a one-position shift?
- Where do BOS, EOS, padding, and ignored labels enter the alignment?
- Which positions contribute to the final mean loss?

## Mechanism introduced

For a token sequence:

\[
x_0,x_1,\ldots,x_{T-1},
\]

the contextual state at position $t$ can use only $x_{\le t}$. Its target is
the next token:

\[
y_t=x_{t+1}, \qquad 0\le t<T-1.
\]

In explicit tensor form:

```python
shift_logits = logits[:, :-1, :]   # [B, T-1, V]
shift_labels = input_ids[:, 1:]    # [B, T-1]
```

The shift is an alignment of predictions with targets; it is not permission for
the hidden state to inspect the future. The causal attention boundary must still
ensure that `logits[:, t, :]` depends only on tokens through position $t$.

Comparing `logits[:, t, :]` directly with `input_ids[:, t]` trains the model to
recover a token it has already received at the same position. That objective can
look numerically easy while failing to train next-token prediction.

EOS is normally a meaningful shifted target after the final content token. BOS,
when used, can provide a position from which to predict the first ordinary token.
Padding or deliberately unsupervised positions must be excluded after alignment,
and the loss denominator must count only valid targets.

Some causal language-model APIs accept `labels=input_ids` and perform the shift
inside the model loss. The exact implementation must be inspected rather than
assuming every framework has the same contract.

## Logits, labels, and their dimensions

Let:

- $B$ be the number of sequences in a batch;
- $T$ be the number of token positions in each sequence;
- $D$ be the hidden or embedding dimension;
- $V$ be the tokenizer vocabulary size.

The main tensor path is:

| Object | Shape | Meaning at one position |
|---|---|---|
| `input_ids` | `[B,T]` | one integer vocabulary ID |
| embeddings | `[B,T,D]` | one $D$-dimensional token vector |
| contextual hidden states | `[B,T,D]` | one $D$-dimensional context-dependent vector |
| logits | `[B,T,V]` | one raw score for every vocabulary candidate |
| labels | `[B,T]` before alignment | one integer ID naming the observed target |

At a single batch item and position, the model emits a logit vector
$z\in\mathbb{R}^V$, while the stored label $y$ is a scalar integer in
$\{0,\ldots,V-1\}$. Softmax turns the $V$ logits into $V$ probabilities, and
cross-entropy uses $y$ to select the observed token's probability. Conceptually,
$y$ corresponds to a one-hot target distribution $q\in\mathbb{R}^V$, but
implementations normally store only the integer ID rather than materializing
the large one-hot vector.

For next-token alignment, the tensors used by the loss become:

```python
shift_logits = logits[:, :-1, :]   # [B, T-1, V]
shift_labels = labels[:, 1:]       # [B, T-1]
```

Thus the two tensors agree on batch and supervised-position axes. The logits
retain one extra vocabulary axis because each prediction is a competition among
all $V$ candidates; each label needs only one integer to identify the observed
winner. Labels are required for training-time evaluation of the prediction but
are absent during ordinary generation, when the model emits logits and a decoding
rule selects the next token.

## Loss masks: which aligned predictions count

Correct target alignment is necessary but does not imply that every tensor
position should contribute to optimization. Let $m_{b,t}\in\{0,1\}$ indicate
whether one aligned next-token target is valid. The mean loss is

\[
L=
\frac{
\sum_{b,t}m_{b,t}\left[-\log p_{b,t}(y_{b,t})\right]
}{
\sum_{b,t}m_{b,t}
}.
\]

The denominator is the number of valid target tokens, not necessarily $B(T-1)$.
This matters when sequences have different lengths or when prompt, padding, or
otherwise unsupervised regions are present.

For example, after shifting two padded sequences may have:

```text
targets A:    The    cat    slept  EOS
loss mask A:   1      1       1     1

targets B:    Hello  EOS    PAD    PAD
loss mask B:   1      1       0      0
```

There are six supervised targets, so the mean divides by six rather than eight.
EOS is normally meaningful because predicting the end of a sequence is part of
language modeling. Padding is storage structure rather than language evidence and
normally contributes no loss. Frameworks often encode an ignored label with a
sentinel such as `-100` rather than store a separate loss-mask tensor.

Three masks answer different questions:

| Mechanism | Question it answers |
|---|---|
| causal attention mask | May this hidden state read a future token? |
| padding/document attention mask | May this hidden state read this padded or unrelated position? |
| loss mask / ignored label | Should this prediction be graded and update parameters? |

A position may be readable as context while still being excluded from the loss.
For example, supervised fine-tuning can let answer tokens attend to a user prompt
while assigning loss only to the answer. Conversely, excluding a position from
loss does not by itself prevent other positions from attending to it.

This explicit masked-mean mathematics is already within approved task
`ANIM-CE-001`; it strengthens that animation's aggregation act and does not create
a duplicate proposal. Production remains on the Mac Studio.

## Evidence state

- `introduced`: position $t$ predicts token $t+1$.
- `introduced`: explicit shifted shapes `[B,T-1,V]` and `[B,T-1]`.
- `developing`: the learner initially connected an unshifted loss with seeing
  future tokens. The discussion separated two independent failure modes:
  unshifted target alignment asks position $t$ to recover the already-visible
  token $x_t$, while a broken causal attention mask lets position $t$ inspect
  future tokens $x_{>t}$.
- `introduced`: loss masks select which correctly aligned targets contribute,
  and the mean-loss denominator counts valid targets rather than padded tensor
  capacity.
- `not yet demonstrated`: learner explanation-back of the corrected distinction,
  exact BOS/EOS trace, separation of attention and loss masks, and
  manual/PyTorch agreement.

## Animation opportunity check

This important tensor alignment automatically triggers animation review. Existing
`CAND-ANIM-001` already preserves token identity while logits, next-token targets,
and per-position losses align, so no duplicate candidate is created. It remains
`discuss` and all production remains on the Mac Studio after approval and
verification.

## Open edges

- Trace one concrete sequence from input IDs to every supervised target.
- Show the failure produced by unshifted labels.
- Verify framework shifting and ignore-index behavior in PyTorch.
- Distinguish sequence boundaries from padding and packed-document boundaries.
