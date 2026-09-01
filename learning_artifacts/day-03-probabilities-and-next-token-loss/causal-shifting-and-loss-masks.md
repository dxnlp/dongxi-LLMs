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

## Evidence state

- `introduced`: position $t$ predicts token $t+1$.
- `introduced`: explicit shifted shapes `[B,T-1,V]` and `[B,T-1]`.
- `not yet demonstrated`: learner explanation-back, exact BOS/EOS trace,
  padding-mask alignment, and manual/PyTorch agreement.

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
