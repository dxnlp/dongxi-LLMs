# Padding, Attention Masks, and Special Tokens

- Day: 02
- Date opened: 2026-08-30
- Status: demonstrated and executable verification present
- Book destination: Chapter 2 batching bridge and later Chapter 8 loss masking
- Related evidence: `experiments/reports/2026-08-31-embedding-gradient-paths.md`
- Related production tasks: `ANIM-CE-001` depends on correct loss masking

## Questions that drove the discussion

- How can unequal-length sequences form one rectangular batch?
- What do attention-mask values 1 and 0 mean?
- Why is an attention mask insufficient to remove padding from the loss?
- Should `<eos>` be treated like padding?

## Learner's initial model

The learner correctly interpreted attention-mask values immediately: 1 means a
real position that can participate, while 0 means padding that should not.

## Refined mental model

Padding creates rectangular `[B, T]` tensors but is not ordinary language data.
Two independent protections are normally required:

- the attention mask prevents padded positions from being used as ordinary
  context;
- the label mask, commonly target value `-100` in PyTorch cross-entropy, excludes
  padded positions from the loss.

If padding ID 0 remains a normal target, cross-entropy trains the model to assign
probability to PAD at artificial batch positions. Attention masking does not
automatically guarantee loss masking.

`<eos>` has a different role: it is a meaningful target that teaches the model
when a sequence ends, so it is normally included in the loss.

More generally, masks define distinct boundaries in the learning system rather
than merely cleaning up tensor shapes:

- an attention or visibility mask controls which positions may influence a
  representation;
- a loss or supervision mask controls which prediction errors contribute to the
  optimization objective;
- a semantic boundary token such as `<eos>` remains visible and supervised
  because learning when to stop is part of the language task.

A position can therefore be visible to the model while being excluded from the
loss. This distinction later enables supervised fine-tuning to expose prompt
tokens as context while applying direct supervision only to assistant response
tokens. Visibility and supervision answer different questions and should never
be treated as one interchangeable mask.

A loss mask determines where scalar loss terms originate; it does not by itself
detach earlier visible positions from the computation graph. If a supervised
response state attends to prompt states, the response loss can backpropagate
through that dependency into attention parameters and prompt-token embedding
rows even though no local loss was assigned at the prompt positions. Stopping
that gradient would require a separate operation such as detaching or freezing
parameters. Thus `not a target` is different from `cannot influence learning`.

## Concrete examples and derivations

```text
input_ids =
[
  [12, 24, 31],
  [47, 18,  0],
]

attention_mask =
[
  [1, 1, 1],
  [1, 1, 0],
]

labels =
[
  [12, 24, 31],
  [47, 18, -100],
]
```

If the final label were `0`, its token-level contribution would be:

```text
L_pad = -log P(PAD | preceding context)
```

That gradient optimizes batch formatting rather than the intended language
target.

## Demonstrated understanding

- Correctly explained `1` as a real attended position and `0` as padding to
  exclude.
- Understood after explanation that an unmasked padding label explicitly trains
  prediction of the padding ID.
- Distinguished PAD from EOS: PAD is normally ignored; EOS teaches stopping.

## Evidence and limitations

The conceptual distinction is demonstrated. In the fixed executable lab, the
loss mask `[0,0,1]` supervised only the response position, while that response
assigned positive attention weights to both prompt positions. Prompt embedding
rows 1 and 2 consequently received gradient norms `0.303322` and `0.668485`.

Exact masking behavior can still vary by model and trainer API. Later integration
labs must inspect actual shifted labels and reduction masks rather than assume a
library's attention mask also modifies its loss.

## Open edges

- Build batches with left and right padding and inspect masks.
- Connect causal shifting and label alignment in Day 3.
- Revisit assistant-only loss masks during instruction-data engineering.

## Reuse opportunities

- Chapter 2 bridge from individual sequences to batches.
- Day 3 cross-entropy animation and manual calculation.
- Later Chapter 8 discussion of prompt, response, and padding loss masks.
