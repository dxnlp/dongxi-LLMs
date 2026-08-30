# Padding, Attention Masks, and Special Tokens

- Day: 02
- Date opened: 2026-08-30
- Status: demonstrated; executable verification open
- Book destination: Chapter 2 batching bridge and later Chapter 8 loss masking
- Related evidence: planned Day 2 embedding and batching lab
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

The conceptual distinction is demonstrated, but exact masking behavior can vary
by model and trainer API. The executable lab must inspect actual shifted labels
and reduction masks rather than assume a library's attention mask also modifies
its loss.

## Open edges

- Build batches with left and right padding and inspect masks.
- Verify zero loss contribution at ignored label positions.
- Connect causal shifting and label alignment in Day 3.
- Revisit assistant-only loss masks during instruction-data engineering.

## Reuse opportunities

- Chapter 2 bridge from individual sequences to batches.
- Day 3 cross-entropy animation and manual calculation.
- Later Chapter 8 discussion of prompt, response, and padding loss masks.
