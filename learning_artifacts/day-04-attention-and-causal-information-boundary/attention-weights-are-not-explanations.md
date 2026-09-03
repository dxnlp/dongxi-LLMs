# Why Attention Weights Are Not Complete Explanations

- Day: 04
- Date opened: 2026-09-03
- Status: demonstrated
- Book destination: Chapter 4, interpretation and evidence boundary
- Related evidence: planned redundant-value and intervention notebook examples
- Related production tasks: `CAND-ANIM-008`

## Questions that drove the discussion

- If different attention distributions can produce the same weighted value sum,
  can attention weights alone explain a model prediction?

## Learner's initial model

The learner correctly answered that attention weights alone cannot provide such
an explanation.

## Refined mental model

One attention head produces

$$
O=AV.
$$

Observing $A$ without $V$ is insufficient to determine even that head's output.
Furthermore, the map from attention distributions to outputs need not be unique.
If value vectors are equal or linearly dependent, distinct rows of $A$ can yield
the same weighted sum.

For example, if

$$
v_3=\tfrac12v_1+\tfrac12v_2,
$$

then the two attention distributions

$$
[0.5,0.5,0]
\quad\text{and}\quad
[0,0,1]
$$

produce the same head output. The visible routing pattern changes while the
transported result does not.

Even when the mixture differs, a complete Transformer applies output projection,
residual addition, other heads, later attention layers, and feed-forward layers.
Information can travel through paths not summarized by one attention map. A high
weight is therefore evidence of strong routing in that operation, not proof of
causal importance to the final logit or a faithful human-readable reason.

Stronger interpretability claims require interventions or other evidence: alter
or ablate a source, value, head, or edge; measure changes downstream; inspect
gradients or attribution methods; and test stability across examples. Those
methods also have assumptions and do not turn attention into a uniquely correct
explanation.

## Concrete examples and derivations

The non-identifiability example above demonstrates that $A$ alone cannot recover
$O$. More generally, if a nonzero row vector $\delta$ satisfies

$$
\delta V=0,
$$

then attention rows $a$ and $a+\delta$ produce the same output whenever both are
valid probability distributions:

$$
(a+\delta)V=aV.
$$

The final model prediction adds another unknown downstream mapping from $O$ and
the residual stream to logits, making an attention-only explanation still less
sufficient.

## Demonstrated understanding

The learner rejected the claim that attention weights alone reliably explain a
prediction after recognizing that multiple weight patterns can create the same
value mixture.

## Evidence and limitations

The counterexample is analytical and establishes insufficiency, not that
attention visualization is useless. Attention maps can reveal hypotheses about
routing, diagnose mask failures, and guide targeted interventions. They should
be described as internal measurements whose explanatory value must be tested,
not as automatic causal explanations.

## Open edges

- Implement the redundant-value counterexample.
- Compare raw attention weight with value-weighted contribution magnitude.
- Perturb one attention edge and measure the resulting logit change.
- Revisit interpretability after multi-head and residual pathways are introduced.

## Reuse opportunities

- Chapter 4 precision box: “routing evidence is not a complete explanation.”
- Worked exercise using a null-space perturbation of $A$.
- Notebook intervention connecting attention visualization to evidence levels.
