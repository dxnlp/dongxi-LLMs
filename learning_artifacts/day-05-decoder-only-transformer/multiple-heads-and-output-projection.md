# Multiple Heads and Output Projection

- Introduced: 2026-09-06, as a bridge from the Day 4 review
- Book placement: Chapter 5, multi-head attention
- Learning status: introduced; no learner explanation-back or local experiment
  claimed. Earlier Day 4 gaps remain recorded separately.

## Mechanism

Each standard attention head forms its own learned Q/K/V views of the same
incoming states and its own causal source distribution per receiving position.
Heads do not divide up token positions; every head may read the allowed prefix.
They need not learn distinct, fixed linguistic roles.

For head h, O_h=A_h V_h. Concatenation retains their separate retrieved feature
vectors, then the learned output projection combines their features:

$$
O_{\mathrm{multi}}=\operatorname{Concat}(O_1,\ldots,O_H)W_O.
$$

Concatenation occurs on the feature axis, not the token axis. W_O acts
positionwise; the attention distributions supply token mixing. This is not the
vocabulary output head. A two-head extension of the prior teaching dimensions
uses each O_h:[4,2], concatenated output [4,4], and W_O:[4,8] to return [4,8].
These are illustrative dimensions; common equal-width designs divide the model
width among heads, so additional heads need not multiply the overall width.

## Intuition

At “it” in “The tired animal crossed the river because it ...”, different heads
could retrieve mixtures emphasizing an entity and an earlier description.
This is a motivation, not an attribution claim about a real trained model.
Separate heads make different routing patterns possible; usefulness must be
learned and tested. One head supplies one distribution over sources shared by
its value coordinates; multiple heads can supply several such distributions.

## Animation opportunity

### Live refinement — 2026-09-07

The learner asked whether random initialization explains head diversity and
whether minimizing the same loss should make their weights converge to similar
values. Clarified: separately parameterized, jointly learned—not independently
trained predictors. With row-vector outputs, u=sum_r o_r W_O,r and
dL/do_r=g W_O,r^T. Different paths from one scalar objective can receive
different gradients. Complementarity can help the combined prediction, but
standard training does not enforce unique roles or prevent redundant heads.
If corresponding head and output parameters are exactly symmetric and the
computation/optimizer preserve symmetry, the heads can remain duplicates.
Different initialization helps break symmetry but is not a diversity guarantee.

Canonical treatment: Chapter 5 section 5.3 and worked answer 3. The learner
acknowledged the explanation; independent explanation-back or a trained
head-specialization experiment has not been recorded. Extend CAND-ANIM-012's
backward sequence rather than opening a duplicate animation task.

`CAND-ANIM-012`: keep one receiving position fixed while the same X fans out
into heads with different projections. Show distinct causal weight rows and
value mixtures, concatenate the output features, then apply W_O. Preserve the
token axis throughout. Production waits for approval, the Chapter 5 treatment,
and verified shape/output examples; rendering remains on the Mac Studio.
