# Future Module — Recurrent Depth and Looped Transformers

- Captured during: Day 4, while connecting attention to modern decoder design
- Intended book placement: Chapter 5, after the standard decoder is understood
- Intended schedule: conceptual treatment on Day 6; controlled comparison and
  architecture defense on Day 7
- Status: scheduled frontier topic, not part of the Day 4 completion contract
- Evidence snapshot: 2026-09-04

## Why preserve this topic

A standard decoder normally applies a sequence of separately parameterized
blocks once. A looped or recurrent-depth model can instead apply a shared block
or shared stack repeatedly. This creates an important design axis that ordinary
parameter-count comparisons hide:

> stored parameters, effective depth, and computation per token are different
> quantities.

The topic belongs after the ordinary decoder because recurrence is easier to
understand as a modification of an already precise block-level computation. It
should not interrupt the first-principles attention lesson.

## Core mechanism

One useful abstraction separates a prelude, a recurrent core, and a coda:

$$
e=P(x), \qquad s_0 \sim \mathcal{N}(0,\sigma^2I),
$$

$$
s_j=R(e,s_{j-1};\theta_R), \qquad j=1,\ldots,r,
$$

$$
p(\text{next token}\mid x)=C(s_r).
$$

The same parameters $\theta_R$ are reused at every recurrent step. Increasing
$r$ therefore increases the number of layer applications and usually the
latency and FLOPs without multiplying the stored recurrent-core parameters by
$r$. It does **not** provide free depth.

For a simple stack with $L_P$ prelude blocks, $L_R$ recurrent blocks, and $L_C$
coda blocks, the effective number of block applications is

$$
L_{\mathrm{effective}}=L_P+rL_R+L_C.
$$

This is an accounting identity, not a claim that a shared block applied twice
is equivalent in capacity or optimization behavior to two independently
parameterized blocks.

## Taxonomy the course must keep distinct

1. **Fixed looped Transformer.** Every token passes through the same shared
   block or stack a predetermined number of times.
2. **Variable recurrent depth.** Training exposes the model to varying recurrence
   counts, and inference may spend more recurrent steps for more test-time
   computation.
3. **Adaptive token-level recurrence.** A learned router assigns different
   recurrence depths to different token positions, so computation need not be
   uniform across a sequence.
4. **Visible chain-of-thought.** The model emits intermediate reasoning as
   tokens. This consumes sequence positions and makes that text inspectable.
5. **Latent recurrent computation.** Additional transformations occur in hidden
   states before an output token is emitted. These states are not ordinary text.

The last two are different computation channels. Reusing layers can move more
computation into latent activations, but layer reuse alone neither proves that a
model will emit fewer reasoning tokens nor that its visible reasoning has been
deliberately concealed.

## The central trade-off

Looping exchanges parameters for sequential computation:

- **Possible benefit:** more transformation steps at nearly fixed stored
  recurrent-core parameter count; an adjustable test-time-compute axis; and,
  with routing, computation that can vary by token.
- **Cost:** more layer applications, latency, activation work during training,
  and potentially more difficult optimization. Weight sharing also prevents
  different depths from learning completely independent transformations.
- **Systems consequence:** two passes through a shared stack may approach twice
  the body compute, but the exact ratio is not two because embeddings, output
  heads, routing, kernels, cache policy, and other non-recurrent work remain.
- **Evidence boundary:** parameter efficiency is not token efficiency, compute
  efficiency, wall-clock efficiency, or quality. Each needs its own controlled
  comparison.

## Published evidence and claim status

- Geiping et al., [*Scaling up Test-Time Compute with Latent Reasoning: A
  Recurrent Depth Approach*](https://arxiv.org/abs/2502.05171), study a prelude,
  recurrent core, and coda; train across recurrence depths; and report a 3.5B
  proof-of-concept whose performance can improve with additional recurrent
  compute. This is primary research evidence for that specific model and setup,
  not a universal scaling law.
- Bae et al., [*Mixture-of-Recursions: Learning Dynamic Recursive Depths for
  Adaptive Token-Level Computation*](https://arxiv.org/abs/2507.10524), add
  learned token-level routing over shared recursive computation. This is more
  specific than merely applying the whole stack twice.
- The [Nanbeige4.2 technical report](https://arxiv.org/abs/2607.22083) reports a
  3B non-embedding-parameter model trained on 28T tokens that reuses its
  Transformer stack for another pass. Its reported two-pass result and roughly
  75% token-efficiency comparison are findings from that training regime, not
  proof that two passes are generally optimal.
- **Astra architecture claim:** the social-media account supplied by the learner
  attributes recurrent depth to OpenAI's Astra. The
  [official GPT-6 Astra documentation](https://developers.openai.com/api/docs/models/gpt-6-astra)
  checked on 2026-09-04 describes model availability and behavior but does not
  disclose a looped or recurrent-depth architecture. The course must therefore
  label this attribution as unverified reporting unless primary architecture
  evidence appears.

## Planned Day 6 learning questions

1. What changes when depth shares weights: parameter count, FLOPs, latency,
   activation memory, KV state, or representational capacity?
2. How does fixed recurrence differ from a router that assigns recurrence depth
   per token?
3. Under what training distribution over $r$ can a model remain useful at
   recurrence depths it sees at inference?
4. What stopping or exit criterion could allocate compute without leaking future
   information or destabilizing batching?
5. Which conclusions concern the architecture itself, and which concern a
   particular systems implementation?

## Planned Day 7 controlled experiment

Add an optional shared-block variant only after the ordinary `DongxiGPT` baseline
works. Compare it under three explicitly separate contracts:

1. **parameter-matched:** similar stored non-embedding parameters;
2. **compute-matched:** similar block applications or estimated FLOPs;
3. **wall-clock-matched:** similar measured training or inference time.

Record physical parameter count, effective block applications, train and decode
latency, peak memory, tokens processed, validation loss/perplexity, and fixed
generation examples. Include an unrolled untied baseline where feasible. A toy
result can demonstrate the mechanics and trade-off; it cannot establish that
recurrent depth is generally superior.

## Animation candidate

`CAND-ANIM-011` should preserve one block's visual identity while the hidden
state loops through it repeatedly. Counters for stored parameters, effective
depth, and compute move independently. A second act can contrast fixed recurrence
with token-level adaptive exit. This is captured now but remains unapproved and
all production remains on the Mac Studio.

## Return point

Resume this topic after Day 5 has established an ordinary decoder block. Until
then, Day 4 continues with scaled causal attention, gradient verification, and
the deliberately broken scaling and masking variants.
