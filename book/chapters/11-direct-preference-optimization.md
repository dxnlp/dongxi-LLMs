# 11. Direct Preference Optimization

A preference tells us that one answer is better than another. The previous
chapter turned this judgment into a scalar reward model, then warned that
optimizing the scalar can discover its mistakes. Can we use the comparison to
train the language policy directly? Direct Preference Optimization, or DPO,
answers that question through a change of variables. Its simplicity becomes
useful only when we understand what is being compared, which reference is held
fixed, and what the loss does not guarantee.

This chapter builds on sequence likelihood, Bradley–Terry modeling, and
supervised fine-tuning. Days 17 and 18 form one route: derive the objective,
audit its implementation, then compare policies using a frozen evaluation
contract. The complete derivation below is an independently worked finite-space
example of the construction introduced in
[Rafailov and colleagues' DPO paper](https://arxiv.org/abs/2305.18290).

## 11.1 The reference is a distribution over possible continuations

Fix prompt $x$ and let $\pi_{\text{ref}}(y\mid x)$ be a reference policy over
complete responses. It is commonly a supervised checkpoint at the start of
preference training. A policy $\pi$ may move probability toward higher-reward
answers, but a movement penalty expresses a preference for staying near that
reference:

$$
J(\pi)=\sum_y\pi(y\mid x)r(x,y)
-\beta\sum_y\pi(y\mid x)\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)},
\qquad \beta>0.
$$

The second term is $\beta\,\mathrm{KL}(\pi\Vert\pi_{\text{ref}})$. Its
direction matters: samples would come from the current policy, and their log
ratio is measured against the reference. The finite derivation assumes the
reference is positive on every candidate under discussion and rewards are
finite. A policy placing mass outside the reference support incurs infinite
forward KL. Real softmax models have positive theoretical token probabilities,
but truncation, top-$k$ sampling, and numerical underflow can create a different
effective support. Do not quietly transfer a theorem about one distribution
to another sampler.

This is an objective over response probabilities. It does not mean that two
neural checkpoints have nearby weights, nor that a small KL implies every
rare behavior is preserved. A reference is a behavioral anchor under a specified
prompt population; that population belongs in the experiment contract.

## 11.2 Solve the finite optimization problem

Use a Lagrange multiplier $\lambda$ for $\sum_y\pi_y=1$:

$$
\mathcal{F}=\sum_y\pi_y r_y
-\beta\sum_y\pi_y\log(\pi_y/\pi_{\text{ref},y})
+\lambda\left(\sum_y\pi_y-1\right).
$$

Differentiating with respect to a positive probability gives

$$
r_y-\beta\left(\log\frac{\pi_y}{\pi_{\text{ref},y}}+1\right)+\lambda=0.
$$

Rearrange and absorb the shared constant into normalization:

$$
\pi^*(y\mid x)=\frac{\pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)}{Z(x)},
\qquad Z(x)=\sum_y\pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta).
$$

The optimum exponentially tilts the reference toward reward. Compute it as
$\mathrm{softmax}(\log\pi_{\text{ref}}+r/\beta)$ for stability. With fixed
reward, large $\beta$ resists movement; small $\beta$ concentrates mass more
strongly. Adding a prompt-dependent constant to rewards multiplies numerator
and partition function by the same factor, so the policy is unchanged. This
is Chapter 10's reward gauge appearing inside an optimization problem.

The solution is exact for unrestricted finite distributions. A transformer
shares parameters across prompts and may not represent every independently
specified optimum. Optimization, preference coverage, and finite sample noise
are further limitations. The closed form is the reason for the loss design,
not a certificate that a short neural training run reaches the ideal policy.

## 11.3 Replace the latent reward with policy log ratios

Take logarithms of the optimum:

$$
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)}
+\beta\log Z(x).
$$

For two responses to the same prompt, the $\log Z(x)$ terms cancel. Define
the policy's reference-relative pair margin

$$
m_\theta=\beta\left[
\log\pi_\theta(y_w\mid x)-\log\pi_{\text{ref}}(y_w\mid x)
-\log\pi_\theta(y_l\mid x)+\log\pi_{\text{ref}}(y_l\mid x)
\right].
$$

Substitute this into the Bradley–Terry preference likelihood. For an ordered
winning pair,

$$
L_{\text{DPO}}=\mathrm{softplus}(-m_\theta).
$$

For an explicit soft preference target $q$, use
$\mathrm{softplus}(m_\theta)-q m_\theta$. The model is now a language policy,
but the classification target is still a preference. Offline DPO needs the
recorded pair's likelihoods; it does not require generating a fresh response
for every update. That operational advantage does not remove biases inherited
from the recorded comparisons.

The reference-relative term asks “How has the policy changed the relative odds
since initialization?” It does not simply ask whether the chosen completion
is more probable than the rejected completion. An answer initially unlikely
under the reference can receive a favorable implicit reward after a modest
probability increase while still remaining unlikely in absolute terms.

## 11.4 The gradient and the meaning of beta

Write $c=\log\pi_\theta(y_w\mid x)$ and
$l=\log\pi_\theta(y_l\mid x)$. With a frozen reference,

$$
\frac{\partial L}{\partial c}=\beta(\sigma(m_\theta)-q),\qquad
\frac{\partial L}{\partial l}=-\beta(\sigma(m_\theta)-q).
$$

At initialization $\pi_\theta=\pi_{\text{ref}}$, every relative margin is
zero. For a hard winning pair the derivatives are $-\beta/2$ and $\beta/2$.
At this point larger $\beta$ gives a larger local gradient for fixed learning
rate. Yet in the fixed-reward optimum larger $\beta$ means stronger restraint.
These statements concern different comparisons and are compatible.

For a fixed noisy target $q$, the optimal classifier margin is
$m^*=\log(q/(1-q))$. The corresponding change in pair log odds is $m^*/\beta$.
Thus changing beta alters both gradient scale and the policy displacement
needed to fit the same preference probability. Report beta with learning rate,
update count, and data noise. It is not inference temperature: sampling
temperature modifies decoding after training, while beta defines the training
objective relative to a reference.

The full parameter gradient combines both answer branches through shared
transformer weights. Just as in Chapter 10, independent scalar derivatives do
not imply independent changes in every sequence probability. DPO can increase
the pair ratio while decreasing *both* answers' absolute likelihoods. Probability
mass can move toward a third continuation absent from the comparison.

## 11.5 Sequence likelihood is a sum with a boundary

An autoregressive completion has

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{|y|}\log p_\theta(y_t\mid x,y_{<t}).
$$

Include a termination token under the declared template when it belongs to the
response. Otherwise a model can assign high prefix likelihood without learning
when to stop. Do not sum prompt likelihood into the DPO response score. The
prompt supplies context and can receive gradients through attention, but its
tokens are not the completion being compared.

For a batch, predictive logits have shape $[B,T,V]$, aligned labels and the
completion mask have shape $[B,T]$, and the gathered token log-probabilities
have shape $[B,T]$. Summing the masked values produces $[B]$ sequence scores.
Chosen and rejected branches each produce one vector. The DPO loss reduces the
pair vector only after these sums.

Perform causal shifting exactly once: run the model on `input_ids[:, :-1]`,
score `input_ids[:, 1:]`, and shift the completion mask with the target tokens.
The first response token is predicted at the final prompt position. A mask
shifted by the input positions instead of target positions drops that token and
may accidentally score another boundary. Padding masks affect attention;
completion masks affect which targets enter the score. They have different jobs.

Use `log_softmax`, gather safe labels, then mask. Labels set to `-100` cannot
be gathered directly; replace unscored labels by a valid temporary index before
gathering. Reject an empty scored completion instead of quietly assigning zero
likelihood. Every branch must use the same tokenizer, template, EOS convention,
precision policy, and response boundary for policy and reference.

## 11.6 Length normalization changes the model

Long sequences often have more negative log-probability because they contain
more terms. Summing is nevertheless the log-likelihood of the full response.
Dividing by response length gives average token log-probability, which describes
a different comparison. Replacing sums by averages in the standard derivation
does not preserve its response distribution interpretation.

This does not mean every length-adjusted preference algorithm is invalid; it
means an adjustment requires a named objective and a separate argument. First
audit preference length differences, EOS handling, truncation, and the generation
length distribution. Compare independent quality slices at matched length as
well as ordinary prompts. Do not declare a loss-mask or sequence-sum bug to be
“length debiasing.” The Day 17 heatmap lab exposes prompt, response, and pad
positions so that this boundary is visible.

## 11.7 A frozen reference is an implementation contract

Put the reference in evaluation mode, disable gradients, and score it under
`no_grad`. Do not update it with the policy optimizer. Disabling dropout matters:
otherwise two identical checkpoints can produce different log-probabilities at
initialization because of stochastic layers. Freezing weights alone is not
evaluation mode, and evaluation mode alone is not freezing gradients.

Reference scores may be cached for a fixed dataset. The cache identity must
include checkpoint hashes, tokenized sequence hashes, template and tokenizer
revisions, completion boundaries, dtype, and any normalization convention.
Changing any of these invalidates the cache. A detached tensor from the current
policy is not a fixed reference: it prevents one backward path but changes every
step, producing a different objective.

For model-scale training, reference memory is a real cost. The optional Spark
runner holds two model copies and one optimizer, uses bounded accumulation,
checks host memory reserve, and records its environment. This simple full-weight
path favors clarity over throughput. A production implementation might use
cached scores, a frozen adapter-disabled base, or distributed placement, but
each requires equivalence checks before it replaces the transparent path.

## 11.8 Controlled experiments and an instructive negative result

Two CPU experiments answer different questions. The categorical experiment has
two contexts and three possible complete answers. It uses known soft preferences
and compares DPO, flipped-label DPO, and chosen-only SFT from the same reference.
Exact expected reward and reference KL are available because the entire response
space can be enumerated. An oracle exponential tilt is an analysis benchmark,
not an extra training label secretly supplied to DPO.

The sequence experiment trains the actual tiny decoder from Chapter 5. Six
short prompts prefer an answer token called A over B; both branches include
EOS. A held-out prompt measures $P(A\mid x_{\text{heldout}})$ independently of
the pair loss. It is intentionally a severe simplification: answers are symbolic,
the prompt set is tiny, and no English capability is being tested.

The [report](../../experiments/reports/2026-10-04-preference-policy-cpu.md)
retains the resulting curves, including failures. A favorable pair margin can
coexist with worse absolute probability of the desired answer. That is a concrete
reason to evaluate free generation and independent tasks after preference
training. The chosen-SFT arm provides a useful control, but it sees demonstration
supervision rather than identical information; equal update count does not make
the two datasets semantically equivalent.

For Day 18's Qwen extension, begin with a saved SFT checkpoint, preserve an
untouched SFT control, train a DPO arm, and include a label-shuffled or matched
chosen-only control under a declared purpose. Use independently frozen tasks,
paired decoding seeds, refusal/correctness/coherence slices, and sample-level
outputs. Reward margins are training diagnostics, not the final verdict. The
[Spark runner](../../scripts/run_chapter11_spark_dpo.py) and
[lab guide](../labs/11-direct-preference-optimization.md) provide an executable
path; model-scale execution and its numerical results remain pending.

## 11.9 Companion route and exercises

Study [the finite optimum](../../notebooks/day-17/01_kl_regularized_optimum.ipynb),
[the sequence boundary audit](../../notebooks/day-17/02_sequence_likelihood_and_masks.ipynb),
then [the controlled DPO comparison](../../notebooks/day-18/01_dpo_controlled_comparison.ipynb).
[Worked answers](../solutions/11-direct-preference-optimization.md) explain the
following questions; each notebook also contains adjacent runnable solutions.

1. Derive the exponential-tilt optimum and state its support assumptions.
2. Why does the partition function disappear from a within-prompt preference?
3. Derive the chosen/rejected log-probability gradients with a soft target.
4. Reconcile larger initial gradients at larger beta with stronger fixed-reward restraint.
5. Why can both chosen and rejected likelihoods fall while the DPO margin improves?
6. Draw the single-shift alignment for prompt, first answer token, EOS, and padding.
7. Why is average token log-probability a different objective from sequence likelihood?
8. What invalidates a reference-log-probability cache?
9. Design a DPO/SFT/control comparison and explain unequal information versus unequal compute.
10. If held-out preference accuracy rises but generation correctness falls, what conclusions and next checks are justified?

DPO made preference learning a supervised likelihood problem over log ratios.
Chapter 12 returns to explicit sampled rewards. Its central difficulty is then
estimating how expected reward changes when the policy itself changes the
responses it generates.
