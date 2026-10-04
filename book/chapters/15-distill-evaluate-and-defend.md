# 15. Distill, Evaluate, and Defend

We now have several ways to change model behavior: next-token training,
demonstrations, preferences and sampled rewards. There is another way to improve
the answer delivered to a user: spend more computation after training, then
select among candidates. A further step is to train a smaller or cheaper model
from the outputs or distributions that this process produces.

These choices share a question: where did the improvement come from, and what
did it cost? Days 26–28 connect inference selection, distillation and final
evaluation. The capstone is a defensible model-development account with executable
evidence. A prepared release is distinct from permission to publish it externally.

## 15.1 One model, several answer-producing systems

A checkpoint does not uniquely specify an answer. The prompt template, sampling
temperature, output cap, candidate count, tool use and selection rule also affect
the result. An evaluation comparing a greedy baseline with a sixteen-candidate
system is comparing complete answer-producing systems. That can be a useful
product comparison, but it does not isolate a weight improvement.

Begin with three distinct rules. Single sampling emits one candidate. Majority
voting extracts a canonical answer from several candidates and selects the most
frequent answer. Best-of-$N$ ranks candidates using a verifier or scorer and
returns the highest-ranked one. Record tie rules, invalid-output handling and
candidate token costs for each. Candidate generation may be parallel, so total
compute, latency and peak memory are related but different costs.

The original [self-consistency paper](https://arxiv.org/abs/2203.11171) studies
sampling multiple reasoning paths and aggregating answers. Its mechanism motivates
our comparison; its benchmark gains are not evidence for our tiny simulator or
our trained DongxiGPT checkpoint. A group of sampled answers is useful only to
the extent that the underlying model supplies valid alternatives and the
aggregation rule can identify them.

## 15.2 An oracle ceiling and a real selection rule

If independent candidates are correct with probability $p$, the probability
that at least one is correct is

$$
P(\text{some correct among }N)=1-(1-p)^N.
$$

This is an availability ceiling for an ideal selector. It is not the accuracy
of a deployed ranker. Correlated errors invalidate the independent approximation;
a model that repeats the same wrong answer sixteen times does not gain sixteen
independent opportunities. Retain candidate-level responses so correlation,
diversity and extraction failures can be examined.

Our original categorical simulator samples answer classes with probabilities
0.45correct,0.40one repeated wrong answer,0.15another wrong answer. A fixed
imperfect ranker gives the frequent wrong answer a higher score than the correct
one. Increasing $N$ can increase oracle availability while reducing the ranker's
selected accuracy. Majority voting faces its own question: does the correct
answer have the largest probability mass after canonicalization? Shared wrong
answers can win the vote.

```python
from dongxi_llms.distillation_lab import inference_comparison
comparison = inference_comparison(seed=2628)
comparison['rows']
```

The notebook plots measured Monte Carlo accuracy against an explicitly
illustrative eight-tokens-per-candidate budget. It overlays the independent oracle
formula as a theoretical comparison. These draws are a controlled simulation,
not measured language-model generations. Change the score order or answer
distribution and predict how each rule will change before rerunning.

The [recorded simulation](../../experiments/reports/2026-10-04-grpo-diagnostics-distillation.md)
uses 1200 trials. At 16 candidates, oracle availability is 1.0000 in this finite
sample, majority accuracy is 0.5683, and the imperfect ranker's accuracy is
0.0000. The analytical availability remains slightly below one. These contrasting
results explain why an oracle ceiling must not be presented as a deployed
selector's performance, especially when the scorer rewards the wrong property.

## 15.3 Selection amplifies scorer errors

Let a scorer return $S(y)=Q(y)+e(y)$, where $Q$ denotes the desired quality and $e$
denotes scoring error. Selecting the maximum $S$ favors candidates that combine
good quality with favorable error. Increasing the candidate pool gives the
selection procedure more opportunities to exploit systematic errors. Even
unbiased errors before selection need not remain unbiased among selected winners.

This is the selection form of the reward problem from Chapter 14. A strong
score on selected candidates cannot independently validate the scorer that
selected them. Use an external frozen evaluator, adversarial examples or
human assessment under a clear rubric. [Reward-model overoptimization
research](https://arxiv.org/abs/2210.10760) studies this issue for best-of-$N$
and policy optimization. Our simulator isolates one understandable case rather
than reproducing the paper's scale relationships.

If a verifier is exact for a narrowly defined task, it can select correct final
answers without estimating broad quality. That still leaves cost, reasoning
validity and domain coverage unresolved. If no candidate passes, report failure
or apply a predeclared fallback. Silently returning the best-looking invalid
answer and counting it as verified changes the system contract.

## 15.4 Rejection sampling creates a new dataset

Rejection sampling generates candidates, applies a criterion and keeps accepted
responses for training. The conditional accepted distribution generally differs
from the original teacher's distribution. Easy prompts with frequent success
can contribute many examples; difficult prompts can disappear. If acceptance
rate is $a$ and one candidate costs $c$ tokens on average, the simple expected
generation cost per accepted example is $c/a$. This estimate assumes stable
independent attempts and omits variable lengths and batch overhead.

Record teacher revision, exact prompt, sampling settings, candidate index,
raw output, acceptance result, verifier revision and dataset split. Bound retries
per prompt. Give rejected outputs a place in the audit rather than making them
vanish. Deduplicate accepted responses and preserve source grouping so prompt
paraphrases or near duplicates do not cross training and evaluation boundaries.

Training on accepted tokens is ordinary supervised sequence learning after a
special data-selection stage. It does not automatically reproduce the teacher's
full conditional distribution. It may reinforce only a narrow successful style.
Compare answer correctness, formatting, diversity, general capability regressions
and cost. A small accepted dataset can be valuable without being a complete
replacement for the demonstrations and controls of Chapter 9.

## 15.5 Hard responses and soft distributions

Sequence distillation uses teacher-generated token sequences as targets. At each
selected position, the observed target is one-hot. Distribution distillation
instead provides probabilities for several vocabulary outcomes at the same
prefix. These alternatives communicate different information.

Soft targets can express that two tokens are both plausible, reducing the
pressure to treat an unobserved alternative as equally wrong as every other
token. This connects to Chapter 3's distinction between one observed continuation
and the full conditional language distribution. Soft targets can also preserve
the teacher's mistakes and biases. A teacher probability is a teaching signal,
not ground truth.

The student and teacher must refer to the same outcome vocabulary for a direct
token-distribution KL. Matching token IDs from different tokenizers is invalid.
For different tokenizers, sequence/text supervision or an explicit alignment
method is needed. Even shared tokenization does not guarantee shared prefix
states: teacher-forced training on teacher responses and on-policy training on
student prefixes expose different state distributions.

## 15.6 Distillation loss and its derivative

Let student logits be $z$, frozen teacher logits $u$, temperature $T>0$ and

$$
p_i^{(T)}=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}},\qquad
q_i^{(T)}=\frac{e^{u_i/T}}{\sum_j e^{u_j/T}}.
$$

The distribution-distillation term is

$$
L_{\mathrm{KD}}=T^2 D_{\mathrm{KL}}(q^{(T)}\Vert p^{(T)})
=T^2\sum_i q_i^{(T)}\log\frac{q_i^{(T)}}{p_i^{(T)}}.
$$

Teacher probabilities are detached. The entropy of $q$ is constant with respect
to student parameters, so minimizing this KL has the same student gradient as
minimizing soft-target cross-entropy. Chain differentiation gives

$$
\frac{\partial L_{\mathrm{KD}}}{\partial z_i}
=T\left(p_i^{(T)}-q_i^{(T)}\right).
$$

Without the $T^2$ factor, the derivative is $(p_i^{(T)}-q_i^{(T)})/T$.
At sufficiently high temperature, probability differences shrink roughly as
$1/T$ for fixed centered logits; the multiplier compensates the resulting
gradient shrinkage. It does not make every temperature produce the same
gradient or require that inference use the training temperature. The classical
motivation is described in [Hinton et al.](https://arxiv.org/abs/1503.02531).

A mixed objective may use

$$
L=\alpha L_{\mathrm{KD}}+(1-\alpha)L_{\mathrm{hard}},\quad 0\le\alpha\le1.
$$

Declare whether each term is averaged over tokens or sequences, which positions
are valid and what data each term uses. The scaling changes their relative
influence. If Torch `kl_div` is used, its input/target conventions and reduction
must be checked explicitly; a visually similar expression can reverse the KL or
divide by vocabulary width unexpectedly. The source module writes the sum directly.

## 15.7 A bounded distillation experiment

The executable microscope fits three student logits to a fixed teacher with
logits 2, 0.5, −1. It runs 80 SGD updates and records temperature-scaled loss,
student probabilities and teacher probabilities. Autograd is compared with the
analytical $T(p-q)$ derivative at several temperatures. The teacher receives no
gradient. The experiment verifies target geometry and optimization mechanics.

The [measured scaled KL](../../experiments/reports/2026-10-04-grpo-diagnostics-distillation.md)
falls from 0.657134229 to approximately $9.77\times10^{-8}$ at $T=2$.
This is a successful fit to the specified teacher distribution. The report
retains both probability vectors so the claim can be checked beyond one loss.

It does not verify a smaller neural network's representation capacity, reasoning
transfer, deployment speed, robustness or acceptable regression. A real
teacher/student experiment must declare architecture, tokenization, data,
generation budget, training budget and evaluation. Include the student before
distillation and the teacher in the same panel. If training uses teacher-generated
answers selected by a verifier, distinguish the teacher, selection procedure
and student as three systems.

The [DeepSeek-R1 report](https://arxiv.org/abs/2501.12948) documents model-scale
distillation as part of a broader training process. We use it as an example
of the design category, not as evidence that our small run shares its capabilities.
The course's Qwen distillation pathway should inherit the frozen evaluation
and source identity conventions already established for SFT and RL.

## 15.8 The checkpoint genealogy is part of the claim

A score is ambiguous without the history of the evaluated weights. Record
checkpoint ID, parent ID, weights hash, tokenizer/chat-template identity,
data hash, objective, seed, precision, optimizer configuration and evidence
location. Branches might share a base or SFT checkpoint while using different
preference or reward data. Parent links reveal which comparisons isolate one
change and which compare several stages at once.

The [genealogy audit](../../src/dongxi_llms/distillation_lab.py) rejects unknown
parents, cycles, duplicate IDs and inconsistent frozen evaluation hashes.
Its synthetic three-node fixture tests structure. Its hashes are hashes of
fixture labels, deliberately not claims about real model files. For a capstone,
replace them with hashes computed from actual artifacts and link measured reports.
Preserve missing stages honestly rather than inventing a continuous lineage.

An evaluation contract also includes decoding, candidate budget, extraction,
verifier, evaluator revision, prompts, split and uncertainty method. Equal
evaluation hashes ensure metadata agreement, not that the panel is appropriate.
Apply paired comparisons on shared prompts where possible, report uncertainty
and keep subgroup failures visible. A single small average cannot defend
all capabilities promised in a model card.

## 15.9 Defending the final system

The technical defense has a continuous argument: target capability → data →
model/recipe → measured change → cost → failure cases → next experiment.
Start with the user-facing task and justify each choice using recorded evidence.
Explain why the evaluation asks the right question and which claims remain
unsupported. Retain negative results and regressions. Being able to explain a
method's limitations is part of understanding it.

The model card should name intended uses, excluded uses, training sources and
licenses, architecture/tokenizer, compute budget, evaluation conditions,
representative failures, known contamination risks and reproducibility commands.
The experiment card explains the comparison and selection procedure. The data
card explains source and split identity. Together these let another reader
separate method readiness, measured CPU mechanisms, measured GPU behavior and
unexecuted model-scale plans.

Prepare a release audit: local links and notation pass; source/test/notebook
paths reproduce; original versus borrowed assets and licenses are recorded;
every numerical claim has evidence; examples are identified as measured or
illustrative; checkpoint genealogy and held-out panel are coherent; limitations
are written. Passing an audit creates a reviewable release candidate. Uploading,
publishing or tagging a public release is a separate action under the learner's
instruction.

## 15.10 Deep questions

1. Why can the probability of finding a correct candidate rise while selected accuracy falls?
2. What assumptions are required for the independent oracle formula?
3. How can a scorer's errors become larger among selected winners?
4. Which prompts disappear from a rejection-sampled training set?
5. What information does a soft teacher target add beyond one generated token?
6. Derive the temperature-scaled student-logit gradient and explain the $T^2$ factor.
7. Why can equal token IDs be insufficient when teacher and student tokenizers differ?
8. How do teacher-prefix and student-prefix training distributions differ?
9. What does a genealogy consistency check establish, and what does it leave unverified?
10. What must a final technical defense say when an intended experiment was never run?

[Worked answers](../solutions/15-distill-evaluate-and-defend.md) and the
[Day 26–28 notebook route](../labs/15-distill-evaluate-and-defend.md) close the
book with executable selection, gradient, genealogy and release exercises.
The final product is a set of choices a reader can inspect and reproduce, with
their strengths, failures and evidence boundaries intact.
