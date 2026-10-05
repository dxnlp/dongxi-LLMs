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

### Actual candidates make the distinction testable

Chapter7's [actual-candidate experiment](../../experiments/reports/2026-10-04-inference-selection.md)
keeps this simulator as a reference, then replaces its class draws with864
autoregressive responses from six frozen tiny-decoder policies. Each policy has
its own ordered eight-response pool per item. Prefixes1,2,4,8 reuse that pool;
gold-blind voting and likelihood selection compete on the same evidence. The
decoder receives symbolic instruction/operand IDs, not parsed English. These
are actual generated answers, not natural-language reasoning traces.

For one fixed trained parity item, the correct answer0 appears three times,
while the wrong answer1 appears five times. A correct candidate is available,
but the majority emits the wrong answer. More samples cannot repair a voting
rule when the wrong answer dominates the pool. Train-only likelihood is another
possible ranker, not a correctness label. A constructed longest-output control
has no length contrast here: every eligible answer is one numeral plus EOS.
That absence of a distinction is a result to acknowledge, not an invitation to
infer verbosity effects from the arm's name.

The raw ledger also keeps immediate EOS, repeated numerals, errors and token
caps. Invalid candidates cannot vote but still consume work. A whole-attempt
token-budget replay charges the first overshooting attempt, rejects it from
selection and stops; it does not pretend that completed attempt was free or
that its actual cost fit the cap. Mandatory path rescoring is charged to every
selector in this microscope, although a serving implementation could omit it
for a first-answer rule. Equal attempts, equal nominal token caps and equal
total computation are different comparisons.

The three predeclared seeds retain weak source/template/family generalization.
Across-item error correlations describe this small fixed panel and its related
source siblings; they are not evidence of independent population samples.
Use the [visual notebook](../../notebooks/day-10/05_actual_candidates_and_answer_selection.ipynb)
to inspect available answers, actual decisions and measured work separately.

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

### Acceptance is not the same decision as selection

The [audited teacher-data experiment](../../experiments/reports/2026-10-04-teacher-data.md)
turns this distinction into an executable pipeline. Its teacher is an original
programmatic copy/reverse function, not an LLM or a human. Across twelve training
prompts it executes108 attempts, retaining errors, one bounded retry per error,
raw traces, final answers and stops. A format/length/END/deduplication gate admits
36 candidates: twelve correct and24 wrong. A well-formed answer is not thereby
a good demonstration.

A separate prompt-based task verifier scores that frozen pool. Top-per-prompt,
seeded random and length-matched random each select twelve examples covering
the same twelve source prompts. Their datasets contain12,4,6 correct examples,
respectively. The verifier can compute the training task's answer; it cannot
inspect held-out gold or use the teacher's declared fault label. Fixture
validation separately checks authored references for consistency. None of these
checks makes the programmatic teacher a neural reasoning model.

Global selection asks a different coverage question. Global top6 covers only
half the prompts and loses every three-word source, because equally correct
answers are ordered by shorter target length. One-per-prompt selection preserves
both length slices. A higher mean selected score cannot tell us which sources
disappeared; preserve source IDs and excluded prompts alongside the score.

### Equal demonstrations need not mean equal supervision

Let $n_{a,j}$ count assistant-body plus END targets for example $j$ in arm $a$.
With $U$ full-batch updates, valid target exposure is

$$
E_a=U\sum_j n_{a,j}.
$$

Here the three datasets expose42,46,42 targets per update. Over80 updates that
means3360,3680,3360 targets per student. Top versus primary random matches examples
and prompt coverage but not target exposure; length-random is the declared
sensitivity control. These counts still do not equate gradient information,
padded computation or upstream teacher work.

Nine same-initialization tiny sequence students, across three fixed seeds,
actually train and generate. Better-selected demonstrations do not consistently
win the four-prompt held-out test. Every trained arm scores0/4 on greedy
polite-prefix controls, despite naturally generating END. This is a concrete
reason to separate selected-data correctness, fitting, stopping and transfer.
The small panel does not establish a universal ranking of selection methods.

Use the [teacher-attempt notebook](../../notebooks/day-11/04_teacher_attempts_and_matched_rejection_sft.ipynb)
as a Chapter8 bridge before distillation. Its journal prevents duplicate committed
attempts on resume, not duplicate physical execution after an uncommitted crash.
The latter can repeat and has a visible unknown lost-cost boundary. Programmatic
serialized words and elapsed time are also not LLM inference tokens or API
billing. Keep those boundaries when replacing the fixture teacher with an
approved real-model adapter later.

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

### From three logits to an actual generating student

The [response-distillation experiment](../../experiments/reports/2026-10-04-response-distillation.md)
keeps the exact soft-target lesson below, but asks a different empirical
question. A6552-parameter tiny teacher actually learns symbolic
`STEP total ANS binary EOS` responses. A3088-parameter student learns either
its selected complete five-token bodies or answer-only three-token subsequences
from the same actual parent responses. Teacher and student share a15-ID
vocabulary/template, not hidden dimensions. Neither parses English nor uses a
pretrained reasoning checkpoint.

Each of three predetermined campaigns uses a120-update teacher and two
same-initialization80-update student arms. All train-source attempts happen to
be eligible and correct; adversarial malformed/wrong/missing-coverage controls
are separately tested, not invented main-run failures. The adapter selects
first format-eligible traces, not best-gold answers. Independent evaluation
then samples unrestricted tokens, including learned EOS, from teacher, original
student and both fitted students on the same source/template/family panel.

Both fitted arms learn the six training items, but neither consistently wins
all held-out slices. Printed steps and final answers tell different stories:
among432 teacher evaluation attempts,80 have a wrong step but a correct final,
and24 a valid step but a wrong final. Complete-response students retain26 and15
such contradictions. An independently checked printed total is observable
output validity, not proof that the network used it causally. Answer-only
outputs have no printed step; their step status is missing, not automatically
valid or wrong.

This experiment minimizes response NLL, not the KL of a full teacher vector:

$$
L_{\mathrm{response}}=-\frac{1}{N_{\mathrm{targets}}}
\sum_{t\in\mathrm{response+EOS}}\log p_\theta(y_t\mid x,y_{<t}).
$$

Prompt labels and padding are ignored and the causal shift occurs once.
Complete responses expose7200 student targets over all campaigns versus4320
for answer-only. Teacher fitting exposes10800 more targets and generating its
dataset incurs720 actions before student evaluation. Smaller student weights
therefore do not by themselves prove lower total cost or faster serving.
The [Day26 response notebook](../../notebooks/day-26/03_response_level_distillation.ipynb)
connects this sequence evidence to Chapter9's supervised objective and the
distribution-gradient lesson that follows. The local Spark transfer protocol
remains prepared, not executed.

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

## 15.8 Refinement is a state machine, not a guarantee

Instead of drawing independent alternatives, we can revisit a current draft.
Separate critique, proposed revision and acceptance. For question $x$, draft
$d_r$, critique function $C$ and reviser $R$, write

$$
c_r=C(x,d_r),\qquad u_r=R(x,d_r,c_r),\qquad
d_{r+1}=\begin{cases}u_r,&a_r=1,\\d_r,&a_r=0.\end{cases}
$$

The acceptance decision $a_r$ determines what the user eventually receives.
Advice can be unhelpful, a revision can be worse, and a permissive gate can accept
it. A high score or well-formed output is not a correctness certificate. The
mechanism therefore needs a declared stopping rule, round ceiling, failure path
and evaluator independent of its acceptance score.

Our [critique/revision experiment](../../experiments/reports/2026-10-04-critique-revision.md)
uses explicit programmatic actions, not a neural self-critic. A syntax gate
accepts a single binary symbol followed by natural EOS, including valid ties.
An unchanged accepted token path stops; otherwise the loop has two rounds.
Gold is physically absent from callback views and used only afterward to grade
the proposed and delivered answers. In the exact programmatic panel, contrarian
revision accepts54 helpful and54 harmful changes, then returns to the original
answer. Looking only at the final score hides both kinds of transition.

A second panel replays the unchanged864 actual tiny-decoder candidates from
Chapter7. Its repair rule makes an invalid draft well formed by retaining its
first binary token and appending EOS. This raises delivered correctness from
30/108 to65/108 on the fixed mixed-checkpoint panel, but every revision is
programmatic and many are still wrong. No historical model consumes critiques
or generates a revised answer. This is observable format postprocessing, not
evidence that repeated neural thought improves reasoning.

Count the draft and every critique/revision, even rejected ones:

$$
B=n_{\mathrm{draft}}+\sum_{r=1}^{R}
(n_{\mathrm{critique},r}+n_{\mathrm{revision},r}).
$$

The exact control matches this serialized-symbol budget with three or five
independent binary/EOS draws. It does not match neural computation. Replayed
model attempts can overshoot or underfill the ceiling; those counts and the
original generation/rescoring work remain visible rather than disappearing
when an attempt is ineligible. The no-critique baseline spends less work.

Independent review caught an important interface defect: an EOS-looking critique
marked as capped could trigger revision. The hardened instrument requires real
natural stopping and retains the capped tokens/error while skipping revision.
The old measurements and diagnostic-only intermediate revision are preserved;
the new campaign exactly reproduces their ordinary paths under current source.
The [refinement notebook](../../notebooks/day-26/04_critique_revision_and_acceptance.ipynb)
lets you inspect this boundary and harmful transitions, not merely watch a final
answer count rise.

## 15.9 What histories does the teacher teach on?

KL direction and prefix source are independent choices. Let a state $s$ be the
question plus generated history, $q(\cdot\mid s)$ the frozen teacher distribution
and $p_\theta(\cdot\mid s)$ the student. Offline trajectories supply one state
distribution; the current student supplies another, including its own mistakes.
At those states we may minimize either conditional objective:

$$
L_F(s)=\sum_i q_i(s)\log\frac{q_i(s)}{p_i(s)},\qquad
L_R(s)=\sum_i p_i(s)\log\frac{p_i(s)}{q_i(s)}.
$$

Forward KL asks the student to cover teacher probability. Reverse KL also
weights outcomes by the student's current probability, so its correction is not
the same $p-q$ residual. At temperature1, with full positive support and
$\ell_i=\log(p_i/q_i)$, the student-logit derivatives are

$$
\frac{\partial L_F}{\partial z_i}=p_i-q_i,\qquad
\frac{\partial L_R}{\partial z_i}
=p_i\left(\ell_i-\sum_jp_j\ell_j\right).
$$

Teacher targets are detached and coordinates must denote the same vocabulary
outcomes. Zero teacher support can make reverse KL infinite. Adding an arbitrary
floor would change the objective rather than resolve that fact.

### Sampling student states is not differentiating their occupancy

Our [student-prefix experiment](../../experiments/reports/2026-10-04-student-prefix-distillation.md)
actually trains24 tiny neural students across three fixed seeds. Its teacher is
an authored finite probability rule, not a pretrained neural teacher. The
factorial comparison changes offline versus current-student prefixes, forward
versus reverse KL, and ordinary versus privileged teacher context. All eight
arms within a seed start at identical student weights and use30 updates.

The teacher assigns0.8 to its target and0.2/7 to every other outcome. On a wrong
prefix, the unhinted rule deliberately repeats a previous symbol; a training-only
authored hint recovers the correct next symbol. This engineered teacher failure
isolates context dependence. The hint never enters a student input or evaluation
generation. A sampled HINT token is simply an invalid output, not private help.

The offline pool is collected once from the finite teacher. On-policy arms
collect actual unrestricted student trajectories before every update. Each
visited pre-action state, including the state emitting EOS, contributes to a
global valid-state mean. A cap contributes only visited states and does not
invent an EOS target. Different stop patterns produce different state exposure
despite equal updates. Ordinary failure controls retain costs and skip the
affected prompt group; the recorded main campaign has no such actual skips.

At a sampled state the implementation differentiates conditional KL with fixed
history and detached teacher targets. It does not differentiate the probability
of visiting that history or add a REINFORCE term. Thus “on-policy” describes how
states were collected, not a claim to compute the full derivative of an
expectation over a parameter-dependent state distribution. An optional
correct-sibling diagnostic reports how many groups would lack a demonstration;
it does not filter the fitted data or turn authored hints into a complete SDPO.

More relevant states do not guarantee better optimization or transfer. The
offline-forward-hint arms score5/24,6/24,7/24 on this tiny held-out reversal
panel, versus1/24,0/24,3/24 for the student-prefix-forward-hint arms. All other
arms, EOS/caps, invalid responses, work and negative outcomes remain recorded.
These are results of this frozen task and finite teacher, not a universal
ranking of distillation methods or model-scale reasoning evidence.

### A top-$k$ tail is a coarser lesson

Large vocabularies motivate retaining selected coordinates and aggregating the
rest. Let $T$ contain unselected outcomes, $q_T=\sum_{i\in T}q_i$ and
$p_T=\sum_{i\in T}p_i$. An exact tail bucket conserves both distributions'
mass but loses distinctions within $T$. For forward KL,

$$
D_{\mathrm{KL}}(q\Vert p)=D_{\mathrm{KL}}(q_{\mathrm{bucket}}\Vert
p_{\mathrm{bucket}})+q_TD_{\mathrm{KL}}(q(\cdot\mid T)\Vert p(\cdot\mid T)).
$$

Reverse KL has the corresponding $p_T$-weighted conditional term. This is an
exact decomposition, not a guarantee that bucket and full gradients agree.
Two vectors can match their retained coordinates and total tail mass perfectly
while distributing that mass differently inside the tail. Bucketed KL is then
zero although full KL is positive. At full vocabulary width no detail is lost.
Choosing top-$k$ indices is also discrete; our microscope freezes that selection
for differentiation and does not claim large-vocabulary compute savings. Its
finite tail-detail helper explicitly requires strictly positive full vectors;
zero-support controls use full KL separately, with infinities disclosed rather
than subtracting infinity from infinity. A homogeneous collection cohort is
also required before fitting: individually valid records from different model
states or sampling phases cannot silently become one frozen pool.

The [student-prefix notebook](../../notebooks/day-26/05_student_prefix_and_teacher_context.ipynb)
holds states fixed to compare exact gradients, then varies the collection source.
This separates target geometry, state coverage and free-generation behavior—
three questions that one decreasing training loss cannot answer.

## 15.10 The checkpoint genealogy is part of the claim

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

The [Day9 evidence-limited model card](../../docs/model_cards/day09-dongxigpt.md)
applies this discipline to a real trained checkpoint. It links measured NLL,
exposure, resource use and observed failures while leaving unverified weight
bytes, cross-machine reload and systematic story quality explicit. Its random-
initialization lineage does not acquire SFT/DPO/RLVR descendants merely because
their later teaching modules exist. The first retained staged snapshot preserves45
rows: at that point it contained one measured profile and44 unrun stages.
Auxiliary native jobs do not automatically backfill selected-parent or publication
rows. Later evidence must identify the actual artifact that satisfies each row.

The [historical short-run model defense](../../experiments/reports/2026-10-05-current-model-defense.md)
now follows the acquired Base into disposable full-SFT and fresh full/LoRA
replay artifacts, then into a separately specified FP32 merged validation
export. That historical defense stops at the short-run stage: it contains no
selected400-update SFT parent or trained DPO/RLVR descendant. A changed precision,
different fresh run or longer planned horizon cannot be hidden behind the
label “SFT model.”

The later [equal-exposure comparison](../../experiments/reports/2026-10-05-native-sft400-comparison.md)
extends that lineage with fresh400-update full and rank8 Q/V LoRA runs, each
presenting9,321 training labels. It then evaluates the original120 held-out
instruction items in40 source groups. Full tuning gets120 exact whole answers
and120 natural message endings; Base and the separately verified FP32-merged
LoRA get zero of either and exhaust every64-token cap. The
[source-bound comparison](../../experiments/reports/native-assistant-comparison-20261005-run-02/comparison.json)
retains original tokens, costs and separate parser/format scores. It does not
replace the failed historical BF16 merge with a fictitious success.

This is lexical transfer within three shared task templates, not broad assistant
competence. Both new training recipes use the same learning rate and exposure,
but have different trainable parameter spaces. The result does not identify the
best possible LoRA recipe. The selected full400 export is a concrete downstream
parent; each descendant still requires its own actual acceptance and ancestry.

The [actual assistant-study model card](../../docs/model_cards/native-assistant-interface-study.md)
names those local artifacts, observed weight/interface identities, data slices,
cost boundaries and excluded uses. It is a different card from the September
DongxiGPT baseline: a book can contain several measured branches without
inventing one universal model-development lineage.

The [later matched preference comparison](../../experiments/reports/2026-10-05-native-preference-comparison.md)
now adds actual chosen-only100 and DPO100 descendants of that exact full400
parent. Both have accepted native recovery and fresh pilot receipts; recovery
weights are not their initializations. Their100 updates and2,047 chosen targets
match, while DPO's rejected/reference supervision and forward geometry differ.
The common generation gives0/4,4/4,1/4 strict location answers for unchanged,
chosen-only and DPO respectively, despite DPO's much larger relative margin.
All retain120/120 original instruction answers and natural stops. Both gain
only `math-10` in the separately annotated reasoning diagnostic, moving5/20
to6/20 without losing an initially correct item. Nine seen/development items
and eleven controlled held-outs remain distinct. This ancestry does not connect
either policy to the separate Instruct→RLVR branch or certify broad reasoning.
Chapter11 explains why a preference ratio and a generated answer need not agree.

The separate [reasoning-interface card](../../docs/model_cards/native-reasoning-interface-study.md)
begins from actual Base and Instruct checkpoints, not those preference
descendants. Six complete initial conditions and two partial Base-chat decode
failures retain680 of800 planned responses, with120 missing. Increasing only
the Instruct/thinking-off cap from32 to128 changes held-out sampled correctness
from0/44 to13/44 and greedy correctness from0/11 to7/11. Both thinking-on
caps still truncate every response. This is an intervention on output budget
and rendered interface, not newly trained reasoning weights or a universal
ranking of thinking modes. Two correct off128 samples remain capped, showing
why delivered answers, bounded-parser correctness and completed stopping must
be defended separately. Its own later recovery gates now pass before both
fresh16 training children complete. G8 pilot supervision passes; G4's native
exit0/final16 export survives a separately retained final-logging failure.
Export consistency is not retroactive G4 pilot acceptance or a reason to refit.

The actual common post-training [cap32](../../experiments/reports/native-rlvr-common20-cap32-20261005-run-02.json)
and [cap128](../../experiments/reports/native-rlvr-common20-cap128-20261005-run-02.json)
comparisons retain400 new responses and reuse200 unchanged-Instruct responses.
On the11controlled held-out items, cap128 sampled correctness is13/44,11/44,
13/44 for unchanged/G4/G8; greedy remains7/11 for all three. Both cap32 metrics
remain zero. Yet pooling all100 diagnostic-containing responses at cap128 gives
34/100,35/100,36/100. An increasing aggregate hides no held-out sampled gain
and one small regression; it cannot justify a universal ranking. Only two
items carry the actual RLVR training-overlap flag, not all nine development diagnostics.

Both training runs have zero task rewards/advantages, but nonzero KL gradients
and16 actual optimizer applications. G4/G8 consume1,004/2,335 valid targets and
1,508/4,888 dense slots/sampling draws. Their updates match, their rollout work
does not, and numerical completion is not evidence of reward-driven learning.
Chapter13 connects these actual negative outcomes to the reward/advantage/KL
gradient paths rather than treating more group samples as a guaranteed repair.

The [fresh story comparison](../../experiments/reports/2026-10-05-native-story-first400-comparison.md)
is a separate random-initialization lineage. Both fixed learning-rate arms
complete 400 updates with identical 1,389,548 valid training labels; neither
completes its original 14,000-update schedule. Their 192 real continuations and
two complete, separately submitted AI rating sets show poor narrative quality
despite lower held-out NLL. The control ends 45 of 48 trained continuations
naturally, yet its mean ending-quality rating is only 0.0208 on the 0–2 scale.
EOS is a decoding event, not evidence of a satisfactory ending. The
[model-free archive](../../experiments/reports/native-story-publication-20261005-01-archive/acceptance.json)
retains bound measurements without implying portable model weights; the separate
rating report retains the reviews. The 288 missing later cells, 56 candidates
with at least one disagreement (64 dimension-level disagreements) and non-human reviewer
provenance remain part of the claim, not defects to conceal with a single score.

An evaluation contract also includes decoding, candidate budget, extraction,
verifier, evaluator revision, prompts, split and uncertainty method. Equal
evaluation hashes ensure metadata agreement, not that the panel is appropriate.
Apply paired comparisons on shared prompts where possible, report uncertainty
and keep subgroup failures visible. A single small average cannot defend
all capabilities promised in a model card.

## 15.11 Defending the final system

The [constructive-control case](../../experiments/reports/2026-10-04-reasoning-controls.md)
from Chapter13 provides a useful defense rehearsal. Actual sampled sequence
learning improves a mean while two predetermined seeds still miss a known-solvable
minority item. A constant-answer baseline explains apparently strong slices; an
exact lookup fit does not establish new-key transfer. Neither a positive control
nor a selected correct final answer proves natural-language reasoning or faithful
intermediate work. Preserve those distinctions when selecting teacher responses
or describing a distilled student's gains.

Before a reasoning comparison, distinguish actual base/instruct checkpoints,
raw/chat templates, supported thinking modes and output/candidate budgets. The
[original math/protocol fixtures](../../fixtures/reasoning-controls/README.md)
provide source/template/family slices and predeclared32/128-token interventions.
The fixtures themselves are instruments, not pretrained measurements. The
[later actual baselines](../../experiments/reports/2026-10-05-native-reasoning-rlvr-evidence.md)
retain completed and partial conditions separately, including costs, errors,
caps and missing responses. A thinking toggle does not turn instruct weights
into Base; a long trace does not validate causal faithfulness. These initial
measurements do not establish a distillation or trained RLVR gain.

The technical defense has a continuous argument: target capability → data →
model/recipe → measured change → cost → failure cases → next experiment.
Start with the user-facing task and justify each choice using recorded evidence.
Explain why the evaluation asks the right question and which claims remain
unsupported. Retain negative results and regressions. Being able to explain a
method's limitations is part of understanding it.

The historical short-run native evidence illustrates three independently useful conclusions.
First, exact full/LoRA recovery succeeds even though both final eight-answer
grids remain0/8 correct. Second, the common fifteen-item development panel
improves one automatic phrase-rubric score while every response truncates;
independent unblinded AI review finds no clean completed requested answer.
Third, a failed BF16 merge and passed separately declared FP32 merge can
coexist without contradiction. Each answers a different question.

Read costs at the same boundary as the claim. A successful replay pair has
455 numerical training targets in its final trajectory but684 physical
training presentations because the tail was executed twice. Development
observation adds1,440 more presentations. The failed full-resume attempt also
consumes wall time; restoring state does not refund it. A technical defense
that says only “the model trained successfully” conceals these distinctions.
Use the [actual tables](../../experiments/reports/2026-10-05-current-model-defense.md)
as a worked case, retaining missing campaign results rather than claiming
that this partial lineage is the final controlled comparison.

The later120-item comparison also shows why uncertainty needs a scope. Every
source cluster has the same observed full-versus-Base difference, so its frozen
source-cluster bootstrap interval is degenerate at one. That describes this
panel; it cannot measure uncertainty over unseen task templates or applications.
Likewise, the permissive format rubric passes all three models even though two
never give an exact requested answer. Declare the estimand before celebrating
the score. Chapter9's learning curves and worked example connect this delivered
behavior to the separate development NLL and recorded training cost.

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

## 15.12 Deep questions

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
11. If three of eight candidates are correct, why can more sampling still return the wrong answer?
12. Why can equal candidate counts, equal token caps and equal total compute identify different comparisons?
13. Why should a format gate preserve wrong but well-formed answers for a later selection audit?
14. If twelve examples cover the same prompts in two arms, what additional measurements are needed before calling their training budgets matched?
15. Why can a student learn every selected teacher demonstration without learning the teacher's full next-token distribution?
16. What does a valid printed intermediate step establish, and what does a wrong-step/right-answer response reveal?
17. How can refinement leave the final accuracy unchanged while repeatedly harming correct drafts?
18. Why is a serialization-matched revision comparison not necessarily compute matched?
19. Why should a capped critique be retained but forbidden from invoking revision?
20. Which two independent choices change in an offline-forward-KL versus student-prefix-reverse-KL comparison?
21. Why does differentiating conditional KL at sampled student states not include the state-occupancy derivative?
22. How can a zero-loss top-$k$ tail bucket still hide a positive full KL and a different gradient?
23. Why can120/120 held-out answers and a degenerate bootstrap interval still be insufficient evidence for a general assistant?

[Worked answers](../solutions/15-distill-evaluate-and-defend.md) and the
[Day 26–28 notebook route](../labs/15-distill-evaluate-and-defend.md) close the
book with executable selection, gradient, genealogy and release exercises.
The final product is a set of choices a reader can inspect and reproduce, with
their strengths, failures and evidence boundaries intact.
