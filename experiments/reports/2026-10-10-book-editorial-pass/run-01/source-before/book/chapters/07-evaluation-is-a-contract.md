# 7. Evaluation Is a Contract

A model continues a story without spelling errors, yet changes the main character's name halfway through. Another model has lower validation loss but repeatedly returns a paragraph when the user asks for one word. Which is better? The question has no answer until we name the work we want the model to do, the population of inputs, and the rules for judging its outputs.

Chapter 6 established evidence about prediction on TinyStories. Its completed run improved fixed-development NLL, while inspected generations still contained repetition and inconsistencies. This chapter makes the next decision possible: define behavior precisely enough to compare before choosing another training intervention. Day 10 supplies four executable companions: [metrics](../../notebooks/day-10/01_metrics_and_contracts.ipynb), [uncertainty](../../notebooks/day-10/02_paired_uncertainty.ipynb), [slices and leakage](../../notebooks/day-10/03_slices_and_contamination.ipynb), and [mathematical grading and response replay](../../notebooks/day-10/04_mathematical_grading_and_response_replay.ipynb). The [lab guide](../labs/07-evaluation-is-a-contract.md) and [worked answers](../solutions/07-evaluation-is-a-contract.md) extend the argument.

## 7.1 Name a capability that can fail

“Good at reasoning” is too broad to evaluate directly. “Given an original two-step integer arithmetic problem, return the correct integer within 64 generated tokens” is narrower and testable. It specifies an input family, a behavior, a correctness rule and a generation budget. It still leaves choices: must the model explain? Can it use a calculator? How is an answer extracted? Those belong in the contract too.

For our storyteller, define five separate properties. Grammar concerns sentence structure. Character consistency asks whether entities and attributes remain stable. Causal continuity asks whether events follow from what came before. Repetition asks whether language cycles without advancing the story. Ending quality asks whether the narrative resolves within the permitted budget. These properties can disagree, so a single “story score” obscures useful evidence unless its aggregation is defended.

Write a rubric before looking at candidate outputs. For character consistency, a three-level scale could mean: 0, contradictory identities; 1, an unresolved reference; 2, stable identities throughout. Include examples created independently of candidate models. A second reader should be able to apply the rubric with similar judgments. Disagreement is evidence about the instrument, and should be preserved instead of silently averaged away.

## 7.2 Freeze the complete input-to-score path

An evaluation item is more than a question. It includes an identifier, provenance, expected answers or verifier, slice labels and split membership. A model identity is more than its public name: record the exact checkpoint revision, tokenizer revision, adapter revision if any, and inference mode.

The full path is:

> item → message serialization → tokenization → decoding → output extraction → metric → aggregation.

Changes anywhere in this path can change the result. A chat template can add a reasoning prefix; truncation can remove a decisive condition; a parser can accept an answer buried in an otherwise incorrect response. Version all of them. In the notebook, an immutable contract produces a SHA-256 identity from canonical JSON. The hash proves that two records refer to identical serialized settings; it does not prove that those settings are appropriate.

Training, development and publication test splits serve different purposes. Training examples update parameters. Development examples support recipe selection. Publication test examples support a final claim after the selection process is complete. Once repeatedly consulted, a nominal test split becomes development evidence. Renaming it does not restore independence.

Keep a small development suite for rapid feedback and a larger final suite with a stronger source audit. The same examples should not quietly migrate between roles. When data is scarce, use grouped cross-validation and explain exactly which decisions each fold informed.

## 7.3 Exact match measures a particular agreement

For item $j$, let the model prediction be $\hat a_j$, the set of accepted references be $R_j$, and a fixed normalization function be $f$. Exact match is

$$
e_j=\mathbf{1}\{f(\hat a_j)\in f(R_j)\},\qquad
\hat s=\frac{1}{N}\sum_{j=1}^{N}e_j.
$$

Here $N$ counts evaluation items, not generated tokens. The fixture normalizer applies Unicode NFC, case folding and whitespace collapse. Thus “ RED ” matches “red”, while “1.0” does not match “1”. This is deliberate: numeric equivalence belongs in a separately specified numeric verifier.

Removing punctuation can make a parser accept a response that violates a JSON requirement. Removing all words except the final number can reward an incorrect derivation that happens to end correctly. Conversely, raw string comparison rejects harmless capitalization. The correct policy depends on the capability. Report answer correctness and format validity separately when both matter.

An output such as “The answer is red” can be correct in ordinary conversation yet fail a one-word instruction. The observed failure should be attributed to the contract's format criterion, not automatically to a missing fact. Error categories guide training choices; a scalar score alone cannot.

## 7.4 pass@k includes an attempt budget

Suppose a code task permits $k$ independently sampled attempts and an executable verifier accepts a candidate. The event of interest is “at least one passes.” If the success probability for one independent sample is $p$, its probability is

$$
\mathrm{pass@}k=1-(1-p)^k.
$$

The evaluation usually samples $n$ candidates, counts $c$ correct candidates, and estimates the chance that a uniformly chosen subset of $k$ candidates contains a success:

$$
\widehat{\mathrm{pass@}k}
=1-\frac{\binom{n-c}{k}}{\binom{n}{k}},
\qquad n\ge k.
$$

The fraction counts all-incorrect subsets divided by all subsets. If fewer than $k$ incorrect candidates exist, every subset contains a success and the estimate is one. For $n=10,c=2,k=3$, the estimate is $1-56/120=0.5333\ldots$. The notebook verifies the product implementation against exact combinatorial counts. The estimator is established in the [original code-evaluation paper](https://arxiv.org/abs/2107.03374).

The procedure assumes candidates come from the stated sampling process. Copying one completion ten times supplies no new independent attempts. Picking the best answer with an oracle after generation measures an oracle budget, not single-response deployment quality. If a user must receive one answer without access to the verifier, pass@10 does not describe that experience. Record temperature, top-p, seeds, maximum output length, execution limits and verifier version.

A verifier itself can be incomplete. Passing three tests does not prove a program works on all inputs. Add adversarial and boundary cases, and report verifier coverage. Never execute untrusted generated programs as part of this lightweight lesson; its correctness flags are purpose-built fixtures.

## 7.5 Confidence has several meanings

The probability of a token is not confidence that the entire answer is true. A model can assign high probability to a common misconception, a boilerplate sentence or a copied phrase. Sequence probabilities also shrink with length, so comparing raw probabilities of answers with different lengths is misleading.

Calibration instead asks whether cases assigned a stated confidence succeed at the corresponding frequency. For example, among cases rated 0.8, roughly 80% should be correct under the same population and grading rule. A reliability diagram groups predicted confidence and compares it with observed correctness. Its bins, sample counts and uncertainty must remain visible. Verbal confidence elicited by a prompt is another model output, not a guarantee.

Statistical confidence intervals describe uncertainty in a score estimate under a sampling model. They do not describe an individual model's internal certainty. For $x$ successes in $N$ independent Bernoulli items, $\hat p=x/N$. The Wilson interval uses

$$
\mathrm{center}=\frac{\hat p+z^2/(2N)}{1+z^2/N},\qquad
\mathrm{halfwidth}=
\frac{z\sqrt{\hat p(1-\hat p)/N+z^2/(4N^2)}}{1+z^2/N}.
$$

For a nominal 95% interval, $z$ is approximately 1.96. Unlike the simplest normal approximation, this interval does not pretend zero observed failures establishes a zero failure rate. Ten successes out of ten still leave uncertainty about the underlying population. The confidence level refers to repeated interval construction; it is not a posterior probability that a fixed parameter lies in a particular computed interval.

If many items share a document, author or problem template, independence at the item level is doubtful. Resample the independent groups, or explicitly narrow the interpretation to the fixed evaluated set. A hundred paraphrases of one question are not a hundred independent subject areas.

## 7.6 Compare models on aligned items

Model A succeeds on 70% of a suite and model B on 73%. Separate score intervals discard useful information: perhaps both models fail on nearly the same items, or perhaps B fixes many A failures while creating new ones. Preserve item identifiers and compute differences on the same inputs.

For per-item scores $a_j$ and $b_j$, define

$$
d_j=b_j-a_j,\qquad \hat\Delta=\frac{1}{N}\sum_j d_j.
$$

The paired bootstrap samples item indices with replacement and recalculates $\hat\Delta$ for each resampled set. Percentile endpoints summarize the resampling distribution. Pairing retains shared item difficulty. Our implementation fixes the seed and number of draws; the notebook shows the full resampling distribution and its interval.

This interval has limits. Few rare events, strong grouping or a selected evaluation population can make it uninformative or optimistic. Trying twenty recipes and reporting only the winning interval adds selection bias. Publish the candidates tested, reserve final confirmation data, and avoid interpreting an interval containing zero as proof of equality. It indicates that the evidence does not resolve the sign at the stated precision.

## 7.7 Aggregate scores can hide regressions

A benchmark contains frequent easy tasks and rare structured outputs. A recipe improves frequent tasks but harms JSON validity. Its average rises because common tasks dominate. The rare failure is still consequential.

Report the item count, denominator and uncertainty for each predeclared slice: task family, input length, language, reasoning depth, format constraint and source. Distinguish a micro-average over all items from a macro-average giving each slice equal weight. Neither is automatically right; the deployment distribution should inform the choice.

Our fixture assigns two models to forty items and deliberately creates a rare-format regression. It demonstrates the accounting. It does not measure Qwen or DongxiGPT. Inspect the paired disagreement table, then a fixed sample of failures from each category. Select that sample by a documented rule, such as the first five identifiers per category. Attractive examples should not control the evaluation population.

For stories, a fixed prompt grid should cover different characters, event structures and openings. Include unfamiliar openings alongside common “Once upon a time” prompts. Evaluate greedy and sampled decoding as separate settings. Have reviewers blind to checkpoint identity, retain disagreement, and show how token-limit truncation affects the ending rubric.

## 7.8 Contamination is a provenance problem

An exact overlap between training and test text is easy to detect with normalized hashes. Semantic paraphrases, shared source documents and solutions copied into explanations are harder. Deduplicate before splitting and group related records by their source or generation template.

The leakage notebook creates a deliberate normalized duplicate across train and test, finds it, and demonstrates how a fictional lookup baseline obtains misleading success. Exact matching catches that constructed failure. Its absence cannot prove a clean benchmark. Search near duplicates, inspect source genealogy and audit synthetic-data prompts that may have included evaluation questions.

Leakage can enter through evaluation infrastructure too. A retrieved document may contain the answer key. A judge prompt may reveal the reference when it should assess independent reasoning. A developer may rewrite a parser after inspecting candidate mistakes. Record these changes as contract revisions rather than folding them silently into a favorable score.

## 7.9 A runnable evidence workflow

The CPU route has four stages. First inspect accepted strings and verify metric edge cases. Then compare aligned fixture panels with paired uncertainty. Inspect slices and contamination, then replay complete responses through a frozen mathematical grader. Each notebook puts an explanation and runnable reference answer immediately after the prediction.

For the campaign route, the [experiment specification](../../experiments/specs/2026-10-04-evaluation-and-sft-course.md) declares a fixed development suite and a separately frozen publication suite. The later [actual120-item publication comparison](../../experiments/reports/2026-10-05-native-sft400-comparison.md) measures unchanged Base, fresh full400 and merged LoRA400 under the same prompts and decoding. Those held-out values still use three shared training task templates; they do not establish transfer to arbitrary instructions. A distinct actual Base/short-SFT development replay supplies model events for the smaller mathematical/behavioral instrument in section7.16. Keep its15-item results separate from the120-item publication test. The original CPU report supplies evidence only for the metric, masking and optimization mechanisms.

Before training, answer: what observation would cause us to reject the recipe? A decline in held-out format validity might be unacceptable even if response NLL improves. A confidence interval that is too broad might justify collecting more evaluation items. A parser disagreement might justify improving the instrument before changing model weights.

## 7.10 Mathematical equivalence has a boundary

The strings `1/2`, `0.5` and a boxed fraction can express the same answer. A
strict string matcher would reject some of them. A parser that accepts any
number found anywhere can fail in the other direction: “Perhaps 1; perhaps 2”
does not supply one unambiguous answer. A stronger instrument separates four
questions: where is the claimed answer, is its expression supported, does it
equal the reference, and does the response obey the required interface?

Our independent [bounded grader](../../src/dongxi_llms/reasoning_evaluation.py)
extracts one balanced box or one explicit `Final answer:` line; without either,
it tries the whole response. Nested fraction braces are balanced. Two explicit
answer markers remain ambiguous, even if they agree. This conservative rule can
reject a valid model response. That is a visible contract limitation, not a
reason to quietly rescue the last number after seeing the output.

Supported scalar expressions contain finite integer/decimal constants, exact
fractions, addition, subtraction, multiplication, division and grouping. Values
are stored as rational numerator/denominator pairs. Decimal text is converted
directly, not first through a binary floating-point approximation; Python's
[Fraction documentation](https://docs.python.org/3.12/library/fractions.html)
explains this distinction. Thus `0.5` equals `1/2`, while the finite decimal
`0.3333333333333333` is not exactly `1/3`. A tolerance would be a separately
declared metric, not an invisible kindness to one checkpoint.

Finite sets ignore element order and duplicates. Real intervals preserve both
endpoints and open/closed boundaries: `[1,2)` differs from `[1,2]`. Named units
remain part of the answer. `3 seconds` and `3 s` share a declared alias, but
`100 cm` and `1 m` do not match because unit conversion is outside this grammar.
The instrument deliberately chooses limited coverage over unsafe generality.

Variables, square roots, unknown units and arbitrary code are `UNSUPPORTED`.
Malformed expressions, zero denominators and exceeded size/depth bounds are
`INVALID`. Multiple explicit answers are `AMBIGUOUS`. These are not the same
failure as a supported expression equal to the wrong number. Report all of
them. The parser never calls `eval`, runs generated programs or asks an
unbounded symbolic engine to simplify arbitrary text. The exact limits and
[reviewed adversarial cases](../../fixtures/reasoning-evaluation/README.md)
are part of its reproducible definition.

Correctness and formatting also remain separate. Unsupported grammar does not
automatically fail an unconstrained format criterion: the grader can lack
coverage while the response obeys its interface. `Final answer: 4` can receive
the right arithmetic score while failing “return only one integer.” A JSON
task uses a bounded object parser that rejects duplicate keys, nonfinite values
and surrounding prose; it does not discard the prose and congratulate the
model on valid JSON. The original integer-only RLVR verifier remains unchanged.
Accepting boxed fractions here does not automatically loosen that other task.

## 7.11 Replay complete responses before interpreting a gain

A score should be reproducible without rerunning generation. Freeze item and
source-group IDs, split membership, prompt, reference, task/rubric, parser
version, serialization, thinking mode, decoding and stopping budget. Hash the
suite and those settings. Each response records the contract identity,
checkpoint and sample identities, full raw text, token IDs/counts, stop reason,
truncation, error and cost boundaries. Unknown values are explicit `null`, not
estimated from text length. A response error stays in the ledger; dropping it
would change the denominator in favor of the failing system.

Replay performs extraction, supported equivalence and task scoring on every
record. Our `task_success` metric is correctness AND required format. Natural
termination is reported separately. A correct string stopped by its maximum
token cap is not evidence of an emitted EOS. If a deployment requires natural
termination as part of success, define a new metric before the comparison.

The [authored replay panel](../../fixtures/reasoning-evaluation/) has fifteen
development items from fourteen source groups. Its two response panels are
named `authored-baseline` and `authored-candidate`, not Qwen checkpoints. The
candidate improves the aggregate while deliberately regressing on finite-set
membership and JSON validity. The
[executed report](../../experiments/reports/2026-10-04-reasoning-evaluation.md)
preserves those failures, an ambiguous answer, unsupported grammar and a
simulated execution error. No model generation or token-cost measurement occurs.

Related items must remain paired and grouped. If source group $g$ contains
$n_g$ aligned item/sample differences $d_{gj}$, a cluster-bootstrap draw
resamples groups with replacement and computes

$$
\Delta^*=\frac{\sum_{g\ \mathrm{drawn}}\sum_{j=1}^{n_g}d_{gj}}
{\sum_{g\ \mathrm{drawn}}n_g}.
$$

Every sibling item travels with its group, including repeated groups in a draw.
This preserves the micro-average estimand while avoiding independent resampling
of variants as though they had unrelated sources. Few authored groups still
give fragile, population-dependent uncertainty. A favorable interval on a
constructed instrument panel does not establish a capability gain.

The original panel also includes positive/negative examples for a benign
computer-process request and a false arithmetic premise. Its required/forbidden
phrase checks make the chosen rubric inspectable; they are not a semantic judge
of helpfulness, sycophancy or safety. A polished response can exploit phrase
checks just as a policy can exploit another verifier. Preserve human review and
do not promote these fixtures into a broad behavioral certification.

Finally, candidate selection must not receive held-out reference answers or
correctness flags. The reusable `candidate_view` returns generation metadata
without those fields, even when passed a graded record. Selection uses the
declared ranker or voting rule; evaluation happens afterward. The local generation
adapter below connects the instrument to actual HF forward calls. Its later
pretrained Base/short-SFT comparison is recorded separately in section7.16;
authored response panels never become pretrained evidence retrospectively.

## 7.12 A model event must survive the trip to a score

Generation and grading are different jobs. The
[local adapter](../../src/dongxi_llms/reasoning_generation.py) consumes an existing
frozen contract, exact suite and local checkpoint; replay consumes its saved
records without a model. A tokenizer-only preparation step binds the contract
to actual encoding rules, special IDs and the complete input template. Before
weights load, the chosen tokenizer must match both the frozen interface and the
one saved with the checkpoint. Equal vocabulary sizes cannot establish this.
Actual local file digests identify bytes; a model's familiar name does not.

Raw completion and chat evaluation serialize different inputs. Chat also records
the explicit thinking-template keyword and verifies rendered-text tokenization
against the template's direct tokenization. Passing `enable_thinking` is an
interface setting, not proof of an architecture or a faithful reasoning trace.
Greedy chooses a maximum logit; stochastic sampling uses the frozen temperature
and top-k/top-p transformation. A greedy contract rejects sampling settings it
would otherwise ignore. Each item's sample seed derives from its stable item ID
and sample coordinate, not its position in the execution loop.

Every generated action remains in the token ledger, including a stopping token.
The complete raw decode preserves special tokens. A separately declared
`response_text` excludes only the final chosen EOS or turn stop for grading:
the sequence `4, EOS` should not fail “return one integer” merely because the
instrument prints the special-token spelling. Neither its EOS action nor its
cost disappears. Truncated responses keep all their text; an arbitrary trailing
answer or unwanted explanation is never trimmed into compliance.

| Observed boundary | Recorded stop | Natural termination | Truncated |
|---|---|---|---|
| Chosen declared EOS | `eos` | Yes | No |
| Chosen declared end-of-turn token | `turn_stop` | Yes | No |
| Full output budget consumed | `max_tokens` | No | Yes |
| Context budget exhausted first | `context_limit` | No | Yes |
| Forward, decoding or execution failure | `error` | No | No |

An EOS in the prompt is not an EOS emitted by the model. An EOS-valued padding
ID also does not make padding a generated action. Context exhaustion can leave
zero output tokens; it must not masquerade as a full output-cap response.
On failure, retain the partial IDs, text, exact stage and attempted work.

The transparent implementation recomputes the complete prefix at each forward.
Consequently, generated actions and model-forward token positions differ:
four output actions after a three-token prompt require forwards over 3, 4, 5 and
6 positions. These are counted input positions, not FLOPs or billing units.
The adapter separates attempted from completed forwards and records wall time;
offline grading consumes zero additional model scoring tokens. This is an
evidence microscope, not an optimized KV-cache inference benchmark.

Each invocation uses a new output directory with append-only, flushed records
and progress events. An interruption leaves saved partial evidence, not a promise
of exact resume. Input bytes are checked again after generation; mutations
invalidate the invocation while retaining its outputs. The
[bounded adapter report](../../experiments/reports/2026-10-04-reasoning-generation-adapter.md)
verifies real CPU forward calls from an original tiny random HF model and separate
scripted failure controls. That establishes an executable measurement path,
not pretrained reasoning ability, independent human agreement or GPU readiness.

## 7.13 A correct candidate is not yet a correct decision

Suppose eight sampled solutions include three correct answers and five copies
of the same wrong answer. An evaluator can report that a correct answer exists.
A majority selector can still return the wrong one. These are different events:
one measures the supply of useful candidates, the other measures a decision
made without access to the answer key.

Let $C_N=(c_1,\ldots,c_N)$ be an ordered candidate prefix, let $v(c)$ be a fixed
independent complete-task verifier, and let $s(C_N)$ select one candidate or
abstain. Define

$$
o_N=\mathbf{1}\{\exists c\in C_N:v(c)=1\},\qquad
d_N=\mathbf{1}\{s(C_N)\ne\varnothing\ \mathrm{and}\ v(s(C_N))=1\}.
$$

The diagnostic $o_N$ is any-correct oracle availability. A deployable selector
does not receive $v(c)$ or the reference answer. Always $d_N\le o_N$. Nested
prefixes make $o_N$ nondecreasing: adding candidates cannot remove an available
correct answer. There is no corresponding monotonic guarantee for $d_N$.
New wrong votes can overturn the decision, and a confident wrong candidate can
outrank a correct one.

The [actual candidate lesson](../../notebooks/day-10/05_actual_candidates_and_answer_selection.ipynb)
connects this distinction to real causal forwards from the original tiny
symbolic decoder. Three fixed seeds each supply an initial policy and a
train-only fixed80-update policy. Eighteen original items cover training,
held-out sources, their alias-template siblings and an unseen task family.
Each policy genuinely samples eight EOS/0/1 paths per item, with immediate EOS,
repeated numerals and caps allowed. Its English strings are descriptions, not
model-parsed language; success here is not natural-language reasoning evidence.

First-attempt, majority, unique-answer-support and likelihood selectors reuse
the same prefixes1,2,4,8. Unsupported or malformed outputs cannot vote, but
remain in attempt and cost denominators. Independent draws with identical
strings still count as repeated votes; copying a stored record is not a new
draw. Ties choose the earliest eligible sample. Counting each distinct answer
once instead changes the estimator and can reduce a binary vote to an
earliest-support tie. Deduplication is not automatically an improvement.
The constructed longest-output control is also deliberately weak: every
eligible answer has one numeral and EOS, so length cannot distinguish them.
It supplies no measured evidence of verbosity-ranking damage.

The likelihood ranker uses the mean conditional action log probability,
including EOS:

$$
r(c)=\frac{1}{T_c}\sum_{t=1}^{T_c}\log p_\theta(c_t\mid x,c_{<t}).
$$

Here $T_c$ counts generated actions including the chosen EOS. The fitted policy
supplies learned probabilities, not a separately trained correctness judge.
Length normalization addresses one scoring convention; it does not turn
likelihood into truth. The strict candidate view removes gold fields even when
the evaluator keeps them in a separate ledger. The selector's code and input
contract, not a reassuring name, enforce that separation.

In the [measured CPU panel](../../experiments/reports/2026-10-04-inference-selection.md),
seed10052's training item `train-11` produces answers1,0,0,1,1,1,1,0, while the
reference is0. Availability is1 and majority success is0. Fitting also increases
within-pool raw repetition, while unseen-family majority success at eight
attempts is0.50,0.00,0.00 for the three final policies. The four-item family slice
is imbalanced: a constructed constant1 reference control scores0.75. No learned
reasoning or general inference-scaling benefit follows from these results.

## 7.14 Failed attempts belong in the compute budget

Equal candidate counts need not imply equal work. Immediate EOS consumes one
action; a repeated-numeral cap can consume three and still supply no usable
answer. Prompt processing and path scoring consume additional model work.
Keep generated actions, rescored actions and forward input positions distinct.
With an uncached loop, three actions after a four-token prompt require forwards
over4,5,6 positions. Their sum15 is not the output-token count3, and neither is
a FLOP estimate.

The lesson records all864 candidates and1736 generated actions. Its token-cap
controls accept only complete attempts that fit the remaining cap. If the next
attempt crosses the boundary, that generated attempt is charged but rejected
from selection, and replay stops. It cannot skip ahead for a cheaper correct
candidate. The resulting whole-attempt overshoot is explicit; this is not an
exact token-level interruption policy. Equal caps can yield unequal accepted
attempts, actual actions, prompt positions and time.

For one final policy, majority success on the fixed full panel declines from
0.388889 at cap6 to0.333333 at cap12, while oracle availability stays0.388889.
Extra budget changes votes rather than guaranteeing a better decision. These
descriptive scores include training items and grouped siblings; a population
capability claim would require a larger independent final suite.

Per-attempt first-forward, later decode, path-rescoring and host wall times are
measured separately. The notebook's accuracy–token–time plots sum observed
attempt costs plus retrospective selector time. They are not optimized adaptive
service measurements. Every selector is charged the protocol's mandatory
likelihood-rescoring work, although a production first-answer implementation
could omit it. Replay records lacking measured timing must report it unknown,
not zero or an estimate from character count.

Finally, measure diversity and errors rather than assuming fresh seeds make
all mistakes unrelated. Record distinct answers and strings, wrong-answer
agreement and across-item sample-ordinal error covariance. Identical strings
can come from independent draws; shared wrong answers can persist across many
draws. A small source-related panel supports descriptive diagnostics, not a
universal correlation claim. The evidence question is whether the declared
selector uses extra attempted work well on the specified population.

## 7.15 A card is an index into evidence, not a certificate

A model card can make a weak experiment look finished if its polished summary
quietly fills missing fields. An evaluation card should do the opposite: make
the claim easy to inspect and the missing evidence hard to overlook. Preserve
the frozen contract, raw responses and replay alongside the readable card.
Someone should be able to recover its denominators, failures, slices and
paired comparison without loading the model or trusting the author's prose.

Our [offline exporter](../../src/dongxi_llms/evaluation_model_card.py) uses the
same grader as response replay. It accepts complete recorded panels rather than
selecting favorable examples. Actual input-byte hashes identify what it read;
settings identify what the contract prescribed. Neither proves that a named
pretrained model ran. When no matching run identity is available, checkpoint
provenance and generation resources remain explicitly unverified or unknown.
A configured device is not an observed hardware measurement.

Consider the authored baseline/candidate panel. Its arithmetic scores can be
replayed exactly, including the candidate's set/JSON regressions and simulated
error. Exporting those results creates a useful instrument-check card. It does
not create model accuracy, measured token cost, training lineage or independent
behavioral review. An actual local random-model panel supports a different,
still narrow claim: this identified adapter emitted and graded those responses.
Neither becomes evidence about Qwen by changing the title.

Keep selection history and missing coverage visible. A partial run must not
look like a complete planned panel, and unavailable measurements must not be
printed as zero. A paired interval requires aligned item/sample coverage and
the declared source groups. A model card may link that analysis, but cannot
repair unpaired data. The [lab](../labs/07-evaluation-is-a-contract.md) shows the
export route; final capability and cost defense waits for the actual approved
experiments.

## 7.16 A real response can pass a weak rubric and still fail the user

The [actual development replay](../../experiments/reports/2026-10-05-pretrained-evaluation-replay.md)
uses the unchanged fifteen-item, fourteen-source-group instrument with two
identified local checkpoints: acquired Qwen3-0.6B-Base and the disposable
twenty-update full-SFT profile. One fixed template, greedy decoding and a
64-token cap produce thirty raw responses. Each retains measured IDs, work,
stop reason and complete text. Offline grading uses the original rules,
not rules adjusted to favor the observed responses.

Base receives0/15 automatic passes; short SFT receives1/15. Both have12/15
format-valid records,0/15 natural stops and15/15 truncations. Their paired
source-group interval for the automatic score difference is[0,0.214286].
These are real model measurements, but a small authored development population
does not become a general reasoning benchmark because its outputs are genuine.
Nor does a successful generation-process exit mean that its answers succeeded.
Most instrument items have an unconstrained `any` format policy, so12/15
format-valid records are not twelve clean or helpful answers.

Inspect the SFT `benign-a` answer. It includes the rubric's `terminate` and
`task manager` phrases, but repeats `.nasa` fragments and ends mid-sentence.
The positive phrase match is correctly reported under the frozen instrument;
an independent whole-response review can still judge the text unhelpful or
unfinished. Preserve both observations. Do not replace the automatic score
after seeing it, or promote that score to “one useful assistant answer.”

This is the same distinction as reward optimization: a measurable proxy can
reward a fragment without capturing the whole intended behavior. Diagnose
the instrument and the policy separately. A new semantic rubric would be a
named next evaluation contract, not a silent repair of this comparison.
Likewise, the parser's `UNSUPPORTED` status reports its own bounded grammar;
it cannot by itself diagnose the model's mathematical knowledge.

The two panels cost1,920 generated actions and94,656 full-prefix forward
positions. These are separate units because this transparent adapter has no
KV cache. Both externally supervised processes exit0; every answer still
reaches its output cap. The readable card indexes the raw evidence rather
than certifying training genealogy, independent human agreement or safety.
This instrument case and Chapter9's lower-NLL/failed-generation case connect
measurement choices to the next training decision without inventing a quality
gain, using the publication test or advancing the learner's day.

## 7.17 Story coherence needs a separate rating instrument

A copy/extraction score cannot answer whether a story preserves its characters,
causal links and ending. The original campaign therefore freezes twelve story
openings, five independent0–2 rubric dimensions and four decoding recipes.
Greedy and three seeded temperature0.8 continuations supply repeated views of
an opening, not four independent opening populations. Predetermined checkpoints
keep an attractive story from choosing its own comparison point.

The [story-rating consumer](../../src/dongxi_llms/story_rubric.py) now turns that
contract into two artifacts. Reviewers receive a shuffled anonymous packet
with the original opening and complete emitted text. A private codebook binds
the anonymous IDs back to checkpoint, recipe and source group; it stays out of
the reviewers' packet. Empty templates do not assign neutral scores or imply
that two independent reviews occurred. Rater type, independence and known
prior exposure still need disclosure: an anonymous label is not proof of
successful human blinding.

For each continuation, retain grammar, entity/object consistency, causal
continuity, repetition and ending separately. A readable story can contradict
an object's location; a consistent story can still loop without resolving its
setup. An output stopped by a256-token cap is the actual incomplete response
being rated. Do not silently discard it or invent its ending. Natural EOS and
narrative resolution remain different measurements.

Two ratings can disagree. Preserve both and apply only the adjudication policy
declared before review. A mean score is a declared summary, not evidence of
agreement. Missing responses, abstentions and absent reviews remain visible
in coverage; a missing checkpoint does not receive0 or a flattering inferred
score. Reject changed packet hashes, unknown/duplicate IDs and invalid rubric
values before computing any comparison.

After joining through the codebook, compare matched checkpoint/recipe cells.
Resample source openings with both arms kept together. Forty-eight outputs
from twelve openings and four recipes are not forty-eight independent opening
groups; two raters do not double that sample either. This consumer conservatively
refuses a paired interval until all twelve openings/four recipes have complete
paired scores. Its coverage report retains missing cells instead of substituting
an interval from an easier partial cohort. The
[uncertainty notebook](../../notebooks/day-10/02_paired_uncertainty.ipynb)
supplies the runnable mathematical mechanism; the
[rating lab](../labs/07-evaluation-is-a-contract.md) supplies the operational
failure controls and adjacent empty-rating example.

The fixtures are explicitly authored demonstrations, not generated model
stories or human scores. The corpus audit is separate again: exact and a
defined lexical near-overlap check can inspect the pinned training bytes but
cannot prove all semantic contamination absent. The frozen publication panel
must not choose learning rate, decoder or checkpoint. Implementing this
instrument does not fabricate a trained comparison.

The [actual first400 comparison](../../experiments/reports/2026-10-05-native-story-first400-comparison.md)
now supplies that separate evidence. Two fresh anonymous AI instances each read
all192 continuations from the four available checkpoints. They received only
the public packet and rubric, not the arm codebook or one another's judgments.
The frozen two-rater mean preserves56 continuations with at least one axis
disagreement; neither reader abstained. Separate instances do not establish
independent underlying models, human agreement or perfectly successful blinding.

| Observed at update400 | Control | Half learning rate |
|---|---:|---:|
| Grammar mean,0–2 |0.0729|0.0000|
| Entity/object consistency mean,0–2 |0.2604|0.1458|
| Causal continuity mean,0–2 |0.1979|0.0938|
| Repetition mean,0–2; higher means less repetition |0.8438|0.9792|
| Narrative ending mean,0–2 |0.0208|0.0000|
| Natural EOS |45/48|25/48|

Each mean uses48 continuations and two supplied judgments, not96 independent
stories. Both models remain poor by this rubric despite their lower NLL and
many natural stops. The half-rate arm has a slightly higher mean repetition
score but lower means on the other axes; no single overall winner is defined.
The initial random checkpoints receive repetition1.5 while grammar/continuity
are zero: absence of repetitive loops is not competence.

The [retained rating report](../../experiments/reports/native-story-publication-20261005-01/ratings-evaluation-01/report.json)
keeps both raw ratings, recipe slices and the frozen800-draw seed1010 paired
resampling. It carries four recipes together within each of twelve source
openings. Its intervals describe these supplied ratings and sampled openings,
not training-seed variability or a general population of human judgments.
The288 cells at later checkpoints remain missing, with null scores and
incomplete paired comparisons. Completing this bounded comparison does not
complete the original14,000-update training schedule.

## 7.18 Exercises and transition

1. Define a testable claim about story coherence, including prompt population and rubric.
2. Explain why “1.0” and “1” should or should not match in a chosen task.
3. Derive pass@3 from a pool with ten samples and two successes.
4. Explain why ten copies of one sampled answer do not supply ten attempts.
5. Describe two different meanings of confidence in model evaluation.
6. Compare paired and independent resampling on the same candidate models.
7. Design a slice report that makes a rare format regression visible.
8. Explain what an exact overlap audit can establish and what it cannot.
9. Decide when a repeatedly consulted test split should be relabeled development.
10. Defend a stopping rule for a story evaluation before inspecting the outputs.
11. Explain why a rounded decimal, an interval boundary and a unit require three different equivalence decisions.
12. Can a response be arithmetically correct, format-invalid and truncated at once? State the three resulting metrics.
13. Design a source-group paired comparison that keeps related variants together without converting the estimand to an equal-group mean.
14. Why must a candidate selector be unable to inspect reference answers, even when the evaluator retains them?
15. Why can `4, EOS` have two generated actions but a one-token grading surface? Which quantities must the instrument retain?
16. A forward fails after one output token. Which costs are known, which planned attempts are missing, and why is restarting not automatically exact resume?
17. Why is any-correct availability monotonic on nested prefixes while majority-selected correctness need not be?
18. Explain how eight independently sampled duplicate strings differ from eight copies of one response record. What changes when each distinct answer receives one vote?
19. A generated three-action candidate exceeds a token budget with only two actions remaining. State the accepted pool, charged work and rejected-boundary rule without peeking ahead.
20. Does a higher mean sequence log probability imply a more correct answer? Which additional measurements are needed before claiming inference-time compute helps?
21. Export the authored response panel as an evaluation card. Which fields can the replay support, which remain unknown, and why would renaming the checkpoint to Qwen invalidate the claim?
22. In the real Base/short-SFT replay, a response earns a phrase-rubric pass but repeats artifacts and ends mid-sentence. Which measurements remain valid, what should independent review add, and why must the original score not be silently replaced?
23. Two arms each have four continuations per opening and two raters. Which objects should be blinded, retained and paired? Run the empty-rating control and explain why neither a missing review nor an absent checkpoint should become a score.

[Worked solutions](../solutions/07-evaluation-is-a-contract.md) give reasoning, implementation and evidence limits. The next chapter turns behavioral requirements into training examples. Its data engineering choices must preserve this evaluation contract, because a well-executed optimizer can faithfully learn a badly specified interface.
