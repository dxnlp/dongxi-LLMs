# 7. Evaluation Is a Contract

A model continues a story without spelling errors, yet changes the main character's name halfway through. Another model has lower validation loss but repeatedly returns a paragraph when the user asks for one word. Which is better? The question has no answer until we name the work we want the model to do, the population of inputs, and the rules for judging its outputs.

Chapter 6 established evidence about prediction on TinyStories. Its completed run improved fixed-development NLL, while inspected generations still contained repetition and inconsistencies. This chapter makes the next decision possible: define behavior precisely enough to compare before choosing another training intervention. Day 10 supplies three executable companions: [metrics](../../notebooks/day-10/01_metrics_and_contracts.ipynb), [uncertainty](../../notebooks/day-10/02_paired_uncertainty.ipynb), and [slices and leakage](../../notebooks/day-10/03_slices_and_contamination.ipynb). The [lab guide](../labs/07-evaluation-is-a-contract.md) and [worked answers](../solutions/07-evaluation-is-a-contract.md) extend the argument.

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

The CPU route has three stages. First inspect accepted strings and verify metric edge cases. Then compare aligned fixture models with paired uncertainty. Finally inspect slices and contamination. Each notebook puts an explanation and runnable reference answer immediately after the prediction.

For the real model route, the [experiment specification](../../experiments/specs/2026-10-04-evaluation-and-sft-course.md) declares a fixed development suite and a separately frozen publication suite. The initial baseline run should include exact revision identifiers and raw completions; it remains a proposed Spark run. The executed CPU report supplies evidence only for the metric, masking and optimization mechanisms.

Before training, answer: what observation would cause us to reject the recipe? A decline in held-out format validity might be unacceptable even if response NLL improves. A confidence interval that is too broad might justify collecting more evaluation items. A parser disagreement might justify improving the instrument before changing model weights.

## 7.10 Exercises and transition

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

[Worked solutions](../solutions/07-evaluation-is-a-contract.md) give reasoning, implementation and evidence limits. The next chapter turns behavioral requirements into training examples. Its data engineering choices must preserve this evaluation contract, because a well-executed optimizer can faithfully learn a badly specified interface.
