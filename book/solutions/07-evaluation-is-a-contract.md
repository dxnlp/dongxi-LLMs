# Worked solutions — Evaluation Is a Contract

Read [Chapter 7](../chapters/07-evaluation-is-a-contract.md) first. The three Day 10 notebooks supply runnable reference calculations next to each prediction. These answers emphasize the claim supported by a measurement.

## 1. Operational story coherence

A defensible claim is: “On the frozen set of 60 original story openings, greedy continuations of at most 192 tokens have fewer character-identity contradictions than the baseline.” Preserve the prompt identifiers and split hash. Two blinded readers assign the prewritten 0–2 identity rubric; retain disagreement and use adjudication only according to a declared rule. Separately score causal continuity, repetition and endings. An improvement on identity does not imply every dimension improved.

Choose the item or source group as the sampling unit. If six prompts are variants of the same underlying plot, a bootstrap should resample plots rather than treating all variants as independent. The test compares behavior under this decoding budget; it does not establish unlimited-length storytelling.

## 2. Normalization policy

For an integer-answer task, a numeric parser may reasonably consider “1.0” equal to “1”. Define whether decimals, scientific notation, units and multiple answers are accepted. If the instruction demands exactly one integer token, the strings may instead differ on format while sharing numeric meaning.

Report separate answer and format scores rather than deleting formatting requirements after seeing failures. The reference normalizer intentionally preserves punctuation; its three-string example yields true, false, false. That verifies its chosen policy, not universal language equivalence.

## 3. Derive pass@3

There are $\binom{10}{3}=120$ equally sized three-candidate subsets. With two correct candidates, eight are incorrect, and $\binom{8}{3}=56$ subsets contain only incorrect answers. Therefore

$$
\widehat{\mathrm{pass@}3}=1-\frac{56}{120}=0.533333\ldots.
$$

The fixture tests the product implementation against combinatorial counts for many small $n,c,k$ combinations, including zero successes and all-success cases. The event is at least one verified success; it does not identify which candidate a user should receive.

## 4. Duplicate draws

Generating once and storing the same string ten times does not provide ten independent opportunities. Conditional on that one draw, every copy has the same outcome. The finite-pool combinatorics still describe subsets of the recorded pool, but interpreting the result as fresh independent sampling is unsupported.

Record sampling settings and candidate identities. Deduplication alone is not enough to establish independence: genuinely independent draws can coincidentally produce the same output. Independence concerns the generation process, not merely distinct strings.

## 5. Two kinds of confidence

Token confidence is a model distribution over vocabulary candidates. It is conditional on the current context and learned training distribution. It does not guarantee answer truth.

A statistical score interval describes uncertainty in an estimate over a declared input population and sampling model. Ten successes out of ten leave a nontrivial Wilson interval. Neither quantity is equivalent to a model saying “I am certain.” Verbal confidence requires its own calibration evaluation.

## 6. Paired comparison

Use the same item index for A and B, compute $d_j=b_j-a_j$, and bootstrap indices once for both scores. Shared task difficulty remains paired. Independent resampling breaks that correlation and can obscure a precise comparison.

The notebook also duplicates all items ten times. The naive binomial interval narrows, although the source information is unchanged. That deliberate failure illustrates why the unit of independence belongs in the contract. A percentile interval that crosses zero means the sign is unresolved at the chosen precision, not that the models are equivalent.

## 7. Slice report

Publish the total score and a table with task family, count, score, interval and paired change. Freeze the slice definitions before evaluating candidates. Include a rare-format slice even when its denominator is small; show the broad interval rather than hiding it.

The purpose-built forty-item fixture gives model B a better aggregate but a worse rare-format score. This verifies the accounting and shows how an aggregate conceals a regression. It supplies no measured model capability.

## 8. Exact contamination audit

Normalized prompt-plus-reference hashes detect the deliberately inserted cross-split duplicate. Grouping those two records repairs that particular exact leak.

The procedure does not detect every paraphrase, translated question, shared document or leaked solution in a reasoning trace. Preserve provenance, run near-duplicate analysis, and document what was inspected. “No exact collisions found” is a narrower and more accurate statement than “the test is uncontaminated.”

## 9. Repeatedly used tests

Once examples inform learning-rate selection, prompt engineering, parser edits or checkpoint choice, they contribute to development. Rename the split and freeze a new final suite, or explicitly describe the evaluation as exploratory. Calling a repeatedly consulted set “publication test” does not restore independence.

If no new data is available, report that limitation and avoid a final-test claim. Cross-validation can support some comparisons, but requires nesting the selection process appropriately.

## 10. Stopping rule

Before inspecting candidate outputs, declare the item count, review rubric, generation budget, seed policy and minimum useful effect. Stop after all fixed items are scored, or on an implementation failure. If uncertainty is too broad, a separately specified expansion can add independent items.

Do not stop when the first excellent example appears. Do not extend only the losing recipe's budget after reviewing its results. Changes remain legitimate experiments when their selection and new evidence are recorded.

## Notebook pathway

1. [Metrics and contracts](../../notebooks/day-10/01_metrics_and_contracts.ipynb): exact match and attempt budgets.
2. [Paired uncertainty](../../notebooks/day-10/02_paired_uncertainty.ipynb): intervals, aligned resampling and false independence.
3. [Slices and contamination](../../notebooks/day-10/03_slices_and_contamination.ipynb): aggregate regression and leakage repair.

Each lesson contains prediction, reference answer, runnable computation and a labeled data-derived plot.
