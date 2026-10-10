# Worked solutions — Supervised Fine-Tuning

These answers accompany [Chapter 9](../chapters/09-supervised-fine-tuning.md) and the four Days 12–14 notebooks. The CPU mechanisms, seen symbolic requests and native 0.6B held-out-value comparison are measured. The proposed 1.7B transfer remains unexecuted.

## 1. Conditional sequence NLL

The assistant response factorizes causally:

$$
p_\theta(y\mid x)=\prod_t p_\theta(y_t\mid x,y_{<t}).
$$

Apply negative natural logs to get a sum of token surprise. Sum across demonstrations and divide by the total supervised count for a token-mean objective. Include the intended ending in the response targets. The logarithm does not choose the most likely token; it scores the demonstrated token's probability.

## 2. Shapes

Input IDs and full labels are $[B,n]$; logits are $[B,n,V]$; hidden states are $[B,n,D]$; the output weight is $[V,D]$. Ownership and padding masks are $[B,n]$. After one shift, logits are $[B,n-1,V]$ and target labels are $[B,n-1]$.

Flattening for cross-entropy produces $[B(n-1),V]$ logits and $[B(n-1)]$ labels. This preserves the vocabulary axis. A tensor with correct dimensions can still have a wrong shift, so print position/target identities in addition to shapes.

## 3. Prompt gradient path

The answer logit depends on its hidden state. Attention at that answer position reads earlier prompt representations, so the answer derivative reaches those representations and their embeddings. The user prediction's own logit gradient can be zero while its state receives a later path's derivative.

For a tied embedding table, output projection supplies an additional gradient path. An unused-as-input row can still update because it is a vocabulary candidate. The Day 12 notebook verifies that case numerically.

## 4. Unequal answer lengths

With two responses of lengths 2 and 6, the token mean gives the longer response three times the total weight, assuming equal average loss. The example mean gives each response half the objective weight. Neither is a harmless formatting choice.

Specify the objective before recipe comparisons. If a model becomes verbose, examine how many explanation tokens it receives, not only how many explanation examples.

## 5. Accumulation test

Start two identical dropout-free models. Compute a joint batch gradient for all four fixture conversations. Compute split gradients using one-example and three-example microbatches, scaling every summed microbatch loss by the same total target count.

The gradients agree within numerical tolerance. A deliberately wrong average of microbatch means produces a measured mismatch. This test checks weighting and step placement; a same-size microbatch test can miss the error because equal counts conceal it.

## 6. Restart boundary

A weights-only restore reproduces the immediate forward pass if the architecture and backend match. AdamW's next update also depends on moment state and step counters. Copying optimizer state restores those local dependencies.

The notebook copies weights plus optimizer state after one update, applies the same next batch, and obtains matching parameters. A full experiment must also preserve scheduler, RNG, data cursor, tokenizer and template. Equality in this CPU fixture does not establish cross-hardware bitwise reproducibility.

## 7. LoRA count and initial gradient

A $d_{\mathrm{out}}\times d_{\mathrm{in}}$ matrix has $d_{\mathrm{out}}d_{\mathrm{in}}$ scalars. Rank-$r$ LoRA trains $r(d_{\mathrm{in}}+d_{\mathrm{out}})$ adapter scalars. For a $5\times6$ teaching matrix at rank 2, that is 22 adapter parameters alongside 30 frozen base scalars.

With $B=0$ and random $A$, the initial effective update is zero. The derivative of $A$ includes $B$ and is zero on the first step; the derivative of $B$ can be nonzero because $A$ is present. The tested merged matrix gives the same linear output as the separate adapter path.

## 8. Memory accounting

LoRA reduces trainable gradients and optimizer moments, but the frozen base weights are still stored. Activation tensors are needed to compute adapter gradients. Quantization is an additional independent change, not implied by low rank.

Report actual peak allocation and sampled host memory during a profiled run. A scalar parameter count alone excludes backend workspaces, attention temporaries and allocator overhead.

## 9. Comparison budget

The [pinned tokenizer sizing case](../../experiments/reports/2026-10-05-base-tokenizer-sizing.md)
illustrates why training exposure and whole-run target allowance differ. The
selected 80 training examples supply 455 supervised targets, while two complete
development panels supply 360 each. The whole-run likelihood allowance is 1,175;
the optimizer's training exposure is still 455. Include message-end and separator
targets according to the exact mask, not a word count. Reserve generation
separately. These counts do not measure parameter updates or establish learning.

Choose equal supervised-token exposure as a primary budget and freeze data order, evaluation contract and seed policy. Report optimizer updates, runtime, memory and trainable count. A matched learning rate supplies a fixed-recipe control; it may be suboptimal for either method.

If also tuning each method, declare equal search budgets and a final confirmation split. The executed microscopic comparison is fixed-recipe: 100 updates, seed 1212, AdamW learning rate 0.015, no weight decay, rank-four Q/V adapters.

## 10. Stopping regression

Check preprocessing first: was the assistant end target retained? Then inspect generation settings: does the stop ID match the training marker? Inspect raw IDs, because decoded text can hide special markers. Compare the same token cap across checkpoints.

Improved answer likelihood does not establish reliable free-generation termination. Count natural endings separately from token-limit stops. If endings regress, a longer generation limit alone can conceal rather than repair the learned behavior.

## 11. Starting checkpoint identity

Fine-tuning Qwen3-0.6B starts from a post-trained model. That tests continued adaptation. Starting from Qwen3-0.6B-Base tests a first instruction-stage transition under the stated data.

The distinction affects causal interpretation even if both runs improve. Exact checkpoint revisions prevent an upstream update from silently changing the starting point. The lab pins official base revisions retrieved on 2026-10-04.

## 12. Larger-run gate

Before a 1.7B transfer check, require finite loss and gradients, successful checkpoint recovery, host-memory reserve compliance, correct target alignment, and improvements on frozen held-out task/format slices. Define tolerated regression bounds in advance.

A healthy runtime is necessary but does not establish behavior. If the smaller run improves only seen values, revise the data or claim before scaling. A larger model is not a substitute for an instrument that measures the intended outcome.

## Actual microscopic evidence

The source-hashed [CPU report](../../experiments/reports/2026-10-04-evaluation-and-sft-course.md) records full-SFT answer NLL decreasing from 3.101167 to 0.000555 and LoRA from 3.101167 to 2.802457. On the four seen requests, answer-plus-END sequence exact match was respectively 1.00 and 0.75. These are measurements from the fixed microscopic recipe.

The adapter model stores 6,504 total parameters, of which 384 train; the full model stores/trains 6,120. Do not turn this outcome into a broad claim that full SFT always outperforms LoRA.

## Distillation forward-reading extension

Exercises 13–16 use Chapter 15's response-level distillation argument and
[Day 26 notebook](../../notebooks/day-26/03_response_level_distillation.ipynb).
The exercise IDs and retained measured outcomes below remain unchanged.

## 13. Response supervision versus a teacher distribution

An actual emitted response supplies target IDs. Sequence NLL pushes the student toward each observed target under the demonstrated prefix. It does not reveal the teacher's probabilities for alternative tokens. Forward KL needs the teacher distribution, an aligned vocabulary and a declared temperature; the existing three-logit $\tau^2$ microscope remains a separate mechanism lesson.

A teacher can emit a wrong but well-formed response. Response NLL still imitates that response unless a separately declared review/filter intervention changes the dataset. It cannot infer semantic truth simply because the example came from a larger model.

## 14. Eligibility and teacher coverage

The frozen adapter selects the first full trace with compatible provenance/interface, strict form, natural EOS and no error. It does not inspect gold totals or answers. Best-correct selection would instead condition training on a verifier and change the experiment. All unselected attempts and costs stay visible.

If one of the six training sources has no eligible trace, both student arms are blocked for that campaign, without replacement by a gold demonstration. The actual three-seed run had complete coverage and 18 correct selected traces; the missing-coverage and wrong-content cases are separately tested. The notebook's authored wrong-step/right-answer control remains eligible and is not falsely described as a neural sample.

## 15. Three distinct trace observations

A wrong step with a correct answer satisfies final correctness but fails joint correctness. A valid step with a wrong answer proves that the printed intermediate statement alone was insufficient to obtain a correct final. An answer-only output supplies no printed step, so its step correctness is null rather than true or false.

The actual teacher evaluation records contain 80 and 24 cases in the first two cells; the complete-response student contains 26 and 15. These counts preserve observed errors instead of rescuing them with an answer-only score. Even a valid step and correct final do not establish that the model used that step internally.

## 16. Distillation work accounting

The two student arms use the same six parent responses, same initial weights and 80 updates. Complete-response supervision contributes 30 targets per update; the answer-only transformation contributes 18. Over a campaign that is 2,400 versus 1,440 supervised targets, with 4,320 versus 3,360 padded forward positions. The comparison is matched in update count, not compute or token exposure.

Across three campaigns, the teacher incurs 360 fitting updates and 144 data-generation attempts totaling 720 actions and 4,320 uncached prefix positions. Student fitting and 1,728 common evaluation attempts are additional work. Parameter counts, generation prefixes and local wall timers must not be conflated with optimized deployment latency. The [actual report](../../experiments/reports/2026-10-04-response-distillation.md) provides each boundary and retains the negative held-out slices.

## Evidence-reading extension

Exercise 17 keeps the conceptual durable-boundary question. Operational snapshot
and merge details are linked from [Appendix D](../appendices/d-reproduction-and-environments.md#supervised-fine-tuning-state-and-adapter-exports).

## 17. Committed state versus a journal attempt

If the update 3 snapshot did not commit, update 2 is the last durable training
state. Restore its policy, Adam moments, applicable RNG, cursor and cumulative
counts, then retry update 3. Preserve the failed invocation's log as an attempt;
do not add that attempted update to the restored trajectory's completed count.

If update 3 did commit before metric append failed, restore update 3 and recover
its numerical metric from the snapshot-carried history. Do not apply update 3
again. The resumed invocation has its own output and identity, but the scientific
recipe and cumulative work remain bound. A visible marker after a failed fsync
requires verification and a retained durability failure, not an invented success.
Neither local branch proves CUDA/BF16 or pretrained recovery.

## 18. Lower NLL, but no exact or naturally ended responses

The native 20-update profile's development NLL fell from 4.061965 to 1.100189,
while both generation panels remained 0/8 exact answers and 0/8 message-end stops.
That is successful execution and measured likelihood improvement, not successful
assistant behavior. Teacher forcing evaluates each gold target under the gold
prefix; greedy free generation conditions on its own choices. A target can
gain probability without becoming the largest logit. Once another token wins,
the continuation can enter contexts not represented by the demonstration.

The end marker was supervised, but did not win in these sampled trajectories.
All responses exhausted 64 tokens, so termination failure remains explicit.
Do not repair the generated text, remove awkward tokens or select a nicer decode
to turn a fixed negative result into a positive one. Use the already declared
longer comparison with unchanged evaluation, and retain the short profile as
resource evidence. The [raw report](../../experiments/reports/2026-10-05-native-base-profile.md)
separates the 455 training targets from 720 evaluation target presentations.

## 19. A controlled recipe is not a universal method ranking

The actual fixed full/LoRA400 intervention establishes different local learning
outcomes under common data, seed, target exposure and evaluation. Full tuning
learned the eight observed instruction/ending behaviors; the adapter improved
teacher-forced likelihood but failed whole-response completion. Safe exits and
five committed snapshots establish execution, not quality.

The separate [actual publication comparison](../../experiments/reports/native-assistant-comparison-20261005-run-02/README.md)
now observes 120/120 strict whole decoded answers and 120/120 natural message-end
stops for full400, versus 0/120 answers/stops and 120 caps for both unchanged Base
and explicitly merged LoRA400. There are 40 copy,40 reverse and 40 extract items
over 40 held-out lexical-value groups; each family shows the same contrast.
Generation uses common prompt token IDs, original template, greedy seed 1010,
context 512, cap 64 and BF16. Every actual producer exits 0 without response errors.

Generic parser correctness agrees here, but its case-folded text rule is not
the strict case-sensitive rule. Every response passes `format_policy=any`,
including the capped incorrect responses: format validity is therefore not
evidence that an answer is correct or naturally ended. Neither rule extracts a
favorable prefix. Raw generated stop tokens are retained; only the terminal
special token is excluded from decoded scoring content.

The 2,000-draw seed 1010 paired bootstrap samples the 40 original source groups,
retaining their 120 aligned greedy item/sample IDs. Full−Base is+100 percentage
points with interval[+100,+100]; LoRA−Base is 0[0,0]; LoRA−full is−100[−100,−100].
Every group has constant outcomes, so these descriptive intervals are
degenerate. They do not measure variation over training seeds, learning rates,
new task templates or genuinely different assistant domains. The physical
receipts stay distinct; only the common logical+actual encoded analysis
contract is shared for resampling.

The trainable subspaces differ. LoRA may require another learning rate, rank or
target-module choice; the one common recipe does not optimize either method.
One seed provides no between-run variability estimate. Shared synthetic task
templates limit both the eight development prompts and the separately measured
held-out-value publication panel; neither establishes broad instruction
generalization. Development NLL falls from 4.061965 to 0.000276425 for full and
to 1.320048 for LoRA, but those teacher-forced 60-item observations do not replace
the generated 120-item results. The adapter's lower NLL and failed whole answers
are compatible measurements, not a contradiction. The [actual comparison report](../../experiments/reports/2026-10-05-native-sft400-comparison.md)
also distinguishes labels, padded positions, measured seconds and overlapping
snapshot cost counters.

For example, full400 emits 760 tokens and processes 29,720 full-prefix forward
positions; Base and LoRA each emit 7,680 and process 515,840. The difference partly
follows from correct short stopping, not an optimized inference-engine speed
advantage. Summed response-attempt times are 27.540/159.574/157.873 seconds for
full/Base/LoRA and have a different boundary from external supervision. FP32
names the verified LoRA merge/storage; all publication generation is BF16.
A passed merge/reload identity check does not imply favorable task scores or
equivalence to an unmerged BF16 adapter path.

## Notebook pathway

1. [Objective and gradient paths](../../notebooks/day-12/01_sft_objective_and_gradient_paths.ipynb).
2. [Accumulation and checkpoint identity](../../notebooks/day-12/02_accumulation_and_checkpoint_identity.ipynb).
3. [Tiny assistant training](../../notebooks/day-13/01_tiny_assistant_training.ipynb).
4. [Full tuning, LoRA and defense](../../notebooks/day-14/01_full_sft_lora_and_recipe_defense.ipynb).
5. [Complete-response distillation, masks and printed-step checks](../../notebooks/day-26/03_response_level_distillation.ipynb), a Chapter 15 extension bridged here through the SFT objective.
