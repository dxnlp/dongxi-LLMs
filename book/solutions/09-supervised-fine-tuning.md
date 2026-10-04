# Worked solutions — Supervised Fine-Tuning

These answers accompany [Chapter 9](../chapters/09-supervised-fine-tuning.md) and the four Days 12–14 notebooks. The executed evidence concerns CPU mechanisms and seen symbolic requests; real Qwen results remain proposed.

## 1. Conditional sequence NLL

The assistant response factorizes causally:

$$
p_\theta(a\mid c)=\prod_t p_\theta(a_t\mid c,a_{<t}).
$$

Apply negative natural logs to get a sum of token surprise. Sum across demonstrations and divide by the total supervised count for a token-mean objective. Include the intended ending in the response targets. The logarithm does not choose the most likely token; it scores the demonstrated token's probability.

## 2. Shapes

Input IDs and full labels are $[B,T]$; logits are $[B,T,V]$; hidden states are $[B,T,D]$; the output weight is $[V,D]$. Ownership and padding masks are $[B,T]$. After one shift, logits are $[B,T-1,V]$ and target labels are $[B,T-1]$.

Flattening for cross-entropy produces $[B(T-1),V]$ logits and $[B(T-1)]$ labels. This preserves the vocabulary axis. A tensor with correct dimensions can still have a wrong shift, so print position/target identities in addition to shapes.

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

## Notebook pathway

1. [Objective and gradient paths](../../notebooks/day-12/01_sft_objective_and_gradient_paths.ipynb).
2. [Accumulation and checkpoint identity](../../notebooks/day-12/02_accumulation_and_checkpoint_identity.ipynb).
3. [Tiny assistant training](../../notebooks/day-13/01_tiny_assistant_training.ipynb).
4. [Full tuning, LoRA and defense](../../notebooks/day-14/01_full_sft_lora_and_recipe_defense.ipynb).
