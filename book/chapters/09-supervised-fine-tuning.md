# 9. Supervised Fine-Tuning

A pretrained decoder can continue text without reliably answering a request. Supervised fine-tuning supplies demonstrations of the conditional behavior we want: given this conversation, produce this response and stop at this boundary. The model remains a next-token predictor. We change the distribution of contexts, the targets receiving loss, and often which parameters can move.

This chapter spans Days 12–14. It combines the [objective and gradient notebook](../../notebooks/day-12/01_sft_objective_and_gradient_paths.ipynb), [accumulation and restart notebook](../../notebooks/day-12/02_accumulation_and_checkpoint_identity.ipynb), [bounded training notebook](../../notebooks/day-13/01_tiny_assistant_training.ipynb), and [full versus LoRA notebook](../../notebooks/day-14/01_full_sft_lora_and_recipe_defense.ipynb). The [lab guide](../labs/09-supervised-fine-tuning.md) provides the run order; [worked solutions](../solutions/09-supervised-fine-tuning.md) explain the deep questions.

## What you should be able to explain

- Derive assistant-only token NLL and follow its gradients through the decoder.
- Implement one correct update and token-weighted gradient accumulation.
- Compare full tuning and LoRA while separating likelihood, whole-answer correctness and stopping.

**Prerequisites:** causal decoding and cross-entropy from Chapters 3–5, the evaluation contract from Chapter 7, and role ownership/label alignment from Chapter 8.

## 9.1 What transfers from pretraining?

Pretraining establishes representations and continuation patterns across a broad corpus. SFT supplies a narrower distribution of demonstrations. It can make an existing ability accessible through an interface, strengthen a task behavior, teach format compliance or introduce new factual associations. These are distinct outcomes and require distinct evaluation slices.

A model that already knows a fact may answer it correctly after learning when to give a direct response. That does not prove SFT created the fact. Conversely, a model can memorize a new answer without learning a transferable instruction. Compare held-out values, unseen phrasings and unrelated regression tasks.

Model naming matters. The official [Qwen3-0.6B-Base card](https://huggingface.co/Qwen/Qwen3-0.6B-Base) describes a base checkpoint; [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) is a separately released post-trained model. Fine-tuning the latter is useful continued adaptation, but does not test a first base-to-assistant transition. Record exact revisions so the comparison cannot change when a repository updates.

Our CPU model begins randomly initialized. Its task is deliberately tiny: map four symbolic copy requests to a word and END. Its results verify optimization and gradient mechanics. A real base-to-assistant claim belongs to the separately specified Spark experiment.

## 9.2 Derive the supervised objective

For demonstration $j$, let $x_j$ be its conversation context and $y_j=(y_{j,1},\ldots,y_{j,n_j})$ its assistant sequence, including the intended ending. The causal model gives

$$
p_\theta(y_j\mid x_j)=\prod_{t=1}^{n_j}
p_\theta(y_{j,t}\mid x_j,y_{j,<t}).
$$

Taking negative natural logs turns products into sums:

$$
L_j(\theta)=-\sum_{t=1}^{n_j}
\log p_\theta(y_{j,t}\mid x_j,y_{j,<t}).
$$

The common token-mean objective divides the total sum by the number of supervised targets, $N=\sum_jn_j$:

$$
L(\theta)=\frac{1}{N}\sum_jL_j(\theta).
$$

For a padded full transcript, use ownership mask $m_{b,t}$ and validity mask $v_{b,t}$. Logits $Z$ have shape $[B,n,V]$ and labels have shape $[B,n]$. After one causal shift, the logit gradient is

$$
\frac{\partial L}{\partial z_{b,t,i}}=
\frac{m_{b,t+1}v_{b,t+1}}{N}
\left(p_{b,t,i}-\mathbf{1}\{i=x_{b,t+1}\}\right).
$$

At a directly ignored prediction position, this gradient is zero. At an answer prediction, the observed target gets upward pressure through gradient descent and competing logits get downward pressure. The hidden state, output head and earlier context receive gradients according to the chain rule. The equation gives a derivative with respect to logits; actual parameter updates depend on the entire computation and optimizer.

An example-mean objective instead averages $L_j/n_j$ across examples. This weights each conversation equally even when answer lengths differ. Choose intentionally and retain the denominator in logs; the two objectives need not prefer the same parameters.

## 9.3 Gradients reach the shared decoder

Let the final hidden state be $h_t\in\mathbb{R}^D$ and output matrix $W_{\mathrm{out}}\in\mathbb{R}^{V\times D}$. With row-vector batch notation, $z_t=h_tW_{\mathrm{out}}^\top$. The loss derivative sends pressure to both sides:

$$
\frac{\partial L}{\partial h_t}
=\frac{\partial L}{\partial z_t}W_{\mathrm{out}},
\qquad
\frac{\partial L}{\partial W_{\mathrm{out}}}
=\left(\frac{\partial L}{\partial z_t}\right)^\top h_t.
$$

Backward propagation traverses final normalization, residual additions, MLP transformations and attention. Attention reaches earlier prompt states through the value path and changes routing through the Q/K paths. It eventually reaches token embeddings unless those parameters are frozen.

The gradient notebook displays parameter-family norms from the actual tiny model. A nonzero Q gradient proves that a local training signal reaches that projection for this batch. It does not prove useful specialization, a large update or general capability. A zero gradient may arise from masking, symmetry, frozen parameters or a saturated/unused path; its cause should be investigated in the graph.

Tied input/output embeddings deserve care. A row unused as an input can still receive an output-head gradient because the softmax considers the whole vocabulary. This is why “only looked-up embedding rows update” is true for an isolated lookup experiment but incomplete for a tied language model.

## 9.4 A readable update loop

The transparent implementation separates loss sum from target count. Its main loop performs: clear old gradients, run the model, align labels once, compute summed answer NLL, divide by the correct denominator, backpropagate, check finiteness, clip, and step the optimizer.

For a small experiment, AdamW is a reasonable declared choice. Learning rate controls the scale of updates; clipping bounds the accumulated gradient norm before the step. Clipping is not a substitute for finite-value checks. Weight decay acts through the optimizer and should be reported separately from loss terms.

For a production BF16 path, autocast changes eligible forward operations. It does not imply every parameter, optimizer state or reduction uses two bytes. The runner loads BF16 weights explicitly, uses SDPA, checks the hardware, and records the actual choices. It does not claim a throughput estimate before profiling.

Teacher forcing supplies the demonstrated answer prefix during training. During free generation, the model consumes its own previous outputs. Low answer NLL can coexist with drifting generations, as Chapter 6 showed. Evaluation therefore needs both teacher-forced likelihood and free-response task behavior.


### From the objective to one update

This is the CPU reference loop from [the SFT module](../../src/dongxi_llms/sft_lab.py).
The module's `token_loss_sum` shifts logits and labels once and returns summed
cross-entropy plus the valid target count. Here `batch` is the audited symbolic
fixture, and `parameters` contains only the parameters allowed to train:

```python
optimizer.zero_grad(set_to_none=True)
total, count = token_loss_sum(model(batch["input_ids"]), batch["labels"])
loss = total / count
loss.backward()
norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
if not torch.isfinite(loss) or not torch.isfinite(norm):
    raise RuntimeError("Nonfinite optimization state")
optimizer.step()
```

The denominator is supervised targets, including END, rather than padded input
positions. The finite check precedes `step`; this snippet demonstrates the
tiny CPU path, while the real-model runner separately supplies attention masks,
precision and checkpointing controls.

## 9.5 Accumulation must preserve weighting

A logical batch can be split into microbatches when memory is limited. Suppose microbatch $r$ has summed loss $S_r$ and supervised count $N_r$. The desired accumulated objective is

$$
L=\frac{\sum_rS_r}{\sum_rN_r}.
$$

Backward each $S_r/\sum_qN_q$, then perform one optimizer step. Averaging microbatch means instead gives $\frac{1}{R}\sum_rS_r/N_r$, which overweights short-answer microbatches when counts differ.

The accumulation notebook compares gradients from a joint batch with token-weighted split batches. It also deliberately averages means to expose the failure. Use a shared initialization, no stochastic dropout and identical context masks so the only difference is weighting. Floating-point reduction order may introduce tiny numerical differences; define a justified tolerance rather than demanding impossible bitwise equality.

An incomplete final accumulation window needs its own denominator and update policy. Silently dividing it by a fixed full-window count weakens its gradient. The Spark runner materializes the bounded window and counts its labels before backpropagation.


### One denominator across the accumulation window

The canonical [accumulation helper](../../src/dongxi_llms/sft_lab.py) uses:

```python
model.zero_grad(set_to_none=True)
count = sum(int((b["labels"][:, 1:] != -100).sum()) for b in batches)
if not count:
    raise ValueError("Empty accumulation window")
for batch in batches:
    total, _ = token_loss_sum(model(batch["input_ids"]), batch["labels"])
    (total / count).backward()
```

Perform clipping and one optimizer step only after the window. The notebook
compares these gradients with a joint batch, then breaks the denominator by
averaging microbatch means. Shared initial weights and disabled dropout isolate
the weighting change; numerical equality uses the declared floating-point tolerance.

### Count labels before setting a budget

The [pinned tokenizer measurement](../../experiments/reports/2026-10-05-base-tokenizer-sizing.md)
makes this distinction concrete without loading a model. The proposed 20-update
profile selects 80 original examples. Their full transcripts contain 3,154 input
positions but only 455 shifted assistant targets. The mask includes each answer's
message-end marker and trailing template newline; counting answer words would
give the wrong denominator. The maximum context is a ceiling, not the number of
targets actually present.

Both baseline and final evaluation score all 60 development examples, adding
360 targets each. A whole-run work allowance therefore needs 1,175 target
presentations, while training exposure remains 455. Evaluation does not add
optimizer supervision. Generation has a separate allowance: sixteen proposed
attempts of at most 64 new tokens, not another teacher-forced target count.
These are measured token encodings and schedule requirements, not executed
updates, sampled completions, memory-fit evidence or a learned assistant.

### A real profile can improve NLL without producing a correct answer

The subsequent [native Spark profile](../../experiments/reports/2026-10-05-native-base-profile.md)
executed this exact schedule. All 20 updates had finite loss and gradient norms;
the process exited 0 in 61.636 seconds and the external sampled memory minimum
was 105.10455GiB, above the 25GiB reserve. Both committed snapshots fit their
declared envelopes. This is actual pinned pretrained BF16/SDPA evidence, not the
tiny random model's result extrapolated to Qwen.

Development answer NLL fell from 4.061965 to 1.100189. Yet both fixed eight-prompt
generation panels produced zero exact answers and zero message-end stops; every
continuation hit 64 tokens. This is not contradictory. NLL asks how much
probability the model assigns to each demonstrated token under the demonstrated
prefix. Generation asks which path the model follows under its own selected
tokens, including whether it chooses the ending. Increasing the right token's
probability need not make it the winner, and correct-prefix likelihood need not
protect a wrong-prefix continuation.

The labels already included the message-end marker and template newline, so
“put END in the loss” is necessary supervision, not proof of learned stopping.
Keep likelihood, task correctness and natural termination as separate outcomes.
The short profile verifies the runtime and supplies a resource measurement; its
weights do not replace a fresh initialization in the predeclared full/LoRA
comparison. No publication-test result or broad assistant improvement follows.

## 9.6 Checkpoint identity is wider than weights

Saving weights permits a forward reconstruction. Resuming optimization also needs optimizer state, scheduler state if present, update index, data order, random states, template, tokenizer, mask policy and configuration. A correct restart should reproduce the next update under the same deterministic assumptions.

The CPU demo verifies state-dictionary forward equality. The second Day 12 notebook adds an optimizer-state continuation check: copy the model and AdamW state, feed the same next batch, and compare the resulting parameters. This verifies the stated local boundary; it does not establish bitwise reproducibility across different hardware or backend versions.

### A saved file is not yet a committed training boundary

Imagine that update 2 is durable, update 3 changes the model in memory, and a
crash occurs before its snapshot commits. Recovery must start from update 2 and
retry update 3. A log saying “update 3 finished” does not supply the missing Adam
moments or establish a durable update. Conversely, a committed update 3 snapshot
can survive a crash before its metric reaches the journal. Carry the numerical
metric history inside the snapshot, so that a new invocation can reconstruct
the record without silently losing or counting the update twice.

Keep scientific identity separate from operational choices:

| Preserved across a same-recipe restart | Separately declared for the new invocation |
|---|---|
| Actual input/source/lock bytes, token meanings, template and ending rules | New output directory, command and invocation ID |
| Objective, optimizer, accumulation, precision and schedule horizon | Deadline, memory reserve and snapshot-size envelope |
| Policy, Adam moments, RNG, fixed data order/cursor and cumulative work | Journal attempts and failed operations after the last durable boundary |

Moving identical input bytes is not changing their scientific identity. Changing
the schedule horizon, tokenizer or loss denominator is. It requires a declared
new experiment or migration, not a relaxed equality check. Cumulative targets
also must not reset merely because the resumed process has a new directory.

The [actual native full/LoRA replay](../../experiments/reports/2026-10-05-native-sft-replay.md)
now tests this on the pinned pretrained model. Both fresh update 10→20 processes
match all ten numerical updates, serialized tensor entries and final generated
token/stop records. The physical journals count 30 executed updates and 684
training targets, although each final numerical trajectory ends at 20 updates
and 455 cumulative targets. Keep those two clocks separate. The earlier full
resume's conflict stop remains an additional failed invocation and wall cost.
This is same-Spark completed-boundary proof, not cross-machine reproducibility
or improved generation quality.

Operational snapshot validation and the separate replay gates are described in
[Appendix D](../appendices/d-reproduction-and-environments.md#supervised-fine-tuning-state-and-adapter-exports).

Checkpoint selection uses development criteria declared in advance. “Choose whichever sample looks best” encourages selection on noise. Save fixed intervals and compare the same prompt grid; retain regressions and stopping reasons.

## 9.7 Full tuning changes all allowed parameters

Full SFT allows the optimizer to update every trainable parameter. It offers flexible adaptation, but its optimizer state and gradients can dominate memory. A rough AdamW accounting for $P$ parameters distinguishes weights, gradients and two moment tensors. Additional master weights, activations, temporary buffers and allocator overhead depend on the implementation. Do not promise memory sufficiency from a parameter-only estimate.

Full tuning can alter useful broad behavior while improving the target interface. Evaluate general continuation, arithmetic, format compliance and concise-answer behavior with the same contract before and after. A narrowly successful recipe may be appropriate for a specialized system; its limitations should remain visible.

The microscopic full run uses fixed seed, a bounded update budget and four known requests. It learns the fixture if the observed report supports that conclusion. There is no unseen-language generalization claim, and no best-seed selection.

## 9.8 LoRA constrains an update

For a linear weight $W_0\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$, LoRA freezes $W_0$ and learns

$$
W=W_0+\frac{\alpha_{\mathrm{LoRA}}}{r}BA,
\qquad
A\in\mathbb{R}^{r\times d_{\mathrm{in}}},
\quad B\in\mathbb{R}^{d_{\mathrm{out}}\times r}.
$$

The trainable adapter count is $r(d_{\mathrm{in}}+d_{\mathrm{out}})$ rather than $d_{\mathrm{in}}d_{\mathrm{out}}$. The rank of the update is at most $r$. The base matrix remains present, so small trainable state does not mean the whole base model occupies little memory. Activations also remain necessary for adapter training.

The [LoRA paper](https://arxiv.org/abs/2106.09685) introduces this parameterization. Our implementation uses a random $A$ and zero $B$. Initially the adapter output is exactly zero. The first gradient of $A$ is zero because it is multiplied by $B$; $B$ can receive a nonzero gradient through the initialized $A$. This asymmetry explains why “all trainable parameters must have nonzero gradients on the first step” is a poor health check.

For row-batch input $X$, compute $XW_0^\top+\frac{\alpha_{\mathrm{LoRA}}}{r}(XA^\top)B^\top$. The notebook compares this with a merged matrix and verifies equality. The microscopic adapter targets Q and V projections only. This is a declared design, not a universal optimal target set.

## 9.9 Compare recipes under named constraints

A full-versus-LoRA comparison changes capacity, optimizer state and often preferred learning rate. Using the same learning rate is a useful mechanism control, but may not give each method its best recipe. Conversely, tuning LoRA extensively while giving full tuning one attempt is an unequal selection budget.

Report two comparisons when feasible: a fixed-recipe intervention and a separately declared tuned-recipe comparison. Match training data, token exposure, seed policy and evaluation contract. State whether compute, wall time, updates or supervised tokens define the budget; all four cannot usually be equal at once.

The Day 14 notebook runs both microscopic modes from the same base initialization and plots observed loss against optimizer updates. Its evidence concerns the four symbolic examples, not the general usefulness of LoRA. The Spark plan proposes full SFT versus LoRA at a frozen data/token budget, followed by a Qwen3-1.7B-Base transfer check only after the smaller recipe is healthy. The smaller fixed-recipe runs are now measured below; the larger transfer check remains unexecuted.

The short native acceptance runs now supply a narrower observation: full and
rank-8 Q/V LoRA both recover exactly, but neither answers the eight fixed
development prompts correctly after 20 updates. Full tuning has 596,049,920
trainable parameters versus LoRA's 1,146,880; observed CUDA peaks are 6.03GB and
2.06GB respectively. These are specific allocated-byte peaks, not total unified
memory or a universal ratio. The 400-update comparison is a separate fresh run.

That [fixed-recipe comparison](../../experiments/reports/2026-10-05-native-sft400-comparison.md)
has now completed. Both arms see 9,321 successful training labels and 63,378
processed training positions. Full tuning reaches development NLL 0.000276425
and answers/stops correctly on all eight fixed development prompts. LoRA
reaches 1.320048 but answers/stops correctly on none: all eight exhaust 64 tokens,
often after printing a plausible answer prefix. Both run safely and export their
declared artifact. This separates execution success, likelihood learning and
whole-response behavior within one measured experiment.

Equal exposure does not make the update spaces equal. Full tuning can change
the output and embedding layers directly; the constrained Q/V adapter can only
change its selected low-rank paths. The observed result is not a proof that this
capacity difference alone caused the gap, nor that every LoRA recipe would fail.
The learning rate, target modules, rank and single seed remain fixed. The eight
development prompts share task templates with training; they are not the
original 120-item publication panel or a broad assistant benchmark.

### Predict before comparing the generated answers

**Reader prediction:** both methods receive the same labels and both improve
development NLL. Must they produce equally correct, naturally ended answers?
Explain how each update space might change content and the probability of END
before reading the held-out results. This is a reader exercise, not a historical
prediction of the measured winner.

### Test held-out values without changing the answer rule

The subsequent [original publication comparison](../../experiments/reports/native-assistant-comparison-20261005-run-02/README.md)
contains 120 items: copy, reverse and extract for each of 40 held-out lexical-value
source groups. All three policies use identical actual prompt token IDs,
the original template, greedy seed 1010, context 512, cap 64 and BF16 generation.
Strict scoring compares the entire decoded answer case-sensitively, excluding
only a terminal special stop from the scoring text. Raw text and stop tokens
remain stored; a correct prefix followed by unwanted text is not repaired.

| Actual held-out-value observation | Base | Full400 | Merged LoRA400 |
|---|---:|---:|---:|
| Strict correct answers |0/120|120/120|0/120|
| Natural message-end stops |0/120|120/120|0/120|
| Responses reaching 64-token cap |120/120|0/120|120/120|
| Emitted tokens, including stops |7,680|760|7,680|
| Full-prefix forward input positions |515,840|29,720|515,840|


### Read one aligned response, including how it stopped

The selection rule is the first held-out item, `test-100-copy`, in each retained
panel. Its request is `Reply with exactly this word: item100`; the expected
whole answer is `item100`. The following are exact `response_text` records,
not a favorable prefix extracted for scoring:

**Base:** [max_tokens, 64 generated actions](../../experiments/reports/native-assistant-publication-20261005-base-run-01/generation/responses.jsonl).

```text
Reply with exactly this word: item101 ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇ ⚇
```

**Full400:** [turn_stop, 5 generated actions](../../experiments/reports/native-assistant-publication-20261005-full400-run-01/generation/responses.jsonl).

```text
item100
```

**Merged LoRA400:** [max_tokens, 64 generated actions](../../experiments/reports/native-assistant-publication-20261005-lora400-fp32-run-01/generation/responses.jsonl).

```text
item100 dólairement
Follow the requested output format exactly.完整热
.REACTuser
reply with exactly this word: item100 dólairement
.REACTassistant
item100 dólairement
.REACTuser
reply with exactly this word: item100 dólairement
.REACTassistant
item10
```

Full400 chooses the requested value and the natural turn-ending action.
LoRA starts with the requested value, then continues into unwanted content;
Base gives another value and continues. The correct prefix is insufficient
under the frozen whole-answer rule. All emitted stop IDs and capped text remain
in the raw records. This example explains the table; the entire panel supplies
the denominator and the result.

Each task family has 40/40 strict successes for full400 and 0/40 for the other
two policies; all producers completed with no response errors. The generic
case-folded parser gives the same correctness results here, but remains a
different scoring rule. Its unconstrained format check passes all 120 outputs
in every arm, including the capped wrong answers. Neither execution success
nor format validity establishes answer correctness.

Source-group resampling of the aligned 120 greedy IDs uses 2,000 draws and
seed 1010. The observed full-minus-Base strict gain is 100 percentage points;
all group draws return the same gain. A degenerate interval is not a promise
about unseen tasks: there are only three shared synthetic templates and one
training seed. This supports held-out-value interface learning under the
measured recipe, not general assistant capability or a universal method rank.
LoRA's development NLL improvement to 1.320048 remains a separate teacher-forced
observation, despite its 0/120 generated whole answers. The model's update
subspace, rank 8 Q/V targets and common learning rate were not retuned.

Full400 also performs less measured generation work because it ends correctly
after short answers. Summed response-attempt times are 27.540 seconds for full,
159.574 for Base and 157.873 for LoRA. These are uncached full-prefix loop times,
not optimized inference-engine benchmarks; input-position counts are not FLOPs.
Keep the populations, precision, stopping behavior and cost boundaries visible.

Merging is a separate numerical experiment: BF16 rounding can violate a
declared tolerance even when the algebra agrees. The published LoRA panel uses
an independently checked FP32 merged export loaded in BF16. That export check
does not establish equivalence to every unmerged BF16 adapter forward or task
success. [Appendix D](../appendices/d-reproduction-and-environments.md#supervised-fine-tuning-state-and-adapter-exports)
retains the failed BF16 check, distinct later FP32 checks and their evidence.

## 9.10 Interpret gains and regressions

![Measured full/LoRA training NLL and gradient norm against successful training-label exposure](../../experiments/reports/native-sft400-comparison-figures/learning-curves.png)

Both lines come from the actual fixed 400-update runs. They share label exposure,
not trainable capacity; the log-scale plot does not measure publication quality.

Suppose answer accuracy rises while natural endings decline. Inspect whether EOS targets survived preprocessing, whether decoding uses the correct stop IDs, and whether the token cap differs between comparisons. If one-word tasks become verbose, inspect example weighting and the explanation-token share. If held-out values fail but seen values succeed, test whether the model learned an instruction or a lookup table.

A fall in training loss alone cannot settle these diagnoses. Use the item-level evaluation from Chapter 7 and the alignment audit from Chapter 8. Preserve raw completions and exact settings. For stochastic generation, use multiple declared seeds and compare uncertainty at the item/group level.

Training should stop on nonfinite states, exhausted declared runtime, violated memory reserve or completion of the fixed budget. Choosing an unplanned extension after seeing disappointing examples changes the experiment. Write the new hypothesis and budget as a follow-up run.

## 9.11 Read the experiment at its actual boundary

The [original contract](../../experiments/specs/2026-10-04-evaluation-and-sft-course.md)
separates the tiny CPU mechanisms from the real-model comparison. The actual
0.6B runs and held-out-value evaluation above are measured; the proposed
Qwen3-1.7B-Base transfer remains unexecuted. A runnable tool or successful smoke
does not establish the outcome of an unrun stage. [Appendix D](../appendices/d-reproduction-and-environments.md#supervised-fine-tuning-state-and-adapter-exports)
and the [lab](../labs/09-supervised-fine-tuning.md) provide the operational route.

## 9.12 Teacher text uses the same supervised objective

A teacher's emitted sequence can supply SFT targets. Response NLL still scores
those observed targets; it does not reveal the teacher's probability vector or
establish that its answer or printed explanation is true. Chapter 15 develops
[response-level distillation](15-distill-evaluate-and-defend.md), its
actual source-controlled experiment and its complete-response/answer-only comparison.

## 9.13 Follow the distillation extension before exercises 13–16

Use Chapter 15 and [the Day 26 response notebook](../../notebooks/day-26/03_response_level_distillation.ipynb)
for the selection rule, missing-source coverage, printed-step versus final-answer
checks and teacher/student cost boundaries. These are an optional forward-reading
extension of the present SFT mechanism. The numbered exercises and their worked
solutions remain stable.

## 9.14 Exercises and the next question

1. Derive answer-only NLL from the conditional sequence probability.
2. State the shapes of logits, labels, ownership masks and output projection.
3. Explain how a user embedding receives a gradient with zero direct user loss.
4. Compare token and example averaging on unequal response lengths.
5. Design an accumulation test that exposes incorrect microbatch weighting.
6. Identify what a weights-only restart can and cannot reproduce.
7. Calculate LoRA parameter count and explain the zero-$B$ first-step gradients.
8. Explain why the base matrix still consumes memory during LoRA training.
9. Defend a full/LoRA comparison budget and identify its remaining confounders.
10. Diagnose improved likelihood with worse stopping behavior.
11. Explain why a post-trained starting checkpoint changes a base-transition claim.
12. Specify a regression gate before a larger-model comparison.
13. Distinguish response NLL from forward KL when only teacher text is available.
14. Explain why first format-eligible selection differs from best-correct teacher selection, and what missing source coverage should do.
15. Interpret a correct final answer with a wrong printed step, a valid step with a wrong answer, and a missing step without claiming internal faithfulness.
16. Defend the complete-response versus answer-only budget and separate teacher cost from student cost.
17. Update 3 was logged after a durable update 2 snapshot, then saving failed. Which boundary is authoritative? What changes if update 3 committed but its metric append failed?
18. The actual 20-update profile improves NLL from 4.061965 to 1.100189 but generates 0/8 exact answers and 0/8 message-end stops in both panels. Why is this consistent, and which claims does each measurement support?
19. Full400 answers/stops correctly on 120/120 held-out-value publication items while Base and merged rank 8 Q/V LoRA400 remain 0/120 and cap every answer. Both trained arms saw 9,321 labels, and LoRA's development NLL improved. What does this intervention establish, why does generic format validity not rescue the failures, and why does a degenerate 40-group bootstrap interval not establish a universal method ranking?

SFT demonstrates desired answers. It cannot directly express every trade-off between two plausible answers. Chapter 10 introduces preference observations and asks what a reward model can learn when humans disagree about which response is better.
