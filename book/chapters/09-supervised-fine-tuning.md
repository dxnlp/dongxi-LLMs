# 9. Supervised Fine-Tuning

A pretrained decoder can continue text without reliably answering a request. Supervised fine-tuning supplies demonstrations of the conditional behavior we want: given this conversation, produce this response and stop at this boundary. The model remains a next-token predictor. We change the distribution of contexts, the targets receiving loss, and often which parameters can move.

This chapter spans Days 12–14. It combines the [objective and gradient notebook](../../notebooks/day-12/01_sft_objective_and_gradient_paths.ipynb), [accumulation and restart notebook](../../notebooks/day-12/02_accumulation_and_checkpoint_identity.ipynb), [bounded training notebook](../../notebooks/day-13/01_tiny_assistant_training.ipynb), and [full versus LoRA notebook](../../notebooks/day-14/01_full_sft_lora_and_recipe_defense.ipynb). The [lab guide](../labs/09-supervised-fine-tuning.md) provides the run order; [worked solutions](../solutions/09-supervised-fine-tuning.md) explain the deep questions.

## 9.1 What transfers from pretraining?

Pretraining establishes representations and continuation patterns across a broad corpus. SFT supplies a narrower distribution of demonstrations. It can make an existing ability accessible through an interface, strengthen a task behavior, teach format compliance or introduce new factual associations. These are distinct outcomes and require distinct evaluation slices.

A model that already knows a fact may answer it correctly after learning when to give a direct response. That does not prove SFT created the fact. Conversely, a model can memorize a new answer without learning a transferable instruction. Compare held-out values, unseen phrasings and unrelated regression tasks.

Model naming matters. The official [Qwen3-0.6B-Base card](https://huggingface.co/Qwen/Qwen3-0.6B-Base) describes a base checkpoint; [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) is a separately released post-trained model. Fine-tuning the latter is useful continued adaptation, but does not test a first base-to-assistant transition. Record exact revisions so the comparison cannot change when a repository updates.

Our CPU model begins randomly initialized. Its task is deliberately tiny: map four symbolic copy requests to a word and END. Its results verify optimization and gradient mechanics. A real base-to-assistant claim belongs to the separately specified Spark experiment.

## 9.2 Derive the supervised objective

For demonstration $j$, let $c_j$ be its conversation context and $a_j=(a_{j,1},\ldots,a_{j,n_j})$ its assistant sequence, including the intended ending. The causal model gives

$$
p_\theta(a_j\mid c_j)=\prod_{t=1}^{n_j}
p_\theta(a_{j,t}\mid c_j,a_{j,<t}).
$$

Taking negative natural logs turns products into sums:

$$
L_j(\theta)=-\sum_{t=1}^{n_j}
\log p_\theta(a_{j,t}\mid c_j,a_{j,<t}).
$$

The common token-mean objective divides the total sum by the number of supervised targets, $N=\sum_jn_j$:

$$
L(\theta)=\frac{1}{N}\sum_jL_j(\theta).
$$

For a padded full transcript, use ownership mask $m_{b,t}$ and validity mask $v_{b,t}$. Logits $Z$ have shape $[B,T,V]$ and labels have shape $[B,T]$. After one causal shift, the logit gradient is

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

## 9.5 Accumulation must preserve weighting

A logical batch can be split into microbatches when memory is limited. Suppose microbatch $r$ has summed loss $S_r$ and supervised count $N_r$. The desired accumulated objective is

$$
L=\frac{\sum_rS_r}{\sum_rN_r}.
$$

Backward each $S_r/\sum_qN_q$, then perform one optimizer step. Averaging microbatch means instead gives $\frac{1}{R}\sum_rS_r/N_r$, which overweights short-answer microbatches when counts differ.

The accumulation notebook compares gradients from a joint batch with token-weighted split batches. It also deliberately averages means to expose the failure. Use a shared initialization, no stochastic dropout and identical context masks so the only difference is weighting. Floating-point reduction order may introduce tiny numerical differences; define a justified tolerance rather than demanding impossible bitwise equality.

An incomplete final accumulation window needs its own denominator and update policy. Silently dividing it by a fixed full-window count weakens its gradient. The Spark runner materializes the bounded window and counts its labels before backpropagation.

## 9.6 Checkpoint identity is wider than weights

Saving weights permits a forward reconstruction. Resuming optimization also needs optimizer state, scheduler state if present, update index, data order, random states, template, tokenizer, mask policy and configuration. A correct restart should reproduce the next update under the same deterministic assumptions.

The CPU demo verifies state-dictionary forward equality. The second Day 12 notebook adds an optimizer-state continuation check: copy the model and AdamW state, feed the same next batch, and compare the resulting parameters. This verifies the stated local boundary; it does not establish bitwise reproducibility across different hardware or backend versions.

Checkpoint selection uses development criteria declared in advance. “Choose whichever sample looks best” encourages selection on noise. Save fixed intervals and compare the same prompt grid; retain regressions and stopping reasons.

## 9.7 Full tuning changes all allowed parameters

Full SFT allows the optimizer to update every trainable parameter. It offers flexible adaptation, but its optimizer state and gradients can dominate memory. A rough AdamW accounting for $P$ parameters distinguishes weights, gradients and two moment tensors. Additional master weights, activations, temporary buffers and allocator overhead depend on the implementation. Do not promise memory sufficiency from a parameter-only estimate.

Full tuning can alter useful broad behavior while improving the target interface. Evaluate general continuation, arithmetic, format compliance and concise-answer behavior with the same contract before and after. A narrowly successful recipe may be appropriate for a specialized system; its limitations should remain visible.

The microscopic full run uses fixed seed, a bounded update budget and four known requests. It learns the fixture if the observed report supports that conclusion. There is no unseen-language generalization claim, and no best-seed selection.

## 9.8 LoRA constrains an update

For a linear weight $W_0\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$, LoRA freezes $W_0$ and learns

$$
W=W_0+\frac{\alpha}{r}BA,
\qquad
A\in\mathbb{R}^{r\times d_{\mathrm{in}}},
\quad B\in\mathbb{R}^{d_{\mathrm{out}}\times r}.
$$

The trainable adapter count is $r(d_{\mathrm{in}}+d_{\mathrm{out}})$ rather than $d_{\mathrm{in}}d_{\mathrm{out}}$. The rank of the update is at most $r$. The base matrix remains present, so small trainable state does not mean the whole base model occupies little memory. Activations also remain necessary for adapter training.

The [LoRA paper](https://arxiv.org/abs/2106.09685) introduces this parameterization. Our implementation uses a random $A$ and zero $B$. Initially the adapter output is exactly zero. The first gradient of $A$ is zero because it is multiplied by $B$; $B$ can receive a nonzero gradient through the initialized $A$. This asymmetry explains why “all trainable parameters must have nonzero gradients on the first step” is a poor health check.

For row-batch input $X$, compute $XW_0^\top+\frac{\alpha}{r}(XA^\top)B^\top$. The notebook compares this with a merged matrix and verifies equality. The microscopic adapter targets Q and V projections only. This is a declared design, not a universal optimal target set.

## 9.9 Compare recipes under named constraints

A full-versus-LoRA comparison changes capacity, optimizer state and often preferred learning rate. Using the same learning rate is a useful mechanism control, but may not give each method its best recipe. Conversely, tuning LoRA extensively while giving full tuning one attempt is an unequal selection budget.

Report two comparisons when feasible: a fixed-recipe intervention and a separately declared tuned-recipe comparison. Match training data, token exposure, seed policy and evaluation contract. State whether compute, wall time, updates or supervised tokens define the budget; all four cannot usually be equal at once.

The Day 14 notebook runs both microscopic modes from the same base initialization and plots observed loss against optimizer updates. Its evidence concerns the four symbolic examples, not the general usefulness of LoRA. The Spark plan proposes full SFT versus LoRA at a frozen data/token budget, followed by a Qwen3-1.7B-Base transfer check only after the smaller recipe is healthy. Those GPU results remain unmeasured until executed.

## 9.10 Interpret gains and regressions

Suppose answer accuracy rises while natural endings decline. Inspect whether EOS targets survived preprocessing, whether decoding uses the correct stop IDs, and whether the token cap differs between comparisons. If one-word tasks become verbose, inspect example weighting and the explanation-token share. If held-out values fail but seen values succeed, test whether the model learned an instruction or a lookup table.

A fall in training loss alone cannot settle these diagnoses. Use the item-level evaluation from Chapter 7 and the alignment audit from Chapter 8. Preserve raw completions and exact settings. For stochastic generation, use multiple declared seeds and compare uncertainty at the item/group level.

Training should stop on nonfinite states, exhausted declared runtime, violated memory reserve or completion of the fixed budget. Choosing an unplanned extension after seeing disappointing examples changes the experiment. Write the new hypothesis and budget as a follow-up run.

## 9.11 The course experiments

The [combined specification](../../experiments/specs/2026-10-04-evaluation-and-sft-course.md) distinguishes three layers: executed CPU correctness checks, bounded microscopic learning, and a proposed real-model Spark campaign. The [report](../../experiments/reports/2026-10-04-evaluation-and-sft-course.md) preserves actual measurements and source identity. Commands and optional dependencies live in the [lab](../labs/09-supervised-fine-tuning.md).

The Spark runner requires exact model/tokenizer revisions, original audited JSONL, an explicit output directory, CUDA, maximum updates, runtime cap and a host-memory reserve. It writes checkpoint state and raw fixed-prompt evaluations. It is prepared tooling; creating it does not launch a GPU run or establish assistant capability.

## 9.12 Exercises and the next question

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
12. Specify a regression gate before a larger validation run.

SFT demonstrates desired answers. It cannot directly express every trade-off between two plausible answers. Chapter 10 introduces preference observations and asks what a reward model can learn when humans disagree about which response is better.
