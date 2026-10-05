# DXI-04 — constructive controls before reasoning claims

Recorded before model measurements,2026-10-04. Mode: bounded CPU sequence
microscope. No pretrained checkpoint, acquisition, GPU, API, installation or
external service. Preserve the original negative G4/G8 GRPO report unchanged.

## Frozen question and controls

Can a genuinely unsaturated, known-solvable task supply sampled relative-reward
learning to a shared autoregressive decoder? Separately, how much of a positive
control survives new problem sources, templates or task families?

The primary control asks whether the sum of two nonnegative numerals is positive:
emit0 or1 followed by EOS. An independent integer-arithmetic oracle fixes the
target, not a learned judge. A one-layer16-wide TinyDecoder shares token
embeddings, attention and MLP parameters across prompts. It receives no SFT
warm-start. Predeclare seeds2301,2302,2303; use all seeds and the last update,
never choose a favorable checkpoint from held-out measurements.

Budget:120 sampled fresh-policy GRPO updates per seed; four training prompts,
16 responses per prompt, cap2, temperature1, AdamW learning rate0.02,
weight decay0, population group std, epsilon0.2, reference KL beta0.01,
per-response token mean then prompt/response mean, gradient clip1.0, one CPU
thread. The sampler's declared syntax grammar allows answer0/1 at the first
position and answer0/1/EOS thereafter. EOS is sampled and learned, not inserted
as a free termination. Recomputed behavior/current/reference likelihoods use
that same grammar. This is a conditional grammar objective, not raw-vocabulary
PPO or a change to the existing Qwen full-support baseline.

Record actual unsaturated initial probability of a correct complete two-token
path, sampled reward, nonzero gradients, zero-variance groups, token costs and
final generated rows. Exact expected return is diagnostic, not the decoder's
training loss. A paired frozen decoder and an independently fixed oracle are
controls. A separate tabular exact-expectation arm demonstrates that successful
lookup fitting alone says nothing about unseen keys or language reasoning.
That counterexample uses40 exact-expectation SGD steps at learning rate2.0,
with zero-initialized per-training-key answer logits and a frozen unknown row;
it supplies EOS from an oracle and is never described as sampled decoder RL.

Acceptance interpretation recorded before final evidence collection: report
answer-only probability separately from complete-path probability. Include fixed
constant0/1 plus EOS baselines to expose answer imbalance, and retain partially
failed training rows even when mean reward improves. These checks analyze the
declared recipe; they do not change it or tune against held-out scores.

## Splits, failure interventions and wider panel

Freeze original fixtures before training. No item/source group crosses splits.
Report training, unseen-source, unseen-source-plus-template and wholly
unseen-family slices separately. Unseen templates also use new sources; their
effect is not isolated from source novelty. Hold out all odd-sum predicate
examples from the decoder's sum-positive task. Family transfer is an observation
to report, not an expected success or model-selection criterion.

Evaluate decoder/frozen controls before and after training under greedy cap2
and fixed seeded sampling; retain every raw token/stop/score row. Independently
report full-vocabulary greedy decoding and cap1. A correct numeral without EOS
is truncated and receives no complete-path training reward. Invalid/unsupported
answers and truncation stay in denominators; supported mathematical accuracy,
format validity and naturally terminated strict success remain distinct.

A separate original natural-language panel covers arithmetic, linear algebra
and two-step quantity problems, with source/template/family splits. Validate its
authored exact references against independent structured arithmetic, using the
existing bounded mathematical grader. This panel and oracle/parser probes are
not claimed as decoder language generations. Planned raw/chat and base/instruct
protocols identify distinct weight checkpoints, tokenizer/template identities,
thinking support and cap32 versus128. A thinking toggle is not a new checkpoint;
long traces, final-answer correctness and rationale faithfulness are different
properties. No pretrained matrix row is executed by this CPU experiment.

## Acceptance and evidence boundary

- All three predetermined seeds start strictly unsaturated and are retained.
- Actual sampled autoregressive updates change shared decoder parameters with
  finite gradients; compare final complete-path probability/greedy results with
  paired frozen controls. Any seed failing to improve remains a failed control.
- Independent tests cover oracle references, source/template/family isolation,
  behavior likelihood alignment, masks/EOS/cap, parameter/gradient reach,
  deterministic seed replay, and wrong/invalid/truncated grading.
- Preserve original GRPO negative results and distinguish exact lookup training
  from sampled shared-decoder training and from unexecuted pretrained reasoning.
- Run the new visual notebook in a fresh CPU kernel, inspect regenerated figures,
  and preserve source/input hashes, actual environment, commands and test outputs.

This instrument can verify bounded learnability and evaluation accounting. It
cannot prove natural-language mathematical competence, unseen-family transfer,
faithful chain of thought, Qwen improvements or Spark memory/throughput. The
separate DXI-03 pretrained baseline remains approval-gated and unexecuted here.
