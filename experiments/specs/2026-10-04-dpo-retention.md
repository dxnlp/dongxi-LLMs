# DXI-11 — retention is an objective and evaluation choice

Frozen before fitting, 2026-10-04. Mode: original bounded CPU sequence
experiment, not a pretrained-model result. Use the isolated locked CPU
interpreter; no model acquisition, GPU, API, installation or service.
Preserve the historical preference-policy report and its failed DPO arm.

## Question and predictions

Does additional chosen-response likelihood or rehearsal change a shared
decoder's absolute chosen/rejected likelihoods, free generation and a separate
task? A rising preference margin alone does not require rising chosen
probability. Chosen NLL supplies extra demonstration supervision; it may also
imitate noisy winners or unwanted length. Rehearsal supplies extra supervised
tokens from another task, not a universal protection against forgetting.
These predictions are hypotheses, not promised improvements.

## Frozen recipe and data before any fit

The canonical input is `fixtures/dpo-retention/contract.json`. It declares
seeds1811/1812/1813, all retained, and a one-layer shared TinyDecoder with
vocabulary16, width24, four query/two KV heads, head dimension6, hidden48,
RoPE/RMSNorm/SwiGLU and tied embeddings. All calculations are CPU float32,
one Torch thread, no dropout. No source/problem group crosses a train/held-out
boundary. IDs and source groups are checked independently of prompt tokens.

First run exactly120 SFT warm-up updates on eight original demonstrations:
four symbolic first-slot copying prompts and four independent parity prompts.
Use AdamW learning rate0.015, weight decay0 and gradient clip1. The same warm
state is copied into every subsequent arm for that seed; no warm checkpoint is
selected by evaluation. It is an actual learned tiny policy, not a probability
table or a pretrained assistant. First-slot copying and parity use disjoint
input and response alphabets. Parity is a separate task absent from the DPO
pairs; it is not evidence about broad language retention.

Run every Cartesian row of three seeds, four data conditions and four arms:
48 fits of80 full-batch updates, AdamW learning rate0.008, weight decay0,
gradient clip1 and DPO beta0.5. Reference is the fixed warm state. Cache
reference scores only for the exact encoded branches; record the cache and
state identities and verify bitwise reference immutability/no gradients.

| Arm | Chosen-token NLL coefficient alpha | Rehearsal-token NLL coefficient gamma |
|---|---:|---:|
|DPO|0|0|
|DPO+chosen|0.25|0|
|DPO+rehearsal|0|0.25|
|DPO+chosen+rehearsal|0.25|0.25|

The DPO term is the mean pair loss over **summed** response log-likelihoods,
including EOS. Chosen and rehearsal NLL are each total valid response-token
NLL divided by their own valid-token count. Their separate coefficients are
predeclared, not tuned from final-test scores. Alpha=gamma=0 must recover the
existing `dpo_lab.dpo_loss` value and gradient exactly. Alpha=0 alone still
includes rehearsal when gamma is nonzero.

| Data condition | Recorded pair construction |
|---|---|
|clean|Correct first slot+EOS wins over wrong second slot+EOS;2/2 tokens|
|noisy|Swap chosen/rejected for fixed pair indices1 and3;2/2 tokens|
|chosen-longer|Correct first slot+STYLE+STYLE+EOS wins over wrong slot+EOS;4/2 tokens|
|matched-long|Both responses receive STYLE+STYLE before EOS;4/4 tokens|

Length controls keep the semantic first-slot target but alter the recorded
style. Evaluation does not adopt that style: the fixed canonical output is
answer+EOS. This intervention separates semantic answer correctness from
canonical completion success. There is no truncation of recorded training
branches and no replacement of DPO sequence sums by token averages.

## Fixed evaluation, source splits and costs

Evaluate the random initialization, warm initialization and each final update80
with the same full-vocabulary greedy generation, cap4, EOS2 and atomic symbol
table. Retain every generated token and raw rendering, stop/truncation,
first-answer correctness, canonical exact completion, format and task slice.
Teacher-forced full-response likelihood is a separate diagnostic; all
generation scores come from consuming the model's own emitted tokens.

Four unseen ordered color pairs form genuine held-out preference generation;
the answer distribution is balanced. Four trained parity problems measure
separate-task retention; four unseen numeric pairs are additionally reported.
Neither held-out panel is used in warm-up, preference training, rehearsal,
coefficient selection or checkpoint selection. English descriptions are
metadata; the policy consumes symbolic tokens, not natural-language prose.

Record per-update absolute chosen/rejected mean sequence log-likelihood and
reference-relative margin, all loss components/gradient norms, pair and
supervised-token costs, policy/reference forward passes, state hashes, actual
runtime and CPU environment/lock/source identities. NLL branches reuse chosen
policy scores; rehearsal adds an extra policy forward. Equal update counts do
not mean equal token, compute or supervision budgets.

## Acceptance and failure retention

- Independent tests verify alpha=gamma=0 equivalence, explicit combined-loss
  parameter gradients, reference detachment, global token NLL weighting,
  prompt/EOS/pad boundaries, source leakage, noisy/length construction and
  complete generation/stopping accounting.
- Every predeclared seed/arm/condition is retained, including negative or
  unchanged retention and held-out results. Report the final scheduled state;
  never report only an apparent best repair.
- Plot absolute chosen/rejected log-likelihoods and margins, initial/final
  unrelated-task retention and genuine held-out free generation, alongside
  additional supervision/token costs. The noise/length controls are not
  interpreted as identical information.
- Execute the visual notebook in a fresh isolated CPU kernel; include adjacent
  runnable answers, inspect reference figures and record hashes/check outputs.
- Historical negative reports remain unchanged. No universal repair, human
  preference, natural-language capability or Spark-scale result is inferred.
