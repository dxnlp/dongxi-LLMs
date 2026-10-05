# Matched chosen response SFT and DPO control

This protocol is saved before fitting or measuring the new control. It closes
the named callable chosen-SFT source-path gap, not the unexecuted pretrained
assistant comparison. No acquisition, GPU execution, installation, service,
Git mutation or external campaign row is authorized. The learner remains Day9.

## Reuse and scientific comparison

The new helper imports the existing Chapter11 runner's `encode_pair`,
`collate`, `verify_parent_tokenizer`, `completed_dpo_update`, validation and
generation functions without modifying them. Chosen-only SFT uses the existing
Chapter9 `summed_loss`. The Chapter9 shuffled cyclic sampler is not described
as matched to DPO's with-replacement sampler. Instead, chosen-SFT draws from
the same private seeded Torch generator, one pair per accumulation slot.

Three fixed arms start from the same local full parent checkpoint: unchanged
parent, chosen-SFT and DPO. Both training arms use the same train-pair order,
chosen response IDs, complete answer and real termination targets, number of
updates, accumulation, optimizer learning rate and clipping. Chosen-SFT minimizes
the sum of chosen-token cross-entropies divided by all valid chosen targets in
that update. DPO retains its original mean-over-pairs loss, summed completion
log probabilities, frozen parent reference and one causal shift. Those are
different reductions and different supervision, not numerically equal losses.

DPO also scores rejected responses and the frozen reference. Report chosen and
rejected targets, logical sequence tokens, actual submitted policy/reference
forward calls and input positions independently. The existing SFT forward sends
the complete input while DPO sends `ids[:-1]`; retain that difference. Equal
updates and chosen exposure are not equal compute, information, FLOPs, memory,
runtime or broad capability. Do not shorten/truncate any input to claim matching.

## Frozen original data and tiny fixture

Use the existing original `fixtures/chapter11/train.jsonl` (8 location pairs),
`validation.jsonl` (4 pairs) and `evaluation.jsonl` (4 independent location
questions). Preserve their bytes. The new `fixtures/matched-chosen-sft/protocol.json`
provides explicit scenario source groups and the fixed recipe. These authored
source-group assignments are not external corpus provenance. Reject raw and
encoded prompt collisions or source groups shared across splits; require
globally unique record IDs, nonempty completions, prefix-compatible templates,
valid IDs and real ending tokens. Overlength and unknown-token encodings refuse.

The CPU checkpoint is original, random, one-layer Qwen3: hidden width16,
intermediate width32, two query heads, one KV head, head dimension8,
position limit64, no dropout. A WordLevel tokenizer's fixed inventory is the
sorted whitespace atoms in the complete predeclared fixture literals, with
four separately assigned special IDs. This intentionally closed authored
inventory is frozen before training, including publication vocabulary; it is
not tokenizer training or an open-vocabulary/generalization claim. Publication
answers never enter a model loss or select the recipe. The audited template
has explicit user, assistant and EOS tokens; EOS is supervised when real and
masked when used as padding.

Freeze seeds1818 and1819; each training arm has6 updates, accumulation2,
AdamW learning rate0.008, weight decay0.01, clip norm1, beta0.2 for DPO,
CPU FP32 and greedy uncached generation capped at8 tokens. This is24 actual
optimizer updates across the complete two-seed comparison; unchanged arms
have zero. No coefficients, checkpoint or generation changes follow heldout
inspection. Existing intended pretrained assistant rows retain their separate
100-update/accumulation4/lr5e-7/beta0.1 recipe and authority gate.

## Observation and lineage contract

Before training, retain parent artifact hashes, typed tensor digest, actual
tokenizer/template/stops/interface, encoded data hashes, source/input/lock
hashes, seed, objective/reduction, train IDs/source groups and recipe. Verify
the saved parent's semantic interface before model loading. Start each arm
from the same saved bytes; disable stochastic dropout and keep the reference
frozen. Record every sampled index, successful target/call/position count,
gradient norm and final policy/reference digest. Independently check matching
chosen exposure and sampler states. No reference parameters may receive gradients.

Evaluate all arms at their predetermined final boundary on the same independent
publication prompts. Retain every raw generated ID/text, natural stop versus
cap, errors and exact expected-answer grading. Validation pair margins and
absolute chosen/rejected likelihood are diagnostics, not the independent
generation metric. Keep all zero/negative outcomes. No score-based selection.

Final local HF exports may reuse the verified tokenizer and interface lineage.
This thin control does not add a production recovery/resource subsystem.
Interrupted chosen-SFT updates are poisoned; restart from the independently
retained original parent. Accounted resume, shared I/O/artifact routing,
pretrained/CUDA/BF16 replay and actual stage execution remain pending. A prepared
local-only command is not an approved model-scale run.

The thin adapter exposes explicit `--device cpu|cuda` with CPU default; CUDA
source uses native FP32 policy/reference weights and BF16 autocast, SDPA and
policy activation checkpointing. There is no automatic backend fallback.
This task tests CPU only. Its in-process sampled deadline is not the required
external Spark supervisor, and it deliberately has no production accounting or
resume flags. Thus a prepared CUDA command wires the scientific control without
asserting approval, CUDA numerical parity or production readiness.

## Predeclared controls and collection

CPU tests cover exact chosen IDs/masks versus native DPO encoding; unequal
response lengths and EOS-as-pad; independent CE gradient/Adam-equation parity;
exact native DPO-update parity; same private replacement draws and chosen
exposure; zero SFT reference forwards; detached DPO reference; unchanged-parent
and shared initial bytes; split/prefix/ID/template/unknown/overlength refusal;
invalid recipe types including bool aliases; partial-forward poisoning;
output exclusivity; actual local checkpoint export/reload/interface checks;
and the real prepared CLI on the tiny checkpoint. Failure rows and completed
history remain retained if evaluation/export fails.

Use only `/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python`, hidden CUDA,
offline Hugging Face flags and one CPU thread. Each child has an external
60-second timeout. Sample real Linux MemAvailable before/during/after each
child and refuse/stop below25GiB; sampling is not continuous containment.
Retain actual exit codes, command, stdout/stderr, source hashes before/after,
all seed/arm records and failure diagnostics in a new exclusive report directory.
Preserve all earlier reports and original giant-runner/input bytes. No whole
course rerun, hosted CI, Mac verification or model-stage readiness is claimed.
