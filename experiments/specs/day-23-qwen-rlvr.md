# Day 23 — local Qwen3-0.6B synchronous RLVR contract

Status: runner prepared and tiny-model adapter tested; no Qwen/CUDA run measured
by this material build. The CPU decoder experiment has its own report and must
not be described as Qwen execution. This experiment requires separately supplied
local weights, immutable revision, platform lock and authorization to run.

## Exact implementation and launch

`dongxi_llms.qwen_rlvr_lab` accepts HF-style causal logits, one prompt per group,
temperature 1/full-support rollouts, strict one-integer/EOS verifier, population
group standard deviation, response-mean clipped ratios and exact categorical
forward KL at sampled states. Behavior weights remain unchanged during collection;
old log probabilities detach; frozen reference parameters never train. The runner
uses one optimizer step per fresh group and no inference engine.

Substitute an existing local snapshot and its full upstream 40-character revision:

```bash
PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m dongxi_llms.qwen_rlvr_lab \
  --model-dir /absolute/path/to/local/Qwen3-0.6B-snapshot \
  --revision FULL_40_CHARACTER_IMMUTABLE_COMMIT_SHA \
  --output outputs/day23-qwen-rlvr-smoke \
  --updates 2 --group-size 2 --max-new-tokens 16 --lr 0.000001 --device cuda \
  --prompt-mode chat --max-seconds 600 \
  --snapshot-max-bytes APPROVED_PAYLOAD_BYTES \
  --work-limits /absolute/path/to/approved-rlvr23-caps.json \
  --work-journal-max-bytes APPROVED_MODEL_JOURNAL_BYTES \
  --snapshot-io-limits /absolute/path/to/approved-snapshot-io9-contract.json \
  --snapshot-io-ledger /absolute/path/to/owned-new-io-journal.jsonl \
  --environment-lock /home/dongxi/dgx-spark-dongxi/uv.lock
```

The placeholder revision is deliberately rejected by the CLI. This is an
explicit-input command template; the payload/journal integer placeholders also
require approved values. It is not a claim the specified snapshot is already
installed. All loading is `local_files_only=True`; no token or network download
is required. Output must be unused. Never overwrite an earlier run directory.

Chat mode uses the saved template, disables thinking and stops on both tokenizer
EOS and the existing single-token `im_end` marker. A course SFT
`course-genealogy.json` requires the exact recorded template hash and rejects raw
prompting. Supply `--template /absolute/path/to/audited-template.jinja` only when
the base tokenizer lacks one; an SFT checkpoint rejects a changed template.
For an intentionally raw base-only experiment, choose `--prompt-mode raw`;
its different interface is recorded and it cannot silently replace the SFT route.
PEFT adapter directories require an explicit prior merge.

Actual execution requires the selected existing `--environment-lock`. Common
identity records source/Git/Python/packages/lock/command/driver and actual local
checkpoint bytes before model loading. A course parent must also match saved
token meanings, fast-tokenizer encoding/wrapper rules, special IDs, template and
stops. An old course checkpoint lacking a fingerprint requires explicit audited
`--allow-legacy-interface` adoption, not automatic upstream authentication.
The [CPU identity report](../reports/2026-10-04-run-identity.md) records tiny random
local HF execution and merge/reload, not Qwen evidence.

Current-source extension,2026-10-05: actual execution also requires a complete
23-dimensional model/semantic work cap file and a separate9-dimensional shared
snapshot I/O contract, including whole-operation payload/node/tensor/primitive
envelopes and I/O journal size. The payload bound must agree in both arguments.
The explicit tiny-CPU fixture limits are not production estimates or launch
authority. Derive/approve model-scale geometry, evaluation/recovery/publication
capacity and resource headroom separately; omitted limits are not inferred.

Resume additionally supplies `--resume`, independently retained
`--resume-contract`, `--resume-sha256`, `--resume-bytes` and
`--resume-io-receipt`, with the same physical `--work-journal` and
`--snapshot-io-ledger` and a new exclusive output. Bounded no-follow metadata
and phase/horizon checks precede output/journal/model creation; bind both saved
prefixes before payload inspection. Expected hashes remain expectations until
actual inspection verifies the bytes. Generic identity hashing must not read
the selected payload first, including template/lock/source/parent hardlink aliases.

## Resources and acceptance

Use platform lock and actual host/GPU identity. Stop a concurrent large inference
server before launching. Check MemAvailable before loading, after reference copy,
during every generation step and around backward; preserve 25 GiB reserve.
This sampled guard cannot prevent every transient allocation spike. Profile
actual peak host/CUDA memory and process exit using the platform guard before
increasing geometry. The baseline makes a full frozen reference copy and uses
ordinary AdamW; memory must be measured, not inferred from model weights alone.

Smoke budget: 2 updates, G=2, 16 new tokens, FP32 CPU or BF16 CUDA, seed 2323,
learning rate 1e-6, clip 1, beta 0.02, epsilon 0.2. Model remains eval mode while allowing
gradients. Record all outputs, stop reasons, reward/advantages, valid tokens,
gradient norm, initial ratio error, KL, elapsed time and source file hashes.
Check initial behavior/current log-ratio near zero, finite loss/gradient and
reference immutability. Zero reward/advantage groups are legitimate smoke
outcomes and do not establish learning.

The runner also enforces a default 600-second wall-clock budget, starting before
model loading and checked between load, generation and backward stages. An
individual blocking kernel or load cannot be interrupted by this sampled
Python check; use the platform process guard for a hard timeout. Reject a
missing EOS ID and prompt-plus-response lengths above the model's declared
context limit. The supplied upstream SHA is metadata; computed local file
hashes identify the actual input bytes and do not prove their upstream origin.

The validated unused output directory is created before model loading. Atomic
status/report snapshots preserve the current stage, partial baseline rows,
initial evaluation and every completed optimizer update; completed updates also
append to `metrics.jsonl`. A caught load, resource or numerical failure records
its exception and last stage while retaining prior evidence. A killed process
may leave status `running`; inspect the actual process exit rather than treating
that status as proof of liveness. The status/metric journal alone is not the
checkpoint-replay state. The separately published immutable completed/pending
snapshots retain native numerical state and independently bound work receipts.

## Learning comparison after smoke

The runner freezes four training pairs `(2,3),(4,5),(3,6),(1,7)` and four
disjoint held-out pairs `(2,6),(3,4),(5,5),(7,2)` before loading. It records greedy
initial/final strict-integer evaluation, all raw outputs, token-limit failures
and one panel hash. The panel contains related original arithmetic prompts;
exact-pair separation does not establish independence of task families or
broad reasoning coverage. It is never used for update selection or recipe tuning.

For a bounded pilot, compare G4 with G8 at identical initial checkpoint, prompt
schedule, 16 updates and 64-token cap using two unused output directories.
Record unequal token/compute costs; an equal-token follow-up must adjust update
counts and label that changed optimization budget. Do not infer superiority from
one seed or a four-prompt score. A broader extrapolation panel is a separately
designed follow-up, not implied by this ready pilot route.

The source now offers explicit completed/pending recovery of policy, original
reference, Adam, global/rollout RNG, source cursor, retained actions and numerical
history. Four actual random-tiny CPU fresh processes replay2→4 without pending
recollection and without refunding later model/I/O failures; native publication
faults retain the previous committed pointer. The [current CPU evidence](../reports/2026-10-05-rlvr-io-readiness.md)
is not pretrained/CUDA/BF16 or physical-resource proof. HF policy/tokenizer
export and genealogy remain separate artifacts for evaluation, not substitutes
for resumable state. Calling this a pretrained recovery-verified training system
requires separately approved replay/profile evidence; this contract supplies no
automatic acquisition, loading, GPU launch or private platform authority.

vLLM/Open Instruct integration is a later adapter comparison. Require matching
token IDs, chat/prompt templates, EOS/masks, full behavior probabilities and
weight/version hashes before comparing throughput. Current changing documentation:
[Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B),
[Transformers causal LM loading](https://huggingface.co/docs/transformers/main/en/model_doc/auto),
[vLLM](https://docs.vllm.ai/). Pin versions/revisions in the measured report.
