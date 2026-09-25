# DongxiGPT Stories — tokenizer and architecture baseline

Execution update,2026-09-13: the selected baseline now has a separate SDPA
training implementation and a successful bounded BF16 Spark smoke/recovery
check. Only a1,024/128-document TinyStories prefix was prepared. See the
[report](../../experiments/reports/2026-09-13-tinystories-pipeline-smoke.md).
The design history below predates that execution; full-corpus audits and a
multi-hour learning run are still pending.

Date: 2026-09-13. User selected TinyStories and requested settling the tokenizer
and model configuration. This is the working design baseline, not a measured
GPU fit, a trained model, or approval to launch a long run. Exact resource and
optimization settings remain subject to a separately specified smoke profile.

## Data identity

- User-selected corpus: `roneneldan/TinyStories`.
- Working first-run variant: original `TinyStories-train.txt` with its matched
  `TinyStories-valid.txt`, rather than silently mixing original and GPT4-only V2.
- Repository revision resolved from the publisher's HF API on 2026-09-13:
  `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.
- [Pinned dataset card](https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md).
- Metadata/card inspected only; corpus bytes not downloaded or audited. Actual
  file hashes, boundary parsing, lengths and split overlap checks remain pending.

## Tokenizer decision

Reuse GPT-2 byte-level BPE from `openai-community/gpt2`, vocabulary50,257,
without training a new tokenizer for the first run. Pin revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e` (resolved via publisher API).
Inputs to retain/hash later: tokenizer.json, vocab.json, merges.txt,
tokenizer_config.json and relevant special-token settings. Do not download model
weights. Encoder/merge reuse is not pretrained neural-network knowledge.

Sources: [GPT-2 tokenizer documentation](https://huggingface.co/docs/transformers/model_doc/gpt2)
and [vocabulary/special-token configuration](https://huggingface.co/openai-community/gpt2/blob/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json).
The source GPT-2 architecture is NOT the selected DongxiGPT architecture.

Motivation: remove tokenizer-training and vocabulary-selection variables from
the first training experiment. Cost: the vocabulary is larger than this narrow
story task may need. The tied embedding/output matrix alone has25,731,584
parameters. A smaller custom byte-BPE vocabulary could reduce that table and
projection work, but may lengthen sequences; it is a later controlled design
comparison, not a guaranteed improvement or an unannounced change now.

Tokenization contract: no automatic BOS/EOS insertion; construct boundaries
explicitly. Use existing ID50256 as a start-context marker and end-of-story
target (same vocabulary ID, distinct sequence roles). No new PAD vocabulary row:
if right-padding uses50256, mark only padded label POSITIONS as-100. Genuine
EOS targets still contribute loss. Preserve original text/whitespace apart from
document-delimiter parsing, which needs verification against the corpus file.
No chat template. Prompts at evaluation use the same start-context convention.

## Model decision

| Setting | Baseline |
|---|---|
| Type | Dense causal decoder; all trainable weights initialized from scratch |
| Layers | 12 distinct blocks; no looped depth or MoE |
| Residual/embedding width | 512 |
| Query / KV heads | 8 / 8 (ordinary multi-head attention, not GQA) |
| Head dimension | 64 |
| SwiGLU intermediate width | 1536 |
| Normalization | Pre-norm RMSNorm, epsilon1e-6; final RMSNorm |
| Position encoding | RoPE, base10000, adjacent-coordinate pairs, as in course reference |
| Context budget | 1024 tokens, including prompt/history; not1024 words |
| Vocabulary | 50,257 output IDs |
| Embedding/output | Tied, with no output bias |
| Other projection biases | None, matching the modern course reference |
| QK normalization | Off for this baseline |
| Dropout | 0, matching the course reference |
| Initialization | Course reference: normal std.02 for embeddings/linears; RMSNorm gains1 |

MHA keeps the first baseline direct; GQA and QK norm remain later ablations, not
requirements for a modern decoder. This choice is not evidence that they are
inferior. The model is not GPT-2 and not a reproduction of a published TinyStories
checkpoint merely because the tokenizer is reused.

Whole route: IDs[B,T] → embeddings[B,T,512] → 12 causal residual blocks → final
norm → tied output projection → logits[B,T,50257]. At full context, each head
has queries/keys/values[B,8,1024,64]. Generation's prompt plus continuation must
fit the1024-token context contract; RoPE does not by itself establish reliable
extrapolation beyond it.

## Transparent accounting check (not a training experiment)

Using `decoder_lab.analytical_parameters`:

- Embedding/output (one shared matrix):50,257×512 =25,731,584.
- Attention per block:4×512×512 =1,048,576.
- SwiGLU per block:3×512×1536 =2,359,296.
- Two RMSNorm gains per block:1024; final gain:512.
- Total:25,731,584 +12×(1,048,576+2,359,296+1024)+512
  = **66,638,848 unique parameters**.

Shape-only verification with `TinyDecoder` on PyTorch's meta device matched the
analytical count and verified tied parameter identity. No real model storage,
forward pass, GPU allocation, throughput measurement or training was performed.
Command exited0 in the existing Spark Python environment.

Reproduce from repo root with the platform interpreter and `PYTHONPATH=src`:

```python
import torch
from dongxi_llms.decoder_lab import (
    DecoderConfig, TinyDecoder, analytical_parameters, parameter_count,
)
cfg = DecoderConfig(vocab=50257, width=512, heads=8, kv_heads=8, head_dim=64,
                    layers=12, hidden=1536, max_length=1024, modern=True,
                    qk_norm=False, tied=True)
with torch.device("meta"):
    model = TinyDecoder(cfg)
assert model.token.weight is model.lm_head.weight
assert parameter_count(model) == analytical_parameters(cfg) == 66_638_848
assert all(parameter.is_meta for parameter in model.parameters())
```

FP32 weights+gradients+two Adam moment tensors have an estimated persistent
ledger of16P bytes, approximately0.993GiB. This omits activations, workspaces,
loss intermediates, allocator/framework/host memory and other processes. A
single full-context sequence already produces51,463,168 logits; materializing
those as FP32 would require196.316MiB before gradients/loss temporaries. Neither
number is a measured peak or proof of fit.

## Implementation and profiling gates

### Hardware-informed runtime recommendation — 2026-09-13

The learner accepted the baseline and asked the agent to recommend a budget
based on Spark rather than choosing an arbitrary duration themselves.
Recommendation (not yet run authorization): reserve15–30 minutes for a bounded
smoke/profile after implementation and data preparation, then cap the first
learning run at4 hours. Consider an8–12-hour follow-up only after reviewing
stability, held-out loss, fixed story samples and a successful recovery check.
These are proposed time allocations, not estimates of time to coherent writing.

Read-only host snapshot: NVIDIA GB10, driver580.173.02,20 Arm CPU cores;
`free -h` reported121GiB total and118GiB MemAvailable (rounded, point-in-time).
No throughput or GPU peak was measured. The
[official NVIDIA hardware guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
specifies128GB unified memory and273GB/s bandwidth; its1PFLOP headline refers to
FP4 with sparsity, not our dense BF16 training throughput. Memory capacity makes
this model a reasonable candidate, but does not determine tokens/second.

After profiling, convert a proposed token budget to runtime using measured valid
training targets/second, with validation/checkpoint overhead accounted for.
Illustration ONLY:10,000 valid targets/s sustained for14,400 training seconds
would give144M presentations; this is not a Spark throughput prediction. Repeated
presentations are not unique tokens. Fix the schedule's token/update horizon
before the learning run and define any later continuation as a recorded schedule
decision, rather than silently extending a completed cosine schedule.

Keep66.64M parameters and1024 context fixed initially; test modest microbatches,
starting at1, before assigning accumulation. Keep at least25GiB host memory
available as the conservative end of the platform's20–25GiB reserve policy.
No inference server, full training run or new GPU probe was started by this
recommendation. Data preparation time is separate from the4-hour learning cap.

Use the course decoder as the correctness reference, not an assumption that
its eager attention/memory behavior is adequate for training at this size.
Implement/verify an SDPA training path with no retained generation KV caches,
and test logits/gradients against the reference at small shapes. CUDA BF16
autocast with FP32 parameter/optimizer state is the intended precision policy,
pending operation-level and recovery verification; it is not yet tested here.

Audit token lengths before deciding how to handle stories longer than1024.
Preserve document boundaries, do not silently truncate them, and do not assume
most stories fit before measuring. Padding/packing, microbatch, accumulation,
learning rate, target-token budget and wall-clock limit belong in the full run
specification after a safe profile. Start profiling at microbatch1 with the
platform guardrails and20–25GiB host reserve. The first learning run and any
controlled comparison require explicit scoped execution approval.
