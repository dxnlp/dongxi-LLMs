# Verified Base files and measured tokenizer requirements

The approved exact Qwen3-0.6B-Base snapshot is now cached on Spark, and its
original instruction fixture has been measured using only the tokenizer.
All ten downloaded files match the pinned upstream sizes and digests. The
unchanged 20-update profile requires 455 training targets plus 720 development
targets, totaling 1,175 supervised target presentations in its work allowance.
No model weights were deserialized, and no inference or training was performed.

The [premeasurement specification](../specs/2026-10-05-base-tokenizer-sizing.md)
records the narrow authorization. This result resolves local availability and
tokenizer-derived geometry; it does not approve the native profile or establish
memory fit, learning quality, snapshot sufficiency or recovery on pretrained
weights. The learner remains Day 9 and the improvement ledger remains 13 of 18
complete in bounded CPU scope.

## Actual local checkpoint bytes

The [acquisition receipt](2026-10-05-base-tokenizer-inspection/acquisition-01.json)
records the exact revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, downloaded
without authentication using the existing platform environment. The snapshot is
under `/home/dongxi/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.
It contains 1,203,641,805 payload bytes; the weight file has 1,192,135,096 bytes
and observed SHA-256
`cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`.
Every file has a local SHA-256; all nine non-LFS Git blob digests and the weight's
LFS SHA-256 match the earlier retained upstream manifest. Acquisition and
streamed byte verification took 90.640 seconds, with actual command exit 0.

Payload is not measured network traffic or cache overhead. Weights were hashed
in bounded file chunks, not loaded as tensors. The source/lock hashes were
stable during collection. The original [availability report](2026-10-05-checkpoint-availability-and-portability.md)
remains the historical evidence of absence before acquisition; the different
cached post-trained checkpoint was not substituted.

## Tokenizer identity and masks

The successful [sizing receipt](2026-10-05-base-tokenizer-inspection/sizing-02.json)
uses installed Transformers 5.16.1 and tokenizers 0.23.1, the pinned local files
and the unchanged course template. Actual tokenizer/config digests match the
acquisition receipt before and after measurement. The vocabulary contains
151,669 entries, maximum token ID 151668. EOS and padding are 151643;
the end-of-message marker is 151645. These are distinct stop meanings, not a
single interchangeable identifier. The full interface digest is
`e869c7e93ad366530f30cb62e78122577d84f0f95fe2dad5a0e0923d4d031f63`.

All raw split hashes reproduce the original 240/60/120 data card. Only train
and development were tokenized; the publication test was neither tokenized
nor evaluated. Records were regenerated in memory, not materialized into the
future native profile's input directory. That file-preparation step remains.

Native `encode_record` was compiled directly from its source AST without
importing the Torch runner. It supervises assistant body, message-end delimiter
and trailing template newline. Counts use nonignored `labels[1:]`, preserving
the explicit causal shift. All 300 train/dev records passed prefix compatibility,
nonempty-target and maximum-length checks without truncation. Training lengths
range from 32 to 44 tokens; development lengths range from 34 to 44.

## The unchanged profile schedule

Seed 1212 selects the first 80 records of the shuffled 240-record training order:
28 copy, 22 reverse and 30 extraction examples. Twenty updates each consume four
single-record microbatches. This schedule is proposed work, not executed updates.

| Scheduled panel | Records | Shifted supervised targets | Full input positions |
|---|---:|---:|---:|
| Training selection | 80 | 455 | 3,154 |
| Baseline development NLL | 60 | 360 | 2,400 |
| Final development NLL | 60 | 360 | 2,400 |
| Total likelihood work | 200 presentations | 1,175 | 7,954 |

The training-exposure counter is 455, not 1,175: evaluation targets do not drive
optimizer updates. The work allowance covers both. The 20,480 maximum-position
geometry is not an observed target count; masked prompt positions still enter
the forward computation.

Each generation panel uses the first eight development IDs. Prompt lengths are
29–37 tokens, sum 269. Adding the fixed 64-token generation cap yields at most
101 positions per sequence, below 256. Two panels reserve 16 attempts and 1,024
new-token slots. No continuation was generated. Native `generation_upper`
reserves 33,344 conservative uncached positions per panel, or 66,688 overall;
these are neither measured cached dispatch nor FLOPs.

The receipt records all fourteen known SFT16 schedule requirements: 20 updates,
80 sampled examples/selector steps, 1,175 valid targets, 9,516 logical sequence
tokens, 1,224 policy calls/74,642 policy positions, 1,144 evaluation calls/71,488
evaluation positions, 16 generation calls/66,688 generation-position upper
bound/1,024 new-token slots, two recovery-validation operations and 20 history
rows. These overlapping dimensions must not be added into a universal cost.
The remaining tensor-element and RNG-state allowances are model-dependent.

## Verification and retained failures

An [independent replay](2026-10-05-base-tokenizer-inspection/independent-review-01.json)
reconstructed records and prefix masks directly, without the collector's native
functions. It reproduced every encoded ID/label row, the complete interface,
fixed selection, eight prompts and all fourteen known requirements. Tokenizer
bytes and eight source/lock bindings stayed unchanged. Exit 0, 1.122 seconds.
The successful root sizing process exited 0 in 1.516 seconds with Torch absent.

The first [sizing attempt](2026-10-05-base-tokenizer-inspection/sizing-01.json)
failed its no-Torch check: this installed Transformers version imports Torch
through `AutoTokenizer` despite `USE_TORCH=0`. No model was loaded. The
[original collector source](2026-10-05-base-tokenizer-inspection/collector-attempt-01.py)
is retained with its original hash. The corrected collector makes Torch absent
only to the current process's availability probe and independently refuses
actual imports through an audit hook. This is an explicitly reported tokenizer
process restriction, not an installed-environment change or CUDA validation.

Seven standard-library [helper tests](2026-10-05-base-tokenizer-inspection/root-helper-verification-02.json)
pass, including original raw hashes, fixed ordering, causal masks, generation
reservations and Git blob digests. An earlier root receipt-construction
[KeyError](2026-10-05-base-tokenizer-inspection/root-helper-collection-failure-01.json)
is retained with its passing test log; the corrected collector preserves whether
an optional ledger field exists. This is not a full-suite/notebook or Mac/hosted
verification. All 45 campaign rows retain 945 null actual fields, jobs 0 and no
actual genealogy.

## Next step before native execution

Materialize and verify the original profile input files. Declare the two fresh
snapshots' payload/tree/tensor/primitive envelopes, model-dependent semantic
validation allowances, separate I/O9 limits and journal/output byte allowances.
Resolve live conflict/resource clearance and obtain separate permission to load
the model and run the fixed 20-update, 900-second supervised profile. Do not
invent snapshot fit or fill campaign outcome fields from tokenizer sizing.
The successful download does not authorize inference, a pilot, another model,
an installation, service, Git push, publication or animation production.
