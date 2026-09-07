# Day 6 chapter — verification and evidence boundaries

## Scope and result

Extended the existing Chapter 5 narrative with sections 5.11–5.22 and twelve
additional worked answers (24 total). Reused the three Day 6 notebooks and
saved diagrams. No notebook source, teaching implementation, or server was
modified. No larger model was instantiated or trained.

Specification: `experiments/specs/2026-09-07-day6-chapter-verification.md`.
Source base: `b640258` plus the new documentation. Checks ran on CPU in the
existing platform environment, one Torch thread, seed 505 where random inputs
or model initialization were needed. Python 3.12.14, Torch 2.13.0+cu130,
Linux aarch64. Numerical comparisons used float64, atol 1e-10, rtol 1e-8;
explicitly rounded three-decimal examples used absolute tolerance .001.

## Checks

- Executed the exact modern-model Python block extracted from the chapter:
  logits `[2,6,16]`, cached/full suffix agreement, 4,960 unique parameters.
- Checked all three rounded RMSNorm examples, three SiLU values, and the
  nonzero derivative of a zero SiLU gate with nonzero content.
- Checked GELU/SwiGLU parameter counts 1072, 1536, and 1056.
- Checked the RoPE relative-position dot-product identity.
- Reconciled the three tiny-model parameter/cache/FLOP rows with the existing
  source: `(5472,6144,138240)`, `(4960,3072,125952)`, `(4704,1536,119808)`.
- Recalculated larger candidates without allocation: 50,480,512; 100,587,008;
  152,508,544 parameters.
- Checked hypothetical cache payload 201,326,592 bytes = 192 MiB.
- Verification process exited 0. Repository tests and local-link checks are
  recorded in the final review below.

Chapter SHA-256 at numerical verification:
`afbf1d401f68d02a5ec71e6992aa94bfadd2bd6a26f23ea630ef17774fb21252`.
Teaching-source SHA-256:
`52060c8a88093b49c91e0cdfa8ba37731347a9b8e38ab2dc851ca151888e9826`.

## Source-preserving notebook readiness

The immediately preceding readiness run executed all three Day 6 notebooks in
fresh `dgx-spark-native` kernels: 43 code cells, 13 figures, all passed. The
chapter check confirmed the notebook hashes still match that run. Executed
copies and its manifest are temporary at `/tmp/day6-readiness-fxqiwx7c`; source
notebooks, the earlier canonical reports, and these hashes are portable.

| Notebook | Code cells | Figures | Execution seconds |
|---|---:|---:|---:|
| `01_rmsnorm_and_swiglu.ipynb` | 15 | 5 | 2.30 |
| `02_rotary_positions.ipynb` | 14 | 4 | 2.38 |
| `03_gqa_qknorm_and_costs.ipynb` | 14 | 4 | 2.35 |

Notebook SHA-256 values, in the same order:

- `ea3d5e595717faf171b4e1950cba9f523ac3b81128c284d22d34349bc209423b`
- `2554495c9937f49004ea72594b69b32d04f271c85d0c0561b8c0e68e4274a35d`
- `38f5cdf6ff997a51de550abd3930b87c02fdc44c5fb165331af0cc5bf53e5504`

These durations include kernel/notebook execution and are not inference-latency
benchmarks. Kernels emitted local TCP-encryption warnings; no network-security
claim follows from successful execution.

## Sources and limits

Rechecked the pinned Qwen3 config and primary abstracts for RMSNorm, GLU variants,
RoFormer, GQA, original QKNorm, recurrent depth, and Mixture-of-Recursions. Direct
citations and versions are in the chapter. The Qwen comparison derives shapes
from configuration, not a checkpoint load. The lab's RMS-based Q/K normalization
is explicitly distinguished from the original paper's L2-based variant.

Numerical checks establish narrow algebraic and implementation properties. They
do not establish learned specialization, a quality gain for any modern variant,
long-context behavior, training-memory feasibility, serving speed, or learner
mastery. Parameter/cache/FLOP values have different meanings. Any trained
recurrence comparison requires its own predeclared experiment.

## Final review

- All 47 repository tests passed in 3.372 seconds; process exit 0.
- 95 local links and seven heading anchors resolved across changed Markdown.
- Chapter numbering covers sections 5.1–5.22; worked answers cover 1–24.
- Code/math fences balanced; `git diff --check` passed.
- Notebook sources have no diff. No commit, push, or animation rendering was
  performed as part of this writing task.
