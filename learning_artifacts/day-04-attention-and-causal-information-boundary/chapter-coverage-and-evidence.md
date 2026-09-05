# Chapter 4 Coverage and Evidence

- Date: 2026-09-05
- Book: `book/chapters/04-attention-and-the-causal-information-boundary.md`
- Solutions: `book/solutions/04-attention-and-the-causal-information-boundary.md`
- Material status: complete against the Day 4 outline; notebook practice deferred
  at the learner's request. Material completeness is separate from demonstrated
  mastery.

| Required topic | Chapter section | Executable or worked evidence |
|---|---|---|
| Q/K/V roles and shapes | 4.2 | Session 1 projections and shape inspection |
| Learned query-key compatibility | 4.3 | Bilinear derivation; exercise 2 |
| Scaling, variance, softmax sensitivity | 4.4 | Session 2 IID simulation |
| Causal masking and future leakage | 4.5 | Session 1 broken masks and prefix intervention |
| Value mixture versus token output | 4.6 | Session 1 explicit mixture and head distinction |
| Transparent implementation and cost | 4.7 | Source, SDPA comparison, forward tests |
| Gradients and prompt credit | 4.8 | Session 2 manual derivatives, autograd, finite differences, detach |
| Interpretation limits | 4.9 | Session 2 distinct distributions with the same mixture |
| Cache validity and context identity | 4.10 | Session 3 two-layer full/cached replay and stale prefix |
| Lifecycle, memory, systems boundary | 4.11 | Session 3 payload accounting; analytical lifecycle treatment |
| Controlled experiments and limits | 4.12 | Two experiment reports; 26 repository tests pass |
| Exercises and next-chapter bridge | 4.13–4.14 | Twelve worked solutions and Chapter 5 transition |

## Discussion captured

The learner initially predicted that changing the fourth token could change
the earlier output at “crossed.” The correction is that causal attention at
“crossed” has exactly the same allowed prefix, so its output must remain the
same; the fourth output may change. This correction is now explained in the
chapter, notebook 1, and worked exercise 5. No later explanation-back is claimed.

The learner subsequently requested skipping live notebook work and generating
the complete chapter, learning materials, and notebooks. The book now covers
all required topics regardless of the point reached in interactive practice.
Future sessions may proceed to the decoder chapter while retaining Day 4
practice as deferred.

## Evidence and content reuse

- Forward report: `experiments/reports/2026-09-05-causal-attention-forward.md`.
- Gradient/cache report: `experiments/reports/2026-09-05-attention-gradients-cache.md`.
- `ANIM-ATTN-001` now has canonical chapter, gradients, mask/detach examples, and
  numerical evidence. Production stays on the Mac Studio.
- `CAND-ANIM-009` has a verified toy cache example; no serving benchmark is claimed.
- `X-ATTN-KV-001` can draw on the canonical chapter and cache equivalence report.
- Recurrent-depth material remains assigned to Chapter 5/Days 6–7.
