# Foundations editorial relocation map

Scope: Chapters 1–3 and their corresponding solutions. Exact removed source
contexts, final file hashes and all exercise/solution ID mappings are retained
in `foundations-editorial-review.json`. The immutable original bytes remain in
`source-before/book/`.

| Origin | Appendix D destination | Retained in chapter |
|---|---|---|
| 1.5 DPO deadline and RLVR cleanup chronology | D.6 checkpoints, recovery and job supervision | Success is the conjunction of the declared criteria; a passing native child does not erase failed acceptance |
| 1.10 manifests, checkpoint adoption and adapter-export machinery | Supervised fine-tuning state and adapter exports | Eight-ID swapped meanings, changed normalization, full supported semantic interface, source declaration versus actual bytes versus observed computation |
| 2.8 detailed two-cap decoder failure accounting | Policy reference and pending rollout identity | Model output rows can exceed tokenizer coverage; actual unmapped output ID 151768 is neither input UNK nor an embedding-index failure; a support mask changes the distribution |
| 2.10 host-specific reproduction commands | D.6 checkpoints, recovery and job supervision | Mechanism modules, CPU lab route and Appendix A environment navigation |

Chapter 1 now names all four companion notebooks. The interface fixture requires
no pretrained weights; its local random weights remain part of the mechanism.
Exercises 8–9 are explicit optional evidence-reading extensions, with their
original answers and IDs preserved. Chapters 2 and 3 retain all twelve exercise
and solution IDs each.

Chapter 3 defines teacher forcing at its first use and places a reader prediction
before the controlled 70/30 trajectory. The stored output weight is `[V,D]`,
row logits are `h @ W_out.T`, the weight gradient is the outer product of the
logit gradient and state, and the state gradient is `g @ W_out`. All five existing
Python examples across the six files remain byte-identical to the baseline.
The existing code's `T` is mapped to mathematical sequence length `n`.

Final owned-file mechanical and narrative findings are zero. The math source
check passes; one bounded float64 CPU algebra check verified the output-head
gradients against autograd. Local destination files exist, and the owned diff
has no whitespace errors. Root final-source verification supplies global
acceptance after concurrent work.

The focused symbol audit for these files and the earlier six chapters is in
`owned-notation-review.json`. No model job, source-module edit, archived report
rewrite, figure rerender or learner-progress reassessment occurred.
