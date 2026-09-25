# LLM inference: latency, memory and scheduling

Task `X-INFER-001`. Approved local Chinese article production, 2026-09-13.
Base: `ca2cfe1` on `main`, clean fast-forward sync confirmed. Execution: Mac.

## Scope

Short autoregressive primer, then user-visible latency, continuous batching,
paged KV storage, prefix caching, chunked prefill, prefill/decode disaggregation,
speculative decoding, quantization and multi-GPU strategies. The user explicitly
expanded scope beyond token generation. This is a sourced conceptual overview,
not a deployment tutorial, serving benchmark or change to Day 8 priorities.

Chinese public manuscript: `zh/x-editor-draft-body-with-image-placeholders.md`.
English technical terms and figure labels retained. No links in the public body;
claim-level links reside in `source-map.md`. Editorial title and 5:2 cover remain
subject to review. Existing KV article and cloud draft must remain unchanged.

## Deliverables

- `zh/review.html`: self-contained local review with seven static mechanism figures.
- `assets/`: original reproducible PNG figures and 2000×800 concept cover.
- `build.py`: figure renderer and Markdown-to-HTML builder using existing Python environment.
- `zh/x-editor-clean-body*`: skill-generated draft body and seven-image insertion plan.
- `source-map.md`: primary-source claim map and boundaries.
- `visual-plan.md`: proposed motion treatment, not rendered animations.
- `qa.json`: structural/arithmetic/media checks, inspection status and limitations.

## Build and check

```bash
uv run --project visuals/animations --with mistune==3.3.4 python publications/x-articles/x-inference-001/build.py
python3 /Users/yongchao/.codex/skills/x-article-drafter/scripts/build_x_article_body_plan.py publications/x-articles/x-inference-001/zh
uv run --project visuals/animations python publications/x-articles/x-inference-001/check.py
```

No model inference, benchmark, GPU job, new video render, X upload, publication,
commit or push is included in this production step. Diagrams are schematics;
only the explicitly labeled weight-payload example is calculated numerical data.
Browser inspection status is recorded in QA rather than inferred from successful build.

## Course placement

Chapters 3–5 supply the generation, causal cache and resource-accounting base.
Detailed production serving is explicitly deferred beyond v0.1 in `ROADMAP.md`.
This user-approved public overview is a bounded extension, not a new course
chapter or a claim that serving topics have been taught or empirically tested.
