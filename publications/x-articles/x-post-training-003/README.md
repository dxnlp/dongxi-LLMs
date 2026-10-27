# LoRA and QLoRA: Fine-Tuning a Large Model on One GPU

`X-POST-003`, third article of the five-part post-training series (series plan and
conventions: `../README.md`). English at the package root, an independently written Chinese
version in `zh/`.

Scope (2026-10-23): LoRA and QLoRA in one article, organized around the learner's key idea
that the **pattern** (one column times one row, a rank-1 matrix) connects rank, SVD,
low-rank approximation, LoRA, quantization and QLoRA. Basic concepts (low rank,
approximation, quantization, NF4, block scales) are explained as in the Day 11–12 learning
note. Hook: QLoRA's 65B model on one 48 GB GPU.

## Status

- Text drafts in English and Chinese: ready for review (2026-10-23). 3 quote blocks, no
  quotation marks, Chinese banned-pattern scan clean.
- Ten GIFs and one static pattern-lens card (`ANIM-POST-003`), approved and rendered 2026-10-23,
  English only, 0.3–1.75 MB per GIF (03 holds a photo), 1280×720 at 20 fps.
- Cover `assets/cover-post-training-03.png`: frozen grid, three glowing column × row patterns, 4-bit levels.
- Local review: `review.html`, `zh/review.html` (serve the folder, port 8767). X image plans: 11 images, 0 missing.
- Chinese version transferred to X draft 2113912265061593088 on 2026-10-24 and reload-verified
  (`X_DRAFT_TRANSFER.md`). Not published; English not transferred.

## Files

- `x-editor-draft-body-with-image-placeholders.md`, `zh/…`: manuscripts with `[IMAGE 01–11]`.
- `image-upload-order.md`, `zh/image-upload-order.md`: planned figures and captions.
- `source-map.md`: every claim mapped to the LoRA/QLoRA abstracts, SLP3 §8.2 or the learning note.
- `visual-plan.md`: figure storyboard and cover concept. `build_cover.py`, `build.py`: cover and HTML/review build.
- Animation source: `visuals/animations/projects/post-training-lora/`.

No model runs or publication.
