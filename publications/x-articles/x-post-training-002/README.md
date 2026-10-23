# Instruction Tuning: Same Loss, a Different Task

`X-POST-002`, second article of the five-part post-training series (series plan and
conventions: `../README.md`). English at the package root, an independently written
Chinese version in `zh/`.

Agreed scope (2026-10-22): instruction tuning / SFT in depth. LoRA and QLoRA are left for
`X-POST-003`. Hook (revised 2026-10-22): Stanford's Alpaca, 52K AI-written examples for a few hundred
dollars. Revised again the same day to stay big-picture: role and meaning (LIMA, FLAN), the
recipe, then technical details (template, mask, shift), data and history, and limits. No
report of our own experiments in the article. The article has one
plain-text formula and one code line, a short paragraph on reasoning SFT, and stated
evidence limits.

## Status

- Chinese version transferred to X draft 2113363792482480128 and verified after reload (not published; see `X_DRAFT_TRANSFER.md`). English version ready, not transferred.
- Nine GIFs and one static summary (`ANIM-POST-002`), English only and shared by both versions, 0.5–1.0 MB per GIF, 1280×720 at 20 fps.
- Cover `assets/cover-post-training-02.png` (series style; token-chain motif: read → graded → stop).
- Local review: `review.html` and `zh/review.html` (serve the folder, e.g. port 8766).
- X image plans built with `x-article-drafter` (10 images each, 0 missing).

## Files

- `x-editor-draft-body-with-image-placeholders.md`, `zh/…`: manuscripts with `[IMAGE 01–10]`.
- `image-upload-order.md`, `zh/image-upload-order.md`: planned GIFs and captions.
- `source-map.md`: every claim mapped to SLP3 §8.1, the learning note, notebooks or reports.
- `visual-plan.md`: GIF storyboard. `build_cover.py`, `build.py`: cover and HTML/review build.
- Animation source: `visuals/animations/projects/post-training-sft/`.

No model runs or publication.
