# KL article — KL Divergence: One Number for How Far a Model Has Moved

Post-training series. It explains KL from scratch to readers with limited math. The KL/DPO article
(`../x-post-training-kl-dpo/`) comes after it and uses KL as a known quantity. Plan: `plan.md`.

## Status

- 2026-11-04: plan started by the learner; KL article first.
- 2026-11-04: audience set by the learner: limited math knowledge; explain KL in detail and in an easy
  way. Plan revised for that reader; English and Chinese drafts written; numbers in `check_numbers.py`;
  paper facts checked (`source-map.md`). Awaiting the learner's review.
- 2026-11-04: twelve GIFs rendered with the DONGXI LABS mark at the learner's request
  (`visuals/animations/projects/post-training-kl/`), copied to `assets/`; agent-reviewed, awaiting the
  learner's review.
- 2026-11-04: title approved by the learner; cover built (`build_cover.py`).
- 2026-11-04: the Chinese version was transferred to X draft 2118081140015656961 and reload-verified
  (`X_DRAFT_TRANSFER.md`). It has not been published.
- 2026-11-04: the English version was transferred to X draft 2118088509898596353 and reload-verified
  (`X_DRAFT_TRANSFER.md`). It has not been published. No commit or push.

## Files

- `x-editor-draft-body-with-image-placeholders.md` — English draft.
- `zh/x-editor-draft-body-with-image-placeholders.md` — Chinese draft, written independently.
- `image-upload-order.md`, `zh/image-upload-order.md` — figure order and captions .
- `check_numbers.py` — every number in the drafts.
- `source-map.md` — every factual claim and its source.
- `build_cover.py` → `assets/cover-post-training-kl.png`.
- `build.py` — HTML bodies and local `review.html` pages.
