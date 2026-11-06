# DPO article — Your Language Model Is Secretly a Reward Model: How DPO Works

Post-training series, after the KL article (`../x-post-training-kl/`), which it treats as known. Written for LLM
beginners and readers with limited math. Plan: `plan.md`. Working title; article number to be confirmed.

## Status

- 2026-11-06: English and Chinese drafts written at the learner's request; numbers in `check_numbers.py`; paper facts
  checked in the primary PDFs (`source-map.md`). Awaiting the learner's review. No GIFs rendered (image 05 reuses
  KL GIF 12), no cover, no commit, nothing on X.
- 2026-11-06: ten GIFs rendered with the DONGXI LABS mark at the learner's request
  (`visuals/animations/projects/post-training-dpo/`), copied to `assets/`; agent-reviewed, awaiting the learner's
  review. Figure set and invented curves for GIF 11 follow the recommendation given on 2026-11-06.
- 2026-11-06: cover built at the learner's request (`build_cover.py`), labelled 07; the series number is not
  confirmed yet.
- 2026-11-06: Chinese language review (回答对 replaced by 配对), then a full Chinese rewrite in short sentences and short
  paragraphs from the learner's review comments; the English draft has not been restructured to match. Two shortcut GIFs added at the learner's
  request (`CAND-ANIM-079`, `assets/00a-two-routes.gif`, `assets/00b-change-is-the-score.gif`) as images 1 and 2, so there
  are now thirteen figures.

- 2026-11-06: the Chinese version was transferred to X draft 2118652439049986048 and reload-verified
  (`X_DRAFT_TRANSFER.md`). It has not been published. No commit or push.

## Files

- `x-editor-draft-body-with-image-placeholders.md` — English draft.
- `zh/x-editor-draft-body-with-image-placeholders.md` — Chinese draft, written independently.
- `image-upload-order.md`, `zh/image-upload-order.md` — figure order and captions.
- `check_numbers.py` — every number in the drafts.
- `source-map.md` — every factual claim and its source.
- `build_cover.py` → `assets/cover-post-training-dpo.png`.
- `build.py` — HTML bodies and local `review.html` pages.
