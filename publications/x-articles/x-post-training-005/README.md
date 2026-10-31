# X-POST-005 — Reward Models: How a Choice Between Two Answers Becomes a Score

Post-training series, article 5. It introduces reward models to LLM readers. It is solely about
reward models; PPO, GRPO and DPO belong to a later article. The plan, decisions and outline are in
`plan.md`.

## Status

- 2026-10-30: plan set with the learner: 11 sections, running example *Is OpenAI open?*, nine GIFs,
  the title, no details of our own experiments, and the DONGXI LABS mark on every GIF and MP4.
- 2026-10-30: cover built (`build_cover.py`; the upright variant was chosen).
- 2026-10-30: English and Chinese drafts written; paper facts checked against primary sources
  (`source-map.md`). Awaiting the learner's review.
- 2026-10-30: GIFs rendered with the DONGXI LABS mark; after review, the head and softmax clips were
  split into chat-model / reward-model pairs, giving eleven GIFs
  (`visuals/animations/projects/post-training-reward-models/`), copied to `assets/`; agent-reviewed,
  awaiting the learner's review.
- 2026-10-31: the Chinese version was transferred to X draft 2116301274433961984 and reload-verified
  (`X_DRAFT_TRANSFER.md`).
- 2026-10-31: the learner reported that the Chinese article is published on X (published by the
  learner). The English version has not been transferred.

## Files

- `x-editor-draft-body-with-image-placeholders.md` — English draft.
- `zh/x-editor-draft-body-with-image-placeholders.md` — Chinese draft, written independently.
- `image-upload-order.md`, `zh/image-upload-order.md` — figure order and captions (planned files).
- `source-map.md` — every factual claim and its source.
- `build_cover.py` → `assets/cover-post-training-05.png` (`--mirror` gives the unused variant).

```sh
uv run --project visuals/animations python publications/x-articles/x-post-training-005/build_cover.py
```

## Writing rules

Series conventions in `../README.md`, plus:
- Chinese: core RL and LLM terms stay in English (reward model, reward, policy, logit, softmax, …),
  and the 不是……而是 family is not used.
- Few quotation marks; 5 quote blocks per version; no metaphors; few adverbs.
- Numbers are public facts, exact arithmetic on illustrative values, or labelled illustrations.
