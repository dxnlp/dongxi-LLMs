# Dongxi X Article production defaults

These conventions apply to new X Article packages unless a reviewed article has
a specific exception.

## Article packages

- [Prefill and Decode](x-prefill-decode-001/zh/review.html): independent Chinese
  mechanism-first draft, seven GIF animations with static fallbacks and a 5:2
  cover; reusable 1080p MP4s retained locally. Spark/Mac is a short supporting
  case. Local review only; no X upload or publication.
- [LLM inference: latency, memory and scheduling](x-inference-001/zh/review.html):
  Chinese local draft, seven static mechanism figures and a 5:2 concept cover;
  primary-source map and motion proposals retained. Not uploaded or published.
- [Why Does a Language Model Need Post-Training?](x-post-training-001/review.html):
  post-training series 1/4, English and [Chinese](x-post-training-001/zh/review.html)
  drafts, five GIFs, comparison figure and the series cover. Not uploaded or published.
- [Text, tokens and BPE](x-bpe-001/review.html): English and Chinese visual drafts.
- [KV cache: why K/V, not past Q](x-attn-kv-001/review.html): English draft,
  [Chinese adaptation](x-attn-kv-001/zh/review.html), five figures and optional
  DeepSeek companion animations. Local editorial review; not published.

## Cover images

- Use a **5:2 aspect ratio** for X Article covers.
- Default editable/render size: **2000×800 px**. A proportional 1600×640 export
  is also acceptable when a smaller file is needed.
- Keep the title and meaningful artwork inside generous horizontal safe margins;
  do not rely on edge content surviving every preview crop.
- Use one dominant visual idea. Cover typography and artwork must remain editable
  and reproducible.

Inline article figures may use another ratio when their mechanism or data needs
it; the 5:2 rule applies specifically to the X Article cover.

### Post-training series cover style (required for X-POST-001 to X-POST-004)

Approved by the learner on 2026-10-20 for every article in the post-training
series. Reference implementation:
[`x-post-training-001/build_cover.py`](x-post-training-001/build_cover.py);
copy it and change only the title, series number and art motif.

- Deep ink background with a faint cool glow; smooth vector "engraved" ribbons
  (pycairo) with light stipple dust. Random-stipple art was rejected as rough.
- Title: solid white Avenir Next Heavy on one line. No gradient title.
- Accent colors in the art may use the semantic trio blue `#60A5FA`, violet
  `#A78BFA`, green `#34D399` (show / compare / check).
- Gold `#FBBF24` letter-spaced capital labels: `POST-TRAINING SERIES · NN` at top
  left and `DONGXI LABS` at top right. No article ID on the cover.
- One English cover is shared by the English and Chinese versions.
