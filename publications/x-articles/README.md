# Dongxi X Article production defaults

These conventions apply to new X Article packages unless a reviewed article has
a specific exception.

## Article packages

- [The Birth of a Token](x-token-001/review.html): standalone general-reader article (outside the
  post-training series), English and [Chinese](x-token-001/zh/review.html) drafts, seven inference GIFs
  (`CAND-ANIM-040`) and its own cover. Chinese version published on X by the learner (reported
  2026-10-29); English not transferred.
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

### Post-training series cover style (required for every article in the series)

Approved by the learner on 2026-10-20 for every article in the post-training
series. Reference implementation:
[`x-post-training-001/build_cover.py`](x-post-training-001/build_cover.py);
keep its style and change the title and series number, but draw a new art motif that
shows that article's technique. Reusing the previous cover's art was rejected (2026-10-22).

- Deep ink background with a faint cool glow; smooth vector "engraved" ribbons
  (pycairo) with light stipple dust. Random-stipple art was rejected as rough.
- Title: solid white Avenir Next Heavy on one line. No gradient title.
- Accent colors in the art may use the semantic trio blue `#60A5FA`, violet
  `#A78BFA`, green `#34D399` (show / compare / check).
- Gold `#FBBF24` letter-spaced capital labels: `POST-TRAINING SERIES · NN` at top
  left and `DONGXI LABS` at top right. No article ID on the cover.
- One English cover is shared by the English and Chinese versions.

## Post-training series conventions

Learner-approved through X-POST-001 (2026-10-20/21). Later articles in the series
follow these unless the learner changes them. Reference package:
[`x-post-training-001/`](x-post-training-001/).

**Series plan** (revised 2026-10-22: LoRA/QLoRA get their own article; revised 2026-10-27: RL
terms article inserted before preference alignment):
`X-POST-001` why post-training · `X-POST-002` instruction tuning (SFT) ·
`X-POST-003` LoRA and QLoRA · `X-POST-004` RL terms for LLMs · `X-POST-005` reward models ·
`X-POST-006` RLVR. (Revised 2026-10-30: `X-POST-005` is solely about reward models; PPO, GRPO and
DPO need their own article, slot to be decided by the learner.)
Article 1 is beginner-level; later articles go deeper, each on one technique.

`X-POST-004` **RL terms for LLMs** (slot approved by the learner 2026-10-27: before preference
alignment): state, action, reward, return, value, Q, advantage, baseline, TD/GAE, policy
gradient, ratio/clipping, KL, exploration, with many GIFs on one running example. Source: learning
note glossary (`learning_artifacts/day-15-preference-data-and-bradley-terry/learning-from-preferences.md`,
D39); animation set `CAND-ANIM-038`. Thirteen GIFs approved 2026-10-28; Chinese version published by
the learner (reported 2026-10-29); English not transferred (`x-post-training-004/`).

`X-POST-005` **reward models** (planned; scope set by the learner 2026-10-30: solely reward models,
plan in [`x-post-training-005/plan.md`](x-post-training-005/plan.md); **read the learning note
`learning_artifacts/day-15-preference-data-and-bradley-terry/learning-from-preferences.md` before
outlining or drafting**). Must include a section on the **chat model vs reward model head** (learner request, 2026-10-30, S1 Part A): same pretrained body, LM head `1024 → 151,936`
replaced by a `1024 → 1` score head; one forward pass, one number read at the last token, no
generation; the pick is a comparison of two scores. Animation `CAND-ANIM-042`; source: learning note
D28, D30, D42. Must also include ★ a section on **chat-model logits vs reward-model scores** (learner, 2026-10-30:
"important"; Bradley–Terry as a softmax over two answers), animation `CAND-ANIM-043`, source D47. ★ **Key point** (learner, 2026-10-30): every logit is `h · one head row`; the same body gives `h`, 151,936 rows give token logits → softmax → token ID → detokenizer, one row gives one logit = the score (source D48; see "Key points" in the plan).
Beginner Bradley–Terry animation `CAND-ANIM-041` is also a candidate for this article. Full animation list (033–035, 041–051, with a recommended core of 7) in the plan's **Animations** table (2026-10-30).

**Text**

- Two versions: English at the package root, an independently written natural
  Chinese version in `zh/` (not a literal translation).
- Chinese: never use 不是……而是…… or its variants (并非…而是, 不在于…而在于,
  与其…不如); `build.py` scans for them.
- Chinese terms: on first use write the English term with the Chinese in
  parentheses, e.g. `pretraining（预训练）`, `supervised fine-tuning（SFT，监督微调）`;
  afterwards Chinese alone is fine.
- Core technical terms stay in English with no Chinese gloss (learner, 2026-10-28, X-POST-004):
  policy, state, action, reward, return, value, advantage, baseline, TD error, GAE, clipping, KL,
  and compounds built from them (policy gradient, action value, reward model, sparse reward,
  importance ratio, reward hacking, KL penalty), plus trajectory, exploration, credit assignment,
  agent, environment and judge. Other
  terms keep the English (中文) first-use rule.
- Open with a concrete, verified, eye-catching example (article 1: OpenAI's
  GPT-3 vs InstructGPT moon-landing comparison), then the reader's question.
- Quote blocks (`>`) only for a few key sentences, about 4–5 per article. Use quotation
  marks (“” / "") as little as possible (learner, 2026-10-22). Bold terms on first
  definition with a plain gloss.
- Precise wording over figurative language: no empty metaphors (换了一份工作, 猜错了就扣分,
  举一反三, "a new job") and few adverbs (笃定地, confidently). Say the technical thing plainly.
- Stay big-picture (role, meaning, method, data, limits). Technical details such as the
  chat template are welcome; detailed reports of our own experiments are not.
- H2 headings only (X Articles handles H2 more reliably than H3). Short
  sentences, plain words before notation. Beginner article 1 used a few analogies;
  later articles prefer precise statements.
- End with a short summary list and a pointer to the next article.
- No links in the public body; every claim maps to `source-map.md`. State
  evidence limits in prose or captions; never present illustrations as measurements.

**Figures and animations**

- **Animated GIFs are the default for every inline figure** (learner request,
  2026-10-22): motion catches the reader's eye in the X feed. Even a comparison
  or a measured result should move, e.g. a table that builds row by row or bars
  that grow. A static image only when motion would add nothing, and then say why
  in `visual-plan.md`.
- Manim, Dongxi visual system (`visuals/animations/STYLE_GUIDE.md`): white canvas,
  semantic colors, Arial labels at title 40 / labels 24–32, one compact amber badge
  for caveats ("illustrative probabilities"), no equations on screen in beginner
  articles. Source in `visuals/animations/projects/<topic>/` (`scenes.py`, `render.py`).
- Final output: 1920×1080 30 fps MP4 (git-ignored) plus a 1280×720 20 fps looping GIF
  (one global 256-color palette, no dithering, under 5 MB) written to the article's
  `assets/`. Encode GIFs from a lossless render (Manim `-t` → qtrle `.mov`, composited
  on white), never from the H.264 MP4: compression noise made one GIF 14.9 MB, the
  lossless source 0.85 MB (`post-training-sft/render.py`). English labels; both language versions share the GIFs.
- All figures are English only, GIFs and static images alike; the Chinese article uses
  the same English figures (learner, 2026-10-22). Article 1's Chinese comparison table
  predates this rule.
- About 5–6 inline GIFs per article; each caption states what is illustrative.

**Package and transfer**

- Files: drafts with `[IMAGE NN]` placeholders, `image-upload-order.md` (cover listed
  separately), `build.py` (HTML bodies, local `review.html`, hash metadata),
  `build_cover.py`, `source-map.md`, `visual-plan.md`, `README.md`.
- Review locally (`python3 -m http.server`) before any X work. X transfer is
  draft-only, never Publish; method in `x-post-training-001/X_DRAFT_TRANSFER.md`.
