# RL Terms for LLMs: How One Score Reaches Every Token

`X-POST-004`, fourth article of the six-part post-training series (series plan and conventions:
`../README.md`). It is inserted before preference alignment (slot approved 2026-10-27).

## Status

- 2026-10-27: planning. Read `plan.md` for the article outline, running example and open decisions,
  and `visual-plan.md` for the GIF storyboard, visual grammar and cover.
- 2026-10-27/28: the learner approved the GIF 05 style pilot (chat bubbles, user grey, assistant
  blue, no outer frame). The remaining eleven GIFs were rendered in that style.
  - All thirteen are in `assets/` at 1280×720, 20 fps, 0.5–1.5 MB each, except 08 at 4.1 MB (its
    branches change on every frame). All are under the series' 5 MB limit.
  - Review page: `gifs.html`; serve this folder on port 8768.
  - Source: `visuals/animations/projects/post-training-rl-terms/`.
- 2026-10-28: all thirteen GIFs reviewed and approved (log below).
- 2026-10-28: EN and ZH drafts and the cover written, awaiting learner review.
  - Drafts: `x-editor-draft-body-with-image-placeholders.md` and `zh/…`. EN is about 2,850 words;
    ZH is about 3,750 CJK characters. Each has 12 H2 sections, 5 quote blocks and 13 images, and no
    quotation marks.
  - Cover: `build_cover.py`, a tree of token paths whose lit path has a value line that steps down
    at a low-scoring branch.
  - Review pages: `review.html` and `zh/review.html` on port 8768. Claims map to `source-map.md`.
  - The DeepSeek-R1 claims (rewards, AIME numbers, PRM problems, readability, language consistency)
    were checked against arXiv:2501.12948v1 on 2026-10-28.
- 2026-10-28: the Chinese version was transferred to X draft 2115385031132635138 and reload-verified
  (`X_DRAFT_TRANSFER.md`).
- 2026-10-29: the learner reported that the Chinese article is published on X (published by the
  learner). The English version has not been transferred.

## GIF review log (2026-10-28, one by one)

- 01: approved as is.
- 02: the label stays "judge". The article defines it as reward model, verifier, tests or LLM
  judge, and explains why it sits in the environment (see `plan.md`, section 2).
- 03: approved as is.
- 04: approved as is. Discussion of sparse reward, γ = 1, return and right-answer-wrong-trajectory
  is recorded in `plan.md` under "Writing notes from the GIF review".
- 05: the learner could not tell from the figure what V, Q and A mean. Revised (agent design,
  learner asked for clarity without visual load):
  - the state sits on the V line, so V belongs to the state;
  - Q is written next to each branch, so Q belongs to state plus token;
  - samples of one branch score differently (open 3, 1, 2, 1, 3; closed 5, 4, 6; partly 10, 8), so
    Q is visibly an average;
  - the score stamps and the Q labels under the axis were removed.
  All numbers (V = 4.3, Q = 2/5/9, A) are unchanged. Re-rendered; approved by the learner.
- 06: unified with 05's frame (state on the V line, Q beside the cards, branch averages on the
  axis). The reference rises from 0 to V and lands on the state. The stamps and both on-screen
  sentences were removed, and the formula is A = Q − V. R versus Q is explained in `plan.md`.
  Re-rendered; approved by the learner.
- 07: unified with 05's chat style. The answer sits in one assistant bubble, the V line is amber
  and dashed, the TD error pill sits beside the drop, and the zeros were removed from the
  advantage row. Re-rendered; approved by the learner. The plan's claim that λ = 1 equals
  GRPO credit was wrong and is corrected in `plan.md`.
- 08–12 (learner, 2026-10-28: redo every remaining figure on 05's frame with less text; agent
  design):
  - 08: the branches reshape step by step while the state and its V line rise.
  - 09: greedy versus sampling on the same tree, with unknown Q shown as "Q = ?".
  - 10: a ρ meter beside the tree.
  - 11: π branches over grey π_ref branches, beside the stacked bar.
  - 12: the term map became the 05 picture with the core terms placed on it.
  06 was re-rendered after the shared frame was factored out. Awaiting review from 08.
  - Render fix: `always_redraw` copies each frame into the previous frame's structure, so a tag
    group whose size changed lost its text mid-motion. `adv_arrows` now always returns one
    (arrow, tag) pair per branch, with the hidden ones transparent.
  - 08: approved by the learner.
  - 09: approved by the learner. The article notes that greedy's V = 2 is its own estimate from
    the samples it saw; the true V is 4.3.
  - 10: approved by the learner. The article explains π_old: one batch, several update steps.
  - 11: approved by the learner. The article names reward hacking, says real KL is per-token and
    summed, and says β = 1 is illustrative.
  - 12: approved by the learner. The learner chose to keep the five-card map as figure 13 (a
    quick-reference sheet) right after it. All thirteen figures are reviewed.
- Learner: advantage was not clear and must be explained fully in section 5 of the article. The
  build order is recorded in `plan.md`.

## Sources

- D39 glossary and D20–D25, D35–D38 in
  `learning_artifacts/day-15-preference-data-and-bradley-terry/learning-from-preferences.md`.
- Book Chapter 12.
- SLP3 Chapter 8.
- DeepSeek-R1 paper, for the hook.
