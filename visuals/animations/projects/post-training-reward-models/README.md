# Reward models: the X-POST-005 GIF set

Eleven GIFs for the article *Reward Models: How a Choice Between Two Answers Becomes a Score*
(`publications/x-articles/x-post-training-005/`). Candidates `CAND-ANIM-044, 041, 035, 042` (two clips), `048, 043` (two clips), `046, 050, 051`. The learner chose the set and asked for production on 2026-10-30 ("use the 9 GIFs",
"start producing the 9 GIFs"). Storyboards in `visuals/animations/PROPOSALS.md` were recast to the
article's running example, *Is OpenAI open?*.

```sh
uv run --project visuals/animations python visuals/animations/projects/post-training-reward-models/render.py --preview
uv run --project visuals/animations python visuals/animations/projects/post-training-reward-models/render.py
```

Each `rendered/<stem>/` holds:
- a 1280×720 GIF at 20 fps;
- a 1080p MP4 (ignored by Git);
- a poster;
- a contact sheet.

Every final GIF and MP4 carries the DONGXI LABS mark at the bottom-right (`../../brand.py`). Each
scene asserts at every timeline mark that the corner is empty. The review page is `gifs.html`;
serve this folder and open it.

## Evidence

- **Real:** the token strings and IDs (Qwen3 tokenizer); ` Some` (4329) and ` All` (2009) differ at
  one position; 151,936 LM-head rows; 28 layers.
- **Chosen, not generated:** the answers A and B.
- **Illustrative:**
  - scores, logits, hidden vectors and per-token values;
  - the twelve pairs of 03;
  - the arrows of 07 and the curves of 08.
- **Exact on those numbers:** every sigmoid, softmax, loss and gradient step. `render.py` checks the
  key values, such as 0.269 / 1.313 / 0.731, 0.705 / 0.259 / 0.035, and gap 0 → 1.0 with loss 0.313.
- **Simplifications:** 03 moves each pair independently (gap += 2 · push, lr = 1); in a real reward
  model, pairs share weights. 08 shows the shape reported by Gao et al. (2022); the "true quality"
  there was a larger reward model standing in for humans.
- **Not shown:** no numbers from our own notebooks or runs (learner, 2026-10-30).

## Cast

| colour | meaning |
|---|---|
| green | the chosen answer |
| slate | the rejected answer |
| violet | network weights: body, LM head, score head |
| blue | scores, gaps, probabilities |
| amber | the loss, label errors, the illustration badge |

## Captions

1. **One choice, one row.** A person picks A over B. The choice becomes one row: prompt, chosen, rejected. Rows stack into the training set of a reward model.
2. **Bradley–Terry.** Two scores; only the gap matters (+10 to both changes nothing). The sigmoid turns gap −1 into 0.27; loss 1.31. Each step raises A and lowers B by 1 − σ(gap): gap −1 → +2, loss 1.31 → 0.13.
3. **Where the push goes.** Twelve pairs on the gap axis. The push 1 − σ(gap) is large on pairs in the wrong order and fades on pairs ranked right. A vs B: gap 0 → 1.0, loss 0.693 → 0.313, push 0.5 → 0.269.
4. **Chat model: the LM head.** h sits above the LM head, column over column; each logit is h · one row: Part 2.0, Yes 1.0, No −1.0, one of 151,936 rows each. Softmax, pick a token, and the loop runs again.
5. **Reward model: one head row.** Same body, one row in the head: h · that row is the score, 2.0 for A. Answer B takes its own pass and scores 1.0. A > B.
6. **Which position gives the score.** The score head gives a number at every token. Changing Some to All leaves earlier numbers the same and changes every later one. The last token has read the whole answer: its number is the score.
7. **Chat model: softmax over all tokens.** One pass gives logits for every vocabulary entry; softmax turns each column into a probability (0.705 / 0.259 / 0.035, sum 1); one token is sampled. Training: −log p(next token).
8. **Reward model: a two-way softmax.** One pass per answer, one score each. Softmax over the two answers gives 0.731 / 0.269 = σ(1), used only in the loss −log p(A). In use, the raw score goes out as the reward.
9. **Random noise vs systematic bias.** Two learned weights, on accurate and on long. Random flips point both ways: both shrink, same ratio. Flips that always favour the longer answer add up: the weight on length grows. More data of the same kind changes neither.
10. **A proxy, optimized.** A longer answer with no new content scores higher. Optimized against, the reward model score keeps rising while true quality rises, peaks and falls: reward hacking. Illustrative shape after Gao et al. (2022).
11. **One reward model, three uses.** Frozen, it only scores: best-of-n keeps the top answer; labeling turns two scored answers into a new row; in RLHF the score is the reward that updates the policy.

## Status

- 2026-10-30: rendered on the Mac Studio. The agent reviewed the preview contact sheets and fixed:
  - layout and size in 04, 05 and 06;
  - overlapping labels in 02, 03 and 07;
  - fading that turned chips solid in 09;
  - the score order in 05: the wrong variant now scores lower.
- 2026-10-30, learner review: 04 "very rough", unlike the others; arrows "not vertical or horizontal".
  - 04 rebuilt in the set's chip style, with h aligned over the head rows.
  - Every connector in every clip is now horizontal, vertical or a right-angled elbow, with no
    curved arrows.
  - `Loop.mark` fails the render on any slanted or curved arrow.
  - 07 redrawn as weight bars instead of angled vectors.
  - 09's beams became a tree of straight lines.
- 2026-10-30, learner: 04 and 06 "too crowded"; separate the chat LLM and the reward model into their own
  GIFs, placed next to each other in the article. 04 became 04 chat head + 05 score head, and 06 became
  07 chat softmax + 08 two-way softmax (the training part framed as training only). The set is renumbered 01–11.
- Awaiting the learner's review.
