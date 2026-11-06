# DPO: the DPO article GIF set

Thirteen GIFs for the article *Your Language Model Is Secretly a Reward Model: How DPO Works*
(`publications/x-articles/x-post-training-kl-dpo/`). Candidates `CAND-ANIM-079` (two clips), `067`, `057`, `052`, `069` + `053` (one
clip), `066`, `070`, `071`, `073`, `074`, `077`, `078`. Clip numbers are file names: clips 00a and 00b were added last at the learner's request and are the article's
images 1 and 2, so each later file is two images later. Clip 05 is KL GIF 12 (`066`) re-rendered here without its badge, so the KL package and its X drafts stay unchanged. The learner asked for production on 2026-11-06 ("start producing the GIFs"). Storyboards are in
`visuals/animations/PROPOSALS.md`; 052, 053 and 057 were told on the egg question for this set.

```sh
uv run --project visuals/animations python visuals/animations/projects/post-training-dpo/render.py --preview
uv run --project visuals/animations python visuals/animations/projects/post-training-dpo/render.py
```

Each `rendered/<stem>/` holds a 1280×720 GIF at 20 fps, a 1080p MP4 (ignored by Git), a poster and a contact
sheet. Every final GIF and MP4 carries the DONGXI LABS mark at the bottom-right (`../../brand.py`). Each scene
asserts at every timeline mark that the content is inside the canvas, the brand corner is empty and every
arrow is horizontal or vertical. `render.py` asserts each clip's numbers against the values in the article.
The review page is `gifs.html`; serve this folder and open it.

## Changes from the storyboards

- **No badges** (learner, 2026-11-06: "toy example 之类的gif里面的标签一概不要，只保留 DONGXILABS"). The only mark in a GIF
  is DONGXI LABS. The article text and captions state what is illustrative, simulated or invented. The KL set is
  unchanged.

- **069 + 053** are one clip: the bars level, then multiply, add, divide. The β dial stays in the article.
- **057** ends on DPO: stored pairs replace the reward model, and β and the reference go into the loss.
- **074** shows the records first and training second. The bars are the simulated run's own shares at 16
  training steps, including its overshoot: distance 0.33, a low of 0.01 at step 25, settling at 0.05.
- **078** uses invented curves with no values, following the series rule on our own runs. The storyboard's 97%,
  67%, 1.61 and 0.82 do not appear.

## Evidence

- **Illustrative:** the shares, rewards and gold values of the egg example; the token values of GIF 10.
- **Exact on those numbers:** every ratio, log, loss, push and optimum (`check_numbers.py` in the article package).
- **Simulated:** GIF 09 (notebook code, seed 0; the averages use eight datasets).
- **Invented:** the curves of GIF 11.
- **Published:** the paper subtitle in GIF 01.
