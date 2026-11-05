# KL: the KL article GIF set

Twelve GIFs for the article *KL Divergence: One Number for How Far a Model Has Moved*
(`publications/x-articles/x-post-training-kl/`). Candidates `CAND-ANIM-058` (two clips), `059`, `065`, `060`,
`061`, `054`, `055`, `063`, `037`, `064`, `066`. The learner chose the set and asked for production on
2026-11-04 ("start producing the 12 GIFs"). Storyboards are in `visuals/animations/PROPOSALS.md`, with the
updates listed under "KL article updates to existing candidates".

```sh
uv run --project visuals/animations python visuals/animations/projects/post-training-kl/render.py --preview
uv run --project visuals/animations python visuals/animations/projects/post-training-kl/render.py
```

Each `rendered/<stem>/` holds a 1280×720 GIF at 20 fps, a 1080p MP4 (ignored by Git), a poster and a contact
sheet. Every final GIF and MP4 carries the DONGXI LABS mark at the bottom-right (`../../brand.py`). Each scene
asserts at every timeline mark that the content is inside the canvas, the brand corner is empty and every
arrow is horizontal or vertical. `render.py` asserts each clip's numbers against the values in the article.
The review page is `gifs.html`; serve this folder and open it.

GIF 03 shows the KL formula on screen, an exception to the series' "no equations on screen" rule, because the
article is about this formula (recommended in the plan; the learner has not ruled on it separately).

## Evidence

- **Illustrative:** the shares, rewards and gold values of the egg example; the token probabilities of GIF 10.
- **Exact on those numbers:** every ratio, log, sum, score and optimum (`check_numbers.py` in the article package).
- **One seeded draw:** GIF 04 and GIF 09 (NumPy seed 0).
- **Toy:** GIF 11; each point is the best compromise at a smaller β, standing in for longer training.
- **Published:** the two β values in GIF 10 (InstructGPT, DeepSeekMath).
- No numbers from our own notebooks or runs appear.
