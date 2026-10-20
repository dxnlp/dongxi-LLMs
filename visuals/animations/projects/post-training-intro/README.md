# Why post-training: five beginner loops

`ANIM-POST-001` for the X article `X-POST-001`
(`publications/x-articles/x-post-training-001/`). Storyboard and packet:
`publications/x-articles/x-post-training-001/visual-plan.md`.

| Clip | Mechanism | Length |
|---|---|---|
| `01-same-prompt-two-models` | Same request; base model's likeliest next words start a question, post-trained model's start an answer | 11.0 s |
| `02-two-stages` | One model: raw-text river (pretraining), then three small curated streams (examples, comparisons, checked answers) | 17.7 s |
| `03-instruction-tuning` | Instruction dims to context; only response words are graded | 11.0 s |
| `04-preference-alignment` | Preferred answer gains probability, rejected loses; leash to the original model; RLHF/PPO vs DPO routes | 10.8 s |
| `05-rlvr` | Several attempts at 17 × 3; checker marks ✓/✗; chance of 51 rises over two rounds | 14.9 s |

Plus a static comparison figure (`Comparison` scene, English and Chinese via
`COMPARE_LANG`). Dongxi visual system (`../../STYLE_GUIDE.md`): white canvas,
semantic colors, Arial / Songti SC, no equations on screen.

## Render

```bash
uv run --project visuals/animations python visuals/animations/projects/post-training-intro/render.py --preview   # 960×540 check
uv run --project visuals/animations python visuals/animations/projects/post-training-intro/render.py             # final
```

Final output: 1920×1080 30 fps H.264 MP4 per clip (git-ignored) in `rendered/<clip>/`
with poster and contact sheet; 1280×720 20 fps looping GIF (one global
256-color palette, no dithering, 3–5 MB) written straight to the article's
`assets/`. Timelines are validated against the mechanism (response-only
grading, 17 × 3 = 51, probabilities summing to 1, preferred bar up / rejected
down). `metadata.json` keeps commands, hashes, ffprobe data and limitations.

After the 2026-10-20 render, `render.py` was edited only to write GIFs directly
to the article assets instead of copying them; the GIF bytes and hashes are
unchanged.

## Limits

Illustrations, not measurements: A1 probabilities are authored; A2 volumes are
not to scale; A3 reveals words one at a time though training grades all
response positions in one pass; A4 merges the reward-model and DPO routes; A5's
checker reads only the final answer and its updates are not a GRPO computation.
