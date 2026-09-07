# Explanatory Visuals in Course Notebooks

Learner request, 2026-09-07: text and code alone make architecture notebooks
hard to engage with. Treat explanatory visuals as first-class teaching content,
not as optional decoration added after the lesson.

## Design contract

- Open architecture lessons with a whole-model map highlighting the current
  component, then a close-up showing its branches, operations, and tensor shapes.
  Distinguish the baseline learned-position decoder from the modern RoPE decoder.
  Label schematics as structure, not measured activations or performance.
- Choose a visual that answers the current conceptual question: a lookup
  heatmap, head-wise attention map, vector addition, nonlinear curve, architecture
  diagram, or measured learning curve. Do not add charts just to fill space.
- Use the actual lesson tensors, parameters, or measured results whenever a
  visual makes a numerical claim. Separate schematic structure from observations.
- Put the visual next to the mechanism and its explanation. Keep predictions
  before the worked solution; show the visual after the relevant reference step.
- Include a short reading guide and a meaningful controlled change the learner
  can make. The plot code must regenerate from the current notebook state.
- Label axes, token/head/feature dimensions, masks, signed values, units, and
  scale conventions. Use a shared scale for side-by-side comparisons; do not
  suggest semantic geometry from arbitrary feature-coordinate axes.
- Use a quiet white canvas, readable labels, and stable semantic colors. Use
  DejaVu Sans where Arial is unavailable on Spark. Provide meaningful image alt
  text and adjacent interpretation so color is not the only explanation.
- Prefer static figures for notebook mechanisms. Animation candidates remain
  automatically recorded, but animation production stays on Mac Studio and
  requires approval. A request for notebook plots is not a rendering commission.

## Reference previews versus live plots

The Chapter 5 notebooks include 52 saved PNG previews: 27 mechanism figures
plus 25 orientation/component schematics (one model map and one close-up per
notebook, with an extra SwiGLU close-up). Runnable cells regenerate schematics
from their drawing definitions and data plots from live lesson tensors. Previews remain
visible before a kernel starts. They are labeled as saved reference results;
changing a code cell does not automatically refresh its preview. Run the plot
cell to see the current experiment output below the fixed reference.

Generated previews live in `notebooks/figures/chapter-05/`. Their source is
`src/dongxi_llms/decoder_visuals.py` and `decoder_architecture.py`, plus each
notebook's plotting call. Existing
learner cells and their outputs are preserved rather than replaced with an
agent-executed notebook.

## Dependencies and reproduction

Install `notebooks/requirements-visuals.txt` into the intended notebook kernel
environment. It pins Matplotlib 3.10.8 and does not replace the platform Torch
lock. Verify proposed dependency changes before altering a shared environment.

```bash
/home/dongxi/dgx-spark-dongxi/.venv/bin/python scripts/verify_decoder_notebooks.py
# Substitute the freshly printed temporary output directory below.
/home/dongxi/dgx-spark-dongxi/.venv/bin/python scripts/render_decoder_notebook_figures.py /tmp/chapter5-reference-EXAMPLE
```

The first command executes the notebooks without modifying their sources; the
second updates only generated reference PNG assets. No notebook server or
animation process is launched by these commands.

## Validation

Check plotted arrays against source tensors and confirm display functions do
not alter inputs, gradients, or random state. Execute all reference paths,
count expected image outputs, inspect contact sheets and important full-size
figures, and check notebook image paths. Record a new report when source hashes
change; historical reports continue to describe their historical revisions.
