#!/usr/bin/env python3
"""Render scientific-figure review sheets from an actual notebook manifest."""
import argparse
import json
from pathlib import Path
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--days", type=int, nargs="*")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = json.loads(args.manifest.read_text())
    paths = []
    for notebook in data["notebooks"]:
        day = int(Path(notebook["path"]).parent.name[4:])
        if args.days and day not in args.days:
            continue
        paths += [root / preview["path"] for preview in notebook["previews"]]
    output = args.output or Path(tempfile.mkdtemp(prefix="dongxi-figure-review-"))
    output.mkdir(parents=True, exist_ok=True)
    for page, start in enumerate(range(0, len(paths), 12), 1):
        fig, axes = plt.subplots(4, 3, figsize=(18, 14), layout="constrained")
        for ax, path in zip(axes.flat, paths[start:start+12]):
            ax.imshow(mpimg.imread(path))
            ax.set_title(path.stem.replace("_", " "), fontsize=8)
            ax.axis("off")
        for ax in axes.flat[len(paths[start:start+12]):]:
            ax.axis("off")
        target = output / f"sheet-{page:02d}.png"
        fig.savefig(target, dpi=120, facecolor="white")
        plt.close(fig)
        print(target)
    print(f"Reviewed source inventory: {len(paths)} generated figures")


if __name__ == "__main__":
    main()
