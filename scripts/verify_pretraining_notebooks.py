#!/usr/bin/env python3
"""Execute Day 8 source notebooks in fresh kernels; optionally export PNG previews.

Source notebooks are never overwritten. Output manifest and executed copies
live in a new temporary directory. --figures updates only generated PNG assets.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import platform
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import nbclient
import matplotlib
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(tempfile.mkdtemp(prefix="chapter6-reference-"))
    began = time.perf_counter()
    available = []
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            if Path("/proc/meminfo").exists():
                line = next(l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:"))
                available.append(int(line.split()[1])*1024)
            stop.wait(.1)
    threading.Thread(target=monitor, daemon=True).start()
    manifest = {"base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "torch": torch.__version__, "matplotlib": matplotlib.__version__,
                "nbformat": nbformat.__version__, "nbclient": nbclient.__version__,
                "host": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
                "kernel": args.kernel, "notebooks": [], "helpers": {}}
    for source in (root / "src/dongxi_llms").glob("*.py"):
        if source.name.startswith(("pretraining", "decoder_lab")):
            manifest["helpers"][str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
    for relative in ("scripts/verify_pretraining_notebooks.py", "tests/test_pretraining_lab.py"):
        manifest["helpers"][relative] = hashlib.sha256((root/relative).read_bytes()).hexdigest()
    sources = sorted((root / "notebooks/day-08").glob("*.ipynb"))
    if len(sources) != 3:
        raise RuntimeError("Expected the three Day 8 notebooks")
    for source in sources:
        notebook = nbformat.read(source, as_version=4)
        nbformat.validate(notebook)
        previews = []
        for cell in notebook.cells:
            if cell.cell_type == "markdown":
                previews.extend(re.findall(r"\.\./figures/chapter-06/([\w-]+\.png)", cell.source))
        NotebookClient(notebook, timeout=180, kernel_name=args.kernel,
                       resources={"metadata": {"path": str(source.parent)}}).execute()
        nbformat.write(notebook, output / source.name)
        images = [out.data["image/png"] for cell in notebook.cells if cell.cell_type == "code"
                  for out in cell.outputs if out.output_type in ("display_data", "execute_result") and "image/png" in out.data]
        if len(images) != len(previews):
            raise AssertionError(f"{source.name}: {len(images)} images, {len(previews)} preview references")
        if args.figures:
            target = root / "notebooks/figures/chapter-06"
            target.mkdir(parents=True, exist_ok=True)
            for name, data in zip(previews, images):
                (target / name).write_bytes(base64.b64decode(data))
        record = {"path": str(source.relative_to(root)), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "code_cells": sum(c.cell_type == "code" for c in notebook.cells), "images": previews}
        manifest["notebooks"].append(record)
        print(json.dumps(record), flush=True)
    stop.set()
    manifest["elapsed_seconds"] = time.perf_counter()-began
    manifest["sampled_min_mem_available_bytes"] = min(available) if available else None
    manifest["memory_samples"] = len(available)
    if available and min(available) < 25*2**30:
        raise RuntimeError("Observed host reserve below 25 GiB; verification not accepted")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Executed copies and provenance: {output}", flush=True)


if __name__ == "__main__":
    main()
