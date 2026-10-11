"""Bounded original/default comparison and fresh kernels for CS33602/03."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import nbformat
from nbclient import NotebookClient
import torch

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
BASE = "78640952e7e6ffb35c57f4ab48569e6a660e93b6"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def defaults():
    from dongxi_llms.pretraining_lab import TrainingSession
    command = ["git", "show", BASE + ":src/dongxi_llms/pretraining_lab.py"]
    source = subprocess.check_output(command, cwd=ROOT).decode()
    module = types.ModuleType("dongxi_llms._cs336_original_pretraining")
    module.__package__ = "dongxi_llms"
    sys.modules[module.__name__] = module
    exec(compile(source, str(OUT / "baseline-pretraining-read-only.py"), "exec"), module.__dict__)
    original, current = module.TrainingSession(total=4), TrainingSession(total=4)
    assert original.contract == current.contract
    results = []
    for _ in range(4):
        before, after = original.update(), current.update()
        assert before == after
        assert all(torch.equal(p, q) for p, q in zip(original.model.parameters(), current.model.parameters()))
        results.append(after)
    result = {"baseline_commit": BASE, "baseline_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "current_source_sha256": digest(ROOT / "src/dongxi_llms/pretraining_lab.py"),
              "contract_equal": True, "all_four_update_records_equal": True,
              "all_parameter_tensors_bitwise_equal_after_every_update": True,
              "history": results}
    (OUT / "default-compatibility.json").write_text(json.dumps(result, indent=2) + "\n")


def notebooks():
    rows = []
    for name in ("05_tokenizer_contracts", "06_corpus_curation"):
        source = ROOT / "notebooks/extensions/cs336" / (name + ".ipynb")
        before = digest(source)
        notebook = nbformat.read(source, as_version=4)
        started = time.monotonic()
        NotebookClient(notebook, timeout=120, kernel_name="dongxi-cs336",
                       resources={"metadata": {"path": str(ROOT)}}, record_timing=True).execute()
        destination = OUT / "notebooks" / source.name
        destination.parent.mkdir(exist_ok=True)
        nbformat.write(notebook, destination)
        images = []
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            for number, output in enumerate(cell.outputs):
                png = output.get("data", {}).get("image/png")
                if png:
                    figure = OUT / "figures" / f"{cell.id}-{number}.png"
                    figure.parent.mkdir(exist_ok=True)
                    figure.write_bytes(base64.b64decode(png))
                    images.append({"path": str(figure.relative_to(ROOT)), "sha256": digest(figure), "cell_id": cell.id})
        assert digest(source) == before
        rows.append({"source": str(source.relative_to(ROOT)), "source_sha256": before,
                     "executed": str(destination.relative_to(ROOT)), "executed_sha256": digest(destination),
                     "runtime_seconds": time.monotonic() - started, "kernel": "dongxi-cs336",
                     "cells": len(notebook.cells), "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
                     "images": images, "source_unchanged": True, "errors": []})
    (OUT / "notebook-execution.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    defaults()
    notebooks()
