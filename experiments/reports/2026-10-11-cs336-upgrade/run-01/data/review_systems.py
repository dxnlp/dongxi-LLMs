"""Read-only source/evidence binding for independent systems peer review."""
from pathlib import Path
import hashlib
import json
import math
import re

import nbformat

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
SYSTEMS = OUT.parent / "systems"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


files = ["src/dongxi_llms/systems_performance_lab.py", "src/dongxi_llms/online_attention_lab.py",
         "src/dongxi_llms/scaling_lab.py", "tests/test_systems_performance_lab.py",
         "tests/test_online_attention_lab.py", "tests/test_scaling_lab.py",
         "scripts/run_cs336_systems.py", "notebooks/extensions/cs336/SYSTEMS_SOLUTIONS.md",
         *["notebooks/extensions/cs336/"+name+".ipynb" for name in
           ("01_resource_accounting", "02_online_attention", "03_scaling_prediction")]]
author = json.loads((SYSTEMS / "implementation.json").read_text())
author_current = json.loads((SYSTEMS / "implementation-v2.json").read_text())
hashes = {name: sha(ROOT / name) for name in files}
assert all(author_current["source_sha256"][name] == digest for name, digest in hashes.items())
assert all(author["source_sha256"][name] == digest for name, digest in hashes.items()
           if not name.endswith(".ipynb") and name != "scripts/run_cs336_systems.py")
tests = json.loads((SYSTEMS / "focused-tests-v1.json").read_text())
assert tests["status"] == "passed" and tests["tests_run"] == 20
assert all(sha(ROOT / name) == digest for name, digest in tests["source_sha256"].items())
executions = json.loads((SYSTEMS / "notebook-v3/notebook-execution-v3.json").read_text())
figures = []
notebooks = []
for row in executions["rows"]:
    assert row["status"] == "passed" and author["source_sha256"][row["path"]] == row["source_sha256"]
    source = nbformat.read(ROOT / row["path"], as_version=4)
    pairs = []
    for index, cell in enumerate(source.cells):
        if cell.cell_type == "code":
            assert "Prediction" in source.cells[index-1].source
            assert "Checkpoint" in source.cells[index-1].source
            assert source.cells[index-1].source.rstrip().endswith("### Reference solution")
            assert source.cells[index+1].source.startswith("### Reference explanation")
            pairs.append(cell.id)
    assert len(pairs) == 5 and len(set(cell.id for cell in source.cells)) == len(source.cells)
    executed = SYSTEMS / "notebook-v3" / (Path(row["path"]).stem + "-executed.ipynb")
    nb = nbformat.read(executed, as_version=4)
    assert [(c.id,c.cell_type) for c in source.cells] == [(c.id,c.cell_type) for c in nb.cells]
    for current, prior in zip(source.cells, nb.cells):
        assert current.source == prior.source if current.cell_type == "code" else current.source.startswith(prior.source)
    assert not any(output.output_type == "error" for cell in nb.cells if cell.cell_type == "code" for output in cell.outputs)
    notebooks.append({"path":row["path"], "current_integrated_source_sha256":hashes[row["path"]],
                      "author_execution_source_sha256":row["source_sha256"], "executed_sha256":sha(executed), "reference_cells":pairs,
                      "integration_changes": "Static preview Markdown appended to explanations only; code, IDs, prompts and prior explanations preserved"})
    for figure in row["figures"]:
        assert sha(ROOT / figure["path"]) == figure["sha256"]
        figures.append(figure)

preview_path = OUT.parent / "preview-execution-01/manifest.json"
preview = json.loads(preview_path.read_text())
preview_figures = []
for row in preview["notebooks"]:
    if row["path"] not in hashes:
        continue
    assert row["kernel_identity"]["executable"] == "/tmp/dongxi-cs336-g9l7v90r/venv/bin/python"
    assert row["kernel_identity"]["cuda_available"] is False
    assert row["sha256"] == author["source_sha256"][row["path"]]
    for figure in row["previews"]:
        assert sha(ROOT / figure["path"]) == figure["sha256"]
        preview_figures.append(figure)
    source_path = ROOT / row["path"]
    source = nbformat.read(source_path, as_version=4)
    links = [(source_path.parent / link).resolve() for cell in source.cells for link in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', cell.source)]
    assert set(links) == {ROOT / figure["path"] for figure in row["previews"]}
assert len(preview_figures) == 15

receipt = json.loads((SYSTEMS / "tiny-scaling-matrix.json").read_text())
study = receipt["result"]; runs = study["all_runs"]
assert len(runs) == 24 and len(study["calibration"]) == 18
assert sum(row["presented_tokens"] for row in runs) == 18880
for row in runs:
    assert row["presented_tokens"] == 32*row["updates"]
    assert row["actual_6ND_coordinate"] == 6*row["parameters"]*row["presented_tokens"]
    assert row["source_repetitions"] == row["presented_tokens"]/256
assert study["frozen_prediction"]["status"] == "fit-refused-before-holdout"
assert study["frozen_prediction"] == json.loads((SYSTEMS / "matrix-runs/frozen-prediction.json").read_text())
assert len(study["holdout"]["entire_budget_rows"]) == 6
assert all(row["compute_budget"] == 96e6 for row in study["holdout"]["entire_budget_rows"])

result = {
    "status": "accepted", "scope": "Independent content/code/test/evidence review of CS33604/05/06 owned files only; no source edits or new training/GPU jobs",
    "reviewed_source_sha256": hashes,
    "current_author_receipt": {"path":str((SYSTEMS / "implementation-v2.json").relative_to(ROOT)),"sha256":sha(SYSTEMS / "implementation-v2.json"),
                               "binding":"All eleven current owned source hashes match exactly; supplements remain separately scoped"},
    "author_receipt": {"path":str((SYSTEMS / "implementation.json").relative_to(ROOT)), "sha256":sha(SYSTEMS / "implementation.json"),
                       "binding": "Seven module/test/solution source hashes agree; three pre-preview notebook identities match v3; current integrated notebook hashes bound separately with exact code/ID equality. The runner gained separately authorized phase_profile after author receipts; its current source is bound, its unchanged CPU paths were reread, and actual GPU acceptance belongs root. Incidental other-agent hashes are outside this review"},
    "runner_history": {"author_execution_sha256":author["source_sha256"]["scripts/run_cs336_systems.py"],
                       "current_source_sha256":hashes["scripts/run_cs336_systems.py"],
                       "difference":"Added explicit CUDA phase-profile function and dispatch; original CPU reference/smoke/matrix paths retained. Earlier matrix receipts keep their own original source identity; no new execution is claimed by this review."},
    "tests": {"receipt":str((SYSTEMS / "focused-tests-v1.json").relative_to(ROOT)), "sha256":sha(SYSTEMS / "focused-tests-v1.json"),
              "count":20, "independent_basis": ["Closed matmul/traffic counts and actual synchronization event order", "Existing checkpoint gradients/re-entry and independent stored-storage boundary", "PyTorch SDPA output and gradients across dtype/tiles/ragged masks/cache offset", "Numerical gradcheck and direct logsumexp", "Authored analytic optimum, frozen identity, pooling refusal and bootstrap failure accounting"]},
    "notebooks": notebooks,
    "semantic_review": {"pairs":15, "adjacency_and_stable_ids":True, "concrete_controls_match_prompts":True,
                        "observations": ["Traffic plot now contains an adverse tile4 curve under the same dimensions", "Bandwidth-only change20→10GB/s shown separately from actual CPU timing; queue labeled untimed", "Checkpoint storage excludes peak claims and visibly adds two backward block entries", "Late maximum control returns1 versus0.5 and overflow; causal changes preserve early rows", "Invalid/cached/ragged rows have explicit zero-output and negative-infinite LSE contracts", "Online ordinary-autograd saves can exceed dense despite smaller modeled forward tiles", "Scaling prediction freezes before revealing entire384M grid; known floor/design separated from real LM uncertainty", "Bootstrap preserves failures and warns that conditional discrete grid can miss known optimum", "Actual measured study visibly includes both seeds, train/dev/test and target-actual quantization"]},
    "mathematical_review": {"matmul_example": {"flops":2*2*3*4,"bytes":(2*3+3*4+2*4)*8,"intensity":48/208},
                            "roofline_ridges_flops_per_byte": [100e9/20e9,100e9/10e9],
                            "online_recurrence": "Old normalizer/numerator rescale by exp(old_max−new_max); new terms use exp(score−new_max); finalu/l and LSE=m+logl", 
                            "known_surface": {"parameter_exponent":.5,"excess_loss_exponent":-.25,"heldout_optimal_parameters":math.sqrt(384e6/6),"heldout_optimal_loss":2+40/math.sqrt(math.sqrt(384e6/6))},
                            "cached_query": "Added prefix attention4*layers*batch*query_heads*prefix*head_dim; compact KV useskv_heads; projections/head run once"},
    "measured_reference": {"path":str((SYSTEMS / "tiny-scaling-matrix.json").relative_to(ROOT)),"sha256":sha(SYSTEMS / "tiny-scaling-matrix.json"),
                           "verified": "24rows=18calibration+6entireheldout;18,880targets; all32target/update,6ND and256-slot repetition counts reconcile; separate frozen-refusal file matches final receipt",
                           "negative_outcome": "Smallest12M calibration budget prefers the boundary N5456; no empirical exponent/optimal-size prediction asserted"},
    "visual_review": {"count":15, "figures":figures,
                      "method": "All15 v1 PNGs inspected in original-resolution contact sheets; unchanged figure mechanisms retained; three correctedv3 PNGs inspected individually at original resolution",
                      "accepted": "Axis/shape/units/legends readable; modeled, authored-simulated and measured plots labeled; empty rows and adverse storage result visible"},
    "canonical_preview_provenance": {"manifest":str(preview_path.relative_to(ROOT)),"sha256":sha(preview_path),"figures":preview_figures,
                                     "binding":"All15 canonical previews match root fresh preview execution. Resource3 timing PNG differs from authorv3 because it comes from a distinct root execution; that distinction is preserved and its canonical PNG was independently inspected."},
    "findings": [{"id":"SYS-PEER-01","location":"01_resource_accounting checkpoint2","issue":"Worse blockwise traffic predicted but neither displayed tile supplied it","resolution":"Added fixed-dimension tile4 curve; inspected v3 shows traffic exceeds dense","status":"resolved"},
                 {"id":"SYS-PEER-02","location":"01_resource_accounting checkpoint3","issue":"Requested bandwidth-only control absent; untimed queue described as measured interval","resolution":"Second10GB/s curve added and queue described as untimed submission/completion illustration","status":"resolved"},
                 {"id":"SYS-PEER-03","location":"03_scaling_prediction checkpoint5","issue":"Requested train/dev/test and target-vs-actual coordinates missing; dev plot showed one seed","resolution":"Both-seed means, three split means, actual-target percentage panel and all24 independenttest points now present","status":"resolved"}],
    "open_issues": [],
    "evidence_boundaries": ["This review does not validate actual Spark profile or hardware kernel selection", "No Triton/distributed/recomputed-backward implementation claimed", "Finite repeated corpus, discrete widths and budget-specific schedules limit tiny study; refusal is not a general scaling result", "Root integration chapters/labs/manifest and other-author post-training/token/corpus files are outside this receipt"]}
(OUT / "systems-peer-review.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
print("Accepted:",len(files),"owned hashes;15 prompt/reference groups and figures;3 concrete findings resolved")
