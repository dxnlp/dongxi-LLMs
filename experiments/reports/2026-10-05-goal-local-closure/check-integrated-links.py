"""Check local links in the current integration; no fetches or model work."""
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[3]
files = [
    "README.md", "BOOK.md", "PROGRESS.md", "LEARNING_MEMORY.md",
    "docs/COURSE_IMPROVEMENT_PLAN.md", "docs/UPGRADE_ACCEPTANCE_REVIEW.md",
    "docs/COURSE_EVIDENCE_MAP.md", "docs/EXPERIMENT_MATRIX.md", "docs/handoffs/CURRENT.md",
    "book/chapters/07-evaluation-is-a-contract.md", "book/chapters/09-supervised-fine-tuning.md",
    "book/chapters/11-direct-preference-optimization.md",
    "book/solutions/07-evaluation-is-a-contract.md", "book/solutions/09-supervised-fine-tuning.md",
    "book/solutions/11-direct-preference-optimization.md",
    "book/labs/07-evaluation-is-a-contract.md", "book/labs/09-supervised-fine-tuning.md",
    "book/labs/11-direct-preference-optimization.md",
    "experiments/reports/2026-10-05-native-base-profile.md",
    "experiments/reports/2026-10-05-native-sft-replay.md",
    "experiments/reports/2026-10-05-pretrained-evaluation-replay.md",
    "experiments/reports/2026-10-05-current-cpu-closure.md",
    "experiments/reports/2026-10-05-goal-local-closure.md",
]
bindings, checks, failures = [], [], []
for relative in files:
    path = ROOT / relative
    data = path.read_bytes()
    bindings.append({"path": relative, "bytes": len(data),
                     "sha256": hashlib.sha256(data).hexdigest()})
    for match in re.finditer(r"\]\(([^)\n]+)\)", data.decode()):
        target = match.group(1).strip()
        if target.startswith(("http:", "https:", "mailto:", "#")):
            continue
        target = unquote(target.split("#", 1)[0].strip("<>"))
        if not target:
            continue
        destination = (path.parent / target).resolve()
        record = {"source": relative, "target": target,
                  "resolved": str(destination), "exists": destination.exists()}
        checks.append(record)
        if not record["exists"]:
            failures.append(record)
report = {"schema": "dongxi-integrated-local-links-v1", "status": "passed" if not failures else "failed",
          "scope": "Local file-link existence only; not external URL or live GitHub rendering/anchor validation.",
          "files": bindings, "links": checks, "failures": failures}
name = sys.argv[1] if len(sys.argv) == 2 else "integrated-links.json"
if Path(name).name != name or not name.endswith(".json"):
    raise ValueError("Output must be a new JSON basename in this evidence directory")
report["collector_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
destination = Path(__file__).with_name(name)
with destination.open("x") as output:
    output.write(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": report["status"], "files": len(files), "links": len(checks), "failures": failures}))
raise SystemExit(bool(failures))
