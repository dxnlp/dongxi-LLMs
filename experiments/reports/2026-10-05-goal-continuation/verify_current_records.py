"""Read-only acceptance/campaign/notebook audit; writes one new evidence receipt."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[3]


def load(relative):
    return json.loads((ROOT / relative).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline_path = "experiments/reports/2026-10-05-native-base-profile/original-acceptance-before-integration.json"
    ledger_path = "docs/course_improvements.json"
    campaign_path = "experiments/reports/2026-10-05-native-base-profile/current-campaign-evidence-02.json"
    historical_path = "experiments/reports/2026-10-05-current-cpu-closure/run-03/notebooks/manifest.json"
    baseline, ledger, campaign, historical = map(load, (
        baseline_path, ledger_path, campaign_path, historical_path))
    checks, errors = [], []

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(name)

    items = {item["id"]: item for item in ledger["items"]}
    check("original eighteen IDs", set(items) == set(baseline) and len(items) == 18)
    for name, old in baseline.items():
        current = items[name]
        check(name + " unchanged acceptance", current["acceptance"] == old["acceptance"])
        check(name + " unchanged dependencies", current["dependencies"] == old["dependencies"])
        retained = set(current.get("pending_checks", [])) | set(current.get("historical_pending_checks", []))
        resolutions = {row["original_check"] for row in current.get("pending_check_resolution", [])}
        for text in old.get("pending_checks", []):
            check(name + " retained original pending text", text in retained, text)
            if text not in current.get("pending_checks", []):
                check(name + " explicit pending scope resolution", text in resolutions, text)
    counts = dict(Counter(item["status"] for item in ledger["items"]))
    check("fifteen complete and three unfinished", counts == {"complete": 15, "in_progress": 3}, counts)
    check("learner stays Day9", ledger["summary"]["learner_day"] == 9)
    check("campaign retains45 rows and10 native jobs", len(campaign["rows"]) == 45 and campaign["jobs_started"] == 10)
    unrun = [row for row in campaign["rows"] if all(v is None for v in row["actual"].values())]
    check("44 unrun rows remain null", len(unrun) == 44)
    notebook_bindings = []
    for row in historical["notebooks"]:
        actual = digest(ROOT / row["path"])
        notebook_bindings.append({"path": row["path"], "recorded_source_sha256": row["sha256"], "current_source_sha256": actual})
        check("unchanged historical notebook source " + row["path"], actual == row["sha256"])
    check("all76 historical notebook sources inspected", len(notebook_bindings) == 76)
    prose_paths = [
        "README.md", "BOOK.md", "PROGRESS.md", "LEARNING_MEMORY.md", "AGENTS.md",
        "docs/course_improvements.json", "docs/COURSE_IMPROVEMENT_PLAN.md",
        "docs/UPGRADE_ACCEPTANCE_REVIEW.md", "docs/COURSE_EVIDENCE_MAP.md",
        "docs/EXPERIMENT_MATRIX.md", "docs/handoffs/CURRENT.md",
        "docs/handoffs/MAC_CPU_VERIFICATION.md", "book/appendices/d-reproduction-and-environments.md",
        "book/chapters/06-pretraining-as-a-controlled-system.md",
        "book/chapters/07-evaluation-is-a-contract.md", "book/chapters/15-distill-evaluate-and-defend.md",
        "book/labs/07-evaluation-is-a-contract.md", "book/labs/15-distill-evaluate-and-defend.md",
        "book/solutions/07-evaluation-is-a-contract.md", "book/solutions/15-distill-evaluate-and-defend.md",
        "experiments/specs/2026-10-05-story-rubric-evaluation.md",
        "experiments/reports/2026-10-05-story-panel-audit.md",
        "experiments/reports/2026-10-05-story-rubric.md",
        "experiments/reports/2026-10-05-current-model-defense.md",
        "experiments/reports/2026-10-05-goal-continuation.md",
        "fixtures/story-rubric/README.md",
        "learning_artifacts/day-09-pretraining-run-and-diagnosis/README.md",
        "visuals/animations/PROPOSALS.md",
    ]
    links, prose_bindings = [], []
    for relative in prose_paths:
        path = ROOT / relative
        prose_bindings.append({"path": relative, "sha256": digest(path)})
        if path.suffix != ".md":
            continue
        for match in re.finditer(r"\]\(([^)\n]+)\)", path.read_text()):
            target = match.group(1).strip()
            if target.startswith(("http:", "https:", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0].strip("<>"))
            if not target:
                continue
            destination = (path.parent / target).resolve()
            links.append({"source": relative, "target": target, "exists": destination.exists()})
            check("local file link " + relative + ": " + target, destination.exists())
    report = {
        "schema": "dongxi-goal-continuation-integrity-v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "scope": "Local read-only evidence/acceptance integrity. Notebook source identity is not new execution, Mac evidence or whole-goal completion.",
        "checks": checks, "failures": errors, "notebook_bindings": notebook_bindings,
        "prose_bindings": prose_bindings, "local_links": links,
        "input_files": [{"path": p, "sha256": digest(ROOT / p)} for p in (
            baseline_path, ledger_path, campaign_path, historical_path)],
        "collector_sha256": digest(Path(__file__)),
    }
    name = sys.argv[1] if len(sys.argv) == 2 else "protected-and-notebook-integrity-01.json"
    if Path(name).name != name or not name.endswith(".json"):
        raise ValueError("Choose a new JSON basename in this evidence directory")
    with Path(__file__).with_name(name).open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks": len(checks),
                      "local_links": len(links), "prose_files": len(prose_bindings), "failures": errors}))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
