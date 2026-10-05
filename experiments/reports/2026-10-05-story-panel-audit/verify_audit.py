"""Independent receipt/nearest-example checks; no corpus/model imports.

This does not repeat the full corpus search. It validates the carried corpus
reconstruction, independently recalculates each retained nearest-example score,
and checks actual source/archive/index byte bindings. Raw corpus completeness is
the streamed audit's observed evidence, not independently reasserted here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def gram_set(text):
    tokens = re.findall(r"\w+(?:'\w+)?", text.casefold())
    return {tuple(tokens[i:i+3]) for i in range(max(0, len(tokens)-2))}


def main(output):
    first_path, current_path = HERE / "run-01/audit.json", HERE / "run-02/audit.json"
    first, current = [json.loads(path.read_bytes()) for path in (first_path, current_path)]
    checks = {}
    checks["both_actual_scans_passed"] = first["status"] == current["status"] == "passed-bounded-audit"
    checks["corpus_panel_numeric_findings_unchanged"] = all(first[key] == current[key] for key in (
        "panel", "panel_metrics", "split_reconstruction", "corpus_provenance"))
    checks["both_source_closures_stable"] = first["source_stable"] and current["source_stable"]
    checks["original_collector_archive_exact"] = sha(HERE / "run-01-collector.py.txt") == first["source_before"][str((HERE/"collect_story_panel_audit.py").relative_to(ROOT))]["sha256"]
    checks["original_protocol_archive_exact"] = sha(HERE / "run-01-protocol.json.txt") == first["source_before"][str((HERE/"protocol.json").relative_to(ROOT))]["sha256"]
    checks["current_source_bytes_match_all13_bindings"] = all(sha(ROOT / path) == value["sha256"] for path, value in current["source_after"].items())
    checks["retained_ignored_indexes_match"] = all(sha(Path(audit["audit_index"]["declared_path"])) == audit["audit_index"]["sha256"] for audit in (first, current))
    prompts = {item["id"]: item["prompt"] for item in current["panel"]["items"]}
    prompts.update({f"development-{i:02d}": prompt for i, prompt in enumerate(current["panel"]["development_prompts"], 1)})
    score_comparisons = 0
    group_comparisons = 0
    for split, items in current["panel_metrics"].items():
        count = current["split_reconstruction"][split]
        checks[f"{split}_raw_count_partition"] = count["raw_documents"] == count["retained_documents"] + count["duplicate_documents"] + count["excluded_overlap"]
        checks[f"{split}_all_reconstruction_fields_match"] = all(count["matches_actual_manifest"].values())
        checks[f"{split}_raw_complete_and_identity_match"] = current["raw_corpus"][split]["complete"] and current["raw_corpus"][split]["actual_sha256"] == current["raw_corpus"][split]["expected"]["sha256"]
        for identifier, statuses in items.items():
            for status, metrics in statuses.items():
                expected_count = count[{"retained": "retained_documents", "duplicate": "duplicate_documents", "excluded_validation": "excluded_overlap"}[status]]
                assert metrics["stories"] == expected_count
                assert 0 <= metrics["whole_story_equal"] <= metrics["prefix_equal"] <= metrics["substring_equal"] <= expected_count
                assert 0 <= metrics["near_threshold_count"] <= metrics["positive_near_candidates"] <= expected_count
                examples = metrics["nearest_examples"]
                assert metrics["maximum_near_score"] == max((e["score"] for e in examples), default=0.0)
                assert (metrics["near_threshold_count"] > 0) == (metrics["maximum_near_score"] >= .4)
                for example in examples:
                    a, b = gram_set(prompts[identifier]), gram_set(example["opening_words"])
                    actual = len(a & b) / len(a | b) if a | b else 0.0
                    assert actual == example["score"]
                    assert sorted(map(list, a & b)) == example["shared_trigrams"]
                    assert 1 <= example["raw_document_ordinal"] <= count["raw_documents"]
                    score_comparisons += 1
                group_comparisons += 1
    checks["all90_metric_groups_consistent"] = group_comparisons == 90
    checks["all_carried_nearest_scores_recomputed"] = score_comparisons > 0
    publication = [metrics for split in current["panel_metrics"].values() for identifier, statuses in split.items()
        if identifier.startswith("story-") for metrics in statuses.values()]
    checks["publication_no_exact_full_prefix_substring"] = all(not metrics[key] for metrics in publication for key in ("whole_story_equal", "prefix_equal", "substring_equal"))
    checks["publication_defined_near_threshold_zero"] = all(metrics["near_threshold_count"] == 0 for metrics in publication)
    checks["historical_development_nonzero_near_preserved"] = current["panel_metrics"]["train"]["development-01"]["retained"]["maximum_near_score"] == .5 and current["panel_metrics"]["train"]["development-01"]["retained"]["near_threshold_count"] == 1
    checks["publication_vs_development36_exact_relationships_disjoint"] = len(current["panel"]["development_separation"]) == 12 and all(not pair[key]
        for item in current["panel"]["development_separation"] for pair in item["pairs"] for key in ("equal", "either_is_prefix", "either_is_substring"))
    result = {"schema": "dongxi-story-panel-audit-independent-verification-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.orig_argv), "status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "check_count": len(checks), "metric_groups_checked": group_comparisons,
        "nearest_scores_independently_recomputed": score_comparisons,
        "bindings": {"run01": {"path": str(first_path), "sha256": sha(first_path)},
            "run02": {"path": str(current_path), "sha256": sha(current_path)},
            "verifier": {"path": str(Path(__file__)), "sha256": sha(Path(__file__))}},
        "limits": "Receipt consistency and short carried-example scoring, not an independent second implementation of the full corpus search or global semantic contamination absence."}
    with output.open("x", encoding="utf-8") as target:
        json.dump(result, target, sort_keys=True, indent=2)
        target.write("\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().parent != HERE:
        parser.error("Output must stay in the owned report directory")
    raise SystemExit(main(args.output))
