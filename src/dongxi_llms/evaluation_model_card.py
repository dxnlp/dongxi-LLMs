"""Evidence-limited cards from saved response records; no model work or probes.

The authoritative replay instrument computes every score. A checkpoint ID is a
record label, not proof of model architecture, lineage, human review or approval.
Exporting a card neither selects candidates nor runs generation. Input identities
can corroborate retained metadata internally; their hashes are not signatures.
"""
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import stat

from dongxi_llms.reasoning_evaluation import (freeze_contract, replay_records,
                                            paired_group_bootstrap)
from dongxi_llms.run_identity import canonical_hash, validate_interface

CARD_SCHEMA = "dongxi-offline-evaluation-model-card-v1"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_LEDGER_BYTES = 64 * 1024 * 1024
MAX_ITEMS = 4096
MAX_RECORDS = 16384
MAX_COMPARISONS = 16
MAX_BOOTSTRAP_WORK = 2_000_000
SOURCE_PATHS = ("src/dongxi_llms/evaluation_model_card.py",
                "src/dongxi_llms/reasoning_evaluation.py",
                "src/dongxi_llms/evaluation_lab.py",
                "src/dongxi_llms/run_identity.py",
                "scripts/export_evaluation_model_card.py")


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")


def _copy(value):
    return json.loads(_json_bytes(value))


def _unavailable(reason):
    return {"status": "unavailable", "value": None, "reason": reason}


def _identity_index(identities, records, contract):
    if not isinstance(identities, (list, tuple)) or len(identities) > MAX_RECORDS:
        raise ValueError("Recorded identities must be a bounded list")
    referenced = {row.get("input_identity_sha256") for row in records}
    result = {}
    for identity in identities:
        if not isinstance(identity, dict) or identity.get("schema_version") != 1:
            raise ValueError("Need a recorded run-identity object, not a report or declaration")
        digest = identity.get("identity_sha256")
        if digest != canonical_hash({k: v for k, v in identity.items()
                                     if k != "identity_sha256"}):
            raise ValueError("Recorded run identity hash mismatch")
        if digest in result or digest not in referenced:
            raise ValueError("Repeated or unrelated recorded run identity")
        if identity.get("config") != contract["settings"]:
            raise ValueError("Recorded run settings differ from the frozen contract")
        interface = identity.get("checkpoint_interface")
        declared = contract["settings"].get("generation", {}).get("interface")
        if declared is not None:
            validate_interface(declared)
            validate_interface(interface)
            if interface != declared:
                raise ValueError("Recorded checkpoint interface differs from the frozen contract")
        for row in records:
            if row.get("input_identity_sha256") != digest:
                continue
            if row.get("input_sha256") != identity.get("input_sha256"):
                raise ValueError("Response input hashes differ from the recorded run identity")
            if row["checkpoint_id"].startswith("local-hf-sha256:"):
                expected_checkpoint = "local-hf-sha256:" + canonical_hash({
                    "files": identity.get("checkpoint_files"),
                    "tokenizer": identity.get("selected_tokenizer_files"),
                    "interface": interface.get("interface_sha256") if interface else None})
                if row["checkpoint_id"] != expected_checkpoint:
                    raise ValueError("Local-adapter checkpoint ID differs from recorded artifact metadata")
        result[digest] = _copy(identity)
    return result


def _costs(rows):
    names = sorted({key for row in rows for key in row["cost"]})
    return {name: {"known_total": sum(row["cost"].get(name) or 0 for row in rows),
                   "known_rows": sum(row["cost"].get(name) is not None for row in rows),
                   "unknown_rows": sum(row["cost"].get(name) is None for row in rows)}
            for name in names}


def build_evaluation_card(items, records, contract, *, run_identities=(),
                          comparisons=(), draws=2000, seed=1010, evidence=None):
    """Return (card, full replay), without querying hardware or loading models.

    Paired comparisons are explicit (baseline, candidate) pairs. Coverage must
    match exactly; refusal is preferable to silently dropping unmatched attempts.
    Input order is retained in replay.json; card serialization is canonical.
    """
    items, records, contract = _copy(items), _copy(records), _copy(contract)
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_ITEMS:
        raise ValueError("Need a bounded nonempty item suite")
    if not isinstance(records, list) or len(records) > MAX_RECORDS:
        raise ValueError("Response ledger exceeds the bounded record count")
    # Recheck suite/source-split validity even when an externally supplied
    # contract has a self-consistent hash. Preparation metadata may add fields.
    base = freeze_contract(items, contract["settings"])
    if any(base[key] != contract.get(key) for key in base if key != "identity"):
        raise ValueError("Frozen evaluation contract fields differ from the suite")
    replay = replay_records(items, records, contract)
    identities = _identity_index(run_identities, records, contract)
    if type(draws) is not int or not 1 <= draws <= 10000 or type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Bounded positive bootstrap draws and a 63-bit seed required")
    if not isinstance(comparisons, (list, tuple)) or len(comparisons) > MAX_COMPARISONS:
        raise ValueError("Too many paired comparisons")
    paired, seen, bootstrap_work = [], set(), 0
    for pair in comparisons:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(not isinstance(p, str) for p in pair):
            raise ValueError("Comparisons need baseline and candidate checkpoint IDs")
    for pair in sorted(tuple(pair) for pair in comparisons):
        a, b = pair
        if a == b or (a, b) in seen or a not in replay["checkpoints"] or b not in replay["checkpoints"]:
            raise ValueError("Unknown, self or repeated paired comparison")
        seen.add((a, b))
        left = [row for row in replay["rows"] if row["checkpoint_id"] == a]
        right = [row for row in replay["rows"] if row["checkpoint_id"] == b]
        bootstrap_work += draws * len(left)
        if bootstrap_work > MAX_BOOTSTRAP_WORK:
            raise ValueError("Paired bootstrap exceeds the declared work bound")
        comparison = paired_group_bootstrap(left, right, draws=draws, seed=seed)
        comparison.pop("samples")
        paired.append(dict(comparison, baseline=a, candidate=b))
    models = {}
    for checkpoint, measured in replay["checkpoints"].items():
        rows = [row for row in replay["rows"] if row["checkpoint_id"] == checkpoint]
        identity_ids = sorted({row.get("input_identity_sha256") for row in rows
                               if row.get("input_identity_sha256") is not None})
        verified_ids = [key for key in identity_ids if key in identities]
        provenance = {key: identities[key] for key in verified_ids}
        missing = [key for key in identity_ids if key not in identities]
        unlinked_rows = sum(row.get("input_identity_sha256") not in identities for row in rows)
        models[checkpoint] = {
            "checkpoint_id": checkpoint,
            "identity_boundary": "Opaque record ID; no checkpoint bytes loaded or independently rehashed by this exporter",
            "recorded_run_identities": provenance,
            "provenance_status": "recorded internally consistent metadata" if unlinked_rows == 0 else "unverified or partially unavailable",
            "unverified_identity_ids": missing, "unlinked_record_count": unlinked_rows,
            "recorded_costs": _costs(rows),
            "model_architecture": _unavailable("Not established by response records or a tokenizer interface"),
            "training": _unavailable("No training evidence consumed"),
            "genealogy": _unavailable("No independently checked parent-to-child training lineage consumed"),
            "hardware": {"status": "recorded metadata only" if provenance else "unavailable",
                         "values_by_identity": {key: {"device": obj.get("device"),
                             "environment": obj.get("environment"), "gpu_driver": obj.get("gpu_driver")}
                             for key, obj in provenance.items()},
                         "boundary": "Exporter performs no hardware probe; configured device is not an observed resource measurement"},
            "results": measured,
        }
    card = {
        "schema_version": CARD_SCHEMA,
        "title": "Saved response evaluation model card",
        "scope": "Offline instrument replay of the supplied ledger, not a new model or capability experiment",
        "intended_use": "Educational inspection of frozen evaluation evidence and its limits",
        "contract": contract,
        "suite": {"item_count": len(items), "source_group_count": len({i["source_group"] for i in items}),
                  "splits": dict(sorted(Counter(i["split"] for i in items).items())),
                  "tasks": dict(sorted(Counter(i["task"] for i in items).items())),
                  "items": items,
                  "split_boundary": "Recorded split names are declarations; no training-contamination or untouched-test proof"},
        "selection": {"rule": "All supplied attempts, including errors; no best-of-N or gold-informed selection",
                      "upstream_selection": _unavailable("Whether records were selected before export is not established"),
                      "denominator": "Retained response attempts, not missing suite items or unseen model generations"},
        "evaluation": {"raw_record_count": len(records), "replayed_record_count": len(replay["rows"]),
                       "raw_records_sha256": canonical_hash(records),
                       "parser_version": contract["parser_version"], "rubric_version": contract["rubric_version"],
                       "metrics": "Authoritative replay accuracy, task success, format validity, natural termination and truncation",
                       "rubric_boundary": "Executable authored checks; no independent human behavioral review",
                       "errors": sum(row["error"] is not None for row in replay["rows"]),
                       "unsupported": sum(row["status"] == "UNSUPPORTED" for row in replay["rows"]),
                       "paired_comparisons": paired},
        "models": models,
        "generation": {"status": "not executed by exporter", "frozen_settings": contract["settings"],
                       "interface_status": "frozen declaration" if "generation" in contract["settings"] else "unavailable",
                       "boundary": "Record stop/token/cost fields are retained observations or authored fixtures; their origin is not inferred from checkpoint names"},
        "approval": _unavailable("No execution, publication, deployment or release approval consumed or granted"),
        "independent_behavioral_review": _unavailable("No independent human ratings consumed"),
        "broad_capability_and_safety": _unavailable("Finite supplied items cannot certify broad capability, safety or reasoning faithfulness"),
        "resource_boundary": "Costs sum only known retained values; unknowns are not zero measurements. Token positions are not FLOPs, total physical resources or provider bills",
        "evidence": _copy(evidence or {"status": "in-memory inputs; file-byte identities unavailable"}),
        "replay_sha256": canonical_hash(replay),
    }
    card["card_sha256"] = canonical_hash(card)
    return card, replay


def _text(value):
    # Checkpoint names and other external strings must not become active HTML,
    # table rows, headings, links or Markdown directives in a generated card.
    return html.escape(str(value), quote=True).replace("\n", " ").replace("\r", " ").replace("|", "&#124;").replace("`", "&#96;").replace("[", "&#91;").replace("]", "&#93;").replace("*", "&#42;").replace("_", "&#95;")


def render_model_card(card):
    """Render the canonical evidence card; do not invent explanatory measurements."""
    lines = ["# Saved response evaluation model card", "",
        "This card replays the supplied response ledger under a frozen evaluation contract. "
        "It does not run generation, establish training lineage or authorize a release. "
        "The results describe retained attempts on these items, not general model capability.", "",
        "## Evaluation scope", "",
        f"The suite contains {card['suite']['item_count']} items in {card['suite']['source_group_count']} source groups. "
        f"Replay retained all {card['evaluation']['replayed_record_count']} responses, including "
        f"{card['evaluation']['errors']} error rows and {card['evaluation']['unsupported']} unsupported answers.", "",
        "Split labels and task definitions are preserved in model-card.json. They do not prove that evaluation "
        "data was untouched during training or recipe selection. Gold references and executable rubric rules "
        "are grading inputs only; this exporter performs no response selection.", "",
        "## Results on retained attempts", "",
        "Rates use every retained response, including failed and capped attempts. Missing suite items are not "
        "silently inserted into the denominator. Task, source-group and split slices, status counts, missing "
        "items and every raw response/error remain in replay.json.", "",
        "| Checkpoint record ID | Attempts | Accuracy | Task success | Format valid | Natural stop | Truncated |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for checkpoint, model in sorted(card["models"].items()):
        s = model["results"]["summary"]
        lines.append(f"| {_text(checkpoint)} | {s['n']} | {s['accuracy']:.6f} | {s['task_success']:.6f} | {s['format_valid']:.6f} | {s['natural_termination']:.6f} | {s['truncation']:.6f} |")
    if not card["models"]:
        lines.extend(["", "No response rows were supplied; no model score is available."])
    lines.extend(["", "## Paired comparisons", ""])
    pairs = card["evaluation"]["paired_comparisons"]
    if not pairs:
        lines.append("No paired comparison was requested; no improvement estimate is reported.")
    for pair in pairs:
        lines.append(f"{_text(pair['candidate'])} minus {_text(pair['baseline'])}: task-success difference "
                     f"{pair['delta']:.6f}, percentile interval [{pair['interval'][0]:.6f}, {pair['interval'][1]:.6f}], "
                     f"{pair['n_items_samples']} aligned item/sample pairs in {pair['n_source_groups']} source groups "
                     f"({pair['draws']} draws, seed {pair['seed']}).")
        lines.append("")
    lines.extend(["The source-group bootstrap preserves within-group dependence but a small panel yields fragile "
                  "descriptive uncertainty. It is not evidence of population-wide improvement.", "",
        "## Decoding and grading", "",
        "The frozen contract preserves template, thinking mode, decoding, stopping, output cap, parser and "
        "rubric versions. The complete settings, references and format/rubric policies are in model-card.json. "
        "Configured device and dtype are declarations, not exporter measurements. Unsupported mathematics "
        "stays unsupported; the exporter delegates grading to the existing bounded evaluator.", "",
        "## Identity and resource evidence", ""])
    for checkpoint, model in sorted(card["models"].items()):
        lines.extend([f"Record ID {_text(checkpoint)}: {_text(model['provenance_status'])}; "
                      f"{model['unlinked_record_count']} rows lack a matching supplied run identity.", ""])
        for name, cost in sorted(model["recorded_costs"].items()):
            lines.append(f"- {_text(name)}: retained known total {cost['known_total']!r}; "
                         f"{cost['known_rows']} known and {cost['unknown_rows']} unknown rows.")
        lines.append("")
    lines.extend(["Supplied run identities are checked for internal hash, input-map, settings and interface consistency. "
        "Their device, environment and checkpoint-file digests are recorded metadata, not a fresh weight reload, "
        "authenticated signature or hardware benchmark. Unknown costs are not zero measurements. "
        "Generated tokens, completed and attempted forward positions and wall time retain their separate units; "
        "they are not FLOPs, provider bills or whole-job physical resources.", "",
        "## Unavailable evidence", "",
        "Model architecture, training history, verified checkpoint genealogy, independent human behavioral review, "
        "broad capability/safety and execution/publication/deployment approval are unavailable from the consumed "
        "evidence. No pretrained evaluation, independent behavioral review or training claim is inferred. "
        "Authored fixture labels and real local-adapter observations must not be conflated.", "",
        "## Reproduction evidence", "",
        "model-card.json contains exact input-byte/source digests when exported from files, the frozen contract, "
        "all item/rubric definitions, resource completeness and paired settings. replay.json retains the full "
        "graded ledger. Neither artifact modifies its source evidence.", "",
        f"Card identity: {_text(card['card_sha256'])}. Frozen contract: {_text(card['contract']['identity'])}. "
        f"Replay identity: {_text(card['replay_sha256'])}.", ""])
    return "\n".join(lines)


def _reject_duplicates(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate JSON key")
        obj[key] = value
    return obj


def _read(path, maximum):
    path = Path(path).absolute()
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("Evidence paths must not traverse symlinks")
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError("Evidence must be a bounded regular file")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read(maximum + 1)
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if len(data) > maximum or any(getattr(after, key) != getattr(before, key) for key in fields):
            raise ValueError("Evidence changed during bounded read")
        return data, {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    finally:
        os.close(fd)


def _parse(data):
    return json.loads(data, object_pairs_hook=_reject_duplicates,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def export_model_card(*, items_path, contract_path, records_path, output,
                       run_identity_paths=(), comparisons=(), draws=2000, seed=1010):
    """Create an exclusive new evidence bundle; partial outputs are never erased.

    Bounded regular-file reads protect cooperative local usage, not a hostile
    filesystem sandbox or a physical quota. No current environment is probed.
    """
    paths = [("items", items_path, MAX_JSON_BYTES), ("contract", contract_path, MAX_JSON_BYTES),
             ("records", records_path, MAX_LEDGER_BYTES)]
    if len(run_identity_paths) > MAX_RECORDS:
        raise ValueError("Too many identity files")
    paths.extend((f"run_identity_{i:04}", p, MAX_JSON_BYTES) for i, p in enumerate(run_identity_paths))
    output = Path(output).absolute()
    for parent in (output, *output.parents):
        if parent.is_symlink():
            raise ValueError("Output path must not traverse symlinks")
    if output.exists():
        raise FileExistsError("Card bundle output must be a new directory")
    loaded, inputs = {}, {}
    for role, path, maximum in paths:
        path = Path(path).absolute()
        if path == output or output in path.parents:
            raise ValueError("Output overlaps input evidence")
        data, inputs[role] = _read(path, maximum)
        if role == "records":
            lines = [line for line in data.splitlines() if line.strip()]
            if len(lines) > MAX_RECORDS:
                raise ValueError("Too many response rows")
            loaded[role] = [_parse(line) for line in lines]
        else:
            loaded[role] = _parse(data)
    root = Path(__file__).resolve().parents[2]
    sources = {name: _read(root / name, MAX_JSON_BYTES)[1] for name in SOURCE_PATHS}
    evidence = {"status": "actual consumed file bytes and current replay/exporter source bytes",
                "inputs": inputs, "sources": sources,
                "boundary": "Digests bind retained bytes, not authenticity, approval, training provenance or deployed weights"}
    card, replay = build_evaluation_card(loaded["items"], loaded["records"], loaded["contract"],
        run_identities=[loaded[role] for role, _, _ in paths if role.startswith("run_identity_")],
        comparisons=comparisons, draws=draws, seed=seed, evidence=evidence)
    payloads = {"model-card.json": _json_bytes(card), "replay.json": _json_bytes(replay),
                "model-card.md": render_model_card(card).encode("utf-8")}
    os.mkdir(output, mode=0o700)
    for name, data in payloads.items():
        with (output / name).open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    return card
