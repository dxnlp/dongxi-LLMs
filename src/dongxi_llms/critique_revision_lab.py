"""Original bounded critique/revision microscope; no model work or IO on import.

Programmatic callbacks are NOT neural self-critique. Historical tiny-model
candidates may be replayed, with source costs distinguished from replay work.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time

from .evaluation_lab import canonical_hash
from .inference_selection_lab import (EOS, NUMBER_START, budget_prefix, cost_totals,
                                      eligible_answer, select, validate_items)
from .reasoning_evaluation import candidate_view, grade_response
from .reasoning_generation import GenerationJournal

SCHEMA = "dongxi-critique-replay-v1"
MODES = ("identity", "repair", "contrarian")
SEEDS = (26061, 26062, 26063)
ACTION_IDS = {"keep": 20, "flip": 21, "repair": 22, "invalid": 23}
SYMBOLS = {EOS: "<eos>", NUMBER_START: "0", NUMBER_START + 1: "1",
           **{v: k.upper() for k, v in ACTION_IDS.items()}}
DRAFT_KEYS = ("sample_id", "item_id", "raw_response", "response_text", "token_ids",
              "stop_reason", "truncated", "error", "generated_tokens")
ITEM_KEYS = ("id", "source_group", "split", "slice", "task", "prompt")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def callback_views(item, draft):
    """Whitelist separately from saved/evaluated records; nested gold never passes."""
    return ({k: deepcopy(item[k]) for k in ITEM_KEYS if k in item},
            {k: deepcopy(draft[k]) for k in DRAFT_KEYS if k in draft})


def emission(ids, stop="eos", error=None):
    ids = list(ids)
    text = " ".join(SYMBOLS.get(t, f"<token:{t}>") for t in ids)
    response = ids[:-1] if stop == "eos" and ids and ids[-1] == EOS else ids
    return {"raw_response": text,
            "response_text": " ".join(SYMBOLS.get(t, f"<token:{t}>") for t in response),
            "token_ids": ids, "generated_tokens": len(ids), "stop_reason": stop,
            "truncated": stop == "max_tokens", "error": error,
            "origin": "authored/programmatic; no model generation or likelihood",
            "model_forward_calls": 0}


def _valid_binary(row):
    """Gate the actual symbolic path, not only an agreeable decoded string."""
    return bool(row and row.get("error") is None and row.get("stop_reason") == "eos"
                and not row.get("truncated")
                and row.get("token_ids") in ([NUMBER_START, EOS], [NUMBER_START + 1, EOS])
                and row.get("generated_tokens") == 2
                and row.get("response_text") == str(row["token_ids"][0] - NUMBER_START))


def acceptance(current, proposed):
    """A syntax/termination gate cannot establish correctness; valid ties pass."""
    valid = _valid_binary(proposed)
    return {"accepted": valid, "current_format_score": int(_valid_binary(current)),
            "proposed_format_score": int(valid),
            "tie": bool(valid and _valid_binary(current)),
            "reason": "valid-format-tie-or-improvement" if valid else "invalid-or-error-proposal",
            "gold_used": False}


def critique(item_view, draft_view, round_index, *, mode):
    del item_view, round_index
    if mode not in MODES:
        raise ValueError("Undeclared programmatic critique mode")
    action = "keep" if mode == "identity" else (
        "flip" if mode == "contrarian" and _valid_binary(draft_view) else
        "keep" if _valid_binary(draft_view) else "repair")
    return emission([ACTION_IDS[action], EOS])


def revise(item_view, draft_view, critique_view, round_index):
    del item_view, round_index
    action = critique_view["token_ids"][0]
    ids = draft_view["token_ids"]
    if action == ACTION_IDS["keep"]:
        return emission(ids, draft_view["stop_reason"], draft_view["error"])
    if action == ACTION_IDS["flip"]:
        if not _valid_binary(draft_view):
            raise ValueError("FLIP requires a well-formed binary draft")
        return emission([NUMBER_START + 1 - (ids[0] - NUMBER_START), EOS])
    if action == ACTION_IDS["repair"]:
        first = next((t for t in ids if t in (NUMBER_START, NUMBER_START + 1)), NUMBER_START)
        return emission([first, EOS])
    if action == ACTION_IDS["invalid"]:
        return emission([NUMBER_START, NUMBER_START + 1, EOS])
    raise ValueError("Unknown critique action")


def _invoke(callback, arguments, role):
    began = time.perf_counter()
    result, validation_error = None, None
    try:
        result = callback(*arguments)
        if not isinstance(result, dict):
            raise ValueError("Callback must return an emission object")
        ids = result.get("token_ids")
        if (not isinstance(ids, list) or len(ids) > 16 or
                any(type(t) is not int or t < 0 for t in ids)):
            raise ValueError("Bounded exact nonnegative token IDs required")
        if result.get("generated_tokens") != len(ids):
            raise ValueError("Emission count differs from retained token IDs")
        if result.get("stop_reason") not in ("eos", "max_tokens", "error"):
            raise ValueError("Undeclared stopping condition")
        if result.get("raw_response") != emission(ids, result["stop_reason"])["raw_response"]:
            raise ValueError("Raw symbolic serialization differs from token path")
        expected = emission(ids, result["stop_reason"], result.get("error"))
        if result.get("response_text") != expected["response_text"] or result.get("truncated") != expected["truncated"]:
            raise ValueError("Decoded response/stopping metadata disagree")
        if result["stop_reason"] == "eos" and (not ids or ids[-1] != EOS or EOS in ids[:-1]):
            raise ValueError("EOS must occur once at the final emitted position")
        if role == "critique":
            if ids not in ([v, EOS] for v in ACTION_IDS.values()) or result.get("error") is not None:
                raise ValueError("Critique requires a known action then EOS")
            if result["stop_reason"] != "eos" or result["truncated"]:
                raise ValueError("Critique requires natural EOS, without truncation/error")
        # Return only serialized emission fields, not arbitrary callback metadata.
        result = expected
    except Exception as error:
        validation_error = f"{type(error).__name__}: {error}"
        if isinstance(result, dict):
            # Keep an invalid returned emission, including its bytes, for diagnosis.
            retained = result
            try:
                retained = deepcopy(result)
                json.dumps(retained, allow_nan=False)
                result = retained
            except Exception:
                retained_ids = retained.get("token_ids") if isinstance(retained, dict) else None
                known_ids = (isinstance(retained_ids, list) and len(retained_ids) <= 16
                             and all(type(t) is int and t >= 0 for t in retained_ids))
                retained_stop = retained.get("stop_reason") if isinstance(retained, dict) else None
                result = emission(retained_ids if known_ids else [],
                                  retained_stop if known_ids and retained_stop in ("eos", "max_tokens", "error") else "error",
                                  validation_error)
                try:
                    result["raw_return_repr"] = repr(retained)[:4096]
                except Exception:
                    result["raw_return_repr"] = f"Unrepresentable return type: {type(retained).__name__}"
                result["unserializable_return"] = True
                result["emitted_token_count_unknown"] = not known_ids
        else:
            result = emission([], "error", validation_error)
        result["error"] = validation_error
        result["error_stage"] = f"{role}_callback_or_validation"
    return result, {"role": role, "attempted_calls": 1, "completed_calls": int(validation_error is None),
                    "wall_seconds": time.perf_counter() - began, "error": validation_error}


def _grade(item, row):
    if row is None:
        return {"state": "invalid", "correct": False, "complete_success": False, "status": "NOT_ATTEMPTED"}
    evaluated = grade_response(item, row)
    valid = _valid_binary(row)
    return {"state": "right" if valid and evaluated["correct"] else "wrong" if valid else "invalid",
            "correct": evaluated["correct"], "complete_success": bool(valid and evaluated["correct"]),
            "status": evaluated["status"], "format_eligible": valid}


def _same_path(before, after):
    return all(before.get(k) == after.get(k) for k in ("token_ids", "stop_reason", "error"))


def transition(item, before, after):
    a, b = _grade(item, before), _grade(item, after)
    unchanged = after is not None and _same_path(before, after)
    label = ("unchanged" if unchanged else "invalid" if "invalid" in (a["state"], b["state"])
             else f"{a['state']}_to_{b['state']}")
    return {"category": label, "before": a["state"], "after": b["state"],
            "unchanged_path": unchanged, "before_grade": a, "after_grade": b}


def run_loop(item, draft, *, mode="identity", max_rounds=2,
             critique_fn=None, revision_fn=revise, on_round=None):
    if mode not in MODES or type(max_rounds) is not int or not 1 <= max_rounds <= 2:
        raise ValueError("Declared mode and one or two exact rounds required")
    current, rounds = deepcopy(draft), []
    critic = critique_fn or (lambda i, d, r: critique(i, d, r, mode=mode))
    for ordinal in range(max_rounds):
        before = deepcopy(current)
        iv, dv = callback_views(item, current)
        crit, crit_cost = _invoke(critic, (iv, dv, ordinal), "critique")
        proposed, revision_cost = None, {"role": "revision", "attempted_calls": 0,
            "completed_calls": 0, "wall_seconds": 0., "error": None}
        if crit["error"] is None:
            iv, dv = callback_views(item, current)
            proposed, revision_cost = _invoke(revision_fn, (iv, dv, deepcopy(crit), ordinal), "revision")
        decision = acceptance(before, proposed)
        if decision["accepted"]:
            current = deepcopy(proposed)
        stopped = bool(decision["accepted"] and _same_path(before, current))
        row = {"round": ordinal + 1, "draft": before, "critique": crit, "revision": proposed,
               "acceptance": decision, "delivered": deepcopy(current),
               "callback_costs": [crit_cost, revision_cost],
               "attempted_transition": transition(item, before, proposed),
               "delivered_transition": transition(item, before, current),
               "stopping": "unchanged-accepted-path" if stopped else
                           "round-limit" if ordinal + 1 == max_rounds else "continue"}
        rounds.append(row)
        if on_round is not None:
            on_round(deepcopy(row))
        if stopped:
            break
    extra = sum(len(r["critique"].get("token_ids", [])) +
                (len(r["revision"].get("token_ids", [])) if r["revision"] else 0) for r in rounds)
    return {"mode": mode, "initial": deepcopy(draft), "initial_grade": _grade(item, draft),
            "rounds": rounds, "delivered": current, "delivered_grade": _grade(item, current),
            "final_transition": transition(item, draft, current),
            "serialized_tokens": draft["generated_tokens"] + extra,
            "new_model_forwards": 0,
            "boundary": "programmatic critique/revision, evaluation-only gold; not neural self-refinement"}


def independent_control(item, pool, token_budget):
    eligible, attempted, rejections = budget_prefix(pool, tokens=token_budget)
    decision = select(eligible, "majority")
    winner = next((r for r in eligible if r["sample_id"] == decision["selected_sample_id"]), None)
    eligible_tokens = sum(r["generated_tokens"] for r in eligible)
    attempted_tokens = sum(r["generated_tokens"] for r in attempted)
    return {"decision": decision, "selected": deepcopy(winner), "grade": _grade(item, winner),
            "token_ceiling": token_budget, "eligible_tokens": eligible_tokens,
            "attempted_tokens": attempted_tokens, "underfill": token_budget - eligible_tokens,
            "overshoot": max(0, attempted_tokens - token_budget),
            "exact_serialized_match": attempted_tokens == eligible_tokens == token_budget,
            "attempted_sample_ids": [r["sample_id"] for r in attempted],
            "eligible_sample_ids": [r["sample_id"] for r in eligible], "rejections": rejections,
            "historical_source_costs": cost_totals(attempted),
            "boundary": "prefix replay; independent draws conditional on same checkpoint are not independent correctness"}


def programmatic_pool(item, seed):
    rows = []
    for ordinal in range(8):
        coordinate = int(canonical_hash([seed, item["id"], ordinal, "programmatic-binary-v1"])[:16], 16)
        token = NUMBER_START + random.Random(coordinate).randrange(2)
        row = emission([token, EOS])
        row.update(sample_id=f"programmatic-{seed}-{item['id']}-{ordinal}", item_id=item["id"],
                   source_group=item["source_group"], split=item["split"], task=item["task"],
                   checkpoint_id=f"programmatic-{seed}", contract_id=SCHEMA, ordinal=ordinal,
                   attempt_seed=coordinate, cost={"generation_tokens": 0, "scoring_tokens": 0,
                       "wall_seconds": None}, cost_boundary="two serialized tokens; zero neural generation tokens")
        rows.append(row)
    return rows


def _scripted_callbacks(actions):
    def critic(i, d, r):
        del i, d
        action = actions[r]
        if action == "critique_error":
            raise RuntimeError("Authored critique exception")
        return emission([99 if action == "unknown" else ACTION_IDS.get(action, ACTION_IDS["keep"]), EOS])
    def revision(i, d, c, r):
        action = actions[r]
        if action == "revision_error":
            raise RuntimeError("Authored revision exception")
        if action == "empty":
            return emission([EOS])
        if action == "cap":
            return emission([NUMBER_START] * 3, "max_tokens")
        return revise(i, d, c, r)
    return critic, revision


def load_inputs(root, protocol_path):
    root, protocol_path = Path(root), Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if (protocol.get("schema") != SCHEMA or protocol.get("max_rounds") != 2
            or protocol.get("modes") != list(MODES) or protocol.get("programmatic_seeds") != list(SEEDS)
            or protocol.get("programmatic_candidates") != 8
            or protocol.get("acceptance") != "valid-binary-EOS-only; valid ties accepted; no correctness score"
            or protocol.get("stopping") != "accepted identical token path or round limit; never gold"
            or protocol.get("independent_selector") != "majority; earliest eligible answer breaks ties"
            or protocol.get("token_interface") != {"EOS": EOS, "0": NUMBER_START, "1": NUMBER_START + 1,
                **{k.upper(): v for k, v in ACTION_IDS.items()}}):
        raise ValueError("Frozen bounded protocol differs from implemented interface")
    paths = [root / protocol["source_items"], root / protocol["source_responses"]]
    if any(not p.resolve().is_relative_to(root.resolve()) for p in paths):
        raise ValueError("Source inputs must remain in this repository")
    for key, path in zip(("source_items", "source_responses"), paths):
        if digest(path) != protocol[key + "_sha256"]:
            raise ValueError("Frozen source bytes changed")
    items = json.loads(paths[0].read_text())
    audit = validate_items(items)
    records = [json.loads(line) for line in paths[1].read_text().splitlines()]
    if len(records) != protocol.get("source_candidates") or len(records) != 864:
        raise ValueError("All 864 raw source attempts must be retained")
    pools = defaultdict(list)
    by_item = {i["id"]: i for i in items}
    for row in records:
        if row["item_id"] not in by_item:
            raise ValueError("Unknown source item")
        item = by_item[row["item_id"]]
        if row["source_group"] != item["source_group"] or row["split"] != item["split"]:
            raise ValueError("Source/split identity differs")
        pools[(row["checkpoint_id"], row["item_id"])].append(row)
    if len(pools) != 108 or any([r["ordinal"] for r in p] != list(range(8)) for p in pools.values()):
        raise ValueError("Six checkpoints ×18 items ×eight ordered attempts required")
    # Existing selector validation checks context, IDs and fixed ordinals without gold.
    for pool in pools.values():
        select(pool, "first")
    adversarial = json.loads((root / "fixtures/critique-revision/adversarial.json").read_text())
    if len(adversarial) != 12 or len({r["id"] for r in adversarial}) != 12:
        raise ValueError("Twelve unique authored microscope cases required")
    for row in adversarial:
        if (row.get("reference") not in ("0", "1") or len(row.get("actions", [])) != 2
                or any(a not in set(ACTION_IDS) | {"critique_error", "revision_error", "unknown", "empty", "cap"}
                       for a in row["actions"])
                or row.get("draft_stop") not in ("eos", "max_tokens")
                or not isinstance(row.get("draft_ids"), list)
                or any(type(t) is not int or t not in (EOS, NUMBER_START, NUMBER_START + 1) for t in row["draft_ids"])):
            raise ValueError("Adversarial fixture outside the frozen binary/action grammar")
    return protocol, items, records, pools, adversarial, audit


def summarize(cases):
    groups = defaultdict(list)
    for case in cases:
        for label in ("all", case["slice"]):
            groups[(case["panel"], case["mode"], label)].append(case)
    result = []
    for (panel, mode, label), rows in sorted(groups.items()):
        attempted = Counter(r["attempted_transition"]["category"] for c in rows for r in c["loop"]["rounds"])
        delivered = Counter(r["delivered_transition"]["category"] for c in rows for r in c["loop"]["rounds"])
        result.append({"panel": panel, "mode": mode, "slice": label, "cases": len(rows),
            "source_groups": len({c["source_group"] for c in rows}),
            "no_critique_success": sum(c["loop"]["initial_grade"]["complete_success"] for c in rows),
            "revision_success": sum(c["loop"]["delivered_grade"]["complete_success"] for c in rows),
            "independent_success": sum(c["independent"]["grade"]["complete_success"] for c in rows),
            "attempted_transitions": dict(attempted), "delivered_transitions": dict(delivered),
            "rounds": sum(len(c["loop"]["rounds"]) for c in rows),
            "accepted": sum(r["acceptance"]["accepted"] for c in rows for r in c["loop"]["rounds"]),
            "callback_errors": sum(k["error"] is not None for c in rows for r in c["loop"]["rounds"] for k in r["callback_costs"]),
            "serialized_revision_tokens": sum(c["loop"]["serialized_tokens"] for c in rows),
            "independent_attempted_tokens": sum(c["independent"]["attempted_tokens"] for c in rows),
            "exact_matches": sum(c["independent"]["exact_serialized_match"] for c in rows),
            "underfill": sum(c["independent"]["underfill"] for c in rows),
            "overshoot": sum(c["independent"]["overshoot"] for c in rows),
            "boundary": "descriptive fixed-source panel; repeated modes share drafts; no IID or causal model-quality inference"})
    return result


def run_campaign(root, protocol_path, *, journal=None):
    started = time.perf_counter()
    protocol, items, raw, pools, adversarial, audit = load_inputs(root, protocol_path)
    by_item, cases = {i["id"]: i for i in items}, []
    program_pools = []
    panel_pools = [("actual-candidate-replay", model, by_item[item], pool)
                   for (model, item), pool in sorted(pools.items())]
    for seed in SEEDS:
        for item in items:
            pool = programmatic_pool(item, seed)
            program_pools.append({"seed": seed, "item_id": item["id"], "candidates": pool})
            panel_pools.append(("exact-programmatic-control", f"programmatic-{seed}", item, pool))
    for panel, checkpoint, item, pool in panel_pools:
        for mode in MODES:
            coordinate = {"panel": panel, "checkpoint_id": checkpoint, "item_id": item["id"], "mode": mode}
            loop = run_loop(item, pool[0], mode=mode,
                on_round=(lambda row, c=coordinate: journal.record(dict(c, **row))) if journal else None)
            independent = independent_control(item, pool, loop["serialized_tokens"])
            case = dict(coordinate, source_group=item["source_group"], slice=item["slice"],
                        loop=loop, independent=independent,
                        no_critique_serialized_tokens=pool[0]["generated_tokens"],
                        historical_draft_costs=cost_totals(pool[:1]),
                        historical_cost_boundary="reused once-collected source work, not newly executed model forwards")
            cases.append(case)
            if journal:
                journal.event("case_completed", **coordinate)
    microscope = []
    for fixture in adversarial:
        item = dict(id=fixture["id"], source_group="authored-" + fixture["id"], split="test", slice="adversarial",
                    task="binary-fixture", prompt="Authored state-machine fixture", reference=fixture["reference"],
                    kind="math", format_policy="single_integer", extraction="whole")
        critic, reviser = _scripted_callbacks(fixture["actions"])
        draft = emission(fixture["draft_ids"], fixture["draft_stop"])
        loop = run_loop(item, draft, critique_fn=critic, revision_fn=reviser,
                        on_round=(lambda row, i=item: journal.record(dict(panel="authored-adversarial", item_id=i["id"], **row))) if journal else None)
        microscope.append({"fixture": fixture, "loop": loop})
    return {"schema": SCHEMA, "protocol": protocol, "source_item_audit": audit,
            "source_candidates": raw, "programmatic_pools": program_pools,
            "cases": cases, "summary": summarize(cases), "adversarial": microscope,
            "campaign_wall_seconds": time.perf_counter() - started,
            "new_model_forwards": 0,
            "limits": ["authored/programmatic critique and revision, no neural self-critique",
                "historical actual model candidates are not conditioned on critiques",
                "serialized-token matching is not model-token/FLOP/latency/billing matching",
                "whole-attempt replay overshoots and underfills retained; no adaptive stopping claim",
                "related original source siblings; descriptive comparisons not population statistics",
                "no guarantee of correctness, usefulness or transferable revision benefit"]}


def stable_payload(value):
    """Numeric/content replay comparison excludes only explicitly volatile timing."""
    if isinstance(value, dict):
        return {k: stable_payload(v) for k, v in value.items()
                if k not in ("wall_seconds", "campaign_wall_seconds", "source_identity", "created_utc")}
    if isinstance(value, list):
        return [stable_payload(v) for v in value]
    return value


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output directory; historical evidence cannot be overwritten")
    root = Path(__file__).resolve().parents[2]
    sources = [root / p for p in ("src/dongxi_llms/critique_revision_lab.py",
        "tests/test_critique_revision_lab.py", "src/dongxi_llms/inference_selection_lab.py",
        "src/dongxi_llms/reasoning_evaluation.py", "src/dongxi_llms/reasoning_generation.py",
        "src/dongxi_llms/evaluation_lab.py", "fixtures/critique-revision/adversarial.json")]
    protocol = json.loads(args.protocol.read_text())
    paths = sources + [args.protocol, args.spec, root / protocol["source_items"], root / protocol["source_responses"]]
    before = {str(p.relative_to(root) if p.is_relative_to(root) else p): digest(p) for p in paths}
    journal = GenerationJournal(args.output)
    try:
        packages = {}
        for name in ("torch", "transformers"):
            try: packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: packages[name] = None
        identity = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                    "executable": sys.executable, "platform": platform.platform(), "packages": packages,
                    "source_and_input_sha256": before, "executor": __name__,
                    "boundary": "actual CPU replay; no Git operation, model invocation or acquisition"}
        journal.write_new("input-identity.json", identity)
        result = run_campaign(root, args.protocol, journal=journal)
        if any(digest(p) != before[str(p.relative_to(root) if p.is_relative_to(root) else p)] for p in paths):
            raise RuntimeError("Source/input bytes changed during collection")
        result["source_identity"] = identity
        result["inputs_unchanged"] = True
        result["content_sha256"] = canonical_hash(stable_payload(result))
        journal.write_new("results.json", result)
        journal.event("completed", cases=len(result["cases"]), source_candidates=len(result["source_candidates"]),
                      authored_adversarial=len(result["adversarial"]), new_model_forwards=0)
        print(f"Saved bounded critique/revision replay: {args.output}/results.json")
    except (Exception, KeyboardInterrupt) as error:
        journal.event("failure", error_type=type(error).__name__, error=str(error))
        journal.write_new("failure.json", {"error_type": type(error).__name__, "error": str(error),
            "boundary": "partial fsynced round/event records retained; no resume/exactly-once claim"})
        raise
    finally:
        journal.close()


if __name__ == "__main__":
    main()
