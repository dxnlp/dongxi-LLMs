"""Actual tiny causal candidates, gold-blind selectors and retained attempt costs.

Original CPU microscope. The symbolic decoder does not parse English prompts;
its likelihood selector is not a correctness/reward judge. Nothing runs on import.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import time

import torch
from torch.nn import functional as F

from .evaluation_lab import canonical_hash
from .reasoning_controls import BOS, EOS, ALIAS, ODD, SUM, NUMBER_START, make_decoder, state_hash
from .reasoning_evaluation import (SCHEMA_VERSION, candidate_view, freeze_contract,
                                   grade_response, validate_record)
from .reasoning_generation import GenerationJournal
from .run_identity import collect_run_identity

SEEDS = (10051, 10052, 10053)
UPDATES, SAMPLES, CAP = 80, 8, 3
N_VALUES, TOKEN_BUDGETS = (1, 2, 4, 8), (3, 6, 12, 24)
SUPPORT = (EOS, NUMBER_START, NUMBER_START + 1)
SELECTORS = ("first", "majority", "unique_support", "mean_logp", "longest")
SLICES = ("train", "heldout-source", "heldout-template", "heldout-family")


def prompt_ids(item):
    a, b = item["problem"]["a"], item["problem"]["b"]
    instruction = ALIAS if item["template_id"] == "alias" else (
        ODD if item["family"] == "parity" else SUM)
    return [BOS, instruction, NUMBER_START + a, NUMBER_START + b]


def validate_items(items):
    if not isinstance(items, list) or not items:
        raise ValueError("Nonempty original item list required")
    ids, sources, problems, encodings = set(), {}, {}, {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each authored item must be an object")
        for key in ("id", "source_group", "split", "slice", "family", "template_id", "prompt"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError("Nonempty item/source/template identities required")
        if item["id"] in ids or item["split"] not in ("train", "test") or item["slice"] not in SLICES:
            raise ValueError("Repeated ID or undeclared split/slice")
        ids.add(item["id"])
        if (item["slice"] == "train") != (item["split"] == "train"):
            raise ValueError("Training slice/split disagree")
        if item["family"] not in ("parity", "sum-positive") or item["template_id"] not in ("direct", "alias"):
            raise ValueError("Outside fixed symbolic task grammar")
        problem = item.get("problem")
        if not isinstance(problem, dict) or set(problem) != {"a", "b"} or any(
                type(v) is not int or not 0 <= v <= 3 for v in problem.values()):
            raise ValueError("Two exact integer operands0..3 required")
        value = problem["a"] + problem["b"]
        oracle = value % 2 if item["family"] == "parity" else int(value > 0)
        if item.get("reference") != str(oracle) or item.get("kind") != "math" or item.get("format_policy") != "single_integer":
            raise ValueError("Authored reference/interface disagrees with independent arithmetic")
        for mapping, key in ((sources, item["source_group"]),
                (problems, canonical_hash([item["family"], problem])),
                (encodings, canonical_hash(prompt_ids(item)))):
            if mapping.setdefault(key, item["split"]) != item["split"]:
                raise ValueError("Source, underlying problem or encoded prompt crosses splits")
    train = [i for i in items if i["split"] == "train"]
    if not train or any(i["family"] != "parity" or i["template_id"] != "direct" for i in train):
        raise ValueError("Training is direct parity only")
    return {"items": len(items), "sources": len(sources), "train_item_ids": [i["id"] for i in train],
            "train_answers": dict(Counter(i["reference"] for i in train)),
            "slices": dict(Counter(i["slice"] for i in items)), "suite_sha256": canonical_hash(items)}


def fit_policy(items, seed, *, updates=UPDATES, progress=None):
    """Train two teacher-forced conditional actions; no held-out labels enter fit."""
    validate_items(items)
    if type(seed) is not int or type(updates) is not int or not 1 <= updates <= UPDATES:
        raise ValueError("Exact integer seed and bounded positive update count required")
    initial = make_decoder(seed).double().eval()
    policy = deepcopy(initial)
    train = [i for i in items if i["split"] == "train"]
    prompts = torch.tensor([prompt_ids(i) for i in train])
    responses = torch.tensor([[NUMBER_START + int(i["reference"]), EOS] for i in train])
    inputs = torch.cat((prompts, responses[:, :1]), -1)
    targets = torch.tensor([[SUPPORT.index(int(t)) for t in r] for r in responses])
    optimizer = torch.optim.AdamW(policy.parameters(), lr=.02, weight_decay=0.)
    history = []
    for step in range(updates):
        scores = policy(inputs)[:, 3:5, list(SUPPORT)]
        loss = F.cross_entropy(scores.reshape(-1, 3), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient)):
            raise ValueError("Nonfinite fixed SFT fit")
        reach = {name: float(p.grad.abs().sum()) for name, p in policy.named_parameters()
                 if p.grad is not None} if step == 0 else None
        history.append({"update": step + 1, "loss": float(loss.detach()),
                        "gradient_norm_before_clip": float(gradient), "first_gradient_reach": reach})
        optimizer.step()
        if progress is not None:
            progress(deepcopy(history[-1]))
    return initial.requires_grad_(False), policy.eval().requires_grad_(False), {
        "seed": seed, "updates": updates, "train_item_ids": [i["id"] for i in train],
        "initial_state_sha256": state_hash(initial), "final_state_sha256": state_hash(policy),
        "history": history, "target_actions": responses.tolist(),
        "boundary": "train-only conditional SFT; no test checkpoint selection or English parser"}


def coordinate_seed(seed, policy_id, item_id, ordinal):
    if type(seed) is not int or type(ordinal) is not int or ordinal < 0:
        raise ValueError("Exact seed and nonnegative candidate coordinate required")
    return int(canonical_hash([seed, policy_id, item_id, ordinal, "selection-v1"])[:16], 16) % 2**63


def decode(ids, *, raw=False):
    return " ".join("<eos>" if t == EOS else str(t - NUMBER_START) for t in ids
                    if raw or t != EOS)


@torch.no_grad()
def generate_candidate(model, item, contract, *, seed, policy_id, ordinal):
    """Count attempted work and preserve partial failures without inventing outputs."""
    begin = time.perf_counter()
    prompt = prompt_ids(item)
    rng_seed = coordinate_seed(seed, policy_id, item["id"], ordinal)
    record = {"schema_version": SCHEMA_VERSION, "contract_id": contract["identity"],
        "checkpoint_id": policy_id, "sample_id": f"{item['id']}-candidate-{ordinal}",
        "item_id": item["id"], "source_group": item["source_group"], "task": item["task"],
        "split": item["split"], "raw_response": "", "response_text": "", "token_ids": [],
        "prompt_token_ids": prompt, "prompt_tokens": len(prompt), "generated_tokens": 0,
        "stop_reason": "unknown", "truncated": False, "error": None, "error_stage": None,
        "ordinal": ordinal, "attempt_seed": rng_seed, "conditional_support": list(SUPPORT),
        "selected_log_probabilities": [], "rescored_log_probabilities": [],
        "cost": {"wall_seconds": 0., "generation_tokens": 0, "scoring_tokens": 0,
            "generation_forward_positions": 0, "attempted_generation_forward_positions": 0,
            "generation_forward_calls": 0, "attempted_generation_forward_calls": 0,
            "prefill_seconds": 0., "decode_seconds": 0., "scoring_seconds": 0.,
            "scoring_forward_positions": 0, "attempted_scoring_forward_positions": 0,
            "scoring_forward_calls": 0, "attempted_scoring_forward_calls": 0},
        "cost_boundary": "Actual uncached CPU forwards, EOS counted; positions are not FLOPs/billing"}
    rng = torch.Generator().manual_seed(rng_seed)
    stage = "generation_forward"
    try:
        for position in range(CAP):
            prefix = prompt + record["token_ids"]
            record["cost"]["attempted_generation_forward_positions"] += len(prefix)
            record["cost"]["attempted_generation_forward_calls"] += 1
            began = time.perf_counter()
            try:
                logits = model(torch.tensor([prefix]))[0, -1, list(SUPPORT)]
            finally:
                record["cost"]["prefill_seconds" if position == 0 else "decode_seconds"] += time.perf_counter() - began
            record["cost"]["generation_forward_positions"] += len(prefix)
            record["cost"]["generation_forward_calls"] += 1
            stage = "sampling"
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Nonfinite sampled logits")
            logp = logits.log_softmax(-1)
            action = int(torch.multinomial(logp.exp(), 1, generator=rng))
            token = SUPPORT[action]
            record["token_ids"].append(token)
            record["selected_log_probabilities"].append(float(logp[action]))
            record["generated_tokens"] = len(record["token_ids"])
            record["cost"]["generation_tokens"] = record["generated_tokens"]
            if token == EOS:
                record["stop_reason"] = "eos"
                break
            stage = "generation_forward"
        if record["stop_reason"] == "unknown":
            record.update(stop_reason="max_tokens", truncated=True)
        stage = "likelihood_rescoring"
        tokens = record["token_ids"]
        input_ids = prompt + tokens[:-1]
        record["cost"]["attempted_scoring_forward_positions"] = len(input_ids)
        record["cost"]["attempted_scoring_forward_calls"] = 1
        began = time.perf_counter()
        try:
            logp = model(torch.tensor([input_ids]))[0, len(prompt)-1:, list(SUPPORT)].log_softmax(-1)
        finally:
            record["cost"]["scoring_seconds"] += time.perf_counter() - began
        indices = torch.tensor([SUPPORT.index(t) for t in tokens])
        scores = logp.gather(-1, indices[:, None]).squeeze(-1)
        if not bool(torch.isfinite(scores).all()) or not torch.allclose(scores,
                torch.tensor(record["selected_log_probabilities"], dtype=scores.dtype), atol=1e-12, rtol=1e-12):
            raise ValueError("Collected and recomputed conditional likelihoods disagree")
        record["rescored_log_probabilities"] = scores.tolist()
        record["cost"].update(scoring_tokens=len(tokens), scoring_forward_positions=len(input_ids), scoring_forward_calls=1)
    except Exception as error:
        record.update(stop_reason="error", truncated=False, error=f"{type(error).__name__}: {error}", error_stage=stage)
    record["raw_response"] = decode(record["token_ids"], raw=True)
    grading_ids = record["token_ids"][:-1] if record["stop_reason"] == "eos" else record["token_ids"]
    record["response_text"] = decode(grading_ids, raw=True)
    record["cost"]["wall_seconds"] = time.perf_counter() - begin
    validate_record(record, contract, item)
    return record


def selection_view(record):
    """Whitelist only generation information even when passed a graded record."""
    view = candidate_view(record)
    for key in ("ordinal", "selected_log_probabilities", "rescored_log_probabilities", "conditional_support"):
        if key in record:
            view[key] = deepcopy(record[key])
    return deepcopy(view)


def _validate_pool(pool):
    if not isinstance(pool, list):
        raise ValueError("Ordered candidate list required")
    seen, context, previous = set(), None, -1
    for row in pool:
        if not isinstance(row, dict):
            raise ValueError("Each candidate must be a record object")
        key = row.get("sample_id")
        ordinal = row.get("ordinal")
        if not isinstance(key, str) or not key or key in seen or type(ordinal) is not int or ordinal <= previous:
            raise ValueError("Unique record IDs and increasing exact ordinals required")
        seen.add(key); previous = ordinal
        identity = tuple(row.get(k) for k in ("contract_id", "checkpoint_id", "item_id"))
        if context is not None and identity != context:
            raise ValueError("Selectors cannot mix models, contracts or items")
        context = identity


def eligible_answer(view):
    """Declared binary/format/termination support, with no answer reference."""
    text = view.get("response_text", view.get("raw_response"))
    if (view.get("error") is not None or view.get("stop_reason") != "eos"
            or not isinstance(text, str) or re.fullmatch(r"[01]", text.strip()) is None):
        return None
    return text.strip()


def select(pool, method):
    _validate_pool(pool)
    if method not in SELECTORS:
        raise ValueError("Undeclared non-oracle selector")
    views = [selection_view(row) for row in pool]
    eligible = [(r, eligible_answer(r)) for r in views]
    eligible = [(r, a) for r, a in eligible if a is not None]
    result = {"method": method, "selected_sample_id": None, "eligible": len(eligible),
              "support": sorted({a for _, a in eligible}), "tie": False, "reason": "empty-or-invalid-pool"}
    if method == "first" and views:
        winner = views[0]
    elif not eligible:
        return result
    elif method in ("majority", "unique_support"):
        counts = Counter(a for _, a in eligible)
        if method == "unique_support":
            counts = Counter({a: 1 for a in counts})
        maximum = max(counts.values())
        tied = {a for a, n in counts.items() if n == maximum}
        winner = next(r for r, a in eligible if a in tied)
        result.update(votes=dict(counts), tie=len(tied) > 1)
    else:
        values = []
        for row, _ in eligible:
            if method == "longest":
                value = row["generated_tokens"]
            else:
                scores = row.get("rescored_log_probabilities")
                if not isinstance(scores, list) or not scores or len(scores) != row["generated_tokens"] or any(
                        type(s) not in (int, float) or not math.isfinite(s) for s in scores):
                    raise ValueError("Likelihood ranker needs actual finite EOS-inclusive path scores")
                value = sum(scores) / len(scores)
            values.append(value)
        maximum = max(values)
        winner = eligible[values.index(maximum)][0]
        result["tie"] = values.count(maximum) > 1
    return dict(result, selected_sample_id=winner["sample_id"], reason="selected-by-declared-rule")


def budget_prefix(pool, *, n=None, tokens=None):
    """Retrospective whole-attempt budget; overshoot attempt charged but rejected."""
    _validate_pool(pool)
    if (n is None) == (tokens is None):
        raise ValueError("Exactly one attempt or generation-token limit required")
    value = n if n is not None else tokens
    if type(value) is not int or value < 1:
        raise ValueError("Positive exact budget required")
    if n is not None:
        return pool[:n], pool[:n], []
    accepted, attempted, rejections, used = [], [], [], 0
    for row in pool:
        count = row.get("generated_tokens")
        if type(count) is not int or count < 0:
            raise ValueError("Token budget needs measured output-action counts")
        attempted.append(row)
        if used + count > tokens:
            rejections.append({"sample_id": row["sample_id"], "reason": "whole-attempt-exceeds-remaining-budget",
                               "generated_tokens": count, "remaining": tokens-used})
            break
        accepted.append(row); used += count
        if used == tokens:
            break
    return accepted, attempted, rejections


def cost_totals(rows):
    keys = ("wall_seconds", "generation_tokens", "scoring_tokens", "prefill_seconds", "decode_seconds",
            "scoring_seconds", "generation_forward_positions", "attempted_generation_forward_positions",
            "scoring_forward_positions", "attempted_scoring_forward_positions")
    return {key: {"known_total": sum(r.get("cost", {}).get(key) or 0 for r in rows),
                  "unknown_rows": sum(r.get("cost", {}).get(key) is None for r in rows)} for key in keys}


def pool_diagnostics(pool, grades):
    answers = [eligible_answer(selection_view(r)) for r in pool]
    valid = [a for a in answers if a is not None]
    pairs, wrong_same, correctness_same = 0, 0, 0
    for i in range(len(pool)):
        for j in range(i+1, len(pool)):
            pairs += 1
            wrong_same += bool(answers[i] is not None and answers[i] == answers[j]
                               and not grades[i]["complete_success"] and not grades[j]["complete_success"])
            correctness_same += grades[i]["complete_success"] == grades[j]["complete_success"]
    return {"attempts": len(pool), "eligible": len(valid), "answer_support": len(set(valid)),
        "distinct_raw_responses": len({r["raw_response"] for r in pool}),
        "duplicate_raw_fraction": 1-len({r["raw_response"] for r in pool})/len(pool) if pool else None,
        "pair_count": pairs, "wrong_same_answer_pairs": wrong_same,
        "correctness_agreement_pairs": correctness_same,
        "pairwise_wrong_agreement": wrong_same/pairs if pairs else None,
        "pairwise_correctness_agreement": correctness_same/pairs if pairs else None,
        "errors": sum(r["error"] is not None for r in pool),
        "truncated": sum(r["truncated"] for r in pool),
        "boundary": "descriptive within-pool agreements, not estimated IID or causal correlation"}


def evaluate_pool(item, pool):
    _validate_pool(pool)
    grades = []
    for row in pool:
        grade = grade_response(item, row)
        grade["complete_success"] = bool(grade["task_success"] and grade["natural_termination"])
        grades.append(grade)
    by_id = {r["sample_id"]: r for r in grades}
    decisions = []
    for kind, budgets in (("attempts", N_VALUES), ("generation_tokens", TOKEN_BUDGETS)):
        for budget in budgets:
            eligible, attempted, rejections = budget_prefix(pool, **({"n": budget} if kind == "attempts" else {"tokens": budget}))
            ids = [r["sample_id"] for r in eligible]
            oracle = any(by_id[i]["complete_success"] for i in ids)
            for method in SELECTORS:
                begin = time.perf_counter(); decision = select(eligible, method)
                decision["selection_wall_seconds"] = time.perf_counter()-begin
                selected = by_id.get(decision["selected_sample_id"])
                decision.update(item_id=item["id"], source_group=item["source_group"], slice=item["slice"],
                    budget_kind=kind, budget=budget, eligible_candidate_ids=ids,
                    attempted_candidate_ids=[r["sample_id"] for r in attempted], rejections=rejections,
                    selected_complete_success=bool(selected and selected["complete_success"]),
                    selected_correct=bool(selected and selected["correct"]),
                    oracle_any_complete=oracle, costs=cost_totals(attempted),
                    generated_token_overshoot=max(0, sum(r["generated_tokens"] for r in attempted)-budget) if kind == "generation_tokens" else 0)
                decisions.append(decision)
    return {"item_id": item["id"], "source_group": item["source_group"], "slice": item["slice"],
            "raw_candidates": pool, "graded_candidates": grades,
            "diagnostics": pool_diagnostics(pool, grades), "decisions": decisions}


def summarize(pools):
    grouped = defaultdict(list)
    for pool in pools:
        for row in pool["decisions"]:
            for slice_name in ("all", row["slice"]):
                grouped[(slice_name, row["budget_kind"], row["budget"], row["method"])].append(row)
    result = []
    for (slice_name, kind, budget, method), rows in sorted(grouped.items()):
        result.append({"slice": slice_name, "budget_kind": kind, "budget": budget, "method": method,
            "items": len(rows), "selected_success": sum(r["selected_complete_success"] for r in rows)/len(rows),
            "oracle_any_complete": sum(r["oracle_any_complete"] for r in rows)/len(rows),
            "abstentions": sum(r["selected_sample_id"] is None for r in rows),
            "ties": sum(r["tie"] for r in rows),
            "generation_tokens": sum(r["costs"]["generation_tokens"]["known_total"] for r in rows),
            "scoring_tokens": sum(r["costs"]["scoring_tokens"]["known_total"] for r in rows),
            "wall_seconds": sum(r["costs"]["wall_seconds"]["known_total"]+r["selection_wall_seconds"] for r in rows),
            "overshoot_tokens": sum(r["generated_token_overshoot"] for r in rows),
            "boundary": "aligned original items; source siblings related; descriptive only"})
    return result


def error_dependence(pools):
    """Across-item ordinal error covariance; descriptive, not an IID assertion."""
    out = {}
    for label in ("all", *SLICES):
        selected = [p for p in pools if label == "all" or p["slice"] == label]
        if not selected:
            continue
        counts = {len(p["graded_candidates"]) for p in selected}
        if len(counts) != 1:
            raise ValueError("Aligned ordinal counts required for error dependence")
        errors = torch.tensor([[float(not r["complete_success"]) for r in p["graded_candidates"]]
                               for p in selected], dtype=torch.float64)
        mean = errors.mean(0)
        centered = errors - mean
        covariance = centered.T @ centered / len(selected)
        variance = covariance.diag()
        denominator = (variance[:, None] * variance[None, :]).sqrt()
        correlation = [[float(covariance[i,j]/denominator[i,j]) if denominator[i,j]>0 else None
                        for j in range(errors.shape[1])] for i in range(errors.shape[1])]
        out[label] = {"items": len(selected), "source_groups": len({p["source_group"] for p in selected}),
            "ordinal_error_rates": mean.tolist(), "population_covariance": covariance.tolist(),
            "correlation": correlation,
            "boundary": "descriptive across original items; sibling sources related; null means zero variance"}
    return out


def memory_available_gib():
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    return next(int(line.split()[1])/1024**2 for line in path.read_text().splitlines() if line.startswith("MemAvailable:"))


def run_experiment(items, *, seeds=SEEDS, updates=UPDATES, journal=None):
    audit = validate_items(items)
    settings = {"template_id": "original-symbolic-parity-selection-v1", "thinking_mode": "not-supported",
                "decoding": "coordinate-seeded-EOS0or1-temperature1", "stopping": [EOS], "max_new_tokens": CAP}
    contract = freeze_contract(items, settings)
    result = {"protocol": "actual-candidate-selection-v1", "suite_audit": audit, "contract": contract,
              "seeds": list(seeds), "runs": [], "failures": [], "mem_available_start_gib": memory_available_gib()}
    started = time.perf_counter()
    for seed in seeds:
        run = {"seed": seed, "policies": []}; result["runs"].append(run)
        initial, final, fit = fit_policy(items, seed, updates=updates,
            progress=(lambda row: journal.event("fit_update", seed=seed, evidence=row)) if journal else None)
        run["fit"] = fit
        if journal: journal.event("fit_completed", seed=seed, evidence=fit)
        for label, model in (("initial", initial), ("sft-final", final)):
            checkpoint = f"tiny-selection-{seed}-{label}-sha256:{state_hash(model)}"
            policy = {"label": label, "checkpoint_id": checkpoint, "state_sha256": state_hash(model),
                      "model_config": vars(model.cfg), "pools": []}; run["policies"].append(policy)
            for item in items:
                pool = []
                for ordinal in range(SAMPLES):
                    record = generate_candidate(model, item, contract, seed=seed, policy_id=checkpoint, ordinal=ordinal)
                    pool.append(record)
                    if journal: journal.record(record)
                    if record["error"]: result["failures"].append({"seed": seed, "policy": label, "record": record})
                policy["pools"].append(evaluate_pool(item, pool))
            policy["summary"] = summarize(policy["pools"])
            policy["error_dependence"] = error_dependence(policy["pools"])
            if state_hash(model) != policy["state_sha256"]:
                raise RuntimeError("Frozen candidate policy changed during collection")
    result.update(mem_available_end_gib=memory_available_gib(),
                  peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  wall_seconds=time.perf_counter()-started,
                  memory_boundary="Linux process lifetime peakRSS; two host available-memory observations, not monitored minimum",
                  limits=["original finite symbolic tasks, no English parser or pretrained reasoning",
                    "conditional syntax/support and train-only likelihood ranker, not reward calibration",
                    "pool prefix reuse and whole-attempt token overshoot, not adaptive token interrupt",
                    "CPU full-prefix/rescore wall measurements, not optimized inference benchmark",
                    "all selector arms charged mandatory path rescoring; not an optimized first/vote serving baseline",
                    "related source siblings and tiny fixed authored suite, not independent population estimate"])
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists(): parser.error("Choose a new run directory; prior evidence cannot be overwritten")
    root = Path(__file__).resolve().parents[2]
    sources = ["src/dongxi_llms/inference_selection_lab.py", "src/dongxi_llms/decoder_lab.py",
        "src/dongxi_llms/reasoning_controls.py", "src/dongxi_llms/reasoning_evaluation.py",
        "src/dongxi_llms/reasoning_generation.py", "src/dongxi_llms/evaluation_lab.py",
        "src/dongxi_llms/run_identity.py", "tests/test_inference_selection_lab.py"]
    journal = GenerationJournal(args.output)
    try:
        identity = collect_run_identity(root, source_files=sources, input_files=[args.items, args.spec])
        before = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [*(root/s for s in sources),args.items,args.spec]}
        journal.write_new("input-identity.json", identity)
        torch.set_num_threads(1)
        items = json.loads(args.items.read_text())
        result = run_experiment(items, journal=journal)
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest for p,digest in before.items()):
            raise RuntimeError("Source/input bytes changed; retained candidates do not establish a valid run")
        result.update(source_and_input_sha256=before, run_identity=identity, inputs_unchanged=True)
        journal.write_new("results.json", result)
        journal.event("completed", candidates=sum(len(p["raw_candidates"]) for r in result["runs"] for m in r["policies"] for p in m["pools"]))
        print(f"Saved actual candidates and aligned selectors: {args.output}/results.json")
    except (Exception, KeyboardInterrupt) as error:
        journal.event("failure", error_type=type(error).__name__, error=str(error))
        journal.write_new("failure.json", {"error_type": type(error).__name__, "error": str(error),
            "boundary": "partial fit events and candidate rows remain; no successful final result"})
        raise
    finally:
        journal.close()


if __name__ == "__main__":
    main()
