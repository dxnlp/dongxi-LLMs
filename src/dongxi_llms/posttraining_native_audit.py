"""Optional frozen-parent native capability/reward audit; no optimizer or updates.

The CLI requires CUDA, one offline pinned snapshot and a frozen protocol hash.
CPU tests use a new randomly initialized HF decoder for the gradient helper.
The full-vocabulary sampler/stop contract and strict verifier are reused.
"""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

from . import qwen_rlvr_lab as native
from .grpo_lab import group_advantages
from .posttraining_assignment import FIXTURE, ROOT, _instrument, file_hash, identity
from .run_identity import artifact_hashes, tokenizer_interface


def native_oracle(item):
    """Independent bounded integer calculation, no expression parser or eval."""
    if item["split"] not in ("train", "development"):
        raise ValueError("Native audit cannot access heldout items")
    a, b, c = item["a"], item["b"], item["c"]
    if any(type(v) is not int or abs(v) > 100 for v in (a, b)):
        raise ValueError("Bounded exact integer operands required")
    if item["operation"] == "add" and c is None:
        return a+b
    if item["operation"] == "multiply_sum" and type(c) is int and abs(c) <= 10:
        return (a+b)*c
    raise ValueError("Unknown independent task grammar")


def load_native_spec(directory=FIXTURE):
    directory = Path(directory)
    items = json.loads((directory/"posttraining_native_items.json").read_text())
    protocol = json.loads((directory/"posttraining_native_protocol.json").read_text())
    if identity(items) != protocol["items_sha256"] or len(items) != 12:
        raise ValueError("Native fixture identity/count mismatch")
    if len({i["id"] for i in items}) != len(items) or len({i["source_group"] for i in items}) != len(items):
        raise ValueError("Native source IDs must be distinct")
    for item in items:
        if type(item["expected"]) is not int or item["expected"] != native_oracle(item):
            raise ValueError("Native authored reference disagrees with independent oracle")
    if Counter((i["split"], i["tier"]) for i in items) != Counter({(s, t): 2 for s in ("train", "development") for t in ("easy", "signed", "composed")}):
        raise ValueError("Two independent sources per split/tier required")
    if (protocol["group_size"] != 4 or protocol["max_new_tokens"] != 32 or
            protocol["runtime_cap_seconds"] > 900 or protocol["max_allocated_gib"] > 8 or
            protocol["min_host_available_gib"] < 25):
        raise ValueError("Native audit caps differ from authorization")
    return items, protocol


def head_task_gradient(model, pool, *, before_stage=lambda: None):
    """Actual sampled task-only score gradient in the output projection space.

    Transformer hidden states are detached. A private output-weight clone gets
    the gradient; neither the model nor its original output weight is changed.
    Returns a dense FP32 vector so the caller can measure group cancellation.
    """
    device = next(model.parameters()).device
    prompts = pool["prompt_ids"].to(device).repeat(len(pool["responses"]), 1)
    responses = pool["responses"].to(device)
    mask = pool["mask"].to(device)
    inputs = torch.cat((prompts, responses), -1)[:, :-1]
    start = prompts.shape[1]-1
    layer = model.get_output_embeddings()
    before_stage()
    with torch.no_grad():
        hidden = model.model(input_ids=inputs, use_cache=False).last_hidden_state[:, start:]
    weight = layer.weight.detach().clone().requires_grad_(True)
    bias = layer.bias.detach() if getattr(layer, "bias", None) is not None else None
    before_stage()
    logits = F.linear(hidden.detach(), weight, bias).float()
    logp = logits.log_softmax(-1).gather(-1, responses[..., None]).squeeze(-1)
    alignment = float(((logp.detach()-pool["old_logp"].to(device)).abs()*mask).max())
    if alignment > 1e-4:
        raise ValueError(f"Head/full-policy fresh likelihood mismatch: {alignment}")
    rewards = torch.tensor(pool["rewards"], device=device, dtype=torch.float32)
    advantages = group_advantages(rewards[None]).flatten()
    loss = -((logp*advantages.detach()[:, None]*mask).sum(-1)/mask.sum(-1)).mean()
    before_stage()
    gradient, = torch.autograd.grad(loss, weight)
    before_stage()
    vector = gradient.detach().float()
    return {"task_gradient_norm": float(vector.norm()), "task_loss": float(loss.detach()),
            "advantage_norm": float(advantages.norm()), "initial_log_ratio_max_abs": alignment,
            "head_shape": list(weight.shape), "teacher_forced_input_tokens": inputs.numel(),
            "valid_scored_response_tokens": int(mask.sum()),
            "scope": "Detached-hidden output-weight task gradient only; no KL and no optimizer step."}, vector


def audit_gate(rows, gradients, aggregate_norms):
    gates = {}
    for split in ("train", "development"):
        selected = [r for r in rows if r["split"] == split]
        groups = [g for g in gradients if g["split"] == split]
        successes = sum(r["strict_terminated_integer_reward"] for r in selected)
        failures = sum(not r["strict_terminated_integer_reward"] for r in selected)
        mixed = sum(0 < g["strict_successes"] < g["requested_responses"] for g in groups)
        nonzero = any(g["task_gradient_norm"] > 1e-8 for g in groups) and aggregate_norms.get(split, 0.) > 1e-8
        gates[split] = {"successes": successes, "failures": failures, "mixed_groups": mixed,
                        "nonzero_task_head_gradient": nonzero,
                        "passes": bool(successes and failures and mixed and nonzero)}
    return {"by_split": gates, "passes": all(g["passes"] for g in gates.values()),
            "boundary": "Audit mechanism gate, not authorization or evidence for a same-parent SFT/RL training campaign."}


def run_native_audit(snapshot, output, expected_protocol_sha256):
    items, protocol = load_native_spec()
    if identity(protocol) != expected_protocol_sha256:
        raise ValueError("Protocol hash differs from the independently frozen launch spec")
    snapshot, output = Path(snapshot), Path(output)
    if snapshot.name != protocol["revision"] or not snapshot.is_dir():
        raise ValueError("Only the pinned local snapshot is permitted")
    if not torch.cuda.is_available():
        raise RuntimeError("Optional native audit requires the actual CUDA host")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model = None
    rows, gradients = [], []
    report = {"status": "running", "protocol": protocol, "protocol_sha256": identity(protocol),
        "requested_responses": len(items)*protocol["group_size"], "attempts": [], "rows": rows,
        "gradients": gradients, "execution": {"python": sys.executable, "torch": torch.__version__},
        "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in
            (Path(__file__), ROOT/"src/dongxi_llms/posttraining_assignment.py", ROOT/"src/dongxi_llms/qwen_rlvr_lab.py",
             ROOT/"src/dongxi_llms/grpo_lab.py", ROOT/"src/dongxi_llms/reasoning_evaluation.py",
             FIXTURE/"posttraining_native_items.json", FIXTURE/"posttraining_native_protocol.json")}}
    (output/"frozen-protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    (output/"frozen-items.json").write_text(json.dumps(items, indent=2)+"\n")

    def flush():
        report["runtime_seconds"] = time.monotonic()-started
        report["completed_responses"] = len(rows)
        report["uncollected_responses"] = report["requested_responses"]-len(rows)
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
        temporary = output/"report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(output/"report.json")

    def guard():
        if time.monotonic()-started > protocol["runtime_cap_seconds"]:
            raise TimeoutError("Frozen native runtime cap reached")
        native.guard_memory(protocol["min_host_available_gib"])
        if torch.cuda.memory_allocated()/2**30 > protocol["max_allocated_gib"] or torch.cuda.max_memory_allocated()/2**30 > protocol["max_allocated_gib"]:
            raise MemoryError("Frozen8GiB allocated-memory cap reached")

    torch.cuda.reset_peak_memory_stats()
    try:
        guard()
        report["snapshot"] = {"path": str(snapshot), "files": artifact_hashes(snapshot, guard),
                               "license_sha256": file_hash(snapshot/"LICENSE")}
        report["initial_host_available_gib"] = native.available_gib()
        flush()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation=protocol["attention"]).to("cuda").eval().requires_grad_(False)
        guard()
        encode, stops, prompt_contract = native.prompt_contract(tokenizer, "chat")
        report["interface"] = dict(prompt_contract, tokenizer=tokenizer_interface(tokenizer,
            template=tokenizer.chat_template, stop_ids=stops, tokenizer_id=protocol["model_id"],
            tokenizer_revision=protocol["revision"]))
        report["architecture"] = model.config.to_dict()
        parent_hash = native.state_digest(model.state_dict())
        report["parent_state_sha256_before"] = parent_hash
        generator = torch.Generator(device="cuda").manual_seed(protocol["seed"])
        aggregate = {split: torch.zeros_like(model.get_output_embeddings().weight, dtype=torch.float32)
                     for split in ("train", "development")}
        group_counts = Counter()
        for cursor, item in enumerate(items):
            guard()
            errors, decode_attempts = [], []
            def safe_decode(ids):
                sample_index = len(decode_attempts)
                decoded = {"sample_index": sample_index, "ids": list(ids), "error": None}
                decode_attempts.append(decoded)
                try:
                    return tokenizer.decode(ids, skip_special_tokens=True)
                except Exception as error:
                    decoded["error"] = f"{type(error).__name__}: {error}"
                    errors.append(decoded)
                    return ""  # strict reward0; richer diagnostic explicitly ERROR below
            attempt = {"item_id": item["id"], "split": item["split"], "tier": item["tier"],
                       "requested_responses": protocol["group_size"], "decode_failures": errors,
                       "decode_attempts": decode_attempts}
            report["attempts"].append(attempt)
            flush()
            ids = encode(item["prompt"]).to("cuda")
            pool = native.collect_rollouts(model, model, ids, native_oracle(item), safe_decode,
                tokenizer.eos_token_id, generator, group_size=protocol["group_size"],
                max_new_tokens=protocol["max_new_tokens"], before_stage=guard,
                context_length=model.config.max_position_embeddings, stop_ids=stops,
                source_id=item["source_group"], version=0, cursor=cursor, attempt=attempt)
            for sample, (text, stop, response, valid) in enumerate(zip(pool["texts"], pool["stops"], pool["responses"], pool["mask"])):
                active = response[valid].tolist()
                grade = _instrument(text, stop, item["expected"])
                if decode_attempts[sample]["error"] is not None:
                    grade.update(richer_status="ERROR", richer_reason="decode-failure", reason_code="decode-failure",
                                 richer_format_valid=False, richer_answer_correct=False, richer_complete_reward=False)
                if float(grade["strict_terminated_integer_reward"]) != pool["rewards"][sample]:
                    raise AssertionError("Strict native collector/replay reward mismatch")
                rows.append(dict(grade, item_id=item["id"], source_group=item["source_group"],
                    split=item["split"], tier=item["tier"], sample_index=sample, prompt=item["prompt"],
                    prompt_ids=ids[0].tolist(), expected=item["expected"], token_ids=active,
                    full_padded_response_ids=response.tolist(), response_mask=valid.tolist(),
                    raw_response=text, original_stop=stop, truncated=stop != "eos",
                    old_full_support_logp=pool["old_logp"][sample].tolist()))
            flush()
            gradient, vector = head_task_gradient(model, pool, before_stage=guard)
            aggregate[item["split"]].add_(vector)
            group_counts[item["split"]] += 1
            gradient.update(item_id=item["id"], split=item["split"], tier=item["tier"],
                strict_successes=sum(pool["rewards"]), requested_responses=protocol["group_size"],
                collection_work=pool["work"], parent_state_sha256=pool["policy_sha256"])
            gradients.append(gradient)
            del vector, pool
            flush()
        report["aggregate_task_head_gradient_norm_by_split"] = {
            split: float((vector/group_counts[split]).norm()) for split, vector in aggregate.items()}
        after_hash = native.state_digest(model.state_dict())
        report["parent_state_sha256_after"] = after_hash
        if parent_hash != after_hash:
            raise AssertionError("Frozen native parent changed")
        report["signal_gate"] = audit_gate(rows, gradients, report["aggregate_task_head_gradient_norm_by_split"])
        report["summary_by_split_tier"] = {f"{split}/{tier}": {
            "responses": len(selected), "strict_successes": sum(r["strict_terminated_integer_reward"] for r in selected),
            "richer_correct": sum(r["richer_answer_correct"] for r in selected),
            "richer_complete": sum(r["richer_complete_reward"] for r in selected),
            "natural_stops": sum(r["richer_natural_termination"] for r in selected),
            "truncated": sum(r["truncated"] for r in selected),
            "statuses": dict(Counter(r["richer_status"] for r in selected)),
            "disagreements": sum(r["disagreement"] for r in selected)}
            for split in protocol["splits"] for tier in protocol["tiers"]
            for selected in [[r for r in rows if r["split"] == split and r["tier"] == tier]]}
        report["status"] = "completed"
        report["future_sft_rl"] = "unexecuted; absence of signal is a valid diagnostic and does not extend this cap"
        guard()
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        flush()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    report = run_native_audit(args.snapshot, args.output, args.protocol_sha256)
    print(json.dumps({"status": report["status"], "signal_gate": report["signal_gate"],
                      "runtime_seconds": report["runtime_seconds"], "peak_allocated_gib": report["peak_allocated_gib"]}, indent=2))


if __name__ == "__main__":
    main()
