"""Original matched-rollout microscope, not a named GRPO-stack reproduction.

No fitting or model acquisition occurs on import. Collection uses a tiny shared
decoder and an explicit three-action conditional grammar. Numerical fixtures
are labeled separately from actual neural rollouts.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

from .reasoning_controls import (BOS, EOS, PAD, SUM, NUMBER_START, make_decoder,
                                 state_hash)
from .run_identity import collect_run_identity

SEEDS = (2401, 2402, 2403)
PAIRS = ((0, 0), (0, 1), (1, 0), (1, 1), (0, 2))
GROUP_SIZE, CAP, MAX_ATTEMPTS = 8, 3, 3
SUPPORT = (EOS, NUMBER_START, NUMBER_START + 1)
COMPONENT_WEIGHTS = (1., .4, .2)
EPS = 1e-8
REDUCTIONS = ("response", "token", "fixed")
SCALINGS = ("center", "total_std", "component_std")


def _finite_float(tensor, name):
    if (not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point()
            or not bool(torch.isfinite(tensor).all())):
        raise ValueError(f"{name} must be a finite floating tensor")


def component_advantages(components, weights=COMPONENT_WEIGHTS,
                         scaling="total_std", eps=EPS):
    """[prompts,G,components] -> detached [prompts,G] population advantages."""
    _finite_float(components, "components")
    if components.ndim != 3 or components.shape[0] < 1 or components.shape[1] < 2:
        raise ValueError("Need nonempty [prompts,G>=2,components]")
    if scaling not in SCALINGS or isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Unknown scaling or invalid epsilon")
    weight = torch.as_tensor(weights, dtype=components.dtype, device=components.device)
    _finite_float(weight, "weights")
    if weight.shape != components.shape[-1:]:
        raise ValueError("One weight per component required")
    component = components.detach()
    weight = weight.detach()
    if scaling == "component_std":
        centered = component - component.mean(1, keepdim=True)
        std = component.std(1, correction=0, keepdim=True)
        normalized = torch.where(std == 0, torch.zeros_like(centered),
                                 centered / (std + eps))
        return (normalized * weight).sum(-1)
    total = (component * weight).sum(-1)
    centered = total - total.mean(-1, keepdim=True)
    if scaling == "center":
        return centered
    std = total.std(-1, correction=0, keepdim=True)
    return torch.where(std == 0, torch.zeros_like(centered), centered / (std + eps))


def reduction_weights(mask, reduction, fixed_cap=None, *, dtype=torch.float64):
    """Actual per-action weights; never use padded width as fixed denominator."""
    if (not isinstance(mask, torch.Tensor) or mask.ndim != 2 or mask.dtype != torch.bool
            or mask.shape[0] < 1 or mask.shape[1] < 1):
        raise ValueError("Need nonempty boolean [responses,positions] mask")
    counts = mask.sum(-1)
    if bool((counts == 0).any()) or bool((mask[:, 1:] & ~mask[:, :-1]).any()):
        raise ValueError("Every response needs a nonempty contiguous valid prefix")
    if reduction not in REDUCTIONS:
        raise ValueError("Unknown reduction")
    weights = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    if reduction == "response":
        denominator = counts[:, None].to(dtype) * mask.shape[0]
        return mask.to(dtype) / denominator
    if reduction == "token":
        return mask.to(dtype) / counts.sum()
    if type(fixed_cap) is not int or fixed_cap < 1 or bool((counts > fixed_cap).any()):
        raise ValueError("Fixed cap must be explicit and cover every valid response")
    weights[mask] = 1. / (mask.shape[0] * fixed_cap)
    return weights


def objective(logp, old_logp, advantages, mask, *, reduction="response",
              fixed_cap=None, clip_low=.2, clip_high=.2):
    """Minimization surrogate; no KL/critic/optimizer is hidden in this function.

    Select valid entries BEFORE arithmetic; NaN padded old/current scores cannot
    contaminate an otherwise defined objective. Advantages and old scores detach.
    """
    weights = reduction_weights(mask, reduction, fixed_cap, dtype=logp.dtype)
    if (logp.shape != mask.shape or old_logp.shape != mask.shape
            or advantages.shape != mask.shape[:1]
            or not logp.is_floating_point() or not old_logp.is_floating_point()):
        raise ValueError("Log score and advantage shapes/dtypes must agree")
    for clip in (clip_low, clip_high):
        if isinstance(clip, bool) or not isinstance(clip, (int, float)) or not math.isfinite(clip) or not 0 < clip < 1:
            raise ValueError("Clip widths must be finite and in (0,1)")
    _finite_float(logp[mask], "valid current log probabilities")
    _finite_float(old_logp[mask], "valid old log probabilities")
    _finite_float(advantages, "advantages")
    difference = logp[mask] - old_logp.detach()[mask]
    ratio = difference.exp()
    _finite_float(ratio, "importance ratios")
    if bool((ratio == 0).any()):
        raise ValueError("Importance ratio underflowed; use bounded scores")
    advantage = advantages.detach()[:, None].expand_as(mask)[mask]
    surrogate = torch.minimum(ratio * advantage,
        ratio.clamp(1 - clip_low, 1 + clip_high) * advantage)
    return -(weights[mask] * surrogate).sum()


def analytic_logp_gradient(logp, old_logp, advantages, mask, *,
                           reduction="response", fixed_cap=None,
                           clip_low=.2, clip_high=.2):
    """Independent piecewise derivative away from clipping boundaries."""
    # Validation follows the public value contract, not its autograd derivative.
    objective(logp, old_logp, advantages, mask, reduction=reduction,
              fixed_cap=fixed_cap, clip_low=clip_low, clip_high=clip_high)
    with torch.no_grad():
        weight = reduction_weights(mask, reduction, fixed_cap, dtype=logp.dtype)
        ratio = (logp[mask] - old_logp[mask]).exp()
        advantage = advantages[:, None].expand_as(mask)[mask]
        active = ~(((advantage > 0) & (ratio > 1 + clip_high)) |
                   ((advantage < 0) & (ratio < 1 - clip_low)))
        result = torch.zeros_like(logp)
        result[mask] = -weight[mask] * advantage * ratio * active
        return result


def score_paths(model, prompts, responses, mask):
    """Conditional log probabilities, same support used by collection at all steps."""
    if (prompts.ndim != 2 or responses.shape != mask.shape
            or prompts.shape[0] != responses.shape[0] or responses.dtype != torch.long):
        raise ValueError("Aligned prompt and response ID matrices required")
    reduction_weights(mask, "response")
    allowed = torch.tensor(SUPPORT, device=responses.device)
    match = responses[..., None] == allowed
    if not bool(match.any(-1)[mask].all()):
        raise ValueError("A valid action falls outside the declared conditional support")
    inputs = torch.cat((prompts, responses), -1)[:, :-1]
    start = prompts.shape[1] - 1
    logits = model(inputs)[:, start:start + responses.shape[1], allowed]
    indices = match.long().argmax(-1)
    return logits.log_softmax(-1).gather(-1, indices[..., None]).squeeze(-1)


def response_components(active, expected):
    """Proxy features and independent strict quality from actual generated actions."""
    first_correct = bool(active and active[0] == NUMBER_START + expected)
    format_ok = len(active) == 2 and active[0] in SUPPORT[1:] and active[1] == EOS
    numeral_count = sum(token in SUPPORT[1:] for token in active)
    complete_quality = format_ok and first_correct
    return [float(first_correct), float(format_ok), numeral_count / CAP], float(complete_quality)


@torch.no_grad()
def collect_group(model, pair, *, seed, attempt):
    """One actual neural group; no prefilled answers or inserted EOS."""
    a, b = pair
    prompt = torch.tensor([[BOS, SUM, NUMBER_START + a, NUMBER_START + b]] * GROUP_SIZE)
    current = prompt.clone()
    finished = torch.zeros(GROUP_SIZE, dtype=torch.bool)
    responses = torch.full((GROUP_SIZE, CAP), PAD, dtype=torch.long)
    mask = torch.zeros_like(responses, dtype=torch.bool)
    allowed = torch.tensor(SUPPORT)
    rng = torch.Generator().manual_seed(seed)
    full_prefix_positions, calls = 0, 0
    for position in range(CAP):
        logits = model(current)[:, -1, allowed]
        full_prefix_positions += current.numel()
        calls += 1
        action = torch.multinomial(logits.softmax(-1), 1, generator=rng).squeeze(-1)
        token = allowed[action]
        mask[:, position] = ~finished
        token = torch.where(finished, torch.full_like(token, PAD), token)
        responses[:, position] = token
        finished |= token == EOS
        current = torch.cat((current, token[:, None]), -1)
        if bool(finished.all()):
            break
    old_logp = score_paths(model, prompt, responses, mask)
    rows, components, quality = [], [], []
    for index in range(GROUP_SIZE):
        active = responses[index, mask[index]].tolist()
        values, q = response_components(active, int(a + b > 0))
        components.append(values)
        quality.append(q)
        rows.append({"sample": index, "tokens": responses[index].tolist(),
            "valid_mask": mask[index].tolist(), "active_tokens": active,
            "behavior_logp": old_logp[index, mask[index]].tolist(),
            "components": values, "independent_quality": q,
            "termination": "eos" if active[-1] == EOS else "cap"})
    return {"source_group": f"sum-positive-{a}-{b}", "pair": list(pair),
        "attempt": attempt, "seed": seed, "responses": rows,
        "mixed_quality": 0 < sum(quality) < GROUP_SIZE,
        "valid_tokens": int(mask.sum()), "full_prefix_positions": full_prefix_positions,
        "generation_forward_calls": calls,
        "collection_rescoring_calls": 1,
        "collection_rescoring_positions": int(GROUP_SIZE * (prompt.shape[1] + CAP - 1)),
        "conditional_support": list(SUPPORT),
        "components": components, "quality": quality}


def collect_with_filter(model, seed):
    """Bounded attempts, retry only unresolved prompts; every rejection retained."""
    records, selected = [], {}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        for index, pair in enumerate(PAIRS):
            if index in selected:
                continue
            record = collect_group(model, pair, seed=seed + 1000 * attempt + index,
                                   attempt=attempt)
            records.append(record)
            if record["mixed_quality"]:
                selected[index] = len(records) - 1
    chosen = set(selected.values())
    for index, record in enumerate(records):
        record["selected"] = index in chosen
        record["rejection_reason"] = None if index in chosen else (
            "all-wrong" if sum(record["quality"]) == 0 else "all-correct")
    return records, {"unique_prompts_attempted": len(PAIRS),
        "group_attempts": len(records), "responses_attempted": len(records) * GROUP_SIZE,
        "valid_tokens_attempted": sum(r["valid_tokens"] for r in records),
        "full_prefix_positions_attempted": sum(r["full_prefix_positions"] for r in records),
        "position_count_scope": "generation full-prefix forwards; collection and analysis rescoring are additional work",
        "groups_selected": len(chosen), "responses_selected": len(chosen) * GROUP_SIZE,
        "valid_tokens_selected": sum(records[i]["valid_tokens"] for i in chosen),
        "unresolved_source_groups": [f"sum-positive-{a}-{b}" for i, (a, b) in enumerate(PAIRS)
                                      if i not in selected],
        "boundary": "retry cost includes rejected groups; no equal-budget algorithm claim"}


def _flat_initial(records):
    initial = [record for record in records if record["attempt"] == 1]
    if len(initial) != len(PAIRS):
        raise ValueError("First-attempt fixed comparison pool must contain every prompt")
    prompts, responses, masks = [], [], []
    for record in initial:
        a, b = record["pair"]
        for row in record["responses"]:
            prompts.append([BOS, SUM, NUMBER_START + a, NUMBER_START + b])
            responses.append(row["tokens"])
            masks.append(row["valid_mask"])
    component = torch.tensor([r["components"] for r in initial], dtype=torch.float64)
    return torch.tensor(prompts), torch.tensor(responses), torch.tensor(masks), component


def _tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def clipping_fixture():
    """Constructed score fixture, explicitly not generated by a decoder."""
    ratios = torch.tensor([.6, .9, 1.1, 1.3, 1.6], dtype=torch.float64)
    old = torch.full((2, 5), -2., dtype=torch.float64)
    logp = (old + ratios.log()).requires_grad_()
    mask = torch.ones_like(old, dtype=torch.bool)
    advantages = torch.tensor([1., -1.], dtype=torch.float64)
    result = {"kind": "constructed-log-probability-fixture", "ratios": ratios.tolist()}
    for label, high in (("symmetric", .2), ("asymmetric", .4)):
        loss = objective(logp, old, advantages, mask, clip_high=high)
        gradient, = torch.autograd.grad(loss, logp)
        analytic = analytic_logp_gradient(logp, old, advantages, mask, clip_high=high)
        if not torch.allclose(gradient, analytic, atol=1e-12, rtol=1e-12):
            raise AssertionError("Independent clipping derivative disagrees")
        result[label] = {"loss": float(loss.detach()), "gradient": gradient.tolist(),
                         "analytic_gradient": analytic.tolist()}
    return result


def run_seed(seed):
    start = time.perf_counter()
    model = make_decoder(seed).double().eval()
    initial_hash = state_hash(model)
    behavior = deepcopy(model).requires_grad_(False)
    records, costs = collect_with_filter(behavior, seed)
    prompts, responses, mask, components = _flat_initial(records)
    old = score_paths(behavior, prompts, responses, mask).detach()
    variants, parameter_gradients = [], []
    for scaling in SCALINGS:
        advantages = component_advantages(components, scaling=scaling).flatten()
        for reduction in REDUCTIONS:
            current = score_paths(model, prompts, responses, mask)
            loss = objective(current, old, advantages, mask, reduction=reduction,
                             fixed_cap=CAP)
            parameters = tuple(model.parameters())
            score_gradient, *grads = torch.autograd.grad(loss, (current, *parameters))
            analytic = analytic_logp_gradient(current, old, advantages, mask,
                                             reduction=reduction, fixed_cap=CAP)
            if not torch.allclose(score_gradient, analytic, atol=1e-12, rtol=1e-12):
                raise AssertionError("Actual sampled-score derivative disagrees")
            flat = torch.cat([g.flatten() for g in grads])
            parameter_gradients.append(flat)
            weights = reduction_weights(mask, reduction, CAP)
            variants.append({"name": f"{scaling}/{reduction}", "scaling": scaling,
                "reduction": reduction, "loss": float(loss.detach()),
                "advantages": advantages.tolist(), "token_weights": weights.tolist(),
                "selected_logp_gradient": score_gradient.tolist(),
                "analytic_max_error": float((analytic - score_gradient).abs().max()),
                "parameter_gradient_norm": float(flat.norm()),
                "parameter_gradient_sha256": _tensor_hash(flat),
                "weight_sum": float(weights.sum()), "ratio_min": 1., "ratio_max": 1.,
                "clip_active_fraction": 0., "valid_response_tokens": int(mask.sum())})
    gradients = torch.stack(parameter_gradients)
    norms = gradients.norm(dim=-1)
    cosine = gradients @ gradients.T / (norms[:, None] * norms[None, :]).clamp_min(1e-30)
    original = component_advantages(components, scaling="component_std")
    rescaled = component_advantages(components, weights=tuple(10 * w for w in COMPONENT_WEIGHTS),
                                   scaling="component_std")
    ratio_one = score_paths(model, prompts, responses, mask)
    symmetric = objective(ratio_one, old, original.flatten(), mask, clip_high=.2)
    asymmetric = objective(ratio_one, old, original.flatten(), mask, clip_high=.4)
    if not torch.equal(symmetric, asymmetric):
        raise AssertionError("Clipping variants must agree at initial ratio one")
    if state_hash(model) != initial_hash or state_hash(behavior) != initial_hash:
        raise AssertionError("Matched-gradient analysis changed model weights")
    return {"seed": seed, "initial_state_sha256": initial_hash,
        "final_state_sha256": state_hash(model), "seconds": time.perf_counter() - start,
        "model_config": vars(model.cfg), "dtype": "float64", "device": "cpu",
        "all_rollout_groups": records, "collection_costs": costs,
        "first_attempt_pool": {"groups": len(PAIRS), "responses": len(prompts),
            "valid_tokens": int(mask.sum()), "response_lengths": mask.sum(-1).tolist(),
            "behavior_logp": old.tolist(), "components": components.tolist()},
        "variants": variants, "parameter_gradient_cosines": cosine.tolist(),
        "component_weight_rescaling_max_error": float((rescaled - 10 * original).abs().max()),
        "ratio_one_clipping_equal": True,
        "interpretation": "measured gradients on fixed samples; no optimizer steps or capability gain"}


def run_controls():
    return {"protocol": "grpo-objective-controls-v1", "seeds": list(SEEDS),
        "support": list(SUPPORT), "cap": CAP, "group_size": GROUP_SIZE,
        "max_group_attempts_per_prompt": MAX_ATTEMPTS,
        "component_weights": list(COMPONENT_WEIGHTS), "runs": [run_seed(s) for s in SEEDS],
        "clipping": clipping_fixture(),
        "limits": ["symbolic conditional grammar, not unrestricted text reasoning",
                   "gradients compared on initial untrained parameters; no model fit",
                   "component reward includes deliberate verbosity shortcut",
                   "bounded retry collection is not equal-cost training",
                   "no complete named algorithm, pretrained/GPU/Mac result"]}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Keep prior evidence; choose a new output path")
    root = Path(__file__).resolve().parents[2]
    source_paths = [Path(__file__).relative_to(root),
        Path("src/dongxi_llms/grpo_lab.py"), Path("src/dongxi_llms/reasoning_controls.py"),
        Path("src/dongxi_llms/decoder_lab.py"), Path("tests/test_grpo_objective_controls.py"),
        Path("experiments/specs/2026-10-04-grpo-objective-controls.md")]
    hashes = {str(p): hashlib.sha256((root / p).read_bytes()).hexdigest() for p in source_paths}
    torch.set_num_threads(1)
    identity = collect_run_identity(root, source_files=source_paths,
                                    input_files=[source_paths[-1]])
    result = run_controls()
    final_hashes = {str(p): hashlib.sha256((root / p).read_bytes()).hexdigest() for p in source_paths}
    if hashes != final_hashes:
        raise RuntimeError("Measured source changed during the run")
    result.update({"source_hashes": hashes, "run_identity": identity,
                   "source_unchanged": True, "actual_command": list(getattr(sys, "orig_argv", []))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {len(result['runs'])} seeds; all actual groups retained: {args.output}")


if __name__ == "__main__":
    main()
