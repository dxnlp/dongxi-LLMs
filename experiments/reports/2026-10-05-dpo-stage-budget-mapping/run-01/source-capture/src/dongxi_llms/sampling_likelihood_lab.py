"""Original exact categorical microscope for DXI-13, not an RL engine.

Filtering order: temperature -> top-k -> top-p on the renormalized top-k
distribution. Ties retain smaller action IDs first. All support selection is
detached. A fixed-support target is a declared conditional objective, NOT a
universal repair for full-support/off-policy policy optimization.
"""
from dataclasses import dataclass
import math

import torch


class MissingSupportError(ValueError):
    """Behavior cannot sample an action required by the declared target."""


def _logits(value):
    if (not isinstance(value, torch.Tensor) or value.ndim < 1 or
            value.shape[-1] < 1 or not value.is_floating_point() or
            not bool(torch.isfinite(value).all())):
        raise ValueError("Need finite floating logits [..., vocabulary>=1]")
    return value


def _temperature(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Temperature must be finite and positive")


def target_log_probabilities(logits, temperature=1., support=None):
    """Declared conditional target; support is fixed during differentiation."""
    _logits(logits)
    _temperature(temperature)
    if support is None:
        support = torch.ones_like(logits, dtype=torch.bool)
    if support.shape != logits.shape or support.dtype != torch.bool:
        raise ValueError("Support must be a matching boolean tensor")
    support = support.detach().to(logits.device)
    if not bool(support.any(-1).all()):
        raise ValueError("Every categorical row needs nonempty support")
    scaled = logits / temperature
    if not bool(torch.isfinite(scaled[support]).all()):
        raise ValueError("Temperature scaling overflowed retained logits")
    logp = scaled.masked_fill(~support, -torch.inf).log_softmax(-1)
    if not bool(torch.isfinite(logp[support]).all()):
        raise ValueError("Retained log probabilities overflowed; use bounded logits")
    return logp


def behavior_distribution(logits, temperature=1., top_k=None, top_p=1.):
    """Return detached actual collection probabilities and retained support.

    For top-p retain the first token crossing the threshold. top-p is applied
    after top-k has been renormalized. Neither filtering rule is differentiable.
    """
    _logits(logits)
    _temperature(temperature)
    width = logits.shape[-1]
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= width):
        raise ValueError("top_k must be None or an integer within vocabulary")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0,1]")
    with torch.no_grad():
        scaled = logits.detach() / temperature
        if not bool(torch.isfinite(scaled).all()):
            raise ValueError("Temperature scaling overflowed logits")
        order = scaled.argsort(dim=-1, descending=True, stable=True)
        rank_support = torch.arange(width, device=logits.device) < (top_k or width)
        ranked = scaled.gather(-1, order).masked_fill(~rank_support, -torch.inf)
        probability = ranked.softmax(-1)
        before = probability.cumsum(-1) - probability
        keep_ranked = rank_support.expand_as(ranked).clone()
        if top_p < 1:
            keep_ranked &= before < top_p
        support = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, order, keep_ranked)
        logp = target_log_probabilities(logits.detach(), temperature, support)
    probabilities = logp.exp()
    if bool((probabilities[support] == 0).any()):
        raise ValueError("Retained behavior probability underflowed; use a wider dtype or bounded logits")
    return probabilities, logp, support


@dataclass(frozen=True)
class BehaviorRecord:
    """Detached collection snapshot; mutable tensors must not be overwritten."""
    logits: torch.Tensor
    probabilities: torch.Tensor
    log_probabilities: torch.Tensor
    support: torch.Tensor
    actions: torch.Tensor
    selected_log_probabilities: torch.Tensor
    temperature: float
    top_k: int | None
    top_p: float

    def portable(self):
        return {"sampling_order": "temperature -> top-k -> renormalize -> top-p -> renormalize",
                "temperature": self.temperature, "top_k": self.top_k, "top_p": self.top_p,
                "source_logits": self.logits.tolist(), "probabilities": self.probabilities.tolist(),
                "support": self.support.tolist(), "actions": self.actions.tolist(),
                "selected_behavior_log_probabilities": self.selected_log_probabilities.tolist()}


def collect_categorical(logits, samples=8, seed=2020, temperature=1., top_k=None, top_p=1.):
    """One finite CPU state; records actual transformed probabilities, not raw p."""
    _logits(logits)
    if logits.ndim != 1 or logits.device.type != "cpu" or not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise ValueError("This bounded collector needs one CPU state and positive sample count")
    p, logp, support = behavior_distribution(logits, temperature, top_k, top_p)
    actions = torch.multinomial(p, samples, replacement=True,
                               generator=torch.Generator().manual_seed(seed))
    return BehaviorRecord(logits.detach().clone(), p.clone(), logp.clone(), support.clone(),
                          actions, logp[actions].clone(), float(temperature), top_k, float(top_p))


def _probabilities(p):
    if not isinstance(p, torch.Tensor) or p.ndim < 1 or not p.is_floating_point() or not bool(torch.isfinite(p).all()) or bool((p < 0).any()):
        raise ValueError("Need finite nonnegative probabilities")
    if not torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1))):
        raise ValueError("Each categorical row must sum to one")


def importance_ratios(target, behavior):
    """Check absolute continuity; zero/zero receives zero, not fictitious credit.

    Target remains differentiable. Behavior ALWAYS detaches. No ratios can
    estimate contributions from target-positive/behavior-zero actions.
    """
    _probabilities(target)
    _probabilities(behavior)
    if target.shape != behavior.shape:
        raise ValueError("Target and behavior shapes must agree")
    behavior = behavior.detach()
    missing = (target.detach() > 0) & (behavior == 0)
    if bool(missing.any()):
        raise MissingSupportError(f"Target-positive actions absent from behavior: {missing.nonzero().tolist()}")
    covered = behavior > 0
    return torch.where(covered, target / behavior.masked_fill(~covered, 1.), torch.zeros_like(target))


def importance_expectation(target, behavior, rewards):
    """Exact expectation over a frozen behavior distribution, no sampled noise."""
    if rewards.shape != target.shape or not bool(torch.isfinite(rewards).all()):
        raise ValueError("Finite rewards must match categorical outcomes")
    return (behavior.detach() * importance_ratios(target, behavior) * rewards.detach()).sum(-1)


def exact_forward_kl(logits, reference_logits, temperature=1., support=None):
    """KL(declared target || raw reference), summed over V; ref ALWAYS detached.

    Excluded actions are selected out BEFORE arithmetic: a reference log
    probability may overflow there without changing this conditional objective.
    Retained probabilities/log probabilities must remain representable instead
    of silently converting numerical underflow into mathematical zero support.
    """
    _logits(reference_logits)
    if logits.shape != reference_logits.shape:
        raise ValueError("Reference and target vocabulary shapes must match")
    logp = target_log_probabilities(logits, temperature, support)
    logq = reference_logits.detach().log_softmax(-1)
    valid = torch.isfinite(logp)
    if not bool(torch.isfinite(logq[valid]).all()):
        raise ValueError("Retained reference log probabilities overflowed; use bounded logits")
    p = logp.exp()
    if bool((p[valid] == 0).any()):
        raise ValueError("Retained target probability underflowed; use a wider dtype or bounded logits")
    difference = logp.masked_fill(~valid, 0.) - logq.masked_fill(~valid, 0.)
    value = (p * difference).sum(-1)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Exact KL value is not representable; use a wider dtype or bounded logits")
    return value


def response_surrogate(logits, actions, old_logp, advantages, response_mask,
                       temperature=1., support=None, reference_logits=None, beta=0.):
    """Unclipped minimization surrogate over valid response actions, not full PPO.

    logits [B,T,V], all remaining action tensors [B,T]. Selected old likelihoods
    alone cannot establish global support coverage: validate that contract
    separately. EOS inclusion depends on the caller's tested response mask.
    KL, if requested, is exact target||raw-reference on these prefix states.
    """
    shape = logits.shape[:-1]
    if logits.ndim != 3 or any(x.shape != shape for x in (actions, old_logp, advantages, response_mask)):
        raise ValueError("Need [B,T,V] logits and matching [B,T] action fields")
    if actions.dtype != torch.long or response_mask.dtype != torch.bool or not bool(response_mask.any()):
        raise ValueError("Need long IDs and a nonempty boolean response mask")
    if bool(((actions < 0) | (actions >= logits.shape[-1])).any()):
        raise ValueError("Actions must be valid vocabulary indices")
    if not bool(torch.isfinite(old_logp[response_mask]).all()) or not bool(torch.isfinite(advantages[response_mask]).all()):
        raise ValueError("Valid old probabilities and advantages must be finite")
    if not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta < 0 or (beta and reference_logits is None):
        raise ValueError("Nonnegative beta requires a reference when positive")
    selected = target_log_probabilities(logits, temperature, support).gather(-1, actions[..., None]).squeeze(-1)
    # Select BEFORE arithmetic: padded -inf/NaN logp never contaminates 0*NaN.
    ratio = (selected[response_mask] - old_logp.detach()[response_mask]).exp()
    loss = -(ratio * advantages.detach()[response_mask]).mean()
    if beta:
        loss = loss + beta * exact_forward_kl(logits, reference_logits, temperature, support)[response_mask].mean()
    return loss


def termination_masks(token_ids, prompt_lengths, valid_lengths, generation_caps, stop_ids):
    """Mask full collated tokens [B,T], including first generated stop action.

    valid_lengths exclude padding; EOS may equal the padding ID. If no stop
    occurs, response length MUST equal its declared generation cap. Otherwise
    the reason is unknown, not silently a cap truncation. These are token masks;
    shift/slice once to align a model's next-token prediction positions.
    """
    if token_ids.ndim != 2 or token_ids.dtype != torch.long or not stop_ids:
        raise ValueError("Need long [B,T] token IDs and explicit stop IDs")
    if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in stop_ids):
        raise ValueError("Stop IDs must be nonnegative integers")
    batch, width = token_ids.shape
    if any(x.shape != (batch,) or x.dtype != torch.long for x in (prompt_lengths, valid_lengths, generation_caps)):
        raise ValueError("Lengths/caps must be long [B] tensors")
    if bool(((prompt_lengths < 0) | (valid_lengths > width) |
             (valid_lengths <= prompt_lengths) | (generation_caps < 1) |
             (valid_lengths - prompt_lengths > generation_caps)).any()):
        raise ValueError("Require a nonempty response within its declared cap")
    mask = torch.zeros_like(token_ids, dtype=torch.bool)
    terminal = torch.zeros(batch, dtype=torch.bool, device=token_ids.device)
    for row in range(batch):
        start, end = int(prompt_lengths[row]), int(valid_lengths[row])
        stops = torch.isin(token_ids[row, start:end], token_ids.new_tensor(stop_ids)).nonzero()
        if len(stops):
            end = start + int(stops[0, 0]) + 1
            terminal[row] = True
        elif end - start != int(generation_caps[row]):
            raise ValueError("Unclassified early end: neither stop nor declared token cap")
        mask[row, start:end] = True
    return {"response_mask": mask, "terminal": terminal, "truncated": ~terminal,
            "bootstrap_allowed": ~terminal}


def kl_gradient_audit(logits, reference_logits):
    """At a fresh full-support state, KL value does not specify its gradient.

    k1=log(p/q), k3=q/p-1+log(p/q). Frozen-sample expectations at b=p0
    have equal values but generally wrong gradients for the exact KL objective.
    Differentiating p/b as well restores the state-wise categorical KL gradient.
    This full-support microscope rejects unrepresentable probabilities, k3
    ratios or gradients. Finite input logits alone do not guarantee these fit
    the chosen floating dtype. p/b is evaluated from saved log probabilities
    to avoid an unnecessary reciprocal of a tiny positive behavior probability.
    """
    _logits(logits)
    _logits(reference_logits)
    if logits.ndim != 1 or logits.shape != reference_logits.shape:
        raise ValueError("One matching finite categorical state required")
    theta = logits.detach().clone().requires_grad_(True)
    logp, logq = theta.log_softmax(-1), reference_logits.detach().log_softmax(-1)
    if not bool(torch.isfinite(logp).all() and torch.isfinite(logq).all()):
        raise ValueError("Full-support audit log probabilities overflowed; use bounded logits")
    p, q = logp.exp(), logq.exp()
    if bool(((p == 0) | (q == 0)).any()):
        raise ValueError("Full-support audit probability underflowed; use a wider dtype or bounded logits")
    b = p.detach()
    k1 = logp - logq
    k3 = torch.expm1(-k1) + k1
    if not bool(torch.isfinite(k3).all()):
        raise ValueError("Full-support audit k3 ratio overflowed; use a wider dtype or bounded logits")
    ratio = (logp - logp.detach()).exp()
    terms = {"exact_kl": (p*k1).sum(), "frozen_k1": (b*k1).sum(),
             "frozen_k3": (b*k3).sum(), "weighted_k1": (b*ratio*k1).sum(),
             "weighted_k3": (b*ratio*k3).sum()}
    results = {}
    for name, value in terms.items():
        if not bool(torch.isfinite(value)):
            raise ValueError(f"Audit {name} value is not representable; use bounded logits")
        gradient, = torch.autograd.grad(value, theta, retain_graph=True)
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError(f"Audit {name} gradient is not representable; use bounded logits")
        results[name] = {"value": float(value.detach()), "gradient": gradient.tolist()}
    return results


def sampling_support_experiment():
    """Specified finite fixture; actual exact enumeration, not LLM results."""
    raw = torch.tensor([.55, .30, .15], dtype=torch.float64)
    rewards = torch.tensor([0., 1., 4.], dtype=torch.float64)
    record = collect_categorical(raw.log(), samples=16, seed=2020,
                                 temperature=.5, top_k=2, top_p=.9)
    theta = raw.log().detach().clone().requires_grad_(True)
    target = target_log_probabilities(theta, 1., record.support).exp()
    direct = (target*rewards).sum()
    repaired = importance_expectation(target, record.probabilities, rewards)
    wrong = (record.probabilities * target / raw * rewards).sum()
    results = {}
    for name, value in (("direct_conditional_target", direct), ("correct_denominator", repaired), ("wrong_raw_denominator", wrong)):
        gradient, = torch.autograd.grad(value, theta, retain_graph=True)
        results[name] = {"value": float(value.detach()), "gradient": gradient.tolist()}
    try:
        importance_ratios(raw, record.probabilities)
    except MissingSupportError as error:
        rejection = str(error)
    else:
        raise AssertionError("Missing-support target must be rejected")
    matched = target_log_probabilities(theta, .5, record.support).exp()
    return {"mode": "CPU exact finite categorical enumeration", "behavior": record.portable(),
            "raw_model_probabilities": raw.tolist(), "rewards": rewards.tolist(),
            "conditional_target_temperature1": target.detach().tolist(),
            "matched_ratios_on_support": importance_ratios(matched, record.probabilities)[record.support].detach().tolist(),
            "temperature1_ratios_on_support": importance_ratios(target, record.probabilities)[record.support].detach().tolist(),
            "expectations": results, "full_support_target_rejection": rejection,
            "missing_raw_target_mass": float(raw[~record.support].sum()),
            "missing_raw_reward_contribution": float((raw*rewards)[~record.support].sum()),
            "kl_audit": kl_gradient_audit(raw.log(), torch.tensor([.2, .5, .3], dtype=torch.float64).log())}


def main(argv=None):
    """Reproduce a bounded report and optionally export this notebook's figures.

    Never modifies notebook source or another lesson's preview assets. A report
    path must be unused, keeping earlier evidence recoverable.
    """
    import argparse
    import base64
    from datetime import datetime, timezone
    import hashlib
    import json
    from pathlib import Path
    import platform
    import resource
    import subprocess
    import sys
    import time

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--notebook-manifest", type=Path, required=True)
    parser.add_argument("--export-previews", action="store_true")
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error("Report path must be unused; do not overwrite historical evidence")
    root = Path(__file__).resolve().parents[2]
    source = "notebooks/day-20/03_behavior_probabilities_and_support.ipynb"
    verification = json.loads(args.notebook_manifest.read_text())
    notebook = next((n for n in verification["notebooks"] if n["path"] == source), None)
    if notebook is None or verification.get("failures"):
        parser.error("Need a passed fresh-kernel manifest including the support notebook")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(root/source) != notebook["sha256"]:
        parser.error("Notebook source differs from its executed reference")
    previews = []
    if args.export_previews:
        import nbformat
        executed = nbformat.read(notebook["output"], as_version=4)
        images = [o.data["image/png"] for c in executed.cells if c.cell_type == "code"
                  for o in c.outputs if o.output_type in ("display_data", "execute_result")
                  and "image/png" in o.data]
        if len(images) != 4:
            parser.error("Expected four checked source figures")
        for index, encoded in enumerate(images, 1):
            path = root/f"notebooks/figures/chapter-12/day-20-03_behavior_probabilities_and_support-{index:02}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(base64.b64decode(encoded))
            previews.append({"path": str(path.relative_to(root)), "sha256": sha(path)})
    began = time.perf_counter()
    torch.set_num_threads(1)
    results = sampling_support_experiment()
    checks = []
    commands = ([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_sampling_likelihood_lab.py", "-v"],
                [sys.executable, "scripts/check_book_math.py"],
                [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    import os
    env = dict(os.environ, PYTHONPATH=str(root/"src"), CUDA_VISIBLE_DEVICES="")
    for command in commands:
        completed = subprocess.run(command, cwd=root, env=env, text=True,
                                   capture_output=True, timeout=60)
        checks.append({"command": command, "exit_code": completed.returncode,
                       "stdout": completed.stdout, "stderr": completed.stderr})
    paths = ["src/dongxi_llms/sampling_likelihood_lab.py", "tests/test_sampling_likelihood_lab.py",
             source, "notebooks/day-20/README.md", "experiments/specs/2026-10-04-sampling-support.md",
             "book/chapters/14-when-optimization-goes-wrong.md",
             "book/solutions/14-when-optimization-goes-wrong.md", "book/labs/14-when-optimization-goes-wrong.md"]
    import matplotlib
    report = {"schema_version": 1, "package": "DXI-13", "mode": "bounded CPU reference",
              "date_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
              "base_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
              "dirty_state": subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).splitlines(),
              "python_executable": sys.executable, "python": platform.python_version(),
              "platform": platform.platform(), "machine": platform.machine(),
              "torch": torch.__version__, "matplotlib": matplotlib.__version__,
              "device": "cpu", "dtype": "torch.float64", "seed": 2020,
              "status": "passed" if all(c["exit_code"] == 0 for c in checks) else "failed",
              "results": results, "checks": checks, "source_sha256": {p: sha(root/p) for p in paths},
              "notebook_verification": notebook, "notebook_kernel": verification["kernel"],
              "notebook_manifest_sha256": sha(args.notebook_manifest), "previews": previews,
              "seconds_this_reference_and_checks": time.perf_counter()-began,
              "maximum_rss_report_process_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "minimum_observed_available_gib_notebook_run": verification.get("minimum_observed_available_gib"),
              "memory_boundary": "Notebook pre-execution MemAvailable samples and this Linux report process RSS, not continuous system/GPU peaks",
              "limitations": ["Exact finite action-distribution microscope, no learned or pretrained model",
                              "No GPU job, model/dataset acquisition, API, installation or service",
                              "Not full PPO or universal off-policy/trajectory-KL repair",
                              "Conditional-support target explicitly differs from raw full-support objective",
                              "Mac environment and live GitHub math rendering unverified",
                              "Current temperature-one/full-support Qwen baseline unchanged"]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"report": str(args.report), "status": report["status"],
                      "notebook_cells": notebook["code_cells"], "figures": len(previews),
                      "checks": [c["exit_code"] for c in checks]}))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
