"""Transparent experiment: repeated one-hot labels learn a distribution."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


TARGET_COUNTS = (70, 30)
EXPECTED_PROBABILITIES = (0.70, 0.30)
DEFAULT_SEED = 42


def make_targets(counts: tuple[int, ...] = TARGET_COUNTS) -> torch.Tensor:
    """Create deterministic categorical targets from precommitted counts."""

    return torch.cat(
        [torch.full((count,), index, dtype=torch.long) for index, count in enumerate(counts)]
    )


def empirical_distribution(targets: torch.Tensor, classes: int) -> torch.Tensor:
    """Return class frequencies using the logits' floating-point dtype later."""

    return torch.bincount(targets, minlength=classes).to(torch.float32) / targets.numel()


def loss_and_gradients(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean CE, probabilities, autograd gradient, and analytical p-r."""

    working_logits = logits.detach().clone().requires_grad_(True)
    batch_logits = working_logits.unsqueeze(0).expand(targets.numel(), -1)
    loss = F.cross_entropy(batch_logits, targets)
    (autograd_gradient,) = torch.autograd.grad(loss, working_logits)
    probabilities = torch.softmax(working_logits.detach(), dim=0)
    target_distribution = empirical_distribution(targets, logits.numel()).to(logits.dtype)
    analytical_gradient = probabilities - target_distribution
    return loss.detach(), probabilities, autograd_gradient.detach(), analytical_gradient


def _snapshot(step: int, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    loss, probabilities, autograd_gradient, analytical_gradient = loss_and_gradients(
        logits, targets
    )
    return {
        "step": step,
        "logits": logits.detach().tolist(),
        "probabilities": probabilities.tolist(),
        "mean_cross_entropy": loss.item(),
        "autograd_gradient": autograd_gradient.tolist(),
        "analytical_p_minus_r": analytical_gradient.tolist(),
        "gradient_max_abs_error": (
            autograd_gradient - analytical_gradient
        ).abs().max().item(),
        "gradient_norm": autograd_gradient.norm().item(),
    }


def run_experiment(
    *, steps: int = 500, learning_rate: float = 0.5, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    """Train two shared logits against a fixed 70/30 target batch."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    torch.manual_seed(seed)
    targets = make_targets()
    target_distribution = empirical_distribution(targets, classes=2)
    logits = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
    optimizer = torch.optim.SGD([logits], lr=learning_rate)

    requested_checkpoints = {0, 1, 2, 5, 10, 50, 100, steps}
    checkpoints = [_snapshot(0, logits, targets)]

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        batch_logits = logits.unsqueeze(0).expand(targets.numel(), -1)
        loss = F.cross_entropy(batch_logits, targets)
        loss.backward()
        optimizer.step()
        if step in requested_checkpoints:
            checkpoints.append(_snapshot(step, logits, targets))

    initial = checkpoints[0]
    final = checkpoints[-1]
    empirical_entropy = -(
        target_distribution * torch.log(target_distribution)
    ).sum().item()

    numeric_values = []
    for checkpoint in checkpoints:
        numeric_values.extend(checkpoint["logits"])
        numeric_values.extend(checkpoint["probabilities"])
        numeric_values.extend(checkpoint["autograd_gradient"])
        numeric_values.extend(checkpoint["analytical_p_minus_r"])
        numeric_values.extend(
            [
                checkpoint["mean_cross_entropy"],
                checkpoint["gradient_max_abs_error"],
                checkpoint["gradient_norm"],
            ]
        )

    criteria = {
        "all_numeric_values_finite": all(math.isfinite(value) for value in numeric_values),
        "initial_probabilities_uniform": max(
            abs(actual - expected)
            for actual, expected in zip(initial["probabilities"], (0.5, 0.5))
        )
        <= 1e-6,
        "initial_gradient_matches_p_minus_r": initial["gradient_max_abs_error"] <= 1e-6,
        "final_gradient_matches_p_minus_r": final["gradient_max_abs_error"] <= 1e-6,
        "final_probabilities_match_empirical_distribution": max(
            abs(actual - expected)
            for actual, expected in zip(final["probabilities"], EXPECTED_PROBABILITIES)
        )
        <= 0.01,
        "final_loss_not_greater_than_initial": (
            final["mean_cross_entropy"] <= initial["mean_cross_entropy"]
        ),
        "final_loss_matches_empirical_entropy": (
            abs(final["mean_cross_entropy"] - empirical_entropy) <= 1e-4
        ),
    }

    return {
        "environment": {
            "torch_version": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
        },
        "configuration": {
            "seed": seed,
            "steps": steps,
            "optimizer": "SGD",
            "learning_rate": learning_rate,
            "target_counts": list(TARGET_COUNTS),
        },
        "empirical_target_distribution": target_distribution.tolist(),
        "empirical_entropy": empirical_entropy,
        "checkpoints": checkpoints,
        "criteria": criteria,
        "all_criteria_passed": all(criteria.values()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_experiment(
        steps=args.steps, learning_rate=args.learning_rate, seed=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_criteria_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
