"""Small, inspectable experiments for embedding and masking gradient paths."""

from __future__ import annotations

import json
import math
from typing import Any

import torch
import torch.nn.functional as F


SEED = 7
VOCAB_SIZE = 6
EMBEDDING_WIDTH = 4
NONZERO_TOLERANCE = 1e-9


def _nonzero_rows(gradient: torch.Tensor) -> list[int]:
    """Return row indices whose gradient norm exceeds the fixed tolerance."""

    norms = gradient.norm(dim=1)
    return torch.nonzero(norms > NONZERO_TOLERANCE).flatten().tolist()


def _rounded(values: torch.Tensor, digits: int = 6) -> list[Any]:
    """Convert a tensor to stable, readable nested Python lists."""

    scale = 10**digits
    rounded = torch.round(values.detach().cpu() * scale) / scale
    return rounded.tolist()


def repeated_lookup_demo() -> dict[str, Any]:
    """Show that repeated uses accumulate into one shared embedding row."""

    embedding = torch.nn.Embedding(VOCAB_SIZE, EMBEDDING_WIDTH)
    input_ids = torch.tensor([[2, 5, 2]])
    looked_up = embedding(input_ids)
    looked_up.sum().backward()

    assert embedding.weight.grad is not None
    return {
        "input_ids_shape": list(input_ids.shape),
        "lookup_output_shape": list(looked_up.shape),
        "nonzero_gradient_rows": _nonzero_rows(embedding.weight.grad),
        "row_2_gradient": _rounded(embedding.weight.grad[2]),
        "row_5_gradient": _rounded(embedding.weight.grad[5]),
    }


def _classifier_demo(*, tied: bool) -> dict[str, Any]:
    """Compare lookup-only input gradients with tied classifier gradients."""

    torch.manual_seed(SEED)
    embedding = torch.nn.Embedding(VOCAB_SIZE, EMBEDDING_WIDTH)
    input_ids = torch.tensor([1, 3])
    hidden = embedding(input_ids).mean(dim=0)

    if tied:
        output_weight = embedding.weight
    else:
        output_weight = torch.nn.Parameter(embedding.weight.detach().clone())

    logits = hidden @ output_weight.T
    target = torch.tensor([4])
    loss = F.cross_entropy(logits.unsqueeze(0), target)
    loss.backward()

    assert embedding.weight.grad is not None
    return {
        "tied": tied,
        "input_rows": input_ids.tolist(),
        "target_row": target.item(),
        "loss": round(loss.item(), 6),
        "embedding_gradient_norms": _rounded(embedding.weight.grad.norm(dim=1)),
        "nonzero_embedding_gradient_rows": _nonzero_rows(embedding.weight.grad),
    }


def classifier_path_demo() -> dict[str, Any]:
    """Return matched untied and tied classifier experiments."""

    return {
        "untied": _classifier_demo(tied=False),
        "tied": _classifier_demo(tied=True),
    }


def response_only_loss_demo() -> dict[str, Any]:
    """Show prompt gradients caused by a supervised response dependency.

    This is deliberately one-head causal self-attention with identity query,
    key, and value projections. Positions 0 and 1 are treated as an unsupervised
    prompt; only position 2 contributes a next-token loss.
    """

    torch.manual_seed(SEED)
    embedding = torch.nn.Embedding(VOCAB_SIZE, EMBEDDING_WIDTH)
    output_weight = torch.nn.Parameter(
        torch.randn(VOCAB_SIZE, EMBEDDING_WIDTH)
    )
    input_ids = torch.tensor([1, 2, 3])
    targets = torch.tensor([2, 3, 4])
    loss_mask = torch.tensor([0.0, 0.0, 1.0])

    x = embedding(input_ids)
    scores = x @ x.T / math.sqrt(EMBEDDING_WIDTH)
    causal_mask = torch.triu(
        torch.ones(3, 3, dtype=torch.bool), diagonal=1
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    hidden = x + attention @ x
    logits = hidden @ output_weight.T

    per_position_loss = F.cross_entropy(logits, targets, reduction="none")
    loss = (per_position_loss * loss_mask).sum() / loss_mask.sum()
    loss.backward()

    assert embedding.weight.grad is not None
    return {
        "input_rows": input_ids.tolist(),
        "targets": targets.tolist(),
        "loss_mask": loss_mask.int().tolist(),
        "per_position_loss": _rounded(per_position_loss),
        "optimized_loss": round(loss.item(), 6),
        "response_attention_weights": _rounded(attention[2]),
        "embedding_gradient_norms": _rounded(embedding.weight.grad.norm(dim=1)),
        "nonzero_embedding_gradient_rows": _nonzero_rows(embedding.weight.grad),
        "prompt_row_gradient_norms": {
            "row_1": round(embedding.weight.grad[1].norm().item(), 6),
            "row_2": round(embedding.weight.grad[2].norm().item(), 6),
        },
    }


def run_all_demos() -> dict[str, Any]:
    """Run all fixed demonstrations and return serializable evidence."""

    return {
        "environment": {
            "python_torch_version": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "seed": SEED,
        },
        "repeated_lookup": repeated_lookup_demo(),
        "classifier_paths": classifier_path_demo(),
        "response_only_loss": response_only_loss_demo(),
    }


def main() -> None:
    print(json.dumps(run_all_demos(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

