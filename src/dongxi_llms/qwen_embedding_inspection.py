"""Inspect the pinned Qwen3 tokenizer/embedding boundary and runtime tying."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def inspect_model_interface(
    model_name: str = DEFAULT_MODEL,
    revision: str = DEFAULT_REVISION,
    *,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Load a pinned model and report exact tokenizer/weight relationships."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
        dtype="auto",
    )
    model.eval()

    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    tokenizer_size = len(tokenizer)
    model_vocab_size = model.config.vocab_size
    boundary_ids = [tokenizer_size - 1, tokenizer_size, model_vocab_size - 1]

    with torch.no_grad():
        input_ids = torch.tensor([[tokenizer.eos_token_id]])
        logits = model(input_ids=input_ids).logits

    return {
        "identity": {"model": model_name, "revision": revision},
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "base_vocabulary_size": tokenizer.vocab_size,
            "total_vocabulary_size": tokenizer_size,
            "maximum_assigned_id": max(tokenizer.get_vocab().values()),
            "boundary_pieces": {
                str(token_id): tokenizer.convert_ids_to_tokens(token_id)
                for token_id in boundary_ids
            },
        },
        "model": {
            "class": type(model).__name__,
            "configured_vocabulary_size": model_vocab_size,
            "hidden_size": model.config.hidden_size,
            "tie_word_embeddings": model.config.tie_word_embeddings,
            "input_embedding_shape": list(input_weight.shape),
            "output_embedding_shape": list(output_weight.shape),
            "same_parameter_object": input_weight is output_weight,
            "same_storage_pointer": input_weight.data_ptr()
            == output_weight.data_ptr(),
            "one_token_logits_shape": list(logits.shape),
            "one_token_logits_all_finite": bool(torch.isfinite(logits).all()),
        },
        "relationship": {
            "extra_model_rows_beyond_tokenizer": model_vocab_size
            - tokenizer_size,
            "model_vocabulary_divisible_by_128": model_vocab_size % 128 == 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hub access instead of requiring a complete local snapshot.",
    )
    args = parser.parse_args()
    result = inspect_model_interface(
        args.model,
        args.revision,
        local_files_only=not args.allow_download,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
