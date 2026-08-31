"""Reproducible multilingual tokenizer measurements for Day 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SPEC = Path(
    "experiments/specs/2026-08-30-qwen3-multilingual-tokenization.yaml"
)


def analyze_text(tokenizer: Any, text: str) -> dict[str, Any]:
    """Measure one string without adding model-specific special tokens."""

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    pieces = tokenizer.convert_ids_to_tokens(token_ids)
    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    code_points = len(text)
    utf8_bytes = len(text.encode("utf-8"))
    token_count = len(token_ids)

    return {
        "text": text,
        "unicode_code_points": code_points,
        "utf8_bytes": utf8_bytes,
        "token_count": token_count,
        "code_points_per_token": round(code_points / token_count, 6),
        "utf8_bytes_per_token": round(utf8_bytes / token_count, 6),
        "token_ids": token_ids,
        "vocabulary_pieces": pieces,
        "offsets": [list(pair) for pair in offsets],
        "text_spans": [text[start:end] for start, end in offsets],
        "decoded": decoded,
        "exact_round_trip": decoded == text,
    }


def run_specification(
    spec_path: Path = DEFAULT_SPEC, *, local_files_only: bool = False
) -> dict[str, Any]:
    """Load the tokenizer identity and inputs from the fixed YAML specification."""

    import yaml
    from transformers import AutoTokenizer

    with spec_path.open(encoding="utf-8") as handle:
        specification = yaml.safe_load(handle)

    tokenizer_identity = specification["identity"]["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_identity["name"],
        revision=tokenizer_identity["revision"],
        local_files_only=local_files_only,
    )

    measurements = {
        language: analyze_text(tokenizer, text)
        for language, text in specification["inputs"].items()
    }
    ranking = sorted(
        measurements,
        key=lambda language: measurements[language]["token_count"],
    )

    return {
        "experiment_id": specification["id"],
        "tokenizer": {
            "name": tokenizer_identity["name"],
            "revision": tokenizer_identity["revision"],
            "class": type(tokenizer).__name__,
            "base_vocabulary_size": tokenizer.vocab_size,
            "total_vocabulary_size": len(tokenizer),
            "is_fast": tokenizer.is_fast,
        },
        "configuration": specification["configuration"],
        "measurements": measurements,
        "ranking_fewest_to_most_tokens": ranking,
        "all_exact_round_trips": all(
            result["exact_round_trip"] for result in measurements.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the pinned tokenizer revision to exist in the local cache.",
    )
    args = parser.parse_args()
    result = run_specification(args.spec, local_files_only=args.local_files_only)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
