"""Inspect BPE, Unicode units, preprocessing, and Qwen chat packaging."""

from __future__ import annotations

import argparse
import json
import platform
import unicodedata
from pathlib import Path
from typing import Any

from dongxi_llms.tiny_bpe import encode_with_merges, serialize_corpus, train_bpe
from dongxi_llms.tokenization_lab import analyze_text

DEFAULT_SPEC = Path("experiments/specs/2026-08-31-tokenizer-mechanics.yaml")


def code_point_records(text: str) -> list[dict[str, object]]:
    """Describe every Unicode code point in a string."""

    return [
        {
            "character": character,
            "code_point": f"U+{ord(character):04X}",
            "name": unicodedata.name(character, "UNNAMED"),
        }
        for character in text
    ]


def analyze_unicode(text: str) -> dict[str, object]:
    """Measure code points, UTF-8 bytes, normalization, and grapheme clusters."""

    import regex

    utf8 = text.encode("utf-8")
    graphemes = regex.findall(r"\X", text)
    return {
        "text": text,
        "nfc": unicodedata.normalize("NFC", text),
        "nfd": unicodedata.normalize("NFD", text),
        "code_point_count": len(text),
        "code_points": code_point_records(text),
        "utf8_byte_count": len(utf8),
        "utf8_hex": [f"{byte:02X}" for byte in utf8],
        "grapheme_cluster_count": len(graphemes),
        "grapheme_clusters": graphemes,
    }


def tokenizer_identity(tokenizer: Any) -> dict[str, object]:
    """Record the tokenizer properties needed to interpret token IDs."""

    return {
        "class": type(tokenizer).__name__,
        "base_vocabulary_size": tokenizer.vocab_size,
        "total_vocabulary_size": len(tokenizer),
        "is_fast": tokenizer.is_fast,
        "special_tokens_map": tokenizer.special_tokens_map,
        "special_token_ids": {
            "bos": tokenizer.bos_token_id,
            "eos": tokenizer.eos_token_id,
            "pad": tokenizer.pad_token_id,
            "unk": tokenizer.unk_token_id,
        },
    }


def run_specification(
    spec_path: Path = DEFAULT_SPEC, *, local_files_only: bool = False
) -> dict[str, object]:
    """Run the fixed local and tokenizer-only mechanics specification."""

    import regex
    import tokenizers
    import transformers
    import yaml
    from transformers import AutoTokenizer

    specification = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    corpus = specification["inputs"]["bpe_corpus"]
    num_merges = specification["configuration"]["bpe_merge_rounds"]
    rounds, final_corpus = train_bpe(corpus, num_merges)
    merges = [tuple(round_["selected_pair"]) for round_ in rounds]

    tokenizer_spec = specification["identity"]["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_spec["name"],
        revision=tokenizer_spec["revision"],
        local_files_only=local_files_only,
    )

    normalized_word = specification["inputs"]["normalized_word"]
    nfc_word = unicodedata.normalize("NFC", normalized_word)
    nfd_word = unicodedata.normalize("NFD", normalized_word)
    leading_space_examples = specification["inputs"]["leading_space_examples"]
    messages = specification["inputs"]["messages"]
    raw_chat_text = specification["inputs"]["raw_chat_text"]
    rendered_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return {
        "experiment_id": specification["id"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "regex": regex.__version__,
            "pyyaml": yaml.__version__,
        },
        "tiny_bpe": {
            "corpus": corpus,
            "selection_rule": (
                "highest weighted adjacent-pair count; lexicographic tie break"
            ),
            "rounds": rounds,
            "frozen_merges": [list(pair) for pair in merges],
            "encoded_examples": {
                word: list(encode_with_merges(word, merges)) for word in corpus
            },
            "final_corpus": serialize_corpus(final_corpus),
        },
        "unicode": {
            "nfc_word": analyze_unicode(nfc_word),
            "nfd_word": analyze_unicode(nfd_word),
            "grapheme_example": analyze_unicode(
                specification["inputs"]["grapheme_example"]
            ),
        },
        "tokenizer": {
            "name": tokenizer_spec["name"],
            "revision": tokenizer_spec["revision"],
            **tokenizer_identity(tokenizer),
            "normalization_examples": {
                "nfc": analyze_text(tokenizer, nfc_word),
                "nfd": analyze_text(tokenizer, nfd_word),
            },
            "leading_space_examples": {
                name: analyze_text(tokenizer, text)
                for name, text in leading_space_examples.items()
            },
            "raw_text": analyze_text(tokenizer, raw_chat_text),
            "chat_template": {
                "messages": messages,
                "rendered_text": rendered_chat,
                "encoded": analyze_text(tokenizer, rendered_chat),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    result = run_specification(args.spec, local_files_only=args.local_files_only)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
