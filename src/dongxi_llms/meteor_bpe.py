"""Measured character-level BPE for the user-supplied meteor-shower corpus.

The second merge is not uniquely implied by frequencies. A declared teaching
preference selects it only among maximum-count pairs. IDs are assigned after
training and are a deliberately chosen, complete demonstration mapping.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .tiny_bpe import (
    SymbolPair,
    count_adjacent_pairs,
    encode_with_merges,
    initialize_corpus,
    merge_corpus,
)

TEXTS = ("流星蝴蝶剑", "带你去看流星雨", "流水", "星辰大海", "雨一直下")
TIE_PREFERENCE: tuple[SymbolPair, ...] = (("流星", "雨"),)
DISPLAY_TOKENS = ("流", "星", "雨", "流星", "流星雨")


def choose_pair(
    counts: Mapping[SymbolPair, int], preferred_ties: Sequence[SymbolPair]
) -> tuple[SymbolPair, str]:
    """Keep maximum frequency authoritative; apply preferences only on ties."""

    if not counts:
        raise ValueError("No adjacent pairs remain")
    maximum = max(counts.values())
    candidates = sorted(pair for pair, count in counts.items() if count == maximum)
    if len(candidates) == 1:
        return candidates[0], "unique maximum"
    for pair in preferred_ties:
        if pair in candidates:
            return pair, "predeclared teaching preference among tied maxima"
    return candidates[0], "Unicode lexicographic order among tied maxima"


def build_trace() -> dict[str, object]:
    """Run exactly two rounds and record all competitors, rewrites, and IDs."""

    frequencies = dict.fromkeys(TEXTS, 1)
    corpus = initialize_corpus(frequencies)
    base_alphabet = list(dict.fromkeys("".join(TEXTS)))
    merges: list[SymbolPair] = []
    rounds = []

    def snapshot() -> list[dict[str, object]]:
        return [
            {"text": text, "frequency": 1,
             "tokens": list(encode_with_merges(text, merges))}
            for text in TEXTS
        ]

    initial = snapshot()
    for rank in (1, 2):
        before = snapshot()
        counts = count_adjacent_pairs(corpus)
        pair, reason = choose_pair(counts, TIE_PREFERENCE)
        maximum = max(counts.values())
        corpus = merge_corpus(corpus, pair)
        merges.append(pair)
        rounds.append({
            "rank": rank,
            "selected_pair": list(pair),
            "selected_count": counts[pair],
            "selection_reason": reason,
            "tied_maxima": [list(p) for p in sorted(counts) if counts[p] == maximum],
            "pair_counts": [
                {"pair": list(p), "count": n}
                for p, n in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            ],
            "corpus_before": before,
            "corpus_after": snapshot(),
            "corpus_token_count_after": sum(len(s) * f for s, f in corpus.items()),
        })

    vocabulary_tokens = [*base_alphabet, *("".join(pair) for pair in merges)]
    assert all(text in vocabulary_tokens for text in DISPLAY_TOKENS)
    ordered_vocabulary = [*DISPLAY_TOKENS,
                          *(t for t in vocabulary_tokens if t not in DISPLAY_TOKENS)]
    token_to_id = {text: i for i, text in enumerate(ordered_vocabulary, start=1)}
    id_to_token = {i: text for text, i in token_to_id.items()}
    encoded = []
    for text in (*TEXTS, "流星雨", "流星", "星雨"):
        symbols = encode_with_merges(text, merges)
        ids = [token_to_id[symbol] for symbol in symbols]
        decoded = "".join(id_to_token[i] for i in ids)
        encoded.append({"text": text, "tokens": list(symbols), "ids": ids,
                        "decoded": decoded, "exact_round_trip": decoded == text})
    return {
        "task_id": "ANIM-BPE-002",
        "mode": "transparent CPU teaching example",
        "base_unit": "Unicode character; not byte-level BPE",
        "normalization": "none",
        "pretokenization": "each supplied string is one segment; no cross-string merges",
        "selection_rule": "maximum weighted pair count; declared preference on ties; then lexicographic",
        "tie_preference": [list(pair) for pair in TIE_PREFERENCE],
        "initial_corpus": initial,
        "initial_corpus_token_count": sum(map(len, TEXTS)),
        "base_alphabet_size": len(base_alphabet),
        "rounds": rounds,
        "learned_merges": [list(pair) for pair in merges],
        "vocabulary_size": len(token_to_id),
        "token_to_id": token_to_id,
        "displayed_subset": list(DISPLAY_TOKENS),
        "id_policy": "illustrative 1-based post-training mapping; other characters occupy IDs 6-19",
        "encoding_examples": encoded,
        "limitations": [
            "The second selected pair is one of 15 tied maxima, not a unique frequency winner.",
            "All five input strings appear once; the corpus has not been reweighted.",
            "This character-level vocabulary has no complete byte coverage or special tokens.",
            "The 1-5 display is a subset of a complete 19-entry toy vocabulary, not Qwen IDs.",
            "Tokenizer training is distinct from neural language-model training and semantic understanding.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(build_trace(), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
