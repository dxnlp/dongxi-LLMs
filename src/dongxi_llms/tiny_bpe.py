"""A small deterministic BPE trainer and frozen encoder for teaching."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from itertools import pairwise
from pathlib import Path

SymbolSequence = tuple[str, ...]
SymbolPair = tuple[str, str]
Corpus = dict[SymbolSequence, int]


def initialize_corpus(word_frequencies: Mapping[str, int]) -> Corpus:
    """Represent each pre-tokenized word as characters with a frequency."""

    corpus: Corpus = {}
    for word, frequency in word_frequencies.items():
        if not word:
            raise ValueError("BPE training words must not be empty")
        if frequency <= 0:
            raise ValueError("BPE training frequencies must be positive")
        corpus[tuple(word)] = corpus.get(tuple(word), 0) + frequency
    return corpus


def count_adjacent_pairs(corpus: Mapping[SymbolSequence, int]) -> Counter[SymbolPair]:
    """Count adjacent pairs, weighted by pre-tokenized word frequency."""

    counts: Counter[SymbolPair] = Counter()
    for symbols, frequency in corpus.items():
        for pair in pairwise(symbols):
            counts[pair] += frequency
    return counts


def merge_sequence(symbols: SymbolSequence, pair: SymbolPair) -> SymbolSequence:
    """Apply one non-overlapping pair merge from left to right."""

    merged: list[str] = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and symbols[index : index + 2] == pair:
            merged.append("".join(pair))
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return tuple(merged)


def merge_corpus(corpus: Mapping[SymbolSequence, int], pair: SymbolPair) -> Corpus:
    """Rewrite a corpus with one merge while preserving frequencies."""

    rewritten: Corpus = {}
    for symbols, frequency in corpus.items():
        merged = merge_sequence(symbols, pair)
        rewritten[merged] = rewritten.get(merged, 0) + frequency
    return rewritten


def select_pair(counts: Mapping[SymbolPair, int]) -> tuple[SymbolPair, int]:
    """Select highest count, breaking ties lexicographically for reproducibility."""

    if not counts:
        raise ValueError("No adjacent pairs remain")
    return min(counts.items(), key=lambda item: (-item[1], item[0]))


def train_bpe(
    word_frequencies: Mapping[str, int], num_merges: int
) -> tuple[list[dict[str, object]], Corpus]:
    """Train a transparent BPE rule list and retain every merge-round trace."""

    if num_merges < 0:
        raise ValueError("num_merges must be non-negative")

    corpus = initialize_corpus(word_frequencies)
    rounds: list[dict[str, object]] = []
    for rank in range(1, num_merges + 1):
        counts = count_adjacent_pairs(corpus)
        if not counts:
            break
        pair, selected_count = select_pair(counts)
        corpus = merge_corpus(corpus, pair)
        rounds.append(
            {
                "rank": rank,
                "selected_pair": list(pair),
                "selected_count": selected_count,
                "pair_counts": [
                    {"pair": list(candidate), "count": count}
                    for candidate, count in sorted(
                        counts.items(), key=lambda item: (-item[1], item[0])
                    )
                ],
                "corpus_after": serialize_corpus(corpus),
            }
        )
    return rounds, corpus


def encode_with_merges(text: str, merges: Iterable[SymbolPair]) -> SymbolSequence:
    """Encode one pre-tokenized word by replaying frozen merge rules in rank order."""

    symbols = tuple(text)
    for pair in merges:
        symbols = merge_sequence(symbols, pair)
    return symbols


def serialize_corpus(corpus: Mapping[SymbolSequence, int]) -> list[dict[str, object]]:
    """Return a stable reader-facing representation of a symbolized corpus."""

    return [
        {"symbols": list(symbols), "frequency": frequency}
        for symbols, frequency in sorted(corpus.items())
    ]


def run_demo() -> dict[str, object]:
    """Run the fixed three-round teaching example used by Chapter 2."""

    word_frequencies = {"hug": 5, "hugs": 3, "hugging": 2}
    rounds, final_corpus = train_bpe(word_frequencies, num_merges=3)
    merges = [tuple(round_["selected_pair"]) for round_ in rounds]
    return {
        "word_frequencies": word_frequencies,
        "initial_corpus": serialize_corpus(initialize_corpus(word_frequencies)),
        "selection_rule": "highest weighted adjacent-pair count; lexicographic tie break",
        "rounds": rounds,
        "frozen_merges": [list(pair) for pair in merges],
        "encoded_examples": {
            word: list(encode_with_merges(word, merges)) for word in word_frequencies
        },
        "final_corpus": serialize_corpus(final_corpus),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(run_demo(), ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
