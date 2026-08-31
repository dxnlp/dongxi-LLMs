"""Unit tests for the transparent BPE trainer and frozen encoder."""

import unittest

from dongxi_llms.tiny_bpe import (
    count_adjacent_pairs,
    encode_with_merges,
    initialize_corpus,
    merge_sequence,
    train_bpe,
)


class TinyBPETest(unittest.TestCase):
    def test_pair_counts_are_weighted_by_word_frequency(self) -> None:
        counts = count_adjacent_pairs(
            initialize_corpus({"hug": 5, "hugs": 3, "hugging": 2})
        )

        self.assertEqual(counts[("h", "u")], 10)
        self.assertEqual(counts[("u", "g")], 10)
        self.assertEqual(counts[("g", "s")], 3)

    def test_three_round_trace_is_deterministic(self) -> None:
        rounds, _ = train_bpe({"hug": 5, "hugs": 3, "hugging": 2}, 3)

        self.assertEqual(
            [round_["selected_pair"] for round_ in rounds],
            [["h", "u"], ["hu", "g"], ["hug", "s"]],
        )
        self.assertEqual([round_["selected_count"] for round_ in rounds], [10, 10, 3])

    def test_frozen_rules_replay_without_learning(self) -> None:
        merges = [("h", "u"), ("hu", "g"), ("hug", "s")]

        self.assertEqual(encode_with_merges("hugs", merges), ("hugs",))
        self.assertEqual(
            encode_with_merges("hugging", merges), ("hug", "g", "i", "n", "g")
        )

    def test_overlapping_pair_merge_is_left_to_right(self) -> None:
        self.assertEqual(merge_sequence(("a", "a", "a"), ("a", "a")), ("aa", "a"))


if __name__ == "__main__":
    unittest.main()
