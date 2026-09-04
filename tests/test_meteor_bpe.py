"""Regression checks for every empirical claim in ANIM-BPE-002."""

import unittest

from dongxi_llms.meteor_bpe import DISPLAY_TOKENS, TEXTS, build_trace, choose_pair
from dongxi_llms.tiny_bpe import encode_with_merges


class MeteorBPETest(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = build_trace()

    def test_corpus_is_unweighted_and_unchanged(self):
        initial = self.trace["initial_corpus"]
        self.assertEqual([row["text"] for row in initial], list(TEXTS))
        self.assertEqual([row["frequency"] for row in initial], [1] * 5)
        self.assertEqual(self.trace["initial_corpus_token_count"], 22)
        self.assertEqual(self.trace["base_alphabet_size"], 17)

    def test_two_merges_and_full_tie_evidence(self):
        first, second = self.trace["rounds"]
        self.assertEqual(first["selected_pair"], ["流", "星"])
        self.assertEqual(first["selected_count"], 2)
        self.assertEqual(first["tied_maxima"], [["流", "星"]])
        self.assertEqual(second["selected_pair"], ["流星", "雨"])
        self.assertEqual(second["selected_count"], 1)
        self.assertEqual(len(second["tied_maxima"]), 15)
        self.assertTrue(all(row["count"] == 1 for row in second["pair_counts"]))
        self.assertEqual([r["corpus_token_count_after"] for r in (first, second)], [20, 19])

    def test_teaching_preference_never_overrides_frequency(self):
        counts = {("a", "b"): 3, ("流星", "雨"): 1}
        self.assertEqual(choose_pair(counts, [("流星", "雨")])[0], ("a", "b"))
        with self.assertRaises(ValueError):
            choose_pair({}, [])

    def test_all_constituents_remain_and_ids_are_unique(self):
        vocabulary = self.trace["token_to_id"]
        self.assertEqual(self.trace["vocabulary_size"], 19)
        self.assertEqual([vocabulary[t] for t in DISPLAY_TOKENS], [1, 2, 3, 4, 5])
        self.assertEqual(len(set(vocabulary.values())), 19)
        self.assertTrue(set("".join(TEXTS)).issubset(vocabulary))

    def test_frozen_encoding_round_trips_and_does_not_add_merges(self):
        examples = {row["text"]: row for row in self.trace["encoding_examples"]}
        self.assertTrue(all(row["exact_round_trip"] for row in examples.values()))
        self.assertEqual(examples["流星雨"]["ids"], [5])
        self.assertEqual(examples["星雨"]["ids"], [2, 3])
        merges = [tuple(pair) for pair in self.trace["learned_merges"]]
        self.assertEqual(encode_with_merges("流星流星雨", merges), ("流星", "流星雨"))
        self.assertEqual(len(merges), 2)


if __name__ == "__main__":
    unittest.main()
