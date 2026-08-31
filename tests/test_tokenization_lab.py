"""Unit tests for language-independent tokenizer measurements."""

import unittest

from dongxi_llms.tokenization_lab import analyze_text


class _FakeTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, object]:
        self.text = text
        return {
            "input_ids": [10, 11],
            "offset_mapping": [(0, 1), (1, 2)],
        }

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        self.token_ids = token_ids
        return ["byte-piece-for-number", "a"]

    def decode(self, token_ids: list[int], **_: object) -> str:
        assert token_ids == self.token_ids
        return self.text


class TokenizationLabTest(unittest.TestCase):
    def test_unicode_bytes_tokens_and_round_trip_are_distinct(self) -> None:
        result = analyze_text(_FakeTokenizer(), "数a")

        self.assertEqual(result["unicode_code_points"], 2)
        self.assertEqual(result["utf8_bytes"], 4)
        self.assertEqual(result["token_count"], 2)
        self.assertEqual(result["code_points_per_token"], 1.0)
        self.assertEqual(result["utf8_bytes_per_token"], 2.0)
        self.assertEqual(result["text_spans"], ["数", "a"])
        self.assertTrue(result["exact_round_trip"])


if __name__ == "__main__":
    unittest.main()
