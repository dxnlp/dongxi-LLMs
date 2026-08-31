"""Verification for the transparent Day 2 embedding-gradient lab."""

import unittest

from dongxi_llms.embedding_gradient_lab import (
    classifier_path_demo,
    repeated_lookup_demo,
    response_only_loss_demo,
)


class EmbeddingGradientLabTest(unittest.TestCase):
    def test_repeated_lookup_accumulates_into_shared_rows(self) -> None:
        result = repeated_lookup_demo()

        self.assertEqual(result["input_ids_shape"], [1, 3])
        self.assertEqual(result["lookup_output_shape"], [1, 3, 4])
        self.assertEqual(result["nonzero_gradient_rows"], [2, 5])
        self.assertEqual(result["row_2_gradient"], [2.0, 2.0, 2.0, 2.0])
        self.assertEqual(result["row_5_gradient"], [1.0, 1.0, 1.0, 1.0])

    def test_tying_adds_dense_classifier_gradient_path(self) -> None:
        result = classifier_path_demo()

        self.assertEqual(
            result["untied"]["nonzero_embedding_gradient_rows"], [1, 3]
        )
        self.assertEqual(
            result["tied"]["nonzero_embedding_gradient_rows"], list(range(6))
        )

    def test_response_loss_reaches_visible_prompt_rows(self) -> None:
        result = response_only_loss_demo()

        self.assertEqual(result["loss_mask"], [0, 0, 1])
        self.assertGreater(result["response_attention_weights"][0], 0.0)
        self.assertGreater(result["response_attention_weights"][1], 0.0)
        self.assertGreater(result["prompt_row_gradient_norms"]["row_1"], 0.0)
        self.assertGreater(result["prompt_row_gradient_norms"]["row_2"], 0.0)


if __name__ == "__main__":
    unittest.main()

