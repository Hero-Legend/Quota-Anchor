from __future__ import annotations

import unittest

from tempprov_audit.cli.run_ctd_semantic_huber import (
    build_ranked_results,
    conditioned_text_and_features,
    flatten_rows,
)
from tempprov_audit.cli.run_ctd_provenance_cross_encoder import flatten_rows as flatten_cross_encoder_rows


class SemanticHuberTest(unittest.TestCase):
    def test_cross_encoder_can_use_raw_query_for_model_specific_protocols(self) -> None:
        rows = [{
            "query_text": "Chemical: aspirin. Disease: headache.",
            "gold_document_ids": [],
            "candidates": [{"document_id": "D", "text": "article"}],
        }]
        flattened = flatten_cross_encoder_rows(
            rows, "", include_publication_year=False
        )[0]
        self.assertEqual(flattened["query"], rows[0]["query_text"])
        self.assertEqual(flattened["document"], "article")

    def test_features_ignore_provenance_labels(self) -> None:
        row = {
            "head_text": "aspirin",
            "tail_text": "headache",
        }
        candidate = {
            "text": "Aspirin treats headache. A second sentence.",
            "pub_year": 2025,
            "source": "gold_postcut_ctd_citation",
            "is_ctd_cited": True,
        }
        changed = {**candidate, "source": "random", "is_ctd_cited": False}
        self.assertEqual(
            conditioned_text_and_features(row, candidate),
            conditioned_text_and_features(row, changed),
        )
        _, numeric = conditioned_text_and_features(row, candidate)
        self.assertEqual(numeric[2], 1.0)
        self.assertGreater(numeric[3], 0.0)
        self.assertGreater(numeric[4], 0.0)

    def test_ranked_results_preserve_query_and_gold(self) -> None:
        rows = [{
            "example_id": "Q1",
            "head_id": "H",
            "tail_id": "T",
            "head_text": "aspirin",
            "tail_text": "headache",
            "gold_document_ids": ["D2"],
            "candidates": [
                {"document_id": "D1", "text": "irrelevant"},
                {"document_id": "D2", "text": "aspirin treats headache"},
            ],
        }]
        flat = flatten_rows(rows)
        result = build_ranked_results(rows, flat, [0.1, 0.9])[0]
        self.assertEqual(result["ranked_document_ids"], ["D2", "D1"])
        self.assertEqual(result["gold_document_ids"], ["D2"])


if __name__ == "__main__":
    unittest.main()
