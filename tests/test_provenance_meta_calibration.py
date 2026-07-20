from __future__ import annotations

import unittest

from tempprov_audit.cli.evaluate_provenance_meta_calibration import (
    build_feature_row,
    select_threshold,
    wilson_upper,
)


class ProvenanceMetaCalibrationTests(unittest.TestCase):
    def test_feature_row_combines_seed_scores_without_labels(self) -> None:
        rows = [
            {
                "example_id": "x",
                "tail_id": "d",
                "has_gold": True,
                "ranked_scores": [3.0 + seed, 1.0, 0.0],
                "ranked_document_ids": ["a", "b", "c"],
            }
            for seed in (0.0, 1.0, 2.0)
        ]
        feature = build_feature_row(rows, 0)
        self.assertEqual(feature["label"], 1)
        self.assertEqual(feature["ensemble_score"], 4.0)
        self.assertGreater(len(feature["features"]), 20)

    def test_empirical_fdr_threshold_maximizes_recall_under_constraint(self) -> None:
        selected = select_threshold([0.9, 0.8, 0.7, 0.1], [1, 1, 0, 0], "empirical_fdr_le_0p10")
        self.assertEqual(selected["tp"], 2)
        self.assertEqual(selected["fp"], 0)
        self.assertEqual(selected["threshold"], 0.8)

    def test_wilson_control_is_more_conservative_for_small_samples(self) -> None:
        self.assertGreater(wilson_upper(0, 2), 0.10)
        selected = select_threshold([0.9, 0.8, 0.1], [1, 1, 0], "wilson95_fdr_le_0p10")
        self.assertEqual(selected["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
