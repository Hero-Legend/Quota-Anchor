from __future__ import annotations

import unittest

from tempprov_audit.cli.compare_ctd_provenance_neural_cv import paired_summary


class ProvenanceNeuralComparisonTests(unittest.TestCase):
    def test_paired_summary_preserves_micro_and_tail_macro_weighting(self) -> None:
        summary = paired_summary(
            {"tail_a": [1.0, 1.0, 1.0], "tail_b": [-1.0]},
            bootstrap_resamples=100,
            signflip_resamples=1000,
            seed=7,
        )
        self.assertEqual(summary["examples"], 4)
        self.assertEqual(summary["tail_clusters"], 2)
        self.assertAlmostEqual(summary["micro"]["delta"], 0.5)
        self.assertAlmostEqual(summary["tail_macro"]["delta"], 0.0)

    def test_signflip_pvalue_is_bounded(self) -> None:
        summary = paired_summary(
            {f"tail_{index}": [1.0] for index in range(8)},
            bootstrap_resamples=100,
            signflip_resamples=2000,
            seed=11,
        )
        self.assertGreater(summary["micro"]["paired_tail_signflip_p"], 0.0)
        self.assertLessEqual(summary["micro"]["paired_tail_signflip_p"], 1.0)
        self.assertGreater(summary["micro"]["bootstrap_95_ci"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
