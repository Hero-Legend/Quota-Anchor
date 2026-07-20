from __future__ import annotations

import unittest

from tempprov_audit.cli.analyze_ctd_provenance_tail_dominance import (
    averaged_metric_rows,
    remove_tails,
    summarize_rows,
)


def prediction(example_id: str, tail_id: str, gold_rank: int) -> dict:
    documents = ["D1", "D2", "D3"]
    gold = documents[gold_rank - 1]
    return {
        "example_id": example_id,
        "tail_id": tail_id,
        "has_gold": True,
        "gold_document_ids": [gold],
        "ranked_document_ids": documents,
    }


class TailDominanceTest(unittest.TestCase):
    def test_macro_and_micro_differ_under_tail_imbalance(self) -> None:
        rows = averaged_metric_rows(
            [[
                prediction("A1", "A", 1),
                prediction("A2", "A", 1),
                prediction("A3", "A", 1),
                prediction("B1", "B", 2),
            ]]
        )
        summary = summarize_rows(rows)
        self.assertAlmostEqual(summary["micro"]["map"], 0.875)
        self.assertAlmostEqual(summary["tail_macro"]["map"], 0.75)
        filtered = remove_tails(rows, {"A"})
        self.assertEqual(set(filtered), {"B1"})


if __name__ == "__main__":
    unittest.main()
