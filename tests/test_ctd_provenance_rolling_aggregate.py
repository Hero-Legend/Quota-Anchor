from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tempprov_audit.cli.aggregate_ctd_provenance_rolling_origin import (
    averaged_positive_rows,
)


class RollingOriginAggregateTests(unittest.TestCase):
    def test_averaged_positive_rows_averages_seed_rankings(self) -> None:
        base = {
            "example_id": "pair__publication_year_2024",
            "tail_id": "D1",
            "has_gold": True,
            "gold_document_ids": ["PMID:1"],
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, ranked in enumerate((["PMID:1", "PMID:2"], ["PMID:2", "PMID:1"])):
                path = Path(directory) / f"seed_{index}.json"
                path.write_text(
                    json.dumps([{**base, "ranked_document_ids": ranked}]), encoding="utf-8"
                )
                paths.append([path])
            rows = averaged_positive_rows(paths)
        self.assertAlmostEqual(rows[base["example_id"]]["metrics"]["map"], 0.75)
        self.assertEqual(rows[base["example_id"]]["tail_id"], "D1")


if __name__ == "__main__":
    unittest.main()
