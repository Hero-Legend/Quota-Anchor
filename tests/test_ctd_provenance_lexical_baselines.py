from __future__ import annotations

import unittest
import gzip
import json
import tempfile
from pathlib import Path

from tempprov_audit.cli.run_ctd_provenance_lexical_baselines import (
    bm25_scores,
    classification_metrics,
    per_positive_metrics,
    select_thresholds,
    tfidf_scores,
    read_jsonl,
)


class CTDProvenanceLexicalBaselineTests(unittest.TestCase):
    def test_lexical_scorers_rank_matching_document_first(self) -> None:
        documents = ["unrelated control text", "aspirin treatment for migraine"]
        for scorer in (bm25_scores, tfidf_scores):
            scores = scorer("aspirin migraine", documents)
            self.assertGreater(scores[1], scores[0])

    def test_positive_ranking_metrics(self) -> None:
        metrics = per_positive_metrics(
            {
                "gold_document_ids": ["b", "d"],
                "ranked_document_ids": ["a", "b", "c", "d"],
            }
        )
        self.assertAlmostEqual(metrics["mrr"], 0.5)
        self.assertAlmostEqual(metrics["recall_at_3"], 0.5)
        self.assertAlmostEqual(metrics["recall_at_5"], 1.0)

    def test_threshold_selection_uses_dev_and_obeys_fdr_constraint(self) -> None:
        dev = [
            {"top_score": 0.9, "has_gold": True},
            {"top_score": 0.8, "has_gold": True},
            {"top_score": 0.7, "has_gold": False},
            {"top_score": 0.1, "has_gold": False},
        ]
        policies = select_thresholds(dev)
        safe = policies["dev_fdr_le_0p10_max_recall"]
        self.assertEqual(safe["recall"], 1.0)
        self.assertEqual(safe["fdr"], 0.0)
        test_metrics = classification_metrics(dev, safe["threshold"])
        self.assertEqual(test_metrics["tp"], 2)
        self.assertEqual(test_metrics["fp"], 0)

    def test_read_jsonl_hydrates_normalized_gzip_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with gzip.open(root / "documents.jsonl.gz", "wt", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "document_key": "k1",
                            "document_id": "PMID:1",
                            "text": "aspirin migraine",
                        }
                    )
                    + "\n"
                )
            split = root / "fold_0" / "test.jsonl.gz"
            split.parent.mkdir()
            with gzip.open(split, "wt", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "query_text": "aspirin migraine",
                            "candidates": [
                                {"document_key": "k1", "document_id": "PMID:1", "source": "gold"}
                            ],
                        }
                    )
                    + "\n"
                )
            rows = read_jsonl(split.with_suffix(""))
            self.assertEqual(rows[0]["candidates"][0]["text"], "aspirin migraine")
            self.assertEqual(rows[0]["candidates"][0]["source"], "gold")


if __name__ == "__main__":
    unittest.main()
