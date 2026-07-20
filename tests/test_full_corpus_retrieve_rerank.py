from __future__ import annotations

import unittest

from tempprov_audit.cli.run_full_corpus_retrieve_rerank import (
    merge_reranked_topk,
    oracle_topk_result,
)


class FullCorpusRetrieveRerankTest(unittest.TestCase):
    def test_merge_changes_only_locked_topk(self) -> None:
        retrieval = {
            "example_id": "Q",
            "gold_document_ids": ["D2"],
            "ranked_document_ids": ["D1", "D2", "D3", "D4"],
        }
        reranked = {
            "example_id": "Q",
            "ranked_document_ids": ["D2", "D1"],
            "top_score": 1.0,
        }
        merged = merge_reranked_topk(retrieval, reranked, 2)
        self.assertEqual(merged["ranked_document_ids"], ["D2", "D1", "D3", "D4"])

    def test_oracle_preserves_candidate_set(self) -> None:
        result = {
            "gold_document_ids": ["D2", "D4"],
            "ranked_document_ids": ["D1", "D2", "D3", "D4"],
        }
        oracle = oracle_topk_result(result, 3)
        self.assertEqual(oracle["ranked_document_ids"], ["D2", "D1", "D3", "D4"])


if __name__ == "__main__":
    unittest.main()
