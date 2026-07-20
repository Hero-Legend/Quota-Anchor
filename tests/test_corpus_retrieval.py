import gzip
import json

from tempprov_audit.cli.run_ctd_provenance_lexical_baselines import per_positive_metrics
from tempprov_audit.cli.run_corpus_retrieval import bm25_topk, read_documents


def test_bm25_topk_prioritizes_matching_document():
    queries = [{"query_text": "Chemical: aspirin. Disease: headache."}]
    documents = [
        {"text": "unrelated protein sequence"},
        {"text": "aspirin treatment for headache"},
        {"text": "headache only"},
    ]
    rankings, stats = bm25_topk(queries, documents, 2)
    assert rankings[0][0][0] == 1
    assert stats["indexed_terms"] > 0


def test_truncated_ranking_assigns_zero_when_gold_is_not_retrieved():
    metrics = per_positive_metrics(
        {"gold_document_ids": ["PMID:gold"], "ranked_document_ids": ["PMID:other"]}
    )
    assert metrics["mrr"] == 0.0
    assert metrics["map"] == 0.0
    assert metrics["recall_at_10"] == 0.0


def test_read_documents_accepts_gzipped_jsonl(tmp_path):
    path = tmp_path / "documents.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "document_id": "PMID:1",
                    "pmid": "1",
                    "pub_year": 2025,
                    "text": "example",
                }
            )
            + "\n"
        )
    assert read_documents(path)[0]["document_id"] == "PMID:1"
