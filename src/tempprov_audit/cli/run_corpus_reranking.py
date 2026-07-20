from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Any

from tempprov_audit.cli.run_ctd_provenance_lexical_baselines import summarize_ranking, write_json
from tempprov_audit.cli.run_full_corpus_retrieve_rerank import (
    ensemble_predictions,
    merge_reranked_topk,
    score_cross_encoder,
)
from tempprov_audit.cli.run_corpus_retrieval import read_documents, read_queries, recall_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank locked Top-k results from the 55k-document corpus benchmark.")
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--document-json", required=True)
    parser.add_argument("--retrieval-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--reranker",
        choices=["medcpt_cross_encoder", "biomedbert_3seed_ensemble", "biomedbert_15run_ensemble"],
        required=True,
    )
    parser.add_argument("--neural-cv-root")
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260703)
    return parser.parse_args()


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_rows(queries, documents_by_id, retrieval, depth):
    retrieval_by_id = {row["example_id"]: row for row in retrieval}
    rows = []
    for query in queries:
        top_ids = retrieval_by_id[query["example_id"]]["ranked_document_ids"][:depth]
        rows.append({**query, "candidates": [documents_by_id[item] for item in top_ids]})
    return rows


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    queries = read_queries(Path(args.query_file).resolve())
    documents = read_documents(Path(args.document_json).resolve())
    documents_by_id = {row["document_id"]: row for row in documents}
    retrieval = read_jsonl_gz(Path(args.retrieval_file).resolve())
    retrieval_by_id = {row["example_id"]: row for row in retrieval}
    rows = make_rows(queries, documents_by_id, retrieval, args.depth)

    import torch

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    predictions = []
    if args.reranker == "medcpt_cross_encoder":
        predictions = score_cross_encoder(
            "ncbi/MedCPT-Cross-Encoder",
            rows,
            batch_size=args.batch_size * 2,
            num_workers=args.num_workers,
            query_prefix="",
            include_publication_year=False,
        )
    elif args.reranker == "biomedbert_3seed_ensemble":
        if not args.neural_cv_root:
            raise ValueError("--neural-cv-root is required for the BiomedBERT ensemble.")
        for fold in range(5):
            fold_rows = [row for row in rows if int(row["fold_index"]) == fold]
            seed_predictions = []
            for seed in (13, 42, 3407):
                checkpoint = (
                    Path(args.neural_cv_root)
                    / f"fold_{fold}"
                    / "biomedbert_cross_encoder_3seed"
                    / f"seed_{seed}"
                    / "best_model"
                )
                seed_predictions.append(
                    score_cross_encoder(
                        str(checkpoint),
                        fold_rows,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                    )
                )
            predictions.extend(ensemble_predictions(seed_predictions))
    else:
        if not args.neural_cv_root:
            raise ValueError("--neural-cv-root is required for the BiomedBERT ensemble.")
        run_predictions = []
        for fold in range(5):
            for seed in (13, 42, 3407):
                checkpoint = (
                    Path(args.neural_cv_root)
                    / f"fold_{fold}"
                    / "biomedbert_cross_encoder_3seed"
                    / f"seed_{seed}"
                    / "best_model"
                )
                run_predictions.append(
                    score_cross_encoder(
                        str(checkpoint),
                        rows,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                    )
                )
        predictions = ensemble_predictions(run_predictions)

    merged = [
        merge_reranked_topk(retrieval_by_id[row["example_id"]], row, args.depth)
        for row in predictions
    ]
    merged.sort(key=lambda row: row["example_id"])
    elapsed = time.perf_counter() - started
    summary = {
        "reranker": args.reranker,
        "queries": len(queries),
        "corpus_documents": len(documents),
        "retrieval_depth": args.depth,
        "wall_seconds": elapsed,
        "query_document_pairs_per_second": len(queries) * args.depth / elapsed,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "ranking": summarize_ranking(merged, args.bootstrap_resamples, args.seed),
        "corpus_recall": recall_metrics(merged),
    }
    write_jsonl_gz(output_root / "topk_rankings.jsonl.gz", merged)
    write_json(output_root / "summary.json", summary)
    (output_root / "COMPLETED").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
