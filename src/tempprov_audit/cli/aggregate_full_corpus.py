from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from tempprov_audit.cli.compare_ctd_provenance_neural_cv import compare_methods
from tempprov_audit.cli.run_ctd_provenance_lexical_baselines import per_positive_metrics, write_json


RETRIEVERS = ("bm25", "bge_large", "medcpt_dual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate full-corpus retrieval and reranking evidence.")
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--retrieval-root", required=True)
    parser.add_argument("--reranking-root", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-biomedbert", action="store_true")
    parser.add_argument("--biomedbert-reranker", default="biomedbert_3seed_ensemble")
    parser.add_argument("--flat-layout", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--signflip-resamples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260703)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_methods(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    retrieval_root = Path(args.retrieval_root)
    reranking_root = Path(args.reranking_root)
    if not args.flat_layout:
        retrieval_root /= args.cohort
        reranking_root /= args.cohort
    methods = {}
    for retriever in RETRIEVERS:
        methods[retriever] = read_jsonl(retrieval_root / retriever / "topk_rankings.jsonl.gz")
        methods[f"{retriever}_medcpt_ce"] = read_jsonl(
            reranking_root / retriever / "medcpt_cross_encoder" / "topk_rankings.jsonl.gz"
        )
        if args.include_biomedbert:
            methods[f"{retriever}_biomedbert"] = read_jsonl(
                reranking_root / retriever / args.biomedbert_reranker / "topk_rankings.jsonl.gz"
            )
    return methods


def metric_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        row["example_id"]: {
            "example_id": row["example_id"],
            "tail_id": str(row.get("tail_id") or row["example_id"]),
            "metrics": per_positive_metrics(row),
        }
        for row in rows
    }


def point_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = [per_positive_metrics(row) for row in rows]
    output = {key: mean(row[key] for row in values) for key in values[0]}
    for cutoff in (20, 50, 100, 200, 500, 1000):
        output[f"recall_at_{cutoff}"] = mean(
            len(set(row["gold_document_ids"]) & set(row["ranked_document_ids"][:cutoff]))
            / len(row["gold_document_ids"])
            for row in rows
        )
    return output


def strata(queries: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = {row["example_id"]: row for row in queries}
    dimensions = {
        "fold_index": lambda row: str(row.get("fold_index", "unknown")),
        "first_pub_year": lambda row: str(row.get("first_pub_year", "not_applicable")),
        "direct_evidence": lambda row: str(row.get("direct_evidence", "not_applicable")),
        "gold_document_count": lambda row: str(len(row["gold_document_ids"])),
    }
    output = {}
    for dimension, accessor in dimensions.items():
        groups = defaultdict(list)
        for row in rows:
            groups[accessor(metadata[row["example_id"]])].append(row)
        output[dimension] = {
            key: {"queries": len(group), "metrics": point_metrics(group)}
            for key, group in sorted(groups.items())
        }
    return output


def main() -> None:
    args = parse_args()
    queries = read_jsonl(Path(args.query_file))
    methods = load_methods(args)
    metric_maps = {name: metric_rows(rows) for name, rows in methods.items()}
    comparisons = []
    for retriever in RETRIEVERS:
        comparisons.append((f"{retriever}_medcpt_ce", retriever))
        if args.include_biomedbert:
            comparisons.extend(
                [
                    (f"{retriever}_biomedbert", retriever),
                    (f"{retriever}_biomedbert", f"{retriever}_medcpt_ce"),
                ]
            )
    if args.include_biomedbert:
        comparisons.append(("bge_large_biomedbert", "medcpt_dual_biomedbert"))

    payload = {
        "protocol": "frozen identical-query comparison; disease-tail cluster bootstrap and paired sign-flip randomization",
        "cohort": args.cohort,
        "queries": len(queries),
        "methods": {name: point_metrics(rows) for name, rows in methods.items()},
        "failure_strata": {name: strata(queries, rows) for name, rows in methods.items()},
        "paired_comparisons": {
            f"{left}_minus_{right}": compare_methods(
                metric_maps[left],
                metric_maps[right],
                bootstrap_resamples=args.bootstrap_resamples,
                signflip_resamples=args.signflip_resamples,
                seed=args.seed + index * 10000,
            )
            for index, (left, right) in enumerate(comparisons)
        },
        "multiplicity": "All predeclared pairwise p-values are reported; Holm correction is applied during report synthesis.",
    }
    write_json(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
