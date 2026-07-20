from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from tempprov_audit.cli.compare_ctd_provenance_neural_cv import compare_methods
from tempprov_audit.cli.run_ctd_provenance_lexical_baselines import (
    per_positive_metrics,
    write_json,
)


METHOD_PATHS = {
    "bm25_retrieval": "retrieval/bm25/full_rankings.jsonl.gz",
    "bm25_medcpt": "reranking/bm25/medcpt_cross_encoder/full_rankings.jsonl.gz",
    "bm25_biomedbert": "reranking/bm25/biomedbert_3seed_ensemble/full_rankings.jsonl.gz",
    "bge_biomedbert": "reranking/bge_large/biomedbert_3seed_ensemble/full_rankings.jsonl.gz",
    "medcptdual_biomedbert": "reranking/medcpt_dual/biomedbert_3seed_ensemble/full_rankings.jsonl.gz",
}

COMPARISONS = (
    ("bm25_biomedbert", "bm25_retrieval"),
    ("bm25_biomedbert", "bm25_medcpt"),
    ("bm25_biomedbert", "bge_biomedbert"),
    ("bm25_biomedbert", "medcptdual_biomedbert"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired inference for frozen full-corpus rankings.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--signflip-resamples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260702)
    return parser.parse_args()


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def metric_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row["has_gold"]:
            continue
        output[row["example_id"]] = {
            "example_id": row["example_id"],
            "tail_id": str(row.get("tail_id") or row["example_id"]),
            "metrics": per_positive_metrics(row),
        }
    return output


def main() -> None:
    args = parse_args()
    root = Path(args.input_root).resolve()
    methods = {
        name: metric_rows(read_jsonl_gz(root / relative))
        for name, relative in METHOD_PATHS.items()
    }
    payload = {
        "protocol": (
            "paired full-corpus out-of-fold inference over identical positive queries; "
            "disease-tail cluster bootstrap and sign-flip randomization"
        ),
        "bootstrap_resamples": args.bootstrap_resamples,
        "signflip_resamples": args.signflip_resamples,
        "comparisons": {
            f"{left}_minus_{right}": compare_methods(
                methods[left],
                methods[right],
                bootstrap_resamples=args.bootstrap_resamples,
                signflip_resamples=args.signflip_resamples,
                seed=args.seed + 10000 * index,
            )
            for index, (left, right) in enumerate(COMPARISONS)
        },
    }
    write_json(Path(args.output).resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
