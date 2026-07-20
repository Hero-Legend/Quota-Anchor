from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from statistics import mean
from typing import Any

from tempprov_audit.cli.compare_ctd_provenance_neural_cv import compare_methods
from tempprov_audit.cli.run_ctd_provenance_lexical_baselines import (
    per_positive_metrics,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare frozen CIViC LLM rerankers with identical-query controls."
    )
    parser.add_argument("--queries", required=True)
    parser.add_argument("--medcpt-dual", required=True)
    parser.add_argument("--medcpt-ce", required=True)
    parser.add_argument("--llama", required=True)
    parser.add_argument("--deepseek", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--signflip-resamples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260707)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
    return {key: mean(row[key] for row in values) for key in values[0]}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (total - rank) * p_values[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def main() -> None:
    args = parse_args()
    queries = read_jsonl(Path(args.queries))
    methods = {
        "medcpt_dual": read_jsonl(Path(args.medcpt_dual)),
        "medcpt_ce": read_jsonl(Path(args.medcpt_ce)),
        "llama31_8b_nf4": read_jsonl(Path(args.llama)),
        "deepseek_v4_pro": read_jsonl(Path(args.deepseek)),
    }
    expected_ids = {row["example_id"] for row in queries}
    for name, rows in methods.items():
        row_ids = {row["example_id"] for row in rows}
        if len(rows) != len(queries) or row_ids != expected_ids:
            raise RuntimeError(
                f"{name} does not match the frozen query set: rows={len(rows)}, "
                f"ids={len(row_ids)}, expected={len(expected_ids)}"
            )

    metric_maps = {name: metric_rows(rows) for name, rows in methods.items()}
    comparisons = [
        ("deepseek_v4_pro", "medcpt_ce"),
        ("llama31_8b_nf4", "medcpt_ce"),
        ("deepseek_v4_pro", "llama31_8b_nf4"),
    ]
    paired = {
        f"{left}_minus_{right}": compare_methods(
            metric_maps[left],
            metric_maps[right],
            bootstrap_resamples=args.bootstrap_resamples,
            signflip_resamples=args.signflip_resamples,
            seed=args.seed + index * 10000,
        )
        for index, (left, right) in enumerate(comparisons)
    }
    map_p_values = {
        name: result["map"]["micro"]["paired_tail_signflip_p"]
        for name, result in paired.items()
    }
    payload = {
        "protocol": (
            "Frozen identical-query CIViC comparison over the MedCPT-dual Top-100 pool; "
            "disease-tail cluster bootstrap and paired tail-cluster sign-flip randomization"
        ),
        "queries": len(queries),
        "tail_clusters": len({str(row.get("tail_id") or row["example_id"]) for row in queries}),
        "methods": {name: point_metrics(rows) for name, rows in methods.items()},
        "paired_comparisons": paired,
        "map_signflip_holm_adjusted_p": holm_adjust(map_p_values),
        "bootstrap_resamples": args.bootstrap_resamples,
        "signflip_resamples": args.signflip_resamples,
        "seed": args.seed,
    }
    write_json(Path(args.output), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
