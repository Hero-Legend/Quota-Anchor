from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from tempprov_audit.cli.compare_ctd_provenance_neural_cv import compare_methods
from tempprov_audit.cli.run_ctd_provenance_lexical_baselines import (
    per_positive_metrics,
    write_json,
)


METRICS = (
    "mrr",
    "map",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit dominant disease tails in CTD provenance evaluation."
    )
    parser.add_argument("--fivefold-root", required=True)
    parser.add_argument("--rolling-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 3407])
    parser.add_argument("--test-years", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--signflip-resamples", type=int, default=100000)
    parser.add_argument("--random-seed", type=int, default=20260630)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def averaged_metric_rows(
    prediction_runs: list[list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    indexed = [
        {row["example_id"]: row for row in run if row["has_gold"]}
        for run in prediction_runs
    ]
    example_ids = set(indexed[0])
    if any(set(run) != example_ids for run in indexed[1:]):
        raise RuntimeError("Positive example IDs differ across runs.")
    output: dict[str, dict[str, Any]] = {}
    for example_id in sorted(example_ids):
        source_rows = [run[example_id] for run in indexed]
        tails = {str(row.get("tail_id") or example_id) for row in source_rows}
        if len(tails) != 1:
            raise RuntimeError(f"Tail mismatch for {example_id}.")
        metrics = [per_positive_metrics(row) for row in source_rows]
        output[example_id] = {
            "example_id": example_id,
            "tail_id": tails.pop(),
            "metrics": {
                metric: mean(run[metric] for run in metrics) for metric in METRICS
            },
        }
    return output


def summarize_rows(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_tail: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows.values():
        by_tail[row["tail_id"]].append(row["metrics"])
    return {
        "positive_queries": len(rows),
        "tail_clusters": len(by_tail),
        "micro": {
            metric: mean(row["metrics"][metric] for row in rows.values())
            for metric in METRICS
        },
        "tail_macro": {
            metric: mean(
                mean(row[metric] for row in tail_rows)
                for tail_rows in by_tail.values()
            )
            for metric in METRICS
        },
    }


def remove_tails(
    rows: dict[str, dict[str, Any]], removed: set[str]
) -> dict[str, dict[str, Any]]:
    return {
        example_id: row
        for example_id, row in rows.items()
        if row["tail_id"] not in removed
    }


def audit_scope(
    method_rows: dict[str, dict[str, dict[str, Any]]],
    *,
    primary: str,
    comparators: list[str],
    bootstrap_resamples: int,
    signflip_resamples: int,
    seed: int,
) -> dict[str, Any]:
    tail_counts = Counter(
        row["tail_id"] for row in method_rows[primary].values()
    )
    ranked_tails = sorted(tail_counts, key=lambda tail: (-tail_counts[tail], tail))
    scenarios: dict[str, Any] = {}
    for scenario_index, count in enumerate((0, 1, 3, 5)):
        removed = set(ranked_tails[:count])
        filtered = {
            method: remove_tails(rows, removed)
            for method, rows in method_rows.items()
        }
        scenario = {
            "removed_tail_count": count,
            "removed_tails": [
                {"tail_id": tail, "positive_queries": tail_counts[tail]}
                for tail in ranked_tails[:count]
            ],
            "methods": {
                method: summarize_rows(rows) for method, rows in filtered.items()
            },
            "map_deltas": {
                f"{primary}_minus_{comparator}": (
                    summarize_rows(filtered[primary])["micro"]["map"]
                    - summarize_rows(filtered[comparator])["micro"]["map"]
                )
                for comparator in comparators
            },
        }
        if count in (0, 1):
            scenario["paired_comparisons"] = {
                f"{primary}_minus_{comparator}": compare_methods(
                    filtered[primary],
                    filtered[comparator],
                    bootstrap_resamples=bootstrap_resamples,
                    signflip_resamples=signflip_resamples,
                    seed=seed + 10000 * scenario_index + 1000 * index,
                )
                for index, comparator in enumerate(comparators)
            }
        scenarios[f"remove_top_{count}"] = scenario

    leave_one_out: dict[str, Any] = {}
    for comparator in comparators:
        deltas = []
        for tail in ranked_tails:
            left = summarize_rows(remove_tails(method_rows[primary], {tail}))["micro"]["map"]
            right = summarize_rows(remove_tails(method_rows[comparator], {tail}))["micro"]["map"]
            deltas.append((left - right, tail))
        leave_one_out[f"{primary}_minus_{comparator}"] = {
            "minimum_delta": min(deltas)[0],
            "minimum_delta_removed_tail": min(deltas)[1],
            "maximum_delta": max(deltas)[0],
            "maximum_delta_removed_tail": max(deltas)[1],
            "all_deltas_positive": all(delta > 0 for delta, _ in deltas),
        }
    return {
        "dominant_tails": [
            {"tail_id": tail, "positive_queries": tail_counts[tail]}
            for tail in ranked_tails[:10]
        ],
        "scenarios": scenarios,
        "leave_one_tail_out": leave_one_out,
    }


def load_fivefold_runs(
    root: Path, fold_count: int, method: str, seeds: list[int] | None
) -> list[list[dict[str, Any]]]:
    runs = [None] if seeds is None else seeds
    return [
        [
            row
            for fold in range(fold_count)
            for row in read_json(
                root
                / f"fold_{fold}"
                / method
                / ("test_predictions.json" if seed is None else f"seed_{seed}/test_predictions.json")
            )
        ]
        for seed in runs
    ]


def load_rolling_runs(
    root: Path,
    years: list[int],
    fold_count: int,
    method: str,
    seeds: list[int] | None,
) -> list[list[dict[str, Any]]]:
    runs = [None] if seeds is None else seeds
    return [
        [
            row
            for year in years
            for fold in range(fold_count)
            for row in read_json(
                root
                / f"test_{year}"
                / f"fold_{fold}"
                / method
                / ("test_predictions.json" if seed is None else f"seed_{seed}/test_predictions.json")
            )
        ]
        for seed in runs
    ]


def main() -> None:
    args = parse_args()
    fivefold_root = Path(args.fivefold_root).resolve()
    rolling_root = Path(args.rolling_root).resolve()
    fivefold_methods = {
        "bge": averaged_metric_rows(
            load_fivefold_runs(
                fivefold_root,
                args.fold_count,
                "bge_reranker_v2_m3_zero_shot",
                None,
            )
        ),
        "biomedbert": averaged_metric_rows(
            load_fivefold_runs(
                fivefold_root,
                args.fold_count,
                "biomedbert_cross_encoder_3seed",
                args.seeds,
            )
        ),
        "modernbert": averaged_metric_rows(
            load_fivefold_runs(
                fivefold_root,
                args.fold_count,
                "modernbert_cross_encoder_3seed",
                args.seeds,
            )
        ),
    }
    payload: dict[str, Any] = {
        "protocol": "positive-query disease-tail dominance audit",
        "fivefold": audit_scope(
            fivefold_methods,
            primary="biomedbert",
            comparators=["bge", "modernbert"],
            bootstrap_resamples=args.bootstrap_resamples,
            signflip_resamples=args.signflip_resamples,
            seed=args.random_seed,
        ),
        "rolling": {},
    }
    for index, years in enumerate([[year] for year in args.test_years] + [args.test_years]):
        methods = {
            "bge": averaged_metric_rows(
                load_rolling_runs(
                    rolling_root,
                    years,
                    args.fold_count,
                    "bge_reranker_v2_m3_zero_shot",
                    None,
                )
            ),
            "biomedbert": averaged_metric_rows(
                load_rolling_runs(
                    rolling_root,
                    years,
                    args.fold_count,
                    "biomedbert_cross_encoder_3seed",
                    args.seeds,
                )
            ),
        }
        key = "combined" if len(years) > 1 else str(years[0])
        payload["rolling"][key] = audit_scope(
            methods,
            primary="biomedbert",
            comparators=["bge"],
            bootstrap_resamples=args.bootstrap_resamples,
            signflip_resamples=args.signflip_resamples,
            seed=args.random_seed + 100000 * (index + 1),
        )
    write_json(Path(args.output).resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
