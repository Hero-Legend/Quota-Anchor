from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from tempprov_audit.cli.aggregate_ctd_provenance_cv import aggregate_confusions
from tempprov_audit.cli.run_ctd_provenance_lexical_baselines import (
    summarize_ranking,
    write_json,
)


METHODS = {
    "bge_reranker_v2_m3_zero_shot": {"seeded": False},
    "biomedbert_cross_encoder_3seed": {"seeded": True},
    "modernbert_cross_encoder_3seed": {"seeded": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate neural provenance outer folds.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 3407])
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260623)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_one_run(
    prediction_paths: list[Path],
    summary_paths: list[Path],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    predictions = [item for path in prediction_paths for item in read_json(path)]
    summaries = [read_json(path) for path in summary_paths]
    policies = {
        policy: aggregate_confusions(
            [summary["test_pair_classification"][policy] for summary in summaries]
        )
        for policy in ("max_dev_f1", "dev_fdr_le_0p10_max_recall")
    }
    return {
        "combined_test_examples": len(predictions),
        "combined_positive_examples": sum(item["has_gold"] for item in predictions),
        "combined_unique_tail_clusters": len({str(item.get("tail_id")) for item in predictions}),
        "ranking": summarize_ranking(
            predictions, bootstrap_resamples, bootstrap_seed
        ),
        "pair_classification": policies,
    }


def summarize_seeds(seed_runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for scope in ("metrics", "tail_macro_metrics"):
        summary[scope] = {}
        metric_names = next(iter(seed_runs.values()))["ranking"][scope]
        for metric in metric_names:
            values = [run["ranking"][scope][metric]["estimate"] for run in seed_runs.values()]
            summary[scope][metric] = {
                "mean": mean(values),
                "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
                "minimum": min(values),
                "maximum": max(values),
            }
    summary["pair_classification"] = {}
    for policy in ("max_dev_f1", "dev_fdr_le_0p10_max_recall"):
        summary["pair_classification"][policy] = {}
        for metric in ("precision", "fdr", "recall", "f1", "specificity"):
            values = [run["pair_classification"][policy][metric] for run in seed_runs.values()]
            summary["pair_classification"][policy][metric] = {
                "mean": mean(values),
                "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
            }
    return summary


def main() -> None:
    args = parse_args()
    root = Path(args.input_root).resolve()
    payload: dict[str, Any] = {
        "protocol": "five-fold tail-grouped, PMID-isolated neural evaluation",
        "fold_count": args.fold_count,
        "methods": {},
    }
    for method_index, (method, spec) in enumerate(METHODS.items()):
        if not spec["seeded"]:
            payload["methods"][method] = aggregate_one_run(
                [root / f"fold_{fold}" / method / "test_predictions.json" for fold in range(args.fold_count)],
                [root / f"fold_{fold}" / method / "summary.json" for fold in range(args.fold_count)],
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed + method_index,
            )
            continue
        seed_runs = {}
        for seed_index, seed in enumerate(args.seeds):
            seed_runs[str(seed)] = aggregate_one_run(
                [
                    root / f"fold_{fold}" / method / f"seed_{seed}" / "test_predictions.json"
                    for fold in range(args.fold_count)
                ],
                [
                    root / f"fold_{fold}" / method / f"seed_{seed}" / "summary.json"
                    for fold in range(args.fold_count)
                ],
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed + 100 * method_index + seed_index,
            )
        payload["methods"][method] = {
            "seed_runs": seed_runs,
            "across_seed_summary": summarize_seeds(seed_runs),
        }
    write_json(Path(args.output).resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
