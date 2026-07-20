from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from tempprov_audit.cli.run_ctd_provenance_lexical_baselines import (
    summarize_ranking,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate outer tail-grouped provenance folds.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--methods", nargs="+", default=["bm25", "tfidf_cosine"])
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260623)
    return parser.parse_args()


def aggregate_confusions(items: list[dict[str, Any]]) -> dict[str, float | int]:
    tp = sum(item["tp"] for item in items)
    fp = sum(item["fp"] for item in items)
    tn = sum(item["tn"] for item in items)
    fn = sum(item["fn"] for item in items)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "fdr": 1.0 - precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "threshold_protocol": "selected independently on each outer fold's tail-disjoint dev set",
    }


def fold_variability(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    run_summary = json.loads((input_root / "summary.json").read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "protocol": "outer tail-grouped cross-validation with PMID-isolated train/dev/test",
        "fold_count": args.fold_count,
        "methods": {},
    }
    for method_index, method in enumerate(args.methods):
        combined_predictions: list[dict[str, Any]] = []
        fold_summaries: list[dict[str, Any]] = []
        for fold_index in range(args.fold_count):
            dataset = f"tail_grouped_cv_pmid_isolated/fold_{fold_index}"
            prediction_path = input_root / dataset / method / "test_predictions.json"
            predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
            combined_predictions.extend(predictions)
            fold_summaries.append(run_summary["datasets"][dataset][method])
        ranking = summarize_ranking(
            combined_predictions,
            args.bootstrap_resamples,
            args.seed + method_index,
        )
        policies = {}
        for policy in ("max_dev_f1", "dev_fdr_le_0p10_max_recall"):
            policies[policy] = aggregate_confusions(
                [fold["test_pair_classification"][policy] for fold in fold_summaries]
            )
        metric_names = ranking["metrics"].keys()
        payload["methods"][method] = {
            "combined_test_examples": len(combined_predictions),
            "combined_positive_examples": sum(item["has_gold"] for item in combined_predictions),
            "combined_unique_tail_clusters": len(
                {str(item.get("tail_id")) for item in combined_predictions}
            ),
            "ranking": ranking,
            "pair_classification": policies,
            "outer_fold_variability": {
                metric: fold_variability(
                    [
                        fold["ranking"]["test"]["metrics"][metric]["estimate"]
                        for fold in fold_summaries
                    ]
                )
                for metric in metric_names
            },
        }
    write_json(Path(args.output).resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
