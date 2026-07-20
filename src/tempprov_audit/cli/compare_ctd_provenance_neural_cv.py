from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from tempprov_audit.cli.run_ctd_provenance_lexical_baselines import (
    per_positive_metrics,
    percentile,
    write_json,
)


METHODS = {
    "bge_reranker_v2_m3_zero_shot": {"seeds": None},
    "biomedbert_cross_encoder_3seed": {"seeds": [13, 42, 3407]},
    "modernbert_cross_encoder_3seed": {"seeds": [13, 42, 3407]},
}

COMPARISONS = (
    ("biomedbert_cross_encoder_3seed", "bge_reranker_v2_m3_zero_shot"),
    ("biomedbert_cross_encoder_3seed", "modernbert_cross_encoder_3seed"),
    ("modernbert_cross_encoder_3seed", "bge_reranker_v2_m3_zero_shot"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired tail-cluster inference for five-fold provenance ranking models."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--signflip-resamples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260624)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_path(root: Path, method: str, fold: int, seed: int | None) -> Path:
    base = root / f"fold_{fold}" / method
    return base / "test_predictions.json" if seed is None else base / f"seed_{seed}" / "test_predictions.json"


def load_method_rows(
    root: Path, method: str, fold_count: int, seeds: list[int] | None
) -> dict[str, dict[str, Any]]:
    runs = [None] if seeds is None else seeds
    by_run: list[dict[str, dict[str, Any]]] = []
    for seed in runs:
        rows = [
            row
            for fold in range(fold_count)
            for row in read_json(prediction_path(root, method, fold, seed))
            if row["has_gold"]
        ]
        by_run.append({row["example_id"]: row for row in rows})
    example_ids = set(by_run[0])
    if any(set(run) != example_ids for run in by_run[1:]):
        raise RuntimeError(f"Positive example IDs differ across runs for {method}.")

    output: dict[str, dict[str, Any]] = {}
    for example_id in sorted(example_ids):
        source_rows = [run[example_id] for run in by_run]
        if len({str(row.get("tail_id")) for row in source_rows}) != 1:
            raise RuntimeError(f"Tail ID differs across runs for {method}/{example_id}.")
        run_metrics = [per_positive_metrics(row) for row in source_rows]
        output[example_id] = {
            "example_id": example_id,
            "tail_id": str(source_rows[0].get("tail_id") or example_id),
            "metrics": {
                metric: mean(run[metric] for run in run_metrics)
                for metric in run_metrics[0]
            },
        }
    return output


def signflip_pvalue(
    clusters: list[list[float]], *, macro: bool, resamples: int, seed: int
) -> float:
    import numpy as np

    values = np.asarray(
        [mean(cluster) for cluster in clusters] if macro else [sum(cluster) for cluster in clusters],
        dtype=np.float64,
    )
    denominator = len(values) if macro else sum(len(cluster) for cluster in clusters)
    observed = abs(float(values.sum()) / denominator)
    rng = np.random.default_rng(seed)
    exceedances = 0
    completed = 0
    chunk_size = 10000
    while completed < resamples:
        current = min(chunk_size, resamples - completed)
        signs = rng.integers(0, 2, size=(current, len(values)), dtype=np.int8) * 2 - 1
        statistics = np.abs(signs @ values / denominator)
        exceedances += int(np.count_nonzero(statistics >= observed - 1e-15))
        completed += current
    return (exceedances + 1) / (resamples + 1)


def paired_summary(
    deltas_by_tail: dict[str, list[float]],
    *,
    bootstrap_resamples: int,
    signflip_resamples: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    tails = sorted(deltas_by_tail)
    clusters = [deltas_by_tail[tail] for tail in tails]
    micro_estimate = mean(value for cluster in clusters for value in cluster)
    macro_estimate = mean(mean(cluster) for cluster in clusters)
    cluster_sums = np.asarray([sum(cluster) for cluster in clusters], dtype=np.float64)
    cluster_counts = np.asarray([len(cluster) for cluster in clusters], dtype=np.float64)
    cluster_means = cluster_sums / cluster_counts
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0, len(clusters), size=(bootstrap_resamples, len(clusters))
    )
    micro_draws = (
        cluster_sums[sampled_indices].sum(axis=1)
        / cluster_counts[sampled_indices].sum(axis=1)
    ).tolist()
    macro_draws = cluster_means[sampled_indices].mean(axis=1).tolist()
    return {
        "examples": sum(len(cluster) for cluster in clusters),
        "tail_clusters": len(clusters),
        "micro": {
            "delta": micro_estimate,
            "bootstrap_95_ci": [percentile(micro_draws, 0.025), percentile(micro_draws, 0.975)],
            "paired_tail_signflip_p": signflip_pvalue(
                clusters, macro=False, resamples=signflip_resamples, seed=seed + 1
            ),
        },
        "tail_macro": {
            "delta": macro_estimate,
            "bootstrap_95_ci": [percentile(macro_draws, 0.025), percentile(macro_draws, 0.975)],
            "paired_tail_signflip_p": signflip_pvalue(
                clusters, macro=True, resamples=signflip_resamples, seed=seed + 2
            ),
        },
    }


def compare_methods(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    *,
    bootstrap_resamples: int,
    signflip_resamples: int,
    seed: int,
) -> dict[str, Any]:
    if set(left) != set(right):
        raise RuntimeError("Compared methods do not contain identical positive examples.")
    metric_names = list(next(iter(left.values()))["metrics"])
    output: dict[str, Any] = {}
    for metric_index, metric in enumerate(metric_names):
        deltas_by_tail: dict[str, list[float]] = {}
        for example_id in sorted(left):
            left_row, right_row = left[example_id], right[example_id]
            if left_row["tail_id"] != right_row["tail_id"]:
                raise RuntimeError(f"Tail ID mismatch for {example_id}.")
            deltas_by_tail.setdefault(left_row["tail_id"], []).append(
                left_row["metrics"][metric] - right_row["metrics"][metric]
            )
        output[metric] = paired_summary(
            deltas_by_tail,
            bootstrap_resamples=bootstrap_resamples,
            signflip_resamples=signflip_resamples,
            seed=seed + 100 * metric_index,
        )
    return output


def main() -> None:
    args = parse_args()
    root = Path(args.input_root).resolve()
    method_rows = {
        method: load_method_rows(root, method, args.fold_count, spec["seeds"])
        for method, spec in METHODS.items()
    }
    comparisons: dict[str, Any] = {}
    for comparison_index, (left, right) in enumerate(COMPARISONS):
        comparisons[f"{left}_minus_{right}"] = compare_methods(
            method_rows[left],
            method_rows[right],
            bootstrap_resamples=args.bootstrap_resamples,
            signflip_resamples=args.signflip_resamples,
            seed=args.seed + 10000 * comparison_index,
        )
    payload = {
        "protocol": (
            "paired five-fold OOF inference; three-seed metrics averaged per example; "
            "tail-cluster bootstrap and tail-cluster sign-flip randomization"
        ),
        "bootstrap_resamples": args.bootstrap_resamples,
        "signflip_resamples": args.signflip_resamples,
        "comparisons": comparisons,
    }
    write_json(Path(args.output).resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
