from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

from tempprov_audit.cli.run_ctd_provenance_lexical_baselines import write_json


POLICIES = ("max_f1", "empirical_fdr_le_0p10", "wilson95_fdr_le_0p10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate leave-one-tail-fold-out safety calibration from OOF provenance scores."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="biomedbert_cross_encoder_3seed")
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 3407])
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260624)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def top_statistics(scores: list[float], cutoff: int) -> tuple[float, float]:
    values = scores[:cutoff]
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


def build_feature_row(seed_rows: list[dict[str, Any]], fold: int) -> dict[str, Any]:
    reference = seed_rows[0]
    top1s: list[float] = []
    margins: list[float] = []
    top3_means: list[float] = []
    top5_means: list[float] = []
    features: list[float] = []
    for row in seed_rows:
        scores = [float(value) for value in row["ranked_scores"]]
        top1 = scores[0]
        top2 = scores[1] if len(scores) > 1 else scores[0]
        top3_mean, top3_std = top_statistics(scores, 3)
        top5_mean, top5_std = top_statistics(scores, 5)
        top1s.append(top1)
        margins.append(top1 - top2)
        top3_means.append(top3_mean)
        top5_means.append(top5_mean)
        features.extend((top1, top1 - top2, top3_mean, top3_std, top5_mean, top5_std))
    features.extend(
        (
            mean(top1s),
            pstdev(top1s),
            min(top1s),
            max(top1s),
            mean(margins),
            pstdev(margins),
            mean(top3_means),
            mean(top5_means),
            math.log1p(len(reference["ranked_document_ids"])),
        )
    )
    return {
        "fold": fold,
        "example_id": reference["example_id"],
        "tail_id": str(reference.get("tail_id") or reference["example_id"]),
        "label": int(bool(reference["has_gold"])),
        "ensemble_score": mean(top1s),
        "features": features,
    }


def load_fold_rows(root: Path, method: str, fold: int, seeds: list[int]) -> list[dict[str, Any]]:
    by_seed = [
        read_json(root / f"fold_{fold}" / method / f"seed_{seed}" / "test_predictions.json")
        for seed in seeds
    ]
    maps = [{row["example_id"]: row for row in rows} for rows in by_seed]
    example_ids = list(maps[0])
    if any(set(mapping) != set(example_ids) for mapping in maps[1:]):
        raise RuntimeError(f"Seed prediction example IDs differ in fold {fold}.")
    return [build_feature_row([mapping[example_id] for mapping in maps], fold) for example_id in example_ids]


def confusion(labels: list[int], predictions: list[bool]) -> dict[str, float | int]:
    tp = sum(prediction and label for label, prediction in zip(labels, predictions))
    fp = sum(prediction and not label for label, prediction in zip(labels, predictions))
    tn = sum(not prediction and not label for label, prediction in zip(labels, predictions))
    fn = sum(not prediction and label for label, prediction in zip(labels, predictions))
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
    }


def wilson_upper(errors: int, predictions: int, z: float = 1.959963984540054) -> float:
    if predictions == 0:
        return 0.0
    proportion = errors / predictions
    z2 = z * z
    denominator = 1.0 + z2 / predictions
    centre = proportion + z2 / (2.0 * predictions)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / predictions + z2 / (4.0 * predictions * predictions)
    )
    return (centre + radius) / denominator


def threshold_candidates(scores: list[float]) -> list[float]:
    return [float("inf"), *sorted(set(scores), reverse=True), float("-inf")]


def select_threshold(scores: list[float], labels: list[int], policy: str) -> dict[str, Any]:
    candidates = []
    for threshold in threshold_candidates(scores):
        metrics = confusion(labels, [score >= threshold for score in scores])
        predicted_count = metrics["tp"] + metrics["fp"]
        metrics["threshold"] = threshold
        metrics["wilson95_fdr_upper"] = wilson_upper(int(metrics["fp"]), int(predicted_count))
        candidates.append(metrics)
    if policy == "max_f1":
        return max(
            candidates,
            key=lambda item: (item["f1"], item["recall"], item["precision"], item["threshold"]),
        )
    if policy == "empirical_fdr_le_0p10":
        eligible = [item for item in candidates if item["fdr"] <= 0.10]
    elif policy == "wilson95_fdr_le_0p10":
        eligible = [item for item in candidates if item["wilson95_fdr_upper"] <= 0.10]
    else:
        raise ValueError(f"Unknown threshold policy: {policy}")
    return max(
        eligible,
        key=lambda item: (item["recall"], item["precision"], item["threshold"]),
    )


def choose_logistic_c(rows_by_fold: dict[int, list[dict[str, Any]]], source_folds: list[int]) -> float:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    candidates = (0.01, 0.1, 1.0, 10.0)
    scores: dict[float, list[float]] = {value: [] for value in candidates}
    for validation_fold in source_folds:
        training = [
            row
            for fold in source_folds
            if fold != validation_fold
            for row in rows_by_fold[fold]
        ]
        validation = rows_by_fold[validation_fold]
        x_train = np.asarray([row["features"] for row in training], dtype=np.float64)
        y_train = np.asarray([row["label"] for row in training], dtype=np.int64)
        x_validation = np.asarray([row["features"] for row in validation], dtype=np.float64)
        y_validation = np.asarray([row["label"] for row in validation], dtype=np.int64)
        for c_value in candidates:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c_value, max_iter=5000, solver="lbfgs"),
            )
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_validation)[:, 1]
            scores[c_value].append(float(average_precision_score(y_validation, probabilities)))
    return max(candidates, key=lambda value: (mean(scores[value]), -value))


def logistic_oof_and_target(
    rows_by_fold: dict[int, list[dict[str, Any]]], target_fold: int
) -> tuple[list[dict[str, Any]], list[float], list[float], float]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    source_folds = sorted(set(rows_by_fold) - {target_fold})
    c_value = choose_logistic_c(rows_by_fold, source_folds)
    source_rows: list[dict[str, Any]] = []
    source_probabilities: list[float] = []
    for validation_fold in source_folds:
        training = [
            row
            for fold in source_folds
            if fold != validation_fold
            for row in rows_by_fold[fold]
        ]
        validation = rows_by_fold[validation_fold]
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=c_value, max_iter=5000, solver="lbfgs")
        )
        model.fit(
            np.asarray([row["features"] for row in training], dtype=np.float64),
            np.asarray([row["label"] for row in training], dtype=np.int64),
        )
        source_rows.extend(validation)
        source_probabilities.extend(
            model.predict_proba(
                np.asarray([row["features"] for row in validation], dtype=np.float64)
            )[:, 1].tolist()
        )
    pooled_source = [row for fold in source_folds for row in rows_by_fold[fold]]
    final_model = make_pipeline(
        StandardScaler(), LogisticRegression(C=c_value, max_iter=5000, solver="lbfgs")
    )
    final_model.fit(
        np.asarray([row["features"] for row in pooled_source], dtype=np.float64),
        np.asarray([row["label"] for row in pooled_source], dtype=np.int64),
    )
    target_rows = rows_by_fold[target_fold]
    target_probabilities = final_model.predict_proba(
        np.asarray([row["features"] for row in target_rows], dtype=np.float64)
    )[:, 1].tolist()
    return source_rows, source_probabilities, target_probabilities, c_value


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    position = proportion * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_bootstrap(
    records: list[dict[str, Any]], resamples: int, seed: int
) -> dict[str, Any]:
    import random

    clusters: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        clusters.setdefault(record["tail_id"], []).append(record)
    cluster_ids = sorted(clusters)
    rng = random.Random(seed)
    draws = {metric: [] for metric in ("precision", "fdr", "recall", "f1", "specificity")}
    for _ in range(resamples):
        sampled = [cluster_ids[rng.randrange(len(cluster_ids))] for _ in cluster_ids]
        rows = [row for cluster_id in sampled for row in clusters[cluster_id]]
        metrics = confusion(
            [row["label"] for row in rows], [bool(row["prediction"]) for row in rows]
        )
        for metric in draws:
            draws[metric].append(float(metrics[metric]))
    return {
        "unit": "tail_id_cluster",
        "clusters": len(cluster_ids),
        "resamples": resamples,
        "intervals": {
            metric: [percentile(values, 0.025), percentile(values, 0.975)]
            for metric, values in draws.items()
        },
    }


def main() -> None:
    args = parse_args()
    root = Path(args.input_root).resolve()
    rows_by_fold = {
        fold: load_fold_rows(root, args.method, fold, args.seeds)
        for fold in range(args.fold_count)
    }
    outputs: dict[str, dict[str, list[dict[str, Any]]]] = {
        "ensemble_max": {policy: [] for policy in POLICIES},
        "logistic_stack": {policy: [] for policy in POLICIES},
    }
    fold_audit: dict[str, Any] = {}
    for target_fold in range(args.fold_count):
        source_folds = sorted(set(rows_by_fold) - {target_fold})
        source_rows = [row for fold in source_folds for row in rows_by_fold[fold]]
        target_rows = rows_by_fold[target_fold]
        fold_audit[str(target_fold)] = {"source_folds": source_folds, "methods": {}}

        method_scores = {
            "ensemble_max": (
                source_rows,
                [row["ensemble_score"] for row in source_rows],
                [row["ensemble_score"] for row in target_rows],
                None,
            )
        }
        logistic_source_rows, logistic_source_scores, logistic_target_scores, c_value = (
            logistic_oof_and_target(rows_by_fold, target_fold)
        )
        method_scores["logistic_stack"] = (
            logistic_source_rows,
            logistic_source_scores,
            logistic_target_scores,
            c_value,
        )

        for method, (calibration_rows, calibration_scores, target_scores, selected_c) in method_scores.items():
            calibration_labels = [row["label"] for row in calibration_rows]
            fold_audit[str(target_fold)]["methods"][method] = {
                "selected_logistic_c": selected_c,
                "policies": {},
            }
            for policy in POLICIES:
                selected = select_threshold(calibration_scores, calibration_labels, policy)
                predictions = [score >= selected["threshold"] for score in target_scores]
                target_metrics = confusion([row["label"] for row in target_rows], predictions)
                fold_audit[str(target_fold)]["methods"][method]["policies"][policy] = {
                    "calibration": selected,
                    "target": target_metrics,
                }
                outputs[method][policy].extend(
                    {
                        "fold": target_fold,
                        "example_id": row["example_id"],
                        "tail_id": row["tail_id"],
                        "label": row["label"],
                        "score": score,
                        "prediction": prediction,
                    }
                    for row, score, prediction in zip(target_rows, target_scores, predictions)
                )

    aggregate: dict[str, Any] = {}
    for method, policies in outputs.items():
        aggregate[method] = {}
        for policy_index, (policy, records) in enumerate(policies.items()):
            metrics = confusion(
                [row["label"] for row in records], [bool(row["prediction"]) for row in records]
            )
            metrics["cluster_bootstrap"] = cluster_bootstrap(
                records,
                args.bootstrap_resamples,
                args.bootstrap_seed + 100 * list(outputs).index(method) + policy_index,
            )
            aggregate[method][policy] = metrics

    payload = {
        "protocol": "leave-one-tail-fold-out meta-calibration using only out-of-fold base scores",
        "method": args.method,
        "fold_count": args.fold_count,
        "seeds": args.seeds,
        "feature_count": len(rows_by_fold[0][0]["features"]),
        "fold_audit": fold_audit,
        "aggregate": aggregate,
    }
    write_json(Path(args.output).resolve(), payload)
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
