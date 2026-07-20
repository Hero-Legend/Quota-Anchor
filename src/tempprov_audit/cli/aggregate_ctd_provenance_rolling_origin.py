from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from tempprov_audit.cli.aggregate_ctd_provenance_neural_cv import (
    aggregate_one_run,
    summarize_seeds,
)
from tempprov_audit.cli.compare_ctd_provenance_neural_cv import compare_methods
from tempprov_audit.cli.run_ctd_provenance_lexical_baselines import (
    per_positive_metrics,
    write_json,
)


BGE = "bge_reranker_v2_m3_zero_shot"
BIOMED = "biomedbert_cross_encoder_3seed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate rolling-origin CTD provenance folds and paired inference."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-years", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 3407])
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--signflip-resamples", type=int, default=100000)
    parser.add_argument("--random-seed", type=int, default=20260624)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_paths(
    root: Path,
    test_years: list[int],
    folds: list[int],
    method: str,
    seed: int | None,
    filename: str,
) -> list[Path]:
    paths = []
    for year in test_years:
        for fold in folds:
            base = root / f"test_{year}" / f"fold_{fold}" / method
            paths.append(base / filename if seed is None else base / f"seed_{seed}" / filename)
    return paths


def averaged_positive_rows(
    prediction_paths_by_seed: list[list[Path]],
) -> dict[str, dict[str, Any]]:
    runs: list[dict[str, dict[str, Any]]] = []
    for paths in prediction_paths_by_seed:
        rows = [row for path in paths for row in read_json(path) if row["has_gold"]]
        runs.append({row["example_id"]: row for row in rows})
    example_ids = set(runs[0])
    if any(set(run) != example_ids for run in runs[1:]):
        raise RuntimeError("Positive examples differ across compared rolling-origin runs.")
    output: dict[str, dict[str, Any]] = {}
    for example_id in sorted(example_ids):
        source_rows = [run[example_id] for run in runs]
        tails = {str(row.get("tail_id") or example_id) for row in source_rows}
        if len(tails) != 1:
            raise RuntimeError(f"Tail IDs differ across runs for {example_id}.")
        metrics = [per_positive_metrics(row) for row in source_rows]
        output[example_id] = {
            "example_id": example_id,
            "tail_id": tails.pop(),
            "metrics": {
                metric: mean(run[metric] for run in metrics) for metric in metrics[0]
            },
        }
    return output


def aggregate_scope(
    root: Path,
    test_years: list[int],
    folds: list[int],
    seeds: list[int],
    *,
    bootstrap_resamples: int,
    signflip_resamples: int,
    random_seed: int,
) -> dict[str, Any]:
    bge_predictions = run_paths(root, test_years, folds, BGE, None, "test_predictions.json")
    bge_summaries = run_paths(root, test_years, folds, BGE, None, "summary.json")
    bge = aggregate_one_run(
        bge_predictions,
        bge_summaries,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=random_seed,
    )
    seed_runs = {}
    for seed_index, seed in enumerate(seeds):
        seed_runs[str(seed)] = aggregate_one_run(
            run_paths(root, test_years, folds, BIOMED, seed, "test_predictions.json"),
            run_paths(root, test_years, folds, BIOMED, seed, "summary.json"),
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=random_seed + seed_index + 1,
        )
    biomed = {"seed_runs": seed_runs, "across_seed_summary": summarize_seeds(seed_runs)}

    bge_rows = averaged_positive_rows([bge_predictions])
    biomed_rows = averaged_positive_rows(
        [run_paths(root, test_years, folds, BIOMED, seed, "test_predictions.json") for seed in seeds]
    )
    paired = compare_methods(
        biomed_rows,
        bge_rows,
        bootstrap_resamples=bootstrap_resamples,
        signflip_resamples=signflip_resamples,
        seed=random_seed + 1000,
    )
    return {
        "test_years": test_years,
        "folds": folds,
        "methods": {BGE: bge, BIOMED: biomed},
        "paired_biomedbert_minus_bge": paired,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.input_root).resolve()
    folds = list(range(args.fold_count))
    by_year = {
        str(year): aggregate_scope(
            root,
            [year],
            folds,
            args.seeds,
            bootstrap_resamples=args.bootstrap_resamples,
            signflip_resamples=args.signflip_resamples,
            random_seed=args.random_seed + 10000 * index,
        )
        for index, year in enumerate(args.test_years)
    }
    combined = aggregate_scope(
        root,
        args.test_years,
        folds,
        args.seeds,
        bootstrap_resamples=args.bootstrap_resamples,
        signflip_resamples=args.signflip_resamples,
        random_seed=args.random_seed + 90000,
    )
    payload = {
        "protocol": (
            "publication-year rolling origin; five disease-tail-disjoint folds; "
            "three-seed Cross-Encoder; paired tail-cluster inference"
        ),
        "test_years": args.test_years,
        "fold_count": args.fold_count,
        "seeds": args.seeds,
        "bootstrap_resamples": args.bootstrap_resamples,
        "signflip_resamples": args.signflip_resamples,
        "by_year": by_year,
        "combined": combined,
    }
    write_json(Path(args.output).resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
