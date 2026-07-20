from __future__ import annotations

import itertools
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from analyze_composite_decision_policy_frontier import selected_keys
from analyze_final_policy_baseline_controls import (
    AUDIT,
    HYBRID_ROOT,
    aggregate,
    best_hybrid_spec_by_dev,
    load_auditor_predictions,
    load_frontier_predictions,
    load_hybrid_scores,
    load_hybrid_seed_metrics,
    read_csv,
    score_policy,
    selected_kimi_prompt,
    test_auprc_for_hybrid,
    write_csv,
)
from analyze_frontier_graph_composite_audit import graph_router_paths


OUT = AUDIT / "strong_temporal_neural_rag_control"
POLICIES = ("top20", "top50", "top100")
EVIDENCE_SOURCES = ("graph_evidence", "kimi_evidence")
RETRIEVAL_ENCODERS = {"BGE_large_en_v15", "E5_large_v2"}


def best_neural_rag_spec_by_dev() -> dict[str, Any]:
    rows = [
        row
        for row in read_csv(HYBRID_ROOT / "hybrid_temporal_embedding_controls_aggregate.csv")
        if row["dataset"] == "tail_disjoint"
        and row["split"] == "dev"
        and row["encoder"] in RETRIEVAL_ENCODERS
        and row["input_mode"] == "pair_plus_retrieved"
    ]
    best = max(rows, key=lambda row: float(row["auprc_mean"]))
    spec: dict[str, Any] = {
        "family": "Strong temporal neural RAG control",
        "model": f"{best['encoder']}:{best['input_mode']}:{best['classifier']}",
        "label": f"neural_rag:{best['encoder']}:{best['input_mode']}:{best['classifier']}",
        "encoder": best["encoder"],
        "input_mode": best["input_mode"],
        "classifier": best["classifier"],
        "dev_auprc": float(best["auprc_mean"]),
        "score_column": "score_future_update",
    }
    spec["test_auprc"] = test_auprc_for_hybrid(spec)
    return spec


def primary_spec() -> dict[str, Any]:
    spec: dict[str, Any] = best_hybrid_spec_by_dev()
    spec["family"] = "Primary hybrid ranker"
    spec["test_auprc"] = test_auprc_for_hybrid(spec)
    return spec


def exact_sign_flip_p(deltas: list[float]) -> float:
    observed = abs(mean(deltas))
    extreme = total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        statistic = abs(mean(sign * delta for sign, delta in zip(signs, deltas)))
        extreme += int(statistic >= observed - 1e-15)
        total += 1
    return extreme / total


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    specs = [primary_spec(), best_neural_rag_spec_by_dev()]
    kimi_prompt = selected_kimi_prompt()
    kimi_rows = load_frontier_predictions(kimi_prompt["test_details_path"])
    graph_paths = graph_router_paths()
    seed_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for spec in specs:
        loader_spec = dict(spec)
        loader_spec["family"] = "Hybrid temporal+embedding"
        thresholds = load_hybrid_seed_metrics(loader_spec)
        for seed in sorted(set(thresholds) & set(graph_paths)):
            queue_scores = load_hybrid_scores(loader_spec, seed)
            graph_rows = load_auditor_predictions(graph_paths[seed])
            keys = sorted(set(queue_scores) & set(graph_rows) & set(kimi_rows))
            score_space = {key: queue_scores[key] for key in keys}
            total_gold = sum(queue_scores[key]["gold_future"] == 1 for key in keys)
            for policy in POLICIES:
                selected = selected_keys(score_space, thresholds[seed], policy)
                for evidence_source in EVIDENCE_SOURCES:
                    metrics, candidates = score_policy(
                        selected,
                        queue_scores,
                        graph_rows,
                        kimi_rows,
                        "kimi_relation",
                        evidence_source,
                        total_gold,
                    )
                    seed_rows.append(
                        {
                            "dataset": "tail_disjoint",
                            "queue_family": spec["family"],
                            "queue_model": spec["model"],
                            "queue_dev_auprc": spec["dev_auprc"],
                            "queue_test_auprc": spec["test_auprc"],
                            "seed": seed,
                            "policy": policy,
                            "relation_source": "kimi_relation",
                            "evidence_source": evidence_source,
                            "total_gold": total_gold,
                            **metrics,
                        }
                    )
                    for candidate in candidates:
                        candidate_rows.append(
                            {
                                "dataset": "tail_disjoint",
                                "queue_family": spec["family"],
                                "queue_model": spec["model"],
                                "seed": seed,
                                "policy": policy,
                                "relation_source": "kimi_relation",
                                "evidence_source": evidence_source,
                                **candidate,
                            }
                        )

    by_key = {
        (row["queue_family"], row["policy"], row["evidence_source"], int(row["seed"])): row
        for row in seed_rows
    }
    tests: list[dict[str, Any]] = []
    seeds = sorted({int(row["seed"]) for row in seed_rows})
    metrics = (
        "stage1_precision",
        "accepted_partial_precision",
        "accepted_partial_fdr",
        "accepted_partial_recall",
        "accepted_strict_recall",
    )
    for evidence_source in EVIDENCE_SOURCES:
        for metric in metrics:
            deltas = [
                float(by_key[("Primary hybrid ranker", "top50", evidence_source, seed)][metric])
                - float(by_key[("Strong temporal neural RAG control", "top50", evidence_source, seed)][metric])
                for seed in seeds
            ]
            tests.append(
                {
                    "dataset": "tail_disjoint",
                    "policy": "top50",
                    "evidence_source": evidence_source,
                    "comparison": "primary_hybrid_minus_strong_temporal_neural_rag",
                    "metric": metric,
                    "paired_seeds": len(seeds),
                    "mean_delta": mean(deltas),
                    "std_delta": stdev(deltas),
                    "two_sided_exact_sign_flip_p": "",
                    "inference_status": "descriptive_only_deterministic_queue_scores",
                    "min_seed_delta": min(deltas),
                    "max_seed_delta": max(deltas),
                }
            )
    return specs, seed_rows, aggregate(seed_rows, candidate_rows), tests


def summary(specs: list[dict[str, Any]], rows: list[dict[str, Any]], tests: list[dict[str, Any]]) -> str:
    lines = [
        "# Strong Temporal Neural-RAG End-to-End Control",
        "",
        "The control uses a dev-selected modern retrieval encoder (BGE/E5 candidate set), the same 36 pre-cutoff temporal features, the same retrieved packet, the same Kimi relation auditor, identical top-k reviewer budgets and either Kimi or graph evidence. This is a matched-compute modular control, not a parameter-count claim.",
        "",
        "## Dev-locked queue specifications",
        "",
        "| queue | model | dev AUPRC | test AUPRC |",
        "|---|---|---:|---:|",
    ]
    for spec in specs:
        lines.append(f"| {spec['family']} | {spec['model']} | {spec['dev_auprc']:.4f} | {spec['test_auprc']:.4f} |")
    lines += [
        "",
        "## End-to-end top-50 results",
        "",
        "| queue | evidence | stage-1 precision | accepted | accepted precision | FDR | partial recall | strict recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["policy"] != "top50":
            continue
        lines.append(
            f"| {row['queue_family']} | {row['evidence_source']} | "
            f"{float(row['stage1_precision_mean']):.4f} | {float(row['accepted_mean']):.1f} | "
            f"{float(row['accepted_partial_precision_mean']):.4f} | {float(row['accepted_partial_fdr_mean']):.4f} | "
            f"{float(row['accepted_partial_recall_mean']):.4f} | {float(row['accepted_strict_recall_mean']):.4f} |"
        )
    lines += [
        "",
        "## Paired primary-minus-control tests at top-50",
        "",
        "Queue scores and Kimi outputs are deterministic; repeated graph seeds are not independent replicates for queue-level precision, so no seed-level significance test is reported.",
        "",
        "| evidence | metric | mean delta | range across graph seeds |",
        "|---|---|---:|---:|",
    ]
    for row in tests:
        lines.append(
            f"| {row['evidence_source']} | {row['metric']} | {float(row['mean_delta']):.4f} | "
            f"[{float(row['min_seed_delta']):.4f}, {float(row['max_seed_delta']):.4f}] |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    specs, seed_rows, aggregate_rows, tests = build()
    write_csv(OUT / "strong_control_specs.csv", specs)
    write_csv(OUT / "strong_control_seed.csv", seed_rows)
    write_csv(OUT / "strong_control_aggregate.csv", aggregate_rows)
    write_csv(OUT / "strong_control_paired_tests.csv", tests)
    (OUT / "strong_control_summary.md").write_text(summary(specs, aggregate_rows, tests), encoding="utf-8")
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "description": "Matched end-to-end strong temporal neural-RAG control against the primary hybrid queue.",
                "retrieval_encoder_candidates": sorted(RETRIEVAL_ENCODERS),
                "selection": "highest tail-disjoint dev AUPRC among BGE/E5 pair-plus-retrieved temporal-fusion controls",
                "seeds": sorted({int(row["seed"]) for row in seed_rows}),
                "policies": list(POLICIES),
                "evidence_sources": list(EVIDENCE_SOURCES),
                "evaluation_cells": len(seed_rows),
                "bootstrap_resamples_per_aggregate_policy": 5000,
                "api_calls": 0,
                "claim_boundary": "matched modular control; no parameter-count equivalence claim",
                "inference_note": "Queue scores and Kimi outputs are deterministic. Seed-level sign-flip p-values are suppressed because repeated queue rows are not independent stochastic replicates; candidate-level bootstrap intervals remain in the aggregate CSV.",
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(OUT / "strong_control_summary.md")


if __name__ == "__main__":
    main()
