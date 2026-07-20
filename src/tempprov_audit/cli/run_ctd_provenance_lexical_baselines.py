from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Callable


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-scale lexical CTD provenance baselines.")
    parser.add_argument(
        "--benchmark-root", default="outputs/ctd_provenance_benchmark"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["primary_balanced_pmid_isolated", "tail_disjoint_pmid_isolated"],
    )
    parser.add_argument(
        "--output-root", default="outputs/ctd_provenance_lexical_baselines"
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260623)
    return parser.parse_args()


_DOCUMENT_STORE_CACHE: dict[Path, dict[str, dict[str, Any]]] = {}


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def _compact_document_store(split_path: Path) -> Path | None:
    for parent in split_path.parents:
        candidate = parent / "documents.jsonl.gz"
        if candidate.exists():
            return candidate
    return None


def _load_document_store(path: Path) -> dict[str, dict[str, Any]]:
    if path not in _DOCUMENT_STORE_CACHE:
        with _open_text(path) as handle:
            _DOCUMENT_STORE_CACHE[path] = {
                row["document_key"]: row
                for line in handle
                if line.strip()
                for row in [json.loads(line)]
            }
    return _DOCUMENT_STORE_CACHE[path]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() and path.suffix == ".jsonl":
        compressed = path.with_suffix(path.suffix + ".gz")
        if compressed.exists():
            path = compressed
    with _open_text(path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    store_path = _compact_document_store(path)
    if store_path and rows and rows[0].get("candidates") and "text" not in rows[0]["candidates"][0]:
        store = _load_document_store(store_path)
        for row in rows:
            row["candidates"] = [
                {**store[candidate["document_key"]], **candidate}
                for candidate in row["candidates"]
            ]
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text or "").lower())


def bm25_scores(query: str, documents: list[str], k1: float = 1.2, b: float = 0.75) -> list[float]:
    tokenized = [tokenize(document) for document in documents]
    query_terms = list(dict.fromkeys(tokenize(query)))
    n_docs = len(tokenized)
    avg_len = mean([len(tokens) for tokens in tokenized]) if tokenized else 0.0
    document_frequency = Counter(
        term for tokens in tokenized for term in set(tokens) if term in query_terms
    )
    scores: list[float] = []
    for tokens in tokenized:
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            tf = frequencies[term]
            if not tf:
                continue
            df = document_frequency[term]
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            normalizer = tf + k1 * (1.0 - b + b * len(tokens) / max(avg_len, 1.0))
            score += idf * tf * (k1 + 1.0) / normalizer
        scores.append(score)
    return scores


def tfidf_scores(query: str, documents: list[str]) -> list[float]:
    tokenized = [tokenize(document) for document in documents]
    query_counts = Counter(tokenize(query))
    vocabulary = set(query_counts)
    n_docs = len(tokenized)
    document_frequency = Counter(
        term for tokens in tokenized for term in set(tokens) if term in vocabulary
    )
    idf = {
        term: math.log((1.0 + n_docs) / (1.0 + document_frequency[term])) + 1.0
        for term in vocabulary
    }
    query_vector = {term: count * idf[term] for term, count in query_counts.items()}
    query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
    scores: list[float] = []
    for tokens in tokenized:
        counts = Counter(tokens)
        dot = sum(query_vector[term] * counts[term] * idf[term] for term in vocabulary)
        doc_norm = math.sqrt(
            sum((counts[term] * idf[term]) ** 2 for term in vocabulary if counts[term])
        )
        scores.append(dot / (query_norm * doc_norm) if query_norm and doc_norm else 0.0)
    return scores


def rank_row(
    row: dict[str, Any], scorer: Callable[[str, list[str]], list[float]]
) -> dict[str, Any]:
    candidates = row["candidates"]
    scores = scorer(row["query_text"], [candidate.get("text", "") for candidate in candidates])
    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: (-item[1], item[0]["document_id"]),
    )
    return {
        "example_id": row["example_id"],
        "head_id": row.get("head_id"),
        "tail_id": row.get("tail_id"),
        "has_gold": bool(row["gold_document_ids"]),
        "gold_document_ids": row["gold_document_ids"],
        "top_score": ranked[0][1] if ranked else float("-inf"),
        "ranked_document_ids": [candidate["document_id"] for candidate, _ in ranked],
        "ranked_scores": [score for _, score in ranked],
    }


def per_positive_metrics(result: dict[str, Any]) -> dict[str, float]:
    gold = set(result["gold_document_ids"])
    ranked = result["ranked_document_ids"]
    relevant_ranks = [index + 1 for index, document_id in enumerate(ranked) if document_id in gold]
    reciprocal_rank = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
    average_precision = (
        mean(
            [
                sum(document_id in gold for document_id in ranked[:rank]) / rank
                for rank in relevant_ranks
            ]
        )
        if relevant_ranks
        else 0.0
    )
    output = {"mrr": reciprocal_rank, "map": average_precision}
    for cutoff in (1, 3, 5, 10):
        hits = sum(document_id in gold for document_id in ranked[:cutoff])
        output[f"recall_at_{cutoff}"] = hits / len(gold)
        dcg = sum(
            1.0 / math.log2(index + 2)
            for index, document_id in enumerate(ranked[:cutoff])
            if document_id in gold
        )
        ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(gold), cutoff)))
        output[f"ndcg_at_{cutoff}"] = dcg / ideal if ideal else 0.0
    return output


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = proportion * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_ranking(
    results: list[dict[str, Any]], bootstrap_resamples: int, seed: int
) -> dict[str, Any]:
    positive_results = [result for result in results if result["has_gold"]]
    row_metrics = [per_positive_metrics(result) for result in positive_results]
    keys = list(row_metrics[0]) if row_metrics else []
    rng = random.Random(seed)
    bootstraps = {key: [] for key in keys}
    macro_bootstraps = {key: [] for key in keys}
    cluster_rows: dict[str, list[dict[str, float]]] = {}
    for result, metrics in zip(positive_results, row_metrics):
        cluster_rows.setdefault(str(result.get("tail_id") or result["example_id"]), []).append(metrics)
    clusters = sorted(cluster_rows)
    for _ in range(bootstrap_resamples):
        sampled_clusters = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        sample = [metrics for cluster in sampled_clusters for metrics in cluster_rows[cluster]]
        for key in keys:
            bootstraps[key].append(mean(item[key] for item in sample))
            macro_bootstraps[key].append(
                mean(mean(item[key] for item in cluster_rows[cluster]) for cluster in sampled_clusters)
            )
    macro_estimates = {
        key: mean(mean(item[key] for item in cluster_rows[cluster]) for cluster in clusters)
        for key in keys
    }
    return {
        "positive_examples": len(row_metrics),
        "metrics": {
            key: {
                "estimate": mean(item[key] for item in row_metrics),
                "bootstrap_95_ci": [
                    percentile(bootstraps[key], 0.025),
                    percentile(bootstraps[key], 0.975),
                ],
            }
            for key in keys
        },
        "tail_macro_metrics": {
            key: {
                "estimate": macro_estimates[key],
                "bootstrap_95_ci": [
                    percentile(macro_bootstraps[key], 0.025),
                    percentile(macro_bootstraps[key], 0.975),
                ],
            }
            for key in keys
        },
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_unit": "tail_id_cluster",
        "bootstrap_clusters": len(clusters),
    }


def classification_metrics(results: list[dict[str, Any]], threshold: float) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    for result in results:
        predicted = result["top_score"] >= threshold
        gold = result["has_gold"]
        tp += predicted and gold
        fp += predicted and not gold
        tn += not predicted and not gold
        fn += not predicted and gold
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold,
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


def select_thresholds(dev_results: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    scores = sorted({result["top_score"] for result in dev_results}, reverse=True)
    thresholds = [float("inf"), *scores, float("-inf")]
    candidates = [classification_metrics(dev_results, threshold) for threshold in thresholds]
    max_f1 = max(
        candidates,
        key=lambda item: (item["f1"], item["recall"], item["precision"], item["threshold"]),
    )
    safe = [item for item in candidates if item["fdr"] <= 0.10]
    fdr_safe = max(
        safe,
        key=lambda item: (item["recall"], item["precision"], item["threshold"]),
    )
    return {"max_dev_f1": max_f1, "dev_fdr_le_0p10_max_recall": fdr_safe}


def run_dataset(
    benchmark_root: Path,
    output_root: Path,
    dataset: str,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    scorers = {"bm25": bm25_scores, "tfidf_cosine": tfidf_scores}
    summary: dict[str, Any] = {}
    for method_index, (method, scorer) in enumerate(scorers.items()):
        split_results: dict[str, list[dict[str, Any]]] = {}
        for split in SPLITS:
            rows = read_jsonl(benchmark_root / dataset / f"{split}.jsonl")
            split_results[split] = [rank_row(row, scorer) for row in rows]
            write_json(
                output_root / dataset / method / f"{split}_predictions.json",
                split_results[split],
            )
        policies = select_thresholds(split_results["dev"])
        summary[method] = {
            "ranking": {
                split: summarize_ranking(
                    split_results[split],
                    bootstrap_resamples,
                    seed + 1000 * method_index + split_index,
                )
                for split_index, split in enumerate(SPLITS)
            },
            "threshold_selection": policies,
            "test_pair_classification": {
                policy: classification_metrics(split_results["test"], values["threshold"])
                for policy, values in policies.items()
            },
        }
    return summary


def main() -> None:
    args = parse_args()
    benchmark_root = Path(args.benchmark_root).resolve()
    output_root = Path(args.output_root).resolve()
    payload = {
        "benchmark_root": str(benchmark_root),
        "datasets": {},
        "metric_note": (
            "Retrieval labels denote CTD citation provenance; unlabelled documents are not "
            "asserted to be biologically irrelevant. Pair thresholds are selected only on dev."
        ),
    }
    for dataset_index, dataset in enumerate(args.datasets):
        payload["datasets"][dataset] = run_dataset(
            benchmark_root,
            output_root,
            dataset,
            args.bootstrap_resamples,
            args.seed + dataset_index * 10000,
        )
    write_json(output_root / "summary.json", payload)
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
