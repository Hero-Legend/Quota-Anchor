from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "benchmark_audit" / "retrieval_text_baselines"

DATASETS = {
    "primary_balanced": ROOT
    / "outputs"
    / "temporal_validation"
    / "model_inputs"
    / "ctd_202111_primary_balanced_temporal_update_clean_20260430.model_input",
    "tail_disjoint": ROOT
    / "outputs"
    / "temporal_validation"
    / "model_inputs"
    / "tail_disjoint"
    / "ctd_202111_primary_balanced_temporal_update_tail_disjoint_clean_20260430.model_input",
}
SPLITS = ("train", "dev", "test")
SEEDS = (13, 42, 101, 777, 1337, 2024, 2025, 3407, 8128, 9001)
POSITIVE = {"future_ctd_update_direct", "future_ctd_update_inferred"}
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-]{1,}", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def sentence_text(sentence: Any) -> str:
    if isinstance(sentence, dict):
        return str(sentence.get("text", ""))
    return str(sentence)


def pair_terms(row: dict[str, Any]) -> list[str]:
    return tokens(
        " ".join(
            [
                str(row.get("head_text", "")),
                str(row.get("head_id", "")),
                str(row.get("tail_text", "")),
                str(row.get("tail_id", "")),
            ]
        )
    )


def retrieved_sentences(row: dict[str, Any], k: int = 8) -> list[tuple[int, str, float]]:
    query = Counter(pair_terms(row))
    if not query:
        return []
    scored: list[tuple[int, str, float]] = []
    for idx, sent in enumerate(row.get("sentences", [])):
        text = sentence_text(sent)
        sent_tokens = tokens(text)
        if not sent_tokens:
            continue
        counts = Counter(sent_tokens)
        overlap = sum(min(counts[token], count) for token, count in query.items())
        phrase_bonus = 0.0
        lowered = text.lower()
        if str(row.get("head_text", "")).lower() in lowered:
            phrase_bonus += 2.0
        if str(row.get("tail_text", "")).lower() in lowered:
            phrase_bonus += 2.0
        length_penalty = math.log1p(len(sent_tokens))
        score = (overlap + phrase_bonus) / length_penalty
        scored.append((idx, text, score))
    scored.sort(key=lambda item: (item[2], -item[0]), reverse=True)
    return scored[:k]


def input_text(row: dict[str, Any], mode: str) -> str:
    pair = (
        f"chemical name {row.get('head_text', '')} chemical id {row.get('head_id', '')} "
        f"disease name {row.get('tail_text', '')} disease id {row.get('tail_id', '')}"
    )
    full_text = str(row.get("document_text", ""))
    retrieved = "\n".join(text for _, text, _ in retrieved_sentences(row, k=8))
    if mode == "pair_only":
        return pair
    if mode == "full_text":
        return full_text
    if mode == "retrieved_snippets":
        return retrieved
    if mode == "pair_plus_retrieved":
        return pair + "\n" + retrieved
    if mode == "pair_plus_full_text":
        return pair + "\n" + full_text
    raise ValueError(mode)


def labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([1 if str(row.get("gold_relation", "")) in POSITIVE else 0 for row in rows], dtype=int)


def relation_labels(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("gold_relation", "")) for row in rows]


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, max_fdr: float | None = None) -> dict[str, float]:
    best_rank: tuple[float, ...] | None = None
    best = {"threshold": float("inf"), "precision": 0.0, "recall": 0.0, "f1": 0.0, "fdr": 0.0}
    for threshold in [float("inf"), *sorted(set(scores.tolist()), reverse=True)]:
        pred = scores >= threshold
        tp = int(((y_true == 1) & pred).sum())
        fp = int(((y_true == 0) & pred).sum())
        fn = int(((y_true == 1) & ~pred).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fdr = fp / (tp + fp) if tp + fp else 0.0
        if max_fdr is not None and fdr > max_fdr:
            continue
        rank = (recall, precision, f1, threshold) if max_fdr is not None else (f1, recall, precision, threshold)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1, "fdr": fdr}
    return best


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    tp = int(((y_true == 1) & pred).sum())
    fp = int(((y_true == 0) & pred).sum())
    fn = int(((y_true == 1) & ~pred).sum())
    tn = int(((y_true == 0) & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fdr": fp / (tp + fp) if tp + fp else 0.0,
    }


def score_summary(y_true: np.ndarray, scores: np.ndarray, thresholds: dict[str, dict[str, float]]) -> dict[str, float]:
    out = {
        "auprc": float(average_precision_score(y_true, scores)) if len(set(y_true.tolist())) > 1 else 0.0,
        "auroc": float(roc_auc_score(y_true, scores)) if len(set(y_true.tolist())) > 1 else 0.0,
    }
    for name, threshold in thresholds.items():
        out[f"{name}_threshold"] = threshold["threshold"]
        for key, value in threshold_metrics(y_true, scores, threshold["threshold"]).items():
            out[f"{name}_{key}"] = value
    return out


def model_specs(seed: int) -> dict[str, Any]:
    vectorizer = HashingVectorizer(
        n_features=2**18,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9_\-]{1,}\b",
    )
    return {
        "tfidf_logistic_l2": make_pipeline(
            vectorizer,
            TfidfTransformer(norm="l2", use_idf=True),
            LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=seed),
        ),
        "tfidf_sgd_logloss": make_pipeline(
            vectorizer,
            TfidfTransformer(norm="l2", use_idf=True),
            SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=1e-5,
                l1_ratio=0.05,
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
                tol=1e-4,
            ),
        ),
    }


def run_dataset(dataset: str, prefix: Path) -> list[dict[str, Any]]:
    rows_by_split = {split: read_jsonl(Path(f"{prefix}.{split}.jsonl")) for split in SPLITS}
    y_by_split = {split: labels(rows) for split, rows in rows_by_split.items()}
    modes = ("pair_only", "full_text", "retrieved_snippets", "pair_plus_retrieved", "pair_plus_full_text")
    seed_rows: list[dict[str, Any]] = []
    retrieval_audit = []
    for split in ("dev", "test"):
        for row, y_true in zip(rows_by_split[split], y_by_split[split], strict=True):
            retrieved = retrieved_sentences(row, k=8)
            gold = {int(item) for item in row.get("gold_evidence_sentence_ids", [])}
            pred = {idx for idx, _, _ in retrieved}
            retrieval_audit.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "example_id": row.get("example_id"),
                    "gold_future_update": int(y_true),
                    "gold_evidence_count": len(gold),
                    "retrieved_count": len(pred),
                    "retrieved_gold_overlap": len(gold & pred),
                    "retrieved_evidence_recall": len(gold & pred) / len(gold) if gold else 0.0,
                    "retrieved_evidence_precision": len(gold & pred) / len(pred) if pred else 0.0,
                }
            )
    write_csv(OUT / f"{dataset}_retrieval_audit.csv", retrieval_audit)

    for mode in modes:
        text_by_split = {split: [input_text(row, mode) for row in rows] for split, rows in rows_by_split.items()}
        for seed in SEEDS:
            for model_name, model in model_specs(seed).items():
                model.fit(text_by_split["train"], y_by_split["train"])
                dev_scores = model.predict_proba(text_by_split["dev"])[:, 1]
                test_scores = model.predict_proba(text_by_split["test"])[:, 1]
                thresholds = {
                    "best_f1": choose_threshold(y_by_split["dev"], dev_scores),
                    "fdr_010": choose_threshold(y_by_split["dev"], dev_scores, max_fdr=0.10),
                    "fdr_020": choose_threshold(y_by_split["dev"], dev_scores, max_fdr=0.20),
                }
                for split, scores in (("dev", dev_scores), ("test", test_scores)):
                    row: dict[str, Any] = {
                        "dataset": dataset,
                        "model": model_name,
                        "input_mode": mode,
                        "seed": seed,
                        "split": split,
                        "n": len(y_by_split[split]),
                        "n_positive": int(y_by_split[split].sum()),
                    }
                    row.update(score_summary(y_by_split[split], scores, thresholds))
                    seed_rows.append(row)
                prediction_rows = []
                for raw, y_true, relation, score in zip(
                    rows_by_split["test"],
                    y_by_split["test"],
                    relation_labels(rows_by_split["test"]),
                    test_scores,
                    strict=True,
                ):
                    prediction_rows.append(
                        {
                            "dataset": dataset,
                            "model": model_name,
                            "input_mode": mode,
                            "seed": seed,
                            "example_id": raw.get("example_id"),
                            "pair_id": raw.get("pair_id"),
                            "gold_relation": relation,
                            "gold_future_update": int(y_true),
                            "score_future_update": float(score),
                        }
                    )
                write_csv(OUT / "predictions" / dataset / model_name / mode / f"seed{seed}_test_scores.csv", prediction_rows)
    return seed_rows


def aggregate(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "auprc",
        "auroc",
        "best_f1_f1",
        "best_f1_precision",
        "best_f1_recall",
        "best_f1_fdr",
        "fdr_010_recall",
        "fdr_010_precision",
        "fdr_010_f1",
        "fdr_010_fdr",
        "fdr_020_recall",
        "fdr_020_precision",
        "fdr_020_f1",
        "fdr_020_fdr",
    ]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(row["dataset"], row["model"], row["input_mode"], row["split"])].append(row)
    rows = []
    for (dataset, model, mode, split), group in sorted(grouped.items()):
        out: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "input_mode": mode,
            "split": split,
            "seeds": len(group),
            "n": group[0]["n"],
            "n_positive": group[0]["n_positive"],
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        rows.append(out)
    return rows


def aggregate_retrieval() -> list[dict[str, Any]]:
    rows = []
    for path in OUT.glob("*_retrieval_audit.csv"):
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["split"])].append(row)
    out = []
    for (dataset, split), group in sorted(grouped.items()):
        for label, keep in (
            ("all", group),
            ("future_positive", [row for row in group if row["gold_future_update"] == "1"]),
        ):
            out.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "subset": label,
                    "n": len(keep),
                    "mean_retrieved_evidence_recall": mean(float(row["retrieved_evidence_recall"]) for row in keep) if keep else 0.0,
                    "mean_retrieved_evidence_precision": mean(float(row["retrieved_evidence_precision"]) for row in keep) if keep else 0.0,
                    "mean_gold_evidence_count": mean(float(row["gold_evidence_count"]) for row in keep) if keep else 0.0,
                    "mean_retrieved_count": mean(float(row["retrieved_count"]) for row in keep) if keep else 0.0,
                }
            )
    return out


def fmt(row: dict[str, Any], metric: str) -> str:
    return f"{float(row[f'{metric}_mean']):.4f} +/- {float(row[f'{metric}_std']):.4f}"


def write_summary(aggregate_rows: list[dict[str, Any]], retrieval_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Retrieval-Style Text Baselines",
        "",
        "Full-split text baselines trained on the temporal CTD train split, thresholded on dev and evaluated on test. Retrieved-snippet modes use a deterministic pair-term sentence retriever over the provided pre-cut evidence packet.",
        "",
        "## Test Metrics",
        "",
        "| dataset | model | input | seeds | AUPRC | AUROC | best-F1 | FDR<=0.10 recall | FDR<=0.20 recall |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        if row["split"] != "test":
            continue
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['input_mode']} | {row['seeds']} | "
            f"{fmt(row, 'auprc')} | {fmt(row, 'auroc')} | {fmt(row, 'best_f1_f1')} | "
            f"{fmt(row, 'fdr_010_recall')} | {fmt(row, 'fdr_020_recall')} |"
        )
    lines += [
        "",
        "## Retriever Evidence Audit",
        "",
        "| dataset | split | subset | n | evidence recall | evidence precision | gold evidence count | retrieved count |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in retrieval_rows:
        lines.append(
            f"| {row['dataset']} | {row['split']} | {row['subset']} | {row['n']} | "
            f"{float(row['mean_retrieved_evidence_recall']):.4f} | "
            f"{float(row['mean_retrieved_evidence_precision']):.4f} | "
            f"{float(row['mean_gold_evidence_count']):.2f} | "
            f"{float(row['mean_retrieved_count']):.2f} |"
        )
    (OUT / "retrieval_text_baselines_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict[str, Any]] = []
    for dataset, prefix in DATASETS.items():
        seed_rows.extend(run_dataset(dataset, prefix))
    aggregate_rows = aggregate(seed_rows)
    retrieval_rows = aggregate_retrieval()
    write_csv(OUT / "retrieval_text_baselines_seed.csv", seed_rows)
    write_csv(OUT / "retrieval_text_baselines_aggregate.csv", aggregate_rows)
    write_csv(OUT / "retriever_evidence_audit_aggregate.csv", retrieval_rows)
    write_summary(aggregate_rows, retrieval_rows)
    write_json(
        OUT / "manifest.json",
        {
            "datasets": {key: str(value.relative_to(ROOT)) for key, value in DATASETS.items()},
            "seeds": SEEDS,
            "input_modes": ["pair_only", "full_text", "retrieved_snippets", "pair_plus_retrieved", "pair_plus_full_text"],
            "models": list(model_specs(SEEDS[0])),
            "retrieval": "Deterministic pair-term sentence retrieval over pre-cut evidence packets; no gold evidence IDs are used for retrieval.",
        },
    )
    print(OUT / "retrieval_text_baselines_summary.md")


if __name__ == "__main__":
    main()
