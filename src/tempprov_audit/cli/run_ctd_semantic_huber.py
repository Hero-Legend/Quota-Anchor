from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from tempprov_audit.cli.run_ctd_provenance_lexical_baselines import (
    classification_metrics,
    json_safe,
    read_jsonl,
    select_thresholds,
    summarize_ranking,
    write_json,
)


SPLITS = ("train", "dev", "test")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
RELATION_CUES = (
    "associate",
    "cause",
    "correlat",
    "induc",
    "inhibit",
    "prevent",
    "protect",
    "risk",
    "treat",
    "therap",
    "toxic",
    "alleviat",
    "aggravat",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a leakage-audited adaptation of the historical CTD semantic "
            "modified-Huber article-ranking baseline."
        )
    )
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[1e-6, 1e-5, 1e-4, 1e-3])
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=2_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260630)
    return parser.parse_args()


def normalized_count(text: str, phrase: str) -> int:
    phrase = str(phrase or "").strip().lower()
    if not phrase:
        return 0
    return str(text or "").lower().count(phrase)


def conditioned_text_and_features(
    row: dict[str, Any], candidate: dict[str, Any]
) -> tuple[str, list[float]]:
    """Create query-conditioned text and transparent numeric interaction features.

    Candidate provenance fields are deliberately ignored because source labels and
    CTD-citation flags would leak the target.
    """

    document = str(candidate.get("text") or "")
    head = str(row.get("head_text") or "")
    tail = str(row.get("tail_text") or "")
    head_count = normalized_count(document, head)
    tail_count = normalized_count(document, tail)
    cooccurrence_sentences = [
        sentence
        for sentence in SENTENCE_BOUNDARY.split(document)
        if normalized_count(sentence, head) and normalized_count(sentence, tail)
    ]
    cooccurrence_text = " ".join(cooccurrence_sentences)
    document_lower = document.lower()
    cooccurrence_lower = cooccurrence_text.lower()
    total_cues = sum(document_lower.count(cue) for cue in RELATION_CUES)
    cooccurrence_cues = sum(cooccurrence_lower.count(cue) for cue in RELATION_CUES)
    publication_year = float(candidate.get("pub_year") or 0)

    conditioned = (
        f"Chemical: {head}. Disease: {tail}. Article: {document} "
        f"Entity-cooccurrence context: {cooccurrence_text}"
    )
    numeric = [
        math.log1p(head_count),
        math.log1p(tail_count),
        float(bool(head_count and tail_count)),
        math.log1p(len(cooccurrence_sentences)),
        math.log1p(cooccurrence_cues),
        math.log1p(total_cues),
        (publication_year - 2021.0) / 10.0 if publication_year else 0.0,
        math.log1p(len(document.split())),
    ]
    return conditioned, numeric


def flatten_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        gold = set(row["gold_document_ids"])
        for candidate_index, candidate in enumerate(row["candidates"]):
            text, numeric = conditioned_text_and_features(row, candidate)
            flat.append(
                {
                    "row_index": row_index,
                    "candidate_index": candidate_index,
                    "text": text,
                    "numeric": numeric,
                    "label": int(candidate["document_id"] in gold),
                }
            )
    return flat


def build_ranked_results(
    rows: list[dict[str, Any]], flat: list[dict[str, Any]], scores: list[float]
) -> list[dict[str, Any]]:
    scores_by_row = [[0.0] * len(row["candidates"]) for row in rows]
    for item, score in zip(flat, scores):
        scores_by_row[item["row_index"]][item["candidate_index"]] = float(score)

    results: list[dict[str, Any]] = []
    for row, row_scores in zip(rows, scores_by_row):
        ranked = sorted(
            zip(row["candidates"], row_scores),
            key=lambda item: (-item[1], item[0]["document_id"]),
        )
        results.append(
            {
                "example_id": row["example_id"],
                "head_id": row.get("head_id"),
                "tail_id": row.get("tail_id"),
                "has_gold": bool(row["gold_document_ids"]),
                "gold_document_ids": row["gold_document_ids"],
                "top_score": ranked[0][1] if ranked else float("-inf"),
                "ranked_document_ids": [candidate["document_id"] for candidate, _ in ranked],
                "ranked_scores": [score for _, score in ranked],
            }
        )
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import MaxAbsScaler

    dataset_root = Path(args.benchmark_root).resolve() / args.dataset
    rows = {split: read_jsonl(dataset_root / f"{split}.jsonl") for split in SPLITS}
    flat = {split: flatten_rows(rows[split]) for split in SPLITS}

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_features=args.max_features,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    text_matrices = {
        "train": vectorizer.fit_transform(item["text"] for item in flat["train"])
    }
    text_matrices["dev"] = vectorizer.transform(item["text"] for item in flat["dev"])
    text_matrices["test"] = vectorizer.transform(item["text"] for item in flat["test"])

    scaler = MaxAbsScaler()
    numeric_matrices = {
        "train": scaler.fit_transform(csr_matrix([item["numeric"] for item in flat["train"]]))
    }
    for split in ("dev", "test"):
        numeric_matrices[split] = scaler.transform(
            csr_matrix([item["numeric"] for item in flat[split]])
        )
    matrices = {
        split: hstack([text_matrices[split], numeric_matrices[split]], format="csr")
        for split in SPLITS
    }
    labels = {
        split: np.asarray([item["label"] for item in flat[split]], dtype=np.int8)
        for split in SPLITS
    }

    selection: list[dict[str, float]] = []
    best: tuple[float, float, Any, list[dict[str, Any]]] | None = None
    for alpha in args.alpha_grid:
        model = SGDClassifier(
            loss="modified_huber",
            alpha=alpha,
            class_weight="balanced",
            max_iter=args.max_iter,
            tol=1e-4,
            random_state=args.seed,
            average=True,
        )
        model.fit(matrices["train"], labels["train"])
        dev_scores = model.decision_function(matrices["dev"]).astype(float).tolist()
        dev_results = build_ranked_results(rows["dev"], flat["dev"], dev_scores)
        dev_map = float(summarize_ranking(dev_results, 0, args.seed)["metrics"]["map"]["estimate"])
        selection.append({"alpha": alpha, "dev_map": dev_map})
        candidate = (dev_map, -alpha, model, dev_results)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    assert best is not None
    best_dev_map, negative_alpha, model, dev_results = best
    selected_alpha = -negative_alpha
    test_scores = model.decision_function(matrices["test"]).astype(float).tolist()
    test_results = build_ranked_results(rows["test"], flat["test"], test_scores)
    policies = select_thresholds(dev_results)
    summary = {
        "method": "historical_ctd_semantic_modified_huber_faithful_adaptation",
        "reproduction_scope": (
            "Pair-conditioned TF-IDF uni/bigrams, explicit entity/cooccurrence/relation-cue "
            "features, and modified-Huber classification. The unavailable 2012 PubMed "
            "semantic space, journal, and MeSH feature pipeline are not claimed as reproduced."
        ),
        "leakage_audit": {
            "candidate_source_used": False,
            "ctd_citation_flag_used": False,
            "hyperparameters_selected_on": "development split only",
        },
        "dataset": args.dataset,
        "seed": args.seed,
        "selected_alpha": selected_alpha,
        "alpha_selection": selection,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "candidate_counts": {split: len(flat[split]) for split in SPLITS},
        "positive_candidate_counts": {split: int(labels[split].sum()) for split in SPLITS},
        "ranking": {
            "dev": summarize_ranking(dev_results, args.bootstrap_resamples, args.seed),
            "test": summarize_ranking(test_results, args.bootstrap_resamples, args.seed + 1),
        },
        "threshold_selection": policies,
        "test_pair_classification": {
            policy: classification_metrics(test_results, values["threshold"])
            for policy, values in policies.items()
        },
    }
    output_root = Path(args.output_root).resolve()
    write_json(output_root / "dev_predictions.json", dev_results)
    write_json(output_root / "test_predictions.json", test_results)
    write_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    payload = run(args)
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
