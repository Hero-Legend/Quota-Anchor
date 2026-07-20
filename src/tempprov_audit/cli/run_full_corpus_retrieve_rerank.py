from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from tempprov_audit.cli.run_ctd_provenance_cross_encoder import score_rows
from tempprov_audit.cli.run_ctd_provenance_lexical_baselines import (
    TOKEN_PATTERN,
    read_jsonl,
    summarize_ranking,
    write_json,
)


RETRIEVERS = ("bm25", "bge_large", "medcpt_dual")
RERANKERS = ("medcpt_cross_encoder", "biomedbert_3seed_ensemble")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate full-corpus retrieval followed by locked Top-k reranking."
    )
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--neural-cv-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--retrieval-depth", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260630)
    return parser.parse_args()


def read_document_store(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        documents = [json.loads(line) for line in handle if line.strip()]
    postcut = [document for document in documents if int(document.get("pub_year") or 0) > 2021]
    return sorted(postcut, key=lambda document: document["document_id"])


def load_positive_test_queries(benchmark_root: Path) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fold in range(5):
        rows = read_jsonl(
            benchmark_root
            / "tail_grouped_cv_pmid_isolated"
            / f"fold_{fold}"
            / "test.jsonl"
        )
        for row in rows:
            if not row["gold_document_ids"]:
                continue
            if row["example_id"] in seen:
                raise ValueError(f"Duplicate test query across folds: {row['example_id']}")
            seen.add(row["example_id"])
            queries.append(
                {
                    **{key: value for key, value in row.items() if key != "candidates"},
                    "fold_index": fold,
                }
            )
    return queries


def bm25_rankings(
    queries: list[dict[str, Any]], documents: list[dict[str, Any]]
) -> list[list[int]]:
    tokenized_documents = [TOKEN_PATTERN.findall(document["text"].lower()) for document in documents]
    lengths = [len(tokens) for tokens in tokenized_documents]
    average_length = mean(lengths)
    postings: dict[str, list[tuple[int, int]]] = {}
    for document_index, tokens in enumerate(tokenized_documents):
        for term, frequency in Counter(tokens).items():
            postings.setdefault(term, []).append((document_index, frequency))
    count = len(documents)
    rankings: list[list[int]] = []
    for row in queries:
        scores = [0.0] * count
        query_terms = set(TOKEN_PATTERN.findall(row["query_text"].lower()))
        for term in query_terms:
            entries = postings.get(term, [])
            document_frequency = len(entries)
            if not document_frequency:
                continue
            inverse_frequency = math.log(
                1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for document_index, frequency in entries:
                denominator = frequency + 1.2 * (
                    1.0 - 0.75 + 0.75 * lengths[document_index] / max(average_length, 1.0)
                )
                scores[document_index] += inverse_frequency * frequency * 2.2 / denominator
        rankings.append(sorted(range(count), key=lambda index: (-scores[index], index)))
    return rankings


def encode_texts(
    model_name: str,
    texts: list[str],
    *,
    batch_size: int,
    query_prefix: str = "",
    normalize: bool,
) -> Any:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to("cuda").eval()
    chunks: list[Any] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = [query_prefix + text for text in texts[start : start + batch_size]]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                vectors = model(**encoded).last_hidden_state[:, 0, :].float()
            if normalize:
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            chunks.append(vectors.cpu().numpy().astype(np.float32))
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return np.concatenate(chunks, axis=0)


def dense_rankings(
    queries: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    *,
    query_model: str,
    document_model: str,
    batch_size: int,
    query_prefix: str = "",
    normalize: bool,
) -> list[list[int]]:
    import numpy as np

    query_vectors = encode_texts(
        query_model,
        [row["query_text"] for row in queries],
        batch_size=batch_size,
        query_prefix=query_prefix,
        normalize=normalize,
    )
    document_vectors = encode_texts(
        document_model,
        [document["text"] for document in documents],
        batch_size=batch_size,
        normalize=normalize,
    )
    similarities = query_vectors @ document_vectors.T
    return [
        np.lexsort((np.arange(len(documents)), -row)).astype(int).tolist()
        for row in similarities
    ]


def to_results(
    queries: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    rankings: list[list[int]],
) -> list[dict[str, Any]]:
    return [
        {
            "example_id": row["example_id"],
            "head_id": row.get("head_id"),
            "tail_id": row.get("tail_id"),
            "fold_index": row["fold_index"],
            "has_gold": True,
            "gold_document_ids": row["gold_document_ids"],
            "top_score": float(-ranking[0]) if ranking else float("-inf"),
            "ranked_document_ids": [documents[index]["document_id"] for index in ranking],
        }
        for row, ranking in zip(queries, rankings)
    ]


def corpus_recall_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for cutoff in (10, 20, 50, 100, 200):
        output[f"recall_at_{cutoff}"] = mean(
            sum(document_id in set(result["gold_document_ids"]) for document_id in result["ranked_document_ids"][:cutoff])
            / len(result["gold_document_ids"])
            for result in results
        )
        output[f"query_success_at_{cutoff}"] = mean(
            float(any(document_id in set(result["gold_document_ids"]) for document_id in result["ranked_document_ids"][:cutoff]))
            for result in results
        )
    return output


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def oracle_topk_result(result: dict[str, Any], depth: int) -> dict[str, Any]:
    top = result["ranked_document_ids"][:depth]
    gold = set(result["gold_document_ids"])
    oracle_top = [item for item in top if item in gold] + [item for item in top if item not in gold]
    return {**result, "ranked_document_ids": oracle_top + result["ranked_document_ids"][depth:]}


def make_rerank_rows(
    queries: list[dict[str, Any]],
    documents_by_id: dict[str, dict[str, Any]],
    retrieval_results: list[dict[str, Any]],
    depth: int,
) -> list[dict[str, Any]]:
    by_example = {result["example_id"]: result for result in retrieval_results}
    output: list[dict[str, Any]] = []
    for query in queries:
        top_ids = by_example[query["example_id"]]["ranked_document_ids"][:depth]
        output.append(
            {
                **query,
                "candidates": [documents_by_id[document_id] for document_id in top_ids],
            }
        )
    return output


def score_cross_encoder(
    model_path: str,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    num_workers: int,
    query_prefix: str | None = None,
    include_publication_year: bool = True,
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")
    predictions = score_rows(
        model,
        tokenizer,
        rows,
        device=torch.device("cuda"),
        batch_size=batch_size,
        max_length=512,
        num_workers=num_workers,
        **({} if query_prefix is None else {"query_prefix": query_prefix}),
        include_publication_year=include_publication_year,
    )
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return predictions


def ensemble_predictions(prediction_sets: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for aligned in zip(*prediction_sets):
        first = aligned[0]
        score_maps = [
            dict(zip(prediction["ranked_document_ids"], prediction["ranked_scores"]))
            for prediction in aligned
        ]
        averaged = {
            document_id: mean(score_map[document_id] for score_map in score_maps)
            for document_id in first["ranked_document_ids"]
        }
        ranked_ids = sorted(averaged, key=lambda item: (-averaged[item], item))
        output.append(
            {
                **{key: value for key, value in first.items() if key not in {"ranked_document_ids", "ranked_scores", "top_score"}},
                "top_score": averaged[ranked_ids[0]],
                "ranked_document_ids": ranked_ids,
                "ranked_scores": [averaged[item] for item in ranked_ids],
            }
        )
    return output


def merge_reranked_topk(
    retrieval: dict[str, Any], reranked: dict[str, Any], depth: int
) -> dict[str, Any]:
    reranked_top = reranked["ranked_document_ids"]
    top_set = set(retrieval["ranked_document_ids"][:depth])
    if set(reranked_top) != top_set:
        raise ValueError("Reranker candidate set differs from the locked retrieval Top-k.")
    remainder = [item for item in retrieval["ranked_document_ids"] if item not in top_set]
    return {
        **retrieval,
        "top_score": reranked.get("top_score", 0.0),
        "ranked_document_ids": reranked_top + remainder,
    }


def load_or_run_retrieval(
    name: str,
    queries: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    output_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    path = output_root / "retrieval" / name / "full_rankings.jsonl.gz"
    if path.exists():
        return read_jsonl_gz(path)
    if name == "bm25":
        rankings = bm25_rankings(queries, documents)
    elif name == "bge_large":
        rankings = dense_rankings(
            queries,
            documents,
            query_model="BAAI/bge-large-en-v1.5",
            document_model="BAAI/bge-large-en-v1.5",
            batch_size=args.batch_size,
            query_prefix="Represent this sentence for searching relevant passages: ",
            normalize=True,
        )
    elif name == "medcpt_dual":
        rankings = dense_rankings(
            queries,
            documents,
            query_model="ncbi/MedCPT-Query-Encoder",
            document_model="ncbi/MedCPT-Article-Encoder",
            batch_size=args.batch_size,
            normalize=False,
        )
    else:
        raise ValueError(name)
    results = to_results(queries, documents, rankings)
    write_jsonl_gz(path, results)
    write_json(
        path.parent / "summary.json",
        {
            "retriever": name,
            "corpus_documents": len(documents),
            "positive_queries": len(queries),
            "ranking": summarize_ranking(results, args.bootstrap_resamples, args.seed),
            "full_corpus_recall": corpus_recall_metrics(results),
        },
    )
    return results


def run_reranker(
    retriever: str,
    reranker: str,
    queries: list[dict[str, Any]],
    documents_by_id: dict[str, dict[str, Any]],
    retrieval_results: list[dict[str, Any]],
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    target = output_root / "reranking" / retriever / reranker
    summary_path = target / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    rerank_rows = make_rerank_rows(
        queries, documents_by_id, retrieval_results, args.retrieval_depth
    )
    predictions: list[dict[str, Any]] = []
    for fold in range(5):
        fold_rows = [row for row in rerank_rows if row["fold_index"] == fold]
        if reranker == "medcpt_cross_encoder":
            fold_predictions = score_cross_encoder(
                "ncbi/MedCPT-Cross-Encoder",
                fold_rows,
                batch_size=args.batch_size * 2,
                num_workers=args.num_workers,
                query_prefix="",
                include_publication_year=False,
            )
        elif reranker == "biomedbert_3seed_ensemble":
            seed_predictions = []
            for seed in (13, 42, 3407):
                checkpoint = (
                    Path(args.neural_cv_root)
                    / f"fold_{fold}"
                    / "biomedbert_cross_encoder_3seed"
                    / f"seed_{seed}"
                    / "best_model"
                )
                seed_predictions.append(
                    score_cross_encoder(
                        str(checkpoint),
                        fold_rows,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                    )
                )
            fold_predictions = ensemble_predictions(seed_predictions)
        else:
            raise ValueError(reranker)
        predictions.extend(fold_predictions)

    retrieval_by_example = {result["example_id"]: result for result in retrieval_results}
    merged = [
        merge_reranked_topk(
            retrieval_by_example[prediction["example_id"]],
            prediction,
            args.retrieval_depth,
        )
        for prediction in predictions
    ]
    merged.sort(key=lambda item: item["example_id"])
    summary = {
        "retriever": retriever,
        "reranker": reranker,
        "retrieval_depth": args.retrieval_depth,
        "corpus_documents": len(documents_by_id),
        "positive_queries": len(queries),
        "ranking": summarize_ranking(merged, args.bootstrap_resamples, args.seed + 1),
        "full_corpus_recall": corpus_recall_metrics(merged),
    }
    write_jsonl_gz(target / "full_rankings.jsonl.gz", merged)
    write_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    benchmark_root = Path(args.benchmark_root).resolve()
    output_root = Path(args.output_root).resolve()
    documents = read_document_store(benchmark_root / "documents.jsonl.gz")
    documents_by_id = {document["document_id"]: document for document in documents}
    queries = load_positive_test_queries(benchmark_root)
    missing_gold = sorted(
        {
            document_id
            for row in queries
            for document_id in row["gold_document_ids"]
            if document_id not in documents_by_id
        }
    )
    if missing_gold:
        raise ValueError(f"Gold documents missing from full corpus: {missing_gold[:10]}")

    retrieval_outputs = {
        name: load_or_run_retrieval(name, queries, documents, output_root, args)
        for name in RETRIEVERS
    }
    aggregate: dict[str, Any] = {
        "protocol": {
            "corpus": "all benchmark documents with publication year > 2021",
            "corpus_documents": len(documents),
            "query_population": "all positive fivefold disease-tail-disjoint test queries",
            "positive_queries": len(queries),
            "retrieval_depth": args.retrieval_depth,
            "retrievers": list(RETRIEVERS),
            "rerankers": list(RERANKERS),
            "biomedbert_seeds": [13, 42, 3407],
        },
        "retrieval": {},
        "oracle_topk": {},
        "reranking": {},
    }
    for name, results in retrieval_outputs.items():
        aggregate["retrieval"][name] = json.loads(
            (output_root / "retrieval" / name / "summary.json").read_text(encoding="utf-8")
        )
        oracle = [oracle_topk_result(result, args.retrieval_depth) for result in results]
        aggregate["oracle_topk"][name] = {
            "ranking": summarize_ranking(oracle, args.bootstrap_resamples, args.seed + 2),
            "full_corpus_recall": corpus_recall_metrics(oracle),
        }
        aggregate["reranking"][name] = {}
        for reranker in RERANKERS:
            aggregate["reranking"][name][reranker] = run_reranker(
                name,
                reranker,
                queries,
                documents_by_id,
                results,
                output_root,
                args,
            )
    write_json(output_root / "aggregate_full_corpus.json", aggregate)
    (output_root / "COMPLETED").write_text("complete\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
