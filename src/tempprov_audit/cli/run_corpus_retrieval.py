from __future__ import annotations

import argparse
import gzip
import heapq
import json
import math
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from tempprov_audit.cli.run_ctd_provenance_lexical_baselines import (
    TOKEN_PATTERN,
    summarize_ranking,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scalable Top-k retrieval over the full 55k CTD literature corpus.")
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--document-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--embedding-cache-root",
        help="Optional shared document-embedding cache used across query cohorts.",
    )
    parser.add_argument("--retriever", choices=["bm25", "bge_large", "medcpt_dual"], required=True)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260703)
    return parser.parse_args()


def read_queries(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_documents(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            source = json.load(handle)
    except json.JSONDecodeError:
        with opener(path, "rt", encoding="utf-8") as handle:
            source = [json.loads(line) for line in handle if line.strip()]
    if isinstance(source, list):
        return sorted(source, key=lambda row: row["document_id"])
    if "document_id" in source:
        return [source]
    rows = []
    for pmid, value in source.items():
        rows.append(
            {
                "document_id": f"PMID:{pmid}",
                "pmid": str(pmid),
                "pub_year": int(value.get("pub_year") or 0),
                "text": str(value.get("text") or "").strip(),
            }
        )
    return sorted(rows, key=lambda row: row["document_id"])


def bm25_topk(queries: list[dict[str, Any]], documents: list[dict[str, Any]], top_k: int):
    tokenized = [TOKEN_PATTERN.findall(row["text"].lower()) for row in documents]
    lengths = np.asarray([len(tokens) for tokens in tokenized], dtype=np.float32)
    average_length = float(lengths.mean())
    postings: dict[str, list[tuple[int, int]]] = {}
    for index, tokens in enumerate(tokenized):
        for term, frequency in Counter(tokens).items():
            postings.setdefault(term, []).append((index, frequency))
    count = len(documents)
    output = []
    for query in queries:
        scores: dict[int, float] = {}
        for term in set(TOKEN_PATTERN.findall(query["query_text"].lower())):
            entries = postings.get(term, [])
            df = len(entries)
            if not df:
                continue
            idf = math.log(1.0 + (count - df + 0.5) / (df + 0.5))
            for index, frequency in entries:
                denominator = frequency + 1.2 * (0.25 + 0.75 * lengths[index] / max(average_length, 1.0))
                scores[index] = scores.get(index, 0.0) + idf * frequency * 2.2 / denominator
        ranked = heapq.nlargest(top_k, scores.items(), key=lambda item: (item[1], -item[0]))
        if len(ranked) < top_k:
            chosen = {index for index, _ in ranked}
            ranked.extend((index, 0.0) for index in range(count) if index not in chosen)
            ranked = ranked[:top_k]
        output.append(([index for index, _ in ranked], [float(score) for _, score in ranked]))
    return output, {"indexed_terms": len(postings), "mean_document_tokens": average_length}


def encode(model_name: str, texts: list[str], batch_size: int, *, prefix: str = "", normalize: bool = False):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to("cuda").eval()
    chunks = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                [prefix + text for text in texts[start : start + batch_size]],
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
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(chunks), {"seconds": elapsed, "items_per_second": len(texts) / elapsed, "peak_gpu_gb": peak}


def dense_topk(args, queries, documents, output_root):
    import torch

    if args.retriever == "bge_large":
        query_model = document_model = "BAAI/bge-large-en-v1.5"
        prefix, normalize = "Represent this sentence for searching relevant passages: ", True
    else:
        query_model, document_model = "ncbi/MedCPT-Query-Encoder", "ncbi/MedCPT-Article-Encoder"
        prefix, normalize = "", False
    cache = Path(args.embedding_cache_root).resolve() if args.embedding_cache_root else output_root / "embedding_cache"
    cache.mkdir(parents=True, exist_ok=True)
    doc_path = cache / f"{args.retriever}_documents.npy"
    meta_path = cache / f"{args.retriever}_documents.json"
    if doc_path.exists() and meta_path.exists():
        document_vectors = np.load(doc_path, mmap_mode="r")
        document_stats = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        document_vectors, document_stats = encode(document_model, [d["text"] for d in documents], args.batch_size, normalize=normalize)
        np.save(doc_path, document_vectors)
        write_json(meta_path, document_stats)
    query_vectors, query_stats = encode(query_model, [q["query_text"] for q in queries], args.batch_size, prefix=prefix, normalize=normalize)
    docs = torch.from_numpy(np.asarray(document_vectors)).to("cuda")
    outputs = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(query_vectors), 128):
            q = torch.from_numpy(query_vectors[start : start + 128]).to("cuda")
            values, indices = torch.topk(q @ docs.T, k=min(args.top_k, len(documents)), dim=1)
            outputs.extend(zip(indices.cpu().tolist(), values.cpu().tolist()))
    search_seconds = time.perf_counter() - started
    search_peak = torch.cuda.max_memory_allocated() / (1024**3)
    return outputs, {"document_encoding": document_stats, "query_encoding": query_stats, "search_seconds": search_seconds, "search_peak_gpu_gb": search_peak}


def recall_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    output = {}
    for cutoff in (10, 20, 50, 100, 200, 500, 1000):
        if cutoff > len(results[0]["ranked_document_ids"]):
            continue
        output[f"recall_at_{cutoff}"] = mean(
            len(set(row["gold_document_ids"]) & set(row["ranked_document_ids"][:cutoff])) / len(row["gold_document_ids"])
            for row in results
        )
        output[f"query_success_at_{cutoff}"] = mean(
            bool(set(row["gold_document_ids"]) & set(row["ranked_document_ids"][:cutoff])) for row in results
        )
    return output


def main() -> None:
    args = parse_args()
    query_path = Path(args.query_file).resolve()
    document_path = Path(args.document_json).resolve()
    output_root = Path(args.output_root).resolve() / args.retriever
    output_root.mkdir(parents=True, exist_ok=True)
    queries = read_queries(query_path)
    documents = read_documents(document_path)
    available = {row["document_id"] for row in documents}
    missing = sorted({item for row in queries for item in row["gold_document_ids"] if item not in available})
    if missing:
        raise ValueError(f"Gold documents missing from corpus: {missing[:10]}")

    started = time.perf_counter()
    if args.retriever == "bm25":
        rankings, method_stats = bm25_topk(queries, documents, args.top_k)
    else:
        rankings, method_stats = dense_topk(args, queries, documents, output_root)
    elapsed = time.perf_counter() - started
    results = []
    for query, (indices, scores) in zip(queries, rankings):
        results.append(
            {
                "example_id": query["example_id"],
                "head_id": query.get("head_id"),
                "tail_id": query.get("tail_id"),
                "fold_index": query.get("fold_index"),
                "has_gold": True,
                "gold_document_ids": query["gold_document_ids"],
                "top_score": float(scores[0]),
                "ranked_document_ids": [documents[index]["document_id"] for index in indices],
                "ranked_scores": [float(value) for value in scores],
            }
        )
    prediction_path = output_root / "topk_rankings.jsonl.gz"
    with gzip.open(prediction_path, "wt", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "protocol": "full 55,001-document CTD literature retrieval with truncated Top-k storage",
        "retriever": args.retriever,
        "corpus_documents": len(documents),
        "queries": len(queries),
        "top_k": args.top_k,
        "wall_seconds": elapsed,
        "queries_per_second": len(queries) / elapsed,
        "method_stats": method_stats,
        "ranking": summarize_ranking(results, args.bootstrap_resamples, args.seed),
        "corpus_recall": recall_metrics(results),
    }
    write_json(output_root / "summary.json", summary)
    (output_root / "COMPLETED").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
