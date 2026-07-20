from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from tempprov_audit.cli.run_api_biomedical_llm_batch import (
    anthropic_messages_url,
    build_payload,
    extract_text,
    request_with_error_capture,
)
from tempprov_audit.cli.run_ctd_provenance_lexical_baselines import summarize_ranking, write_json
from tempprov_audit.cli.run_full_corpus_retrieve_rerank import merge_reranked_topk
from tempprov_audit.cli.run_corpus_retrieval import read_documents, read_queries, recall_metrics


SYSTEM_PROMPT = """You are a biomedical publication-provenance reranker.
Score how strongly the supplied PubMed title and abstract support reviewing the specified therapy-disease association.
Use the complete 0-100 range: 100 means direct, specific clinical or experimental evidence; 50 means partial or indirect evidence; 0 means unrelated evidence.
Judge only the supplied text. Return exactly one JSON object with one integer field: {\"score\": <0-100>}."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank the frozen CIViC Top-k pool with an API LLM.")
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--document-file", required=True)
    parser.add_argument("--retrieval-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--api-format", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--max-document-words", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--extra-body-json", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-request-attempts", type=int, default=6)
    parser.add_argument("--retry-initial-seconds", type=float, default=2.0)
    parser.add_argument("--retry-max-seconds", type=float, default=60.0)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--allow-subset-debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-valid-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260705)
    return parser.parse_args()


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truncate_words(text: str, max_words: int) -> str:
    return " ".join(str(text).split()[:max_words])


def build_user_prompt(query: dict[str, Any], document: dict[str, Any], max_words: int) -> str:
    title = str(document.get("title", "")).strip()
    abstract = truncate_words(str(document.get("abstract", "")).strip(), max_words)
    return (
        f"Therapy: {query['head_text']}\n"
        f"Disease: {query['tail_text']}\n"
        f"Candidate title: {title}\n"
        f"Candidate abstract: {abstract}\n"
        'Return only {"score": <integer from 0 to 100>}.'
    )


def parse_score(text: str) -> int:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'(?i)["\']?score["\']?\s*[:=]\s*(-?\d+(?:\.\d+)?)', cleaned)
        if match is None and re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned):
            match = re.match(r"(-?\d+(?:\.\d+)?)", cleaned)
        if match is None:
            raise ValueError("score_not_found")
        value = float(match.group(1))
    else:
        if isinstance(payload, (int, float)):
            value = float(payload)
        elif isinstance(payload, dict) and "score" in payload:
            value = float(payload["score"])
        else:
            raise ValueError("score_not_found")
    if not 0 <= value <= 100:
        raise ValueError("score_out_of_range")
    return int(round(value))


def extract_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0}
    return {
        "input_tokens": int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0),
        "output_tokens": int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0),
    }


def task_key(example_id: str, document_id: str) -> str:
    return f"{example_id}\t{document_id}"


def load_latest_scores(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[task_key(str(row["example_id"]), str(row["document_id"]))] = row
    return rows


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def make_tasks(
    queries: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
    retrieval: list[dict[str, Any]],
    depth: int,
) -> list[dict[str, Any]]:
    retrieval_by_id = {row["example_id"]: row for row in retrieval}
    tasks = []
    for query in queries:
        ranked_ids = retrieval_by_id[query["example_id"]]["ranked_document_ids"][:depth]
        for original_rank, document_id in enumerate(ranked_ids, 1):
            tasks.append(
                {
                    "example_id": query["example_id"],
                    "document_id": document_id,
                    "original_rank": original_rank,
                    "query": query,
                    "document": documents[document_id],
                }
            )
    return tasks


def aggregate(
    args: argparse.Namespace,
    queries: list[dict[str, Any]],
    retrieval: list[dict[str, Any]],
    score_rows: dict[str, dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    retrieval_by_id = {row["example_id"]: row for row in retrieval}
    valid = {
        key: row
        for key, row in score_rows.items()
        if row.get("score") is not None and not row.get("parse_error")
    }
    expected = len(queries) * args.depth
    if len(valid) != expected:
        raise RuntimeError(f"incomplete_valid_scores:{len(valid)}/{expected}")

    merged = []
    for query in queries:
        example_id = query["example_id"]
        candidate_ids = retrieval_by_id[example_id]["ranked_document_ids"][: args.depth]
        scored = [valid[task_key(example_id, document_id)] for document_id in candidate_ids]
        scored.sort(key=lambda row: (-int(row["score"]), int(row["original_rank"])))
        reranked = {
            "example_id": example_id,
            "ranked_document_ids": [row["document_id"] for row in scored],
            "ranked_scores": [int(row["score"]) for row in scored],
            "top_score": int(scored[0]["score"]),
        }
        merged.append(merge_reranked_topk(retrieval_by_id[example_id], reranked, args.depth))
    merged.sort(key=lambda row: row["example_id"])

    latencies = [float(row.get("latency_seconds", 0.0)) for row in valid.values()]
    summary = {
        "provider": args.provider,
        "model_name": args.model_name,
        "queries": len(queries),
        "evaluated_pairs": expected,
        "retrieval_depth": args.depth,
        "tie_break": "original_medcpt_dual_rank",
        "ranking": summarize_ranking(merged, args.bootstrap_resamples, args.seed),
        "corpus_recall": recall_metrics(merged),
        "efficiency": {
            "latency_mean_seconds_per_pair": statistics.fmean(latencies),
            "latency_p50_seconds_per_pair": percentile(latencies, 0.50),
            "latency_p95_seconds_per_pair": percentile(latencies, 0.95),
            "input_tokens": sum(int(row.get("input_tokens", 0)) for row in valid.values()),
            "output_tokens": sum(int(row.get("output_tokens", 0)) for row in valid.values()),
            "request_attempts": sum(int(row.get("request_attempts", 1)) for row in valid.values()),
        },
    }
    write_jsonl_gz(output_root / "topk_rankings.jsonl.gz", merged)
    write_json(output_root / "summary.json", summary)
    (output_root / "COMPLETED").write_text("complete\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.max_pairs is not None and not args.allow_subset_debug:
        raise SystemExit("--max-pairs is restricted to --allow-subset-debug and cannot produce release-grade evidence.")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and not args.aggregate_only:
        raise SystemExit(f"Missing API key env: {args.api_key_env}")

    query_path = Path(args.query_file).resolve()
    document_path = Path(args.document_file).resolve()
    retrieval_path = Path(args.retrieval_file).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    score_path = output_root / "pair_scores.jsonl"
    raw_path = output_root / "raw_responses.jsonl"

    queries = read_queries(query_path)
    documents = {row["document_id"]: row for row in read_documents(document_path)}
    retrieval = read_jsonl_gz(retrieval_path)
    tasks = make_tasks(queries, documents, retrieval, args.depth)
    if args.max_pairs is not None:
        tasks = tasks[: args.max_pairs]

    protocol = {
        "provider": args.provider,
        "model_name": args.model_name,
        "api_base": args.api_base,
        "api_format": args.api_format,
        "system_prompt": SYSTEM_PROMPT,
        "score_range": [0, 100],
        "tie_break": "original_medcpt_dual_rank",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_document_words": args.max_document_words,
        "retrieval_depth": args.depth,
        "query_sha256": sha256_file(query_path),
        "document_sha256": sha256_file(document_path),
        "retrieval_sha256": sha256_file(retrieval_path),
    }
    write_json(output_root / "protocol.json", protocol)

    existing = load_latest_scores(score_path)
    if args.aggregate_only:
        print(json.dumps(aggregate(args, queries, retrieval, existing, output_root), ensure_ascii=False, indent=2))
        return

    if args.resume_valid_only:
        done = {
            key for key, row in existing.items() if row.get("score") is not None and not row.get("parse_error")
        }
    elif args.resume:
        done = set(existing)
    else:
        existing = {}
        done = set()
        score_path.write_text("", encoding="utf-8")
        raw_path.write_text("", encoding="utf-8")
    pending = [task for task in tasks if task_key(task["example_id"], task["document_id"]) not in done]
    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else {}
    api_url = anthropic_messages_url(args.api_base) if args.api_format == "anthropic" else args.api_base
    fatal_billing_error = threading.Event()

    def run_task(task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        user_prompt = build_user_prompt(task["query"], task["document"], args.max_document_words)
        payload = build_payload(
            args.model_name,
            SYSTEM_PROMPT,
            user_prompt,
            args.max_tokens,
            args.temperature,
            extra_body,
            args.api_format,
        )
        started = time.perf_counter()
        response, request_error, attempts = request_with_error_capture(
            api_url,
            api_key,
            payload,
            args.request_timeout,
            args.api_format,
            max_attempts=args.max_request_attempts,
            retry_initial_seconds=args.retry_initial_seconds,
            retry_max_seconds=args.retry_max_seconds,
            retry_http_statuses={408, 409, 425, 429, 500, 502, 503, 504},
        )
        if request_error and any(
            marker in request_error.lower()
            for marker in (
                "http_402",
                "insufficient balance",
                "exceeded_current_quota",
                "overdue-payment",
                "account is in good standing",
                "余额不足",
                "无可用资源包",
            )
        ):
            fatal_billing_error.set()
        latency = time.perf_counter() - started
        text = extract_text(response)
        parse_error = request_error
        score: int | None = None
        if request_error is None:
            try:
                score = parse_score(text)
            except Exception as exc:  # noqa: BLE001
                parse_error = str(exc)
        usage = extract_usage(response)
        row = {
            "example_id": task["example_id"],
            "document_id": task["document_id"],
            "original_rank": task["original_rank"],
            "provider": args.provider,
            "model_name": args.model_name,
            "score": score,
            "latency_seconds": round(latency, 6),
            "request_attempts": attempts,
            **usage,
            "parse_error": parse_error,
        }
        raw = {
            **{key: row[key] for key in ("example_id", "document_id", "provider", "model_name")},
            "raw_text": text,
            "raw_response": response,
            "parse_error": parse_error,
        }
        return row, raw

    started = time.perf_counter()
    processed = 0
    mode = "a" if (args.resume or args.resume_valid_only) else "w"
    with score_path.open(mode, encoding="utf-8") as score_handle, raw_path.open(mode, encoding="utf-8") as raw_handle:
        chunk_size = max(args.workers * 4, 1)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for offset in range(0, len(pending), chunk_size):
                futures = [executor.submit(run_task, task) for task in pending[offset : offset + chunk_size]]
                for future in as_completed(futures):
                    row, raw = future.result()
                    append_jsonl(score_handle, row)
                    append_jsonl(raw_handle, raw)
                    existing[task_key(row["example_id"], row["document_id"])] = row
                    processed += 1
                if processed == len(pending) or processed % max(100, chunk_size) == 0:
                    valid = sum(
                        row.get("score") is not None and not row.get("parse_error") for row in existing.values()
                    )
                    print(
                        json.dumps(
                            {
                                "provider": args.provider,
                                "processed_this_run": processed,
                                "pending_this_run": len(pending),
                                "valid_total": valid,
                                "expected_total": len(queries) * args.depth,
                                "wall_seconds": round(time.perf_counter() - started, 1),
                            }
                        ),
                        flush=True,
                    )
                if fatal_billing_error.is_set():
                    valid_count = sum(
                        row.get("score") is not None and not row.get("parse_error") for row in existing.values()
                    )
                    write_json(
                        output_root / "BILLING_BLOCKED.json",
                        {
                            "provider": args.provider,
                            "valid_scores": valid_count,
                            "expected_scores": len(queries) * args.depth,
                        },
                    )
                    raise SystemExit(12)

    if args.max_pairs is not None:
        print(json.dumps({"debug_pairs_complete": len(tasks), "release_evidence": False}))
        return
    latest = load_latest_scores(score_path)
    try:
        summary = aggregate(args, queries, retrieval, latest, output_root)
    except RuntimeError as exc:
        write_json(output_root / "INCOMPLETE.json", {"error": str(exc)})
        raise SystemExit(3) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
