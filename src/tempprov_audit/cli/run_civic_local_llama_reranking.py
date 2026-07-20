from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tempprov_audit.cli.run_civic_llm_reranking import (
    SYSTEM_PROMPT,
    aggregate,
    append_jsonl,
    build_user_prompt,
    load_latest_scores,
    make_tasks,
    parse_score,
    read_jsonl_gz,
    sha256_file,
    task_key,
)
from tempprov_audit.cli.run_ctd_provenance_lexical_baselines import write_json
from tempprov_audit.cli.run_corpus_retrieval import read_documents, read_queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank the frozen CIViC Top-k pool with local LLaMA.")
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--document-file", required=True)
    parser.add_argument("--retrieval-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--model-name",
        default="hugging-quants/Meta-Llama-3.1-8B-Instruct-BNB-NF4",
    )
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--max-document-words", type=int, default=512)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument(
        "--rerun-legacy-input-limit",
        type=int,
        help=(
            "Recompute legacy rows that reached this input-token limit and do not yet "
            "record max_input_tokens_used. This repairs right-truncated prompts without "
            "recomputing unaffected rows."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--allow-subset-debug", action="store_true")
    parser.add_argument("--resume-valid-only", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260705)
    return parser.parse_args()


def render_prompt(tokenizer: Any, task: dict[str, Any], max_document_words: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(task["query"], task["document"], max_document_words)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def is_complete_row(row: dict[str, Any], rerun_legacy_input_limit: int | None) -> bool:
    if row.get("score") is None or row.get("parse_error"):
        return False
    if (
        rerun_legacy_input_limit is not None
        and row.get("max_input_tokens_used") is None
        and (row.get("input_tokens") or 0) >= rerun_legacy_input_limit
    ):
        return False
    return True


def main() -> None:
    args = parse_args()
    args.provider = "local_llama"
    if args.max_pairs is not None and not args.allow_subset_debug:
        raise SystemExit("--max-pairs is restricted to --allow-subset-debug and cannot produce release-grade evidence.")

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
        "provider": "local_llama",
        "model_name": args.model_name,
        "quantization": "pre-quantized bitsandbytes NF4",
        "system_prompt": SYSTEM_PROMPT,
        "score_range": [0, 100],
        "tie_break": "original_medcpt_dual_rank",
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "rerun_legacy_input_limit": args.rerun_legacy_input_limit,
        "max_document_words": args.max_document_words,
        "retrieval_depth": args.depth,
        "query_sha256": sha256_file(query_path),
        "document_sha256": sha256_file(document_path),
        "retrieval_sha256": sha256_file(retrieval_path),
    }
    write_json(output_root / "protocol.json", protocol)

    existing = load_latest_scores(score_path)
    done = {
        key
        for key, row in existing.items()
        if is_complete_row(row, args.rerun_legacy_input_limit)
    }
    if not args.resume_valid_only:
        existing = {}
        done = set()
        score_path.write_text("", encoding="utf-8")
        raw_path.write_text("", encoding="utf-8")
    pending = [task for task in tasks if task_key(task["example_id"], task["document_id"]) not in done]

    if pending:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()
        torch.cuda.reset_peak_memory_stats()

        started = time.perf_counter()
        processed = 0
        mode = "a" if args.resume_valid_only else "w"
        with score_path.open(mode, encoding="utf-8") as score_handle, raw_path.open(
            mode, encoding="utf-8"
        ) as raw_handle:
            for offset in range(0, len(pending), args.batch_size):
                batch = pending[offset : offset + args.batch_size]
                prompts = [render_prompt(tokenizer, task, args.max_document_words) for task in batch]
                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_input_tokens,
                ).to(model.device)
                batch_started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                batch_latency = time.perf_counter() - batch_started
                input_width = encoded["input_ids"].shape[1]
                outputs = tokenizer.batch_decode(generated[:, input_width:], skip_special_tokens=True)
                input_tokens = encoded["attention_mask"].sum(dim=1).tolist()
                output_tokens = (generated[:, input_width:] != tokenizer.pad_token_id).sum(dim=1).tolist()
                for task, text, in_tokens, out_tokens in zip(
                    batch, outputs, input_tokens, output_tokens, strict=True
                ):
                    parse_error = None
                    score = None
                    try:
                        score = parse_score(text)
                    except Exception as exc:  # noqa: BLE001
                        parse_error = str(exc)
                    row = {
                        "example_id": task["example_id"],
                        "document_id": task["document_id"],
                        "original_rank": task["original_rank"],
                        "provider": "local_llama",
                        "model_name": args.model_name,
                        "score": score,
                        "latency_seconds": round(batch_latency / len(batch), 6),
                        "request_attempts": 1,
                        "input_tokens": int(in_tokens),
                        "output_tokens": int(out_tokens),
                        "max_input_tokens_used": args.max_input_tokens,
                        "parse_error": parse_error,
                    }
                    raw = {
                        "example_id": task["example_id"],
                        "document_id": task["document_id"],
                        "provider": "local_llama",
                        "model_name": args.model_name,
                        "raw_text": text,
                        "max_input_tokens_used": args.max_input_tokens,
                        "parse_error": parse_error,
                    }
                    append_jsonl(score_handle, row)
                    append_jsonl(raw_handle, raw)
                    existing[task_key(task["example_id"], task["document_id"])] = row
                processed += len(batch)
                if processed == len(pending) or processed % max(120, args.batch_size * 10) == 0:
                    valid = sum(
                        row.get("score") is not None and not row.get("parse_error") for row in existing.values()
                    )
                    print(
                        json.dumps(
                            {
                                "provider": "local_llama",
                                "processed_this_run": processed,
                                "pending_this_run": len(pending),
                                "valid_total": valid,
                                "expected_total": len(queries) * args.depth,
                                "pairs_per_second": round(processed / (time.perf_counter() - started), 3),
                            }
                        ),
                        flush=True,
                    )
        local_efficiency = {
            "wall_seconds_this_run": time.perf_counter() - started,
            "pairs_this_run": processed,
            "pairs_per_second_this_run": processed / (time.perf_counter() - started),
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "gpu": torch.cuda.get_device_name(0),
        }
        write_json(output_root / "local_efficiency.json", local_efficiency)

    if args.max_pairs is not None:
        print(json.dumps({"debug_pairs_complete": len(tasks), "release_evidence": False}))
        return
    latest = load_latest_scores(score_path)
    try:
        summary = aggregate(args, queries, retrieval, latest, output_root)
    except RuntimeError as exc:
        write_json(output_root / "INCOMPLETE.json", {"error": str(exc)})
        raise SystemExit(3) from exc
    if (output_root / "local_efficiency.json").exists():
        summary["local_efficiency"] = json.loads(
            (output_root / "local_efficiency.json").read_text(encoding="utf-8")
        )
        write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
