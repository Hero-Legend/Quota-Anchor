from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tempprov_audit.cli.run_ctd_provenance_cross_encoder import score_rows
from tempprov_audit.cli.run_ctd_provenance_lexical_baselines import (
    read_jsonl,
    summarize_ranking,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the locked BioRED external provenance benchmark."
    )
    parser.add_argument(
        "--benchmark-root",
        default="outputs/biored_external_provenance",
    )
    parser.add_argument("--view", default="document_disjoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--minimum-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--query-prefix", default=None)
    parser.add_argument("--raw-document", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load models only from local paths or the local Hugging Face cache.",
    )
    return parser.parse_args()


def score_with_adaptive_batch(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    device: Any,
    batch_size: int,
    minimum_batch_size: int,
    max_length: int,
    num_workers: int,
    query_prefix: str | None = None,
    raw_document: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    import torch

    current = batch_size
    while True:
        try:
            return (
                score_rows(
                    model,
                    tokenizer,
                    rows,
                    device=device,
                    batch_size=current,
                    max_length=max_length,
                    num_workers=num_workers,
                    **({} if query_prefix is None else {"query_prefix": query_prefix}),
                    include_publication_year=not raw_document,
                ),
                current,
            )
        except RuntimeError as error:
            is_cuda_oom = isinstance(error, torch.OutOfMemoryError) or (
                "out of memory" in str(error).lower()
                and "cuda" in str(error).lower()
            )
            if not is_cuda_oom:
                raise
            if current <= minimum_batch_size:
                raise
            current = max(minimum_batch_size, current // 2)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "event": "cuda_oom_retry",
                        "retry_batch_size": current,
                    }
                ),
                flush=True,
            )


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    benchmark_root = Path(args.benchmark_root).resolve()
    output_root = Path(args.output_root).resolve()
    completed = output_root / "COMPLETED"
    if completed.exists() and (output_root / "external_predictions.json").exists():
        print(json.dumps({"status": "already_complete", "output": str(output_root)}))
        return

    rows = read_jsonl(benchmark_root / args.view / "test.jsonl.gz")
    print(
        json.dumps(
            {
                "event": "load_model",
                "model": args.model,
                "model_label": args.model_label,
                "queries": len(rows),
                "candidate_instances": sum(len(row["candidates"]) for row in rows),
            }
        ),
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    device = torch.device("cuda")
    model = model.to(device)
    predictions, effective_batch_size = score_with_adaptive_batch(
        model,
        tokenizer,
        rows,
        device=device,
        batch_size=args.batch_size,
        minimum_batch_size=args.minimum_batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        query_prefix=args.query_prefix,
        raw_document=args.raw_document,
    )
    summary = {
        "model": args.model,
        "model_label": args.model_label,
        "benchmark_root": str(benchmark_root),
        "view": args.view,
        "queries": len(rows),
        "candidate_instances": sum(len(row["candidates"]) for row in rows),
        "requested_batch_size": args.batch_size,
        "effective_batch_size": effective_batch_size,
        "ranking": summarize_ranking(
            predictions, args.bootstrap_resamples, args.seed
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "external_predictions.json", predictions)
    write_json(output_root / "summary.json", summary)
    completed.write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
