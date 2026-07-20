from __future__ import annotations

import argparse
import json
import math
import random
import shutil
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
DEFAULT_QUERY_PREFIX = (
    "Identify a publication that directly supports the CTD chemical-disease relation. "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate a provenance Cross-Encoder.")
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--dataset", default="tail_disjoint_pmid_isolated")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("zero_shot", "finetune"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 3407, 8128, 2025])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--query-prefix",
        default=DEFAULT_QUERY_PREFIX,
        help="Text prepended to each benchmark query; pass an empty string for raw-query models.",
    )
    parser.add_argument(
        "--raw-document",
        action="store_true",
        help="Pass title/abstract text without the benchmark publication-year prefix.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--discard-best-model-after-scoring",
        action="store_true",
        help="Delete only the run-local best_model directory after predictions and summaries are durable.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def discard_run_best_model(run_dir: Path) -> None:
    resolved_run = run_dir.resolve()
    target = (resolved_run / "best_model").resolve()
    if target.parent != resolved_run or target.name != "best_model":
        raise RuntimeError(f"Refusing to remove unsafe checkpoint path: {target}")
    if target.exists():
        shutil.rmtree(target)


def flatten_rows(
    rows: list[dict[str, Any]],
    query_prefix: str = DEFAULT_QUERY_PREFIX,
    include_publication_year: bool = True,
) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        gold = set(row["gold_document_ids"])
        for candidate_index, candidate in enumerate(row["candidates"]):
            publication_year = candidate.get("pub_year") or "unknown"
            document = str(candidate.get("text", ""))
            if include_publication_year:
                document = f"Publication year: {publication_year}. {document}"
            flat.append(
                {
                    "row_index": row_index,
                    "candidate_index": candidate_index,
                    "query": f"{query_prefix}{row['query_text']}",
                    "document": document,
                    "label": float(candidate["document_id"] in gold),
                }
            )
    return flat


def build_collator(tokenizer: Any, max_length: int, include_labels: bool):
    import torch

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [item["query"] for item in batch],
            [item["document"] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["row_index"] = torch.tensor([item["row_index"] for item in batch])
        encoded["candidate_index"] = torch.tensor([item["candidate_index"] for item in batch])
        if include_labels:
            encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.float)
        return encoded

    return collate


def score_rows(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    device: Any,
    batch_size: int,
    max_length: int,
    num_workers: int,
    query_prefix: str = DEFAULT_QUERY_PREFIX,
    include_publication_year: bool = True,
) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    flat = flatten_rows(rows, query_prefix, include_publication_year)
    loader = DataLoader(
        flat,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=build_collator(tokenizer, max_length, include_labels=False),
    )
    scores_by_row = [[0.0] * len(row["candidates"]) for row in rows]
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            row_indices = batch.pop("row_index").tolist()
            candidate_indices = batch.pop("candidate_index").tolist()
            inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(**inputs).logits.reshape(-1).float().cpu().tolist()
            for row_index, candidate_index, score in zip(row_indices, candidate_indices, logits):
                scores_by_row[row_index][candidate_index] = score

    results: list[dict[str, Any]] = []
    for row, scores in zip(rows, scores_by_row):
        ranked = sorted(
            zip(row["candidates"], scores),
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


def train_seed(args: argparse.Namespace, rows: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, Any]:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    seed_everything(seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        ignore_mismatched_sizes=True,
    ).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    run_dir = Path(args.output_root).resolve() / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_flat = flatten_rows(
        rows["train"], args.query_prefix, not args.raw_document
    )
    positive_count = sum(item["label"] for item in train_flat)
    negative_count = len(train_flat) - positive_count
    positive_weight = torch.tensor([negative_count / max(positive_count, 1.0)], device=device)
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_flat,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=build_collator(tokenizer, args.max_length, include_labels=True),
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    best_map = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch.pop("row_index")
            batch.pop("candidate_index")
            labels = batch.pop("labels").to(device, non_blocking=True)
            inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(**inputs).logits.reshape(-1)
                loss = criterion(logits, labels) / args.gradient_accumulation
            scaler.scale(loss).backward()
            running_loss += float(loss.detach()) * args.gradient_accumulation
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        dev_results = score_rows(
            model,
            tokenizer,
            rows["dev"],
            device=device,
            batch_size=args.batch_size * 2,
            max_length=args.max_length,
            num_workers=args.num_workers,
            query_prefix=args.query_prefix,
            include_publication_year=not args.raw_document,
        )
        dev_summary = summarize_ranking(dev_results, 0, seed)
        dev_map = dev_summary["metrics"]["map"]["estimate"]
        history.append({"epoch": epoch, "train_loss_sum": running_loss, "dev_map": dev_map})
        write_json(
            run_dir / "progress.json",
            {
                "status": "training",
                "seed": seed,
                "epoch": epoch,
                "maximum_epochs": args.epochs,
                "train_loss_sum": running_loss,
                "dev_map": dev_map,
                "best_dev_map_before_epoch": best_map,
                "stale_epochs_before_epoch": stale_epochs,
            },
        )
        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss_sum": running_loss,
                    "dev_map": dev_map,
                }
            ),
            flush=True,
        )
        if dev_map > best_map:
            best_map = dev_map
            best_epoch = epoch
            stale_epochs = 0
            model.save_pretrained(run_dir / "best_model")
            tokenizer.save_pretrained(run_dir / "best_model")
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stopping_patience:
                break

    model = AutoModelForSequenceClassification.from_pretrained(run_dir / "best_model").to(device)
    split_results = {
        split: score_rows(
            model,
            tokenizer,
            rows[split],
            device=device,
            batch_size=args.batch_size * 2,
            max_length=args.max_length,
            num_workers=args.num_workers,
            query_prefix=args.query_prefix,
            include_publication_year=not args.raw_document,
        )
        for split in ("dev", "test")
    }
    policies = select_thresholds(split_results["dev"])
    summary = {
        "seed": seed,
        "model": args.model,
        "mode": args.mode,
        "training_candidates": len(train_flat),
        "training_positive_candidates": int(positive_count),
        "positive_weight": float(positive_weight.item()),
        "best_epoch": best_epoch,
        "best_dev_map": best_map,
        "history": history,
        "checkpoint_retained": not args.discard_best_model_after_scoring,
        "ranking": {
            split: summarize_ranking(
                results, args.bootstrap_resamples, seed + index
            )
            for index, (split, results) in enumerate(split_results.items())
        },
        "threshold_selection": policies,
        "test_pair_classification": {
            policy: classification_metrics(split_results["test"], values["threshold"])
            for policy, values in policies.items()
        },
    }
    for split, results in split_results.items():
        write_json(run_dir / f"{split}_predictions.json", results)
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "progress.json",
        {"status": "complete", "seed": seed, "best_epoch": best_epoch, "best_dev_map": best_map},
    )
    if args.discard_best_model_after_scoring:
        discard_run_best_model(run_dir)
    return summary


def zero_shot(args: argparse.Namespace, rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    split_results = {
        split: score_rows(
            model,
            tokenizer,
            rows[split],
            device=device,
            batch_size=args.batch_size * 2,
            max_length=args.max_length,
            num_workers=args.num_workers,
            query_prefix=args.query_prefix,
            include_publication_year=not args.raw_document,
        )
        for split in ("dev", "test")
    }
    policies = select_thresholds(split_results["dev"])
    summary = {
        "model": args.model,
        "mode": args.mode,
        "ranking": {
            split: summarize_ranking(results, args.bootstrap_resamples, args.seeds[0] + index)
            for index, (split, results) in enumerate(split_results.items())
        },
        "threshold_selection": policies,
        "test_pair_classification": {
            policy: classification_metrics(split_results["test"], values["threshold"])
            for policy, values in policies.items()
        },
    }
    output_root = Path(args.output_root).resolve()
    for split, results in split_results.items():
        write_json(output_root / f"{split}_predictions.json", results)
    write_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if not __import__("torch").cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the full-scale Cross-Encoder experiment.")
    benchmark_root = Path(args.benchmark_root).resolve() / args.dataset
    rows = {split: read_jsonl(benchmark_root / f"{split}.jsonl") for split in SPLITS}
    if args.mode == "zero_shot":
        payload = zero_shot(args, rows)
    else:
        runs = []
        output_root = Path(args.output_root).resolve()
        for seed in args.seeds:
            completed_summary = output_root / f"seed_{seed}" / "summary.json"
            if completed_summary.exists():
                print(json.dumps({"event": "resume_skip_completed_seed", "seed": seed}), flush=True)
                runs.append(json.loads(completed_summary.read_text(encoding="utf-8")))
            else:
                runs.append(train_seed(args, rows, seed))
        payload = {
            "model": args.model,
            "mode": args.mode,
            "dataset": args.dataset,
            "runs": runs,
        }
        write_json(Path(args.output_root).resolve() / "all_seeds_summary.json", payload)
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
