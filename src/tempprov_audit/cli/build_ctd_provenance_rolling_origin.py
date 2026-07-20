from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from tempprov_audit.cli.build_ctd_provenance_benchmark import parse_pmids, stable_rng


YEARS = (2022, 2023, 2024, 2025)
SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build disease-tail-disjoint rolling-origin CTD publication-provenance folds."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--ctd-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--test-years", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--tail-grouped-folds", type=int, default=5)
    parser.add_argument("--same-tail-negatives", type=int, default=20)
    parser.add_argument("--same-head-negatives", type=int, default=10)
    parser.add_argument("--random-negatives", type=int, default=20)
    parser.add_argument("--minimum-target-year-candidates", type=int, default=40)
    parser.add_argument("--historical-context", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260624)
    return parser.parse_args()


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".gz" else path.open(encoding="utf-8", errors="replace")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_documents(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_pmid: dict[str, dict[str, Any]] = {}
    by_key: dict[str, dict[str, Any]] = {}
    with open_text(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            by_key[row["document_key"]] = row
            pmid = str(row.get("pmid") or "").strip()
            if pmid:
                by_pmid[pmid] = row
    return by_pmid, by_key


def parse_ctd_direct_pairs(
    lines: Iterable[str], documents_by_pmid: dict[str, dict[str, Any]]
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    data_rows = direct_rows = retained_links = 0
    for line in lines:
        if not line or line.startswith("#"):
            continue
        data_rows += 1
        columns = line.rstrip("\r\n").split("\t")
        if len(columns) < 10 or not columns[5].strip():
            continue
        direct_rows += 1
        linked_pmids = [pmid for pmid in parse_pmids(columns[9]) if pmid in documents_by_pmid]
        if not linked_pmids:
            continue
        key = (columns[1].strip(), columns[4].strip())
        record = pairs.setdefault(
            key,
            {
                "head_id": key[0],
                "head_text": columns[0].strip(),
                "tail_id": key[1],
                "tail_text": columns[3].strip(),
                "pmids_by_year": defaultdict(set),
            },
        )
        for pmid in linked_pmids:
            year = documents_by_pmid[pmid].get("pub_year")
            if isinstance(year, int):
                record["pmids_by_year"][year].add(pmid)
                retained_links += 1
    recent_pairs = {
        key: record
        for key, record in pairs.items()
        if any(record["pmids_by_year"].get(year) for year in YEARS)
    }
    return recent_pairs, {
        "ctd_data_rows": data_rows,
        "ctd_direct_evidence_rows": direct_rows,
        "retained_direct_pmid_links_before_deduplication": retained_links,
        "recent_pairs": len(recent_pairs),
        "recent_heads": len({key[0] for key in recent_pairs}),
        "recent_tails": len({key[1] for key in recent_pairs}),
    }


def build_year_pools(
    pairs: dict[tuple[str, str], dict[str, Any]], years: Iterable[int]
) -> tuple[dict[tuple[str, int], set[str]], dict[tuple[str, int], set[str]], dict[int, set[str]]]:
    head_pool: dict[tuple[str, int], set[str]] = defaultdict(set)
    tail_pool: dict[tuple[str, int], set[str]] = defaultdict(set)
    global_pool: dict[int, set[str]] = defaultdict(set)
    for (head_id, tail_id), record in pairs.items():
        for year in years:
            pmids = set(record["pmids_by_year"].get(year, set()))
            head_pool[(head_id, year)].update(pmids)
            tail_pool[(tail_id, year)].update(pmids)
            global_pool[year].update(pmids)
    return head_pool, tail_pool, global_pool


def assign_tail_groups(
    pairs: dict[tuple[str, str], dict[str, Any]], fold_count: int, seed: int
) -> list[set[str]]:
    if fold_count < 3:
        raise ValueError("Rolling-origin evaluation requires at least three tail folds.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pairs.values():
        grouped[record["tail_id"]].append(record)
    keys = ("pairs", *(f"positive_{year}" for year in YEARS))
    totals = {"pairs": len(pairs) / fold_count}
    totals.update(
        {
            f"positive_{year}": sum(bool(record["pmids_by_year"].get(year)) for record in pairs.values()) / fold_count
            for year in YEARS
        }
    )
    folds = [set() for _ in range(fold_count)]
    stats = [Counter() for _ in range(fold_count)]
    ordered = sorted(
        grouped.items(),
        key=lambda item: (
            -len(item[1]),
            -sum(bool(record["pmids_by_year"].get(year)) for record in item[1] for year in YEARS),
            hashlib.sha256(f"{seed}|{item[0]}".encode()).hexdigest(),
        ),
    )
    for tail_id, records in ordered:
        group = {"pairs": len(records)}
        group.update(
            {
                f"positive_{year}": sum(bool(record["pmids_by_year"].get(year)) for record in records)
                for year in YEARS
            }
        )

        def score(index: int) -> tuple[float, int]:
            return (
                sum(((stats[index][key] + group[key]) / max(totals[key], 1.0)) ** 2 for key in keys),
                index,
            )

        selected = min(range(fold_count), key=score)
        folds[selected].add(tail_id)
        stats[selected].update(group)
    return folds


def sample_pmids(
    values: Iterable[str], count: int, rng: random.Random, excluded: set[str]
) -> list[str]:
    candidates = sorted(set(values) - excluded)
    rng.shuffle(candidates)
    return candidates[: max(0, count)]


def compact_candidate(document: dict[str, Any], source: str, is_gold: bool) -> dict[str, Any]:
    return {
        "document_key": document["document_key"],
        "document_id": document["document_id"],
        "pmid": document.get("pmid"),
        "pub_year": document.get("pub_year"),
        "source": source,
        "is_ctd_cited": is_gold,
    }


def build_pair_year_example(
    record: dict[str, Any],
    year: int,
    documents_by_pmid: dict[str, dict[str, Any]],
    head_pool: dict[tuple[str, int], set[str]],
    tail_pool: dict[tuple[str, int], set[str]],
    global_pool: dict[int, set[str]],
    *,
    seed: int,
    same_tail_negatives: int,
    same_head_negatives: int,
    random_negatives: int,
    minimum_target_year_candidates: int,
    historical_context: int,
) -> dict[str, Any]:
    head_id, tail_id = record["head_id"], record["tail_id"]
    gold_pmids = set(record["pmids_by_year"].get(year, set()))
    rng = stable_rng(seed, head_id, tail_id, str(year))
    candidates: dict[str, dict[str, Any]] = {}

    def add(pmids: Iterable[str], source: str, is_gold: bool = False) -> None:
        for pmid in pmids:
            document = documents_by_pmid.get(pmid)
            if document is None:
                continue
            candidates.setdefault(pmid, compact_candidate(document, source, is_gold))

    add(sorted(gold_pmids), "gold_target_year_ctd_citation", True)
    excluded = set(gold_pmids)
    same_tail = sample_pmids(
        tail_pool.get((tail_id, year), set()), same_tail_negatives, rng, excluded
    )
    add(same_tail, "target_year_same_tail_unlabelled")
    excluded.update(same_tail)
    same_head = sample_pmids(
        head_pool.get((head_id, year), set()), same_head_negatives, rng, excluded
    )
    add(same_head, "target_year_same_head_unlabelled")
    excluded.update(same_head)
    random_year = sample_pmids(global_pool.get(year, set()), random_negatives, rng, excluded)
    add(random_year, "target_year_random_unlabelled")
    excluded.update(random_year)

    target_count = sum(candidate["pub_year"] == year for candidate in candidates.values())
    if target_count < minimum_target_year_candidates:
        fill = sample_pmids(
            global_pool.get(year, set()),
            minimum_target_year_candidates - target_count,
            rng,
            excluded,
        )
        add(fill, "target_year_global_fill_unlabelled")
        excluded.update(fill)

    historical_pmids = {
        pmid
        for publication_year, pmids in record["pmids_by_year"].items()
        if publication_year < year
        for pmid in pmids
    }
    history = sample_pmids(historical_pmids, historical_context, rng, excluded)
    add(history, "prior_pair_ctd_citation_context")
    ordered = list(candidates.values())
    rng.shuffle(ordered)
    pair_id = f"{head_id}__{tail_id}"
    return {
        "dataset_name": "ctd_publication_provenance_rolling_origin",
        "example_id": f"{pair_id}__publication_year_{year}",
        "pair_id": pair_id,
        "head_id": head_id,
        "head_text": record["head_text"],
        "tail_id": tail_id,
        "tail_text": record["tail_text"],
        "query_text": f"Chemical: {record['head_text']}. Disease: {record['tail_text']}.",
        "evaluation_year": year,
        "cutoff_year": year - 1,
        "gold_document_ids": sorted(f"PMID:{pmid}" for pmid in gold_pmids),
        "declared_gold_pmids": sorted(gold_pmids),
        "candidates": ordered,
        "label_semantics": {
            "positive": "target-year PMID directly cited by CTD for this chemical-disease pair",
            "unlabelled": "candidate publication not directly cited by CTD for this pair-year",
            "caution": "Unlabelled is not a confirmed biological negative.",
        },
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_counts = [len(row["candidates"]) for row in rows]
    gold_counts = [len(row["gold_document_ids"]) for row in rows]
    return {
        "examples": len(rows),
        "positive_pair_years": sum(bool(count) for count in gold_counts),
        "negative_unlabelled_pair_years": sum(not count for count in gold_counts),
        "gold_documents": sum(gold_counts),
        "candidate_instances": sum(candidate_counts),
        "candidate_mean": mean(candidate_counts) if candidate_counts else 0.0,
        "candidate_min": min(candidate_counts) if candidate_counts else 0,
        "candidate_max": max(candidate_counts) if candidate_counts else 0,
        "tail_count": len({row["tail_id"] for row in rows}),
        "evaluation_years": sorted({row["evaluation_year"] for row in rows}),
        "nonhistorical_year_violations": sum(
            candidate.get("source") != "prior_pair_ctd_citation_context"
            and candidate.get("pub_year") != row["evaluation_year"]
            for row in rows
            for candidate in row["candidates"]
        ),
    }


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def build_window_folds(
    pairs: dict[tuple[str, str], dict[str, Any]],
    documents_by_pmid: dict[str, dict[str, Any]],
    tail_folds: list[set[str]],
    test_year: int,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    head_pool, tail_pool, global_pool = build_year_pools(pairs, YEARS)
    train_years = list(range(args.start_year, test_year - 1))
    dev_year = test_year - 1
    folds: dict[str, dict[str, list[dict[str, Any]]]] = {}
    audit: dict[str, Any] = {
        "train_years": train_years,
        "dev_year": dev_year,
        "test_year": test_year,
        "folds": {},
    }
    for fold_index in range(len(tail_folds)):
        test_tails = tail_folds[fold_index]
        dev_tails = tail_folds[(fold_index + 1) % len(tail_folds)]
        split_rows = {split: [] for split in SPLITS}
        for record in pairs.values():
            tail_id = record["tail_id"]
            if tail_id in test_tails:
                split_years = (("test", test_year),)
            elif tail_id in dev_tails:
                split_years = (("dev", dev_year),)
            else:
                split_years = tuple(("train", year) for year in train_years)
            for split, year in split_years:
                row = build_pair_year_example(
                    record,
                    year,
                    documents_by_pmid,
                    head_pool,
                    tail_pool,
                    global_pool,
                    seed=args.seed,
                    same_tail_negatives=args.same_tail_negatives,
                    same_head_negatives=args.same_head_negatives,
                    random_negatives=args.random_negatives,
                    minimum_target_year_candidates=args.minimum_target_year_candidates,
                    historical_context=args.historical_context,
                )
                row["split"] = split
                split_rows[split].append(row)
        name = f"fold_{fold_index}"
        folds[name] = split_rows
        split_tails = {
            split: {row["tail_id"] for row in rows} for split, rows in split_rows.items()
        }
        gold_pmids = {
            split: {pmid for row in rows for pmid in row["declared_gold_pmids"]}
            for split, rows in split_rows.items()
        }
        audit["folds"][name] = {
            "splits": {split: summarize_rows(rows) for split, rows in split_rows.items()},
            "tail_overlap": {
                "train_vs_dev": len(split_tails["train"] & split_tails["dev"]),
                "train_vs_test": len(split_tails["train"] & split_tails["test"]),
                "dev_vs_test": len(split_tails["dev"] & split_tails["test"]),
            },
            "gold_pmid_overlap": {
                "train_vs_dev": len(gold_pmids["train"] & gold_pmids["dev"]),
                "train_vs_test": len(gold_pmids["train"] & gold_pmids["test"]),
                "dev_vs_test": len(gold_pmids["dev"] & gold_pmids["test"]),
            },
        }
    return folds, audit


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    ctd_path = Path(args.ctd_path).resolve()
    document_path = source_root / "documents.jsonl.gz"
    documents_by_pmid, documents_by_key = load_documents(document_path)
    with open_text(ctd_path) as handle:
        pairs, source_audit = parse_ctd_direct_pairs(handle, documents_by_pmid)
    tail_folds = assign_tail_groups(pairs, args.tail_grouped_folds, args.seed)

    output_root.mkdir(parents=True, exist_ok=True)
    target_document_path = output_root / "documents.jsonl.gz"
    if target_document_path.resolve() != document_path.resolve():
        temporary = target_document_path.with_suffix(target_document_path.suffix + ".tmp")
        shutil.copy2(document_path, temporary)
        temporary.replace(target_document_path)

    windows: dict[str, Any] = {}
    for test_year in args.test_years:
        if test_year <= args.start_year + 1:
            raise ValueError(f"Test year {test_year} leaves no earlier training year.")
        window_name = f"test_{test_year}"
        folds, audit = build_window_folds(
            pairs, documents_by_pmid, tail_folds, test_year, args
        )
        for fold_name, split_rows in folds.items():
            for split, rows in split_rows.items():
                write_jsonl_gz(
                    output_root / window_name / fold_name / f"{split}.jsonl.gz", rows
                )
        windows[window_name] = audit

    year_summary = {
        str(year): {
            "positive_pairs": sum(bool(record["pmids_by_year"].get(year)) for record in pairs.values()),
            "gold_pmids": len(
                {
                    pmid
                    for record in pairs.values()
                    for pmid in record["pmids_by_year"].get(year, set())
                }
            ),
        }
        for year in YEARS
    }
    manifest = {
        "benchmark": "ctd_publication_provenance_rolling_origin",
        "version": "2026-06-24",
        "protocol": "publication-year rolling origin with disease-tail-disjoint outer folds",
        "scope_caution": "Retrospective CTD publication provenance, not CTD curation-date reconstruction or independent-database validation.",
        "unlabelled_caution": "Non-cited candidates are unlabelled, not confirmed biological negatives.",
        "seed": args.seed,
        "years": list(YEARS),
        "test_years": args.test_years,
        "source": {
            "ctd_path": str(ctd_path),
            "ctd_sha256": sha256(ctd_path),
            "document_store_path": str(document_path),
            "document_store_sha256": sha256(document_path),
            "document_records": len(documents_by_key),
            **source_audit,
        },
        "candidate_budgets": {
            "same_tail": args.same_tail_negatives,
            "same_head": args.same_head_negatives,
            "random": args.random_negatives,
            "minimum_target_year_candidates": args.minimum_target_year_candidates,
            "historical_context": args.historical_context,
        },
        "year_summary": year_summary,
        "tail_fold_count": len(tail_folds),
        "tail_fold_sizes": [len(fold) for fold in tail_folds],
        "unique_tail_assignments": len(set().union(*tail_folds)),
        "windows": windows,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
