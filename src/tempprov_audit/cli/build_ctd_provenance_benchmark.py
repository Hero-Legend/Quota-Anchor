from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import random
import re
import time
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


SPECS = {
    "primary_balanced": (
        Path("outputs/temporal_validation/model_inputs"),
        "ctd_202111_primary_balanced_temporal_update_clean_20260430.model_input",
    ),
    "tail_disjoint": (
        Path("outputs/temporal_validation/model_inputs/tail_disjoint"),
        "ctd_202111_primary_balanced_temporal_update_tail_disjoint_clean_20260430.model_input",
    ),
}
SPLITS = ("train", "dev", "test")
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CTD-citation document-provenance benchmark with matched post-cut distractors."
    )
    parser.add_argument("--workspace", default=".")
    parser.add_argument(
        "--output-root",
        default="outputs/ctd_provenance_benchmark",
    )
    parser.add_argument("--same-tail-negatives", type=int, default=8)
    parser.add_argument("--same-head-negatives", type=int, default=4)
    parser.add_argument("--random-negatives", type=int, default=8)
    parser.add_argument("--ctd-full-path", default="")
    parser.add_argument(
        "--minimum-postcut-year",
        type=int,
        default=2022,
        help="Conservative year filter for a 2021-11-01 cut; excludes ambiguous 2021 records.",
    )
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--email", default="")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--fetch-checkpoint-size", type=int, default=1000)
    parser.add_argument("--fetch-workers", type=int, default=3)
    parser.add_argument("--tail-grouped-folds", type=int, default=5)
    parser.add_argument("--no-fetch", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_pmids(value: Any) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\d+", str(value or ""))))


def element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def pubmed_year(article: ET.Element) -> int | None:
    for xpath in (
        ".//JournalIssue/PubDate/Year",
        ".//ArticleDate/Year",
        ".//PubMedPubDate[@PubStatus='pubmed']/Year",
    ):
        value = element_text(article.find(xpath))
        if value.isdigit():
            return int(value)
    medline_date = element_text(article.find(".//JournalIssue/PubDate/MedlineDate"))
    match = re.search(r"(?:18|19|20)\d{2}", medline_date)
    return int(match.group()) if match else None


def parse_pubmed_xml(payload: bytes) -> dict[str, dict[str, Any]]:
    root = ET.fromstring(payload)
    records: dict[str, dict[str, Any]] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = element_text(article.find(".//MedlineCitation/PMID"))
        if not pmid:
            continue
        title = element_text(article.find(".//Article/ArticleTitle"))
        abstract_parts = [element_text(item) for item in article.findall(".//Article/Abstract/AbstractText")]
        abstract = " ".join(part for part in abstract_parts if part)
        records[pmid] = {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "pub_year": pubmed_year(article),
            "text": " ".join(part for part in (title, abstract) if part).strip(),
        }
    return records


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def fetch_pubmed_records(
    pmids: list[str],
    *,
    email: str,
    batch_size: int,
    workers: int = 1,
    max_attempts: int = 5,
) -> dict[str, dict[str, Any]]:
    batches = list(chunks(sorted(set(pmids)), max(1, batch_size)))

    def fetch_batch(batch_index: int, batch: list[str]) -> dict[str, dict[str, Any]]:
        params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml", "tool": "tempprov-provenance"}
        if email:
            params["email"] = email
        url = EFETCH_URL + "?" + urllib.parse.urlencode(params)
        error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "tempprov-provenance/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    return parse_pubmed_xml(response.read())
            except Exception as exc:  # network errors are retried and surfaced after exhaustion
                error = exc
                time.sleep(min(16.0, 2.0**attempt))
        if error is not None:
            raise RuntimeError(f"NCBI EFetch failed for batch {batch_index + 1}: {error}") from error
        return {}

    records: dict[str, dict[str, Any]] = {}
    if workers <= 1:
        for batch_index, batch in enumerate(batches):
            records.update(fetch_batch(batch_index, batch))
            if batch_index + 1 < len(batches):
                time.sleep(0.4)
        return records

    futures: list[concurrent.futures.Future[dict[str, dict[str, Any]]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for batch_index, batch in enumerate(batches):
            futures.append(executor.submit(fetch_batch, batch_index, batch))
            if batch_index + 1 < len(batches):
                time.sleep(0.36)
        for future in concurrent.futures.as_completed(futures):
            records.update(future.result())
    return records


def extract_ctd_direct_pmid_pools(
    lines: Iterable[str], head_ids: set[str], tail_ids: set[str]
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, Any]]:
    head_pool: dict[str, set[str]] = defaultdict(set)
    tail_pool: dict[str, set[str]] = defaultdict(set)
    data_rows = direct_rows = matched_rows = 0
    for line in lines:
        if not line or line.startswith("#"):
            continue
        data_rows += 1
        columns = line.rstrip("\r\n").split("\t")
        if len(columns) < 10 or not columns[5].strip():
            continue
        direct_rows += 1
        head_id, tail_id = columns[1].strip(), columns[4].strip()
        if head_id not in head_ids and tail_id not in tail_ids:
            continue
        matched_rows += 1
        pmids = parse_pmids(columns[9])
        if head_id in head_ids:
            head_pool[head_id].update(pmids)
        if tail_id in tail_ids:
            tail_pool[tail_id].update(pmids)
    unique_pmids = set().union(*head_pool.values(), *tail_pool.values())
    return head_pool, tail_pool, {
        "data_rows": data_rows,
        "direct_evidence_rows": direct_rows,
        "matched_direct_evidence_rows": matched_rows,
        "matched_heads": len(head_pool),
        "matched_tails": len(tail_pool),
        "unique_pmids": len(unique_pmids),
    }


def load_ctd_direct_pmid_pools(
    path: Path, head_ids: set[str], tail_ids: set[str]
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        return extract_ctd_direct_pmid_pools(handle, head_ids, tail_ids)


def stable_rng(seed: int, *parts: str) -> random.Random:
    digest = hashlib.sha256("|".join((str(seed), *parts)).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def take_sample(values: Iterable[str], count: int, rng: random.Random, excluded: set[str]) -> list[str]:
    candidates = sorted(set(values) - excluded)
    rng.shuffle(candidates)
    return candidates[: max(0, count)]


def strip_sentence_prefix(text: str) -> str:
    return re.sub(r"^\[[^]]+\]\s*", "", str(text or "")).strip()


def historical_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    metadata: dict[str, dict[str, Any]] = {}
    for sentence in row.get("sentences", []):
        pmid = str(sentence.get("pmid") or sentence.get("doc_id") or "").strip()
        if not pmid:
            continue
        grouped[pmid].append(strip_sentence_prefix(sentence.get("text", "")))
        metadata[pmid] = {
            "pmid": pmid if pmid.isdigit() else None,
            "pub_year": sentence.get("pub_year"),
        }
    candidates: list[dict[str, Any]] = []
    for pmid in sorted(grouped):
        text = " ".join(item for item in grouped[pmid] if item).strip()
        if not text:
            continue
        candidates.append(
            {
                "document_id": f"PMID:{pmid}" if pmid.isdigit() else f"HIST:{pmid}",
                "pmid": metadata[pmid]["pmid"],
                "pub_year": metadata[pmid]["pub_year"],
                "title": "",
                "abstract": text,
                "text": text,
                "source": "precut_historical_context",
                "is_ctd_cited": False,
            }
        )
    return candidates


def postcut_candidate(record: dict[str, Any], source: str, is_gold: bool) -> dict[str, Any]:
    pmid = str(record["pmid"])
    return {
        "document_id": f"PMID:{pmid}",
        "pmid": pmid,
        "pub_year": record.get("pub_year"),
        "title": record.get("title", ""),
        "abstract": record.get("abstract", ""),
        "text": record.get("text", ""),
        "source": source,
        "is_ctd_cited": is_gold,
    }


def build_split(
    rows: list[dict[str, Any]],
    pubmed_records: dict[str, dict[str, Any]],
    *,
    dataset_name: str,
    split: str,
    seed: int,
    same_tail_negatives: int,
    same_head_negatives: int,
    random_negatives: int,
    global_tail_pool: dict[str, set[str]] | None = None,
    global_head_pool: dict[str, set[str]] | None = None,
    global_postcut_pool: set[str] | None = None,
) -> list[dict[str, Any]]:
    tail_pool: dict[str, set[str]] = defaultdict(set)
    head_pool: dict[str, set[str]] = defaultdict(set)
    split_pool: set[str] = set()
    for row in rows:
        gold_pmids = parse_pmids(row.get("target", {}).get("future_ctd_post_cut_pmids"))
        tail_pool[str(row.get("tail_id", ""))].update(gold_pmids)
        head_pool[str(row.get("head_id", ""))].update(gold_pmids)
        split_pool.update(gold_pmids)

    if global_tail_pool is not None:
        tail_pool = global_tail_pool
    if global_head_pool is not None:
        head_pool = global_head_pool
    if global_postcut_pool is not None:
        split_pool = global_postcut_pool

    outputs: list[dict[str, Any]] = []
    for row in rows:
        example_id = str(row.get("example_id") or row.get("document_id"))
        gold_pmids = parse_pmids(row.get("target", {}).get("future_ctd_post_cut_pmids"))
        fetched_gold = [pmid for pmid in gold_pmids if pmid in pubmed_records]
        excluded = set(gold_pmids)
        rng = stable_rng(seed, dataset_name, split, example_id)
        same_tail = take_sample(
            tail_pool.get(str(row.get("tail_id", "")), set()),
            same_tail_negatives,
            rng,
            excluded,
        )
        excluded.update(same_tail)
        same_head = take_sample(
            head_pool.get(str(row.get("head_id", "")), set()),
            same_head_negatives,
            rng,
            excluded,
        )
        excluded.update(same_head)
        random_pmids = take_sample(split_pool, random_negatives, rng, excluded)

        candidates_by_id: dict[str, dict[str, Any]] = {}
        for pmid in fetched_gold:
            candidate = postcut_candidate(pubmed_records[pmid], "gold_postcut_ctd_citation", True)
            candidates_by_id[candidate["document_id"]] = candidate
        for source, pmids in (
            ("postcut_same_tail_distractor", same_tail),
            ("postcut_same_head_distractor", same_head),
            ("postcut_random_distractor", random_pmids),
        ):
            for pmid in pmids:
                record = pubmed_records.get(pmid)
                if record is None:
                    continue
                candidate = postcut_candidate(record, source, False)
                candidates_by_id.setdefault(candidate["document_id"], candidate)
        for candidate in historical_candidates(row):
            candidates_by_id.setdefault(candidate["document_id"], candidate)

        candidates = list(candidates_by_id.values())
        rng.shuffle(candidates)
        gold_document_ids = sorted(f"PMID:{pmid}" for pmid in fetched_gold)
        outputs.append(
            {
                "dataset_name": dataset_name,
                "split": split,
                "example_id": example_id,
                "pair_id": row.get("pair_id"),
                "head_id": row.get("head_id"),
                "head_text": row.get("head_text"),
                "tail_id": row.get("tail_id"),
                "tail_text": row.get("tail_text"),
                "gold_relation": row.get("gold_relation"),
                "gold_novelty": row.get("gold_novelty"),
                "query_text": f"Chemical: {row.get('head_text', '')}. Disease: {row.get('tail_text', '')}.",
                "gold_document_ids": gold_document_ids,
                "declared_gold_pmids": gold_pmids,
                "missing_gold_pmids": sorted(set(gold_pmids) - set(fetched_gold)),
                "candidates": candidates,
                "label_semantics": {
                    "positive": "PMID cited by the post-cut CTD relation record",
                    "unlabelled": "matched or historical document not cited for this target pair",
                    "caution": "Unlabelled does not imply biologically irrelevant.",
                },
            }
        )
    return outputs


def legacy_supervision_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sentence_counts = [len(row.get("sentences", [])) for row in rows]
    gold_counts = [len(row.get("gold_evidence_sentence_ids", [])) for row in rows]
    all_gold = 0
    no_future_nonempty = 0
    for row in rows:
        sentence_ids = set(range(len(row.get("sentences", []))))
        gold_ids = {int(value) for value in row.get("gold_evidence_sentence_ids", [])}
        all_gold += sentence_ids == gold_ids
        if row.get("gold_relation") == "no_future_ctd_update" and gold_ids:
            no_future_nonempty += 1
    return {
        "examples": len(rows),
        "sentences": sum(sentence_counts),
        "gold_sentence_ids": sum(gold_counts),
        "all_input_sentences_labelled_gold_examples": all_gold,
        "all_input_sentences_labelled_gold_rate": all_gold / len(rows) if rows else 0.0,
        "no_future_examples_with_nonempty_gold_evidence": no_future_nonempty,
    }


def summarize_benchmark(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_counts = [len(row["candidates"]) for row in rows]
    gold_counts = [len(row["gold_document_ids"]) for row in rows]
    source_counts = Counter(
        candidate["source"] for row in rows for candidate in row["candidates"]
    )
    return {
        "examples": len(rows),
        "future_update_examples": sum(bool(row["gold_document_ids"]) for row in rows),
        "no_future_examples": sum(not row["gold_document_ids"] for row in rows),
        "examples_missing_declared_gold_documents": sum(bool(row["missing_gold_pmids"]) for row in rows),
        "candidate_documents": sum(candidate_counts),
        "candidate_documents_mean": mean(candidate_counts) if candidate_counts else 0.0,
        "candidate_documents_min": min(candidate_counts) if candidate_counts else 0,
        "candidate_documents_max": max(candidate_counts) if candidate_counts else 0,
        "gold_documents": sum(gold_counts),
        "gold_documents_mean": mean(gold_counts) if gold_counts else 0.0,
        "candidate_sources": dict(sorted(source_counts.items())),
    }


def candidate_pmid(candidate: dict[str, Any]) -> str | None:
    pmid = str(candidate.get("pmid") or "").strip()
    return pmid if pmid else None


def build_pmid_isolated_view(
    split_rows: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Reserve test PMIDs from dev/train and dev PMIDs from train.

    Rows whose positive labels intersect a later split are removed rather than
    relabelled. Reserved PMIDs are also removed when they occur as distractors.
    This keeps threshold selection and final evaluation publication-disjoint.
    """
    test_pmids = {
        pmid
        for row in split_rows["test"]
        for pmid in row.get("declared_gold_pmids", [])
    }
    dev_pmids = {
        pmid
        for row in split_rows["dev"]
        for pmid in row.get("declared_gold_pmids", [])
    }
    reserved_by_split = {
        "train": test_pmids | dev_pmids,
        "dev": test_pmids,
        "test": set(),
    }
    isolated: dict[str, list[dict[str, Any]]] = {}
    audit: dict[str, Any] = {}
    for split in SPLITS:
        reserved = reserved_by_split[split]
        kept: list[dict[str, Any]] = []
        dropped_positive_rows = 0
        removed_candidate_documents = 0
        for source_row in split_rows[split]:
            row_gold = set(source_row.get("declared_gold_pmids", []))
            if row_gold & reserved:
                dropped_positive_rows += 1
                continue
            row = dict(source_row)
            original_candidates = row.get("candidates", [])
            row["candidates"] = [
                candidate
                for candidate in original_candidates
                if candidate_pmid(candidate) not in reserved
            ]
            removed_candidate_documents += len(original_candidates) - len(row["candidates"])
            kept.append(row)
        isolated[split] = kept
        audit[split] = {
            "source_examples": len(split_rows[split]),
            "kept_examples": len(kept),
            "dropped_positive_rows_with_later_split_pmid": dropped_positive_rows,
            "removed_reserved_candidate_documents": removed_candidate_documents,
            "reserved_pmid_count": len(reserved),
        }

    gold_sets = {
        split: {
            pmid
            for row in isolated[split]
            for pmid in row.get("declared_gold_pmids", [])
        }
        for split in SPLITS
    }
    residual = {
        "train_vs_dev": sorted(gold_sets["train"] & gold_sets["dev"]),
        "train_vs_test": sorted(gold_sets["train"] & gold_sets["test"]),
        "dev_vs_test": sorted(gold_sets["dev"] & gold_sets["test"]),
    }
    audit["residual_gold_pmid_overlap"] = {
        name: {"count": len(pmids), "pmids": pmids} for name, pmids in residual.items()
    }
    return isolated, audit


def assign_tail_groups_to_folds(
    rows: list[dict[str, Any]], fold_count: int, seed: int
) -> list[set[str]]:
    if fold_count < 3:
        raise ValueError("tail-grouped cross-validation requires at least 3 folds")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("tail_id", ""))].append(row)
    totals = {
        "rows": len(rows) / fold_count,
        "positive": sum(bool(row.get("gold_document_ids")) for row in rows) / fold_count,
        "negative": sum(not row.get("gold_document_ids") for row in rows) / fold_count,
    }
    fold_tails = [set() for _ in range(fold_count)]
    fold_stats = [Counter() for _ in range(fold_count)]
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            -len(item[1]),
            -sum(bool(row.get("gold_document_ids")) for row in item[1]),
            hashlib.sha256(f"{seed}|{item[0]}".encode("utf-8")).hexdigest(),
        ),
    )
    for tail_id, group_rows in ordered_groups:
        group_stats = {
            "rows": len(group_rows),
            "positive": sum(bool(row.get("gold_document_ids")) for row in group_rows),
            "negative": sum(not row.get("gold_document_ids") for row in group_rows),
        }

        def placement_score(fold_index: int) -> tuple[float, int]:
            score = sum(
                ((fold_stats[fold_index][key] + group_stats[key]) / max(totals[key], 1.0)) ** 2
                for key in totals
            )
            return score, fold_index

        chosen = min(range(fold_count), key=placement_score)
        fold_tails[chosen].add(tail_id)
        fold_stats[chosen].update(group_stats)
    return fold_tails


def build_tail_grouped_cross_validation(
    rows: list[dict[str, Any]], fold_count: int, seed: int
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    fold_tails = assign_tail_groups_to_folds(rows, fold_count, seed)
    folds: dict[str, dict[str, list[dict[str, Any]]]] = {}
    audit: dict[str, Any] = {"fold_count": fold_count, "folds": {}}
    for fold_index in range(fold_count):
        test_tails = fold_tails[fold_index]
        dev_tails = fold_tails[(fold_index + 1) % fold_count]
        raw = {"train": [], "dev": [], "test": []}
        for source_row in rows:
            tail_id = str(source_row.get("tail_id", ""))
            split = "test" if tail_id in test_tails else "dev" if tail_id in dev_tails else "train"
            row = dict(source_row)
            row["source_split"] = source_row.get("split")
            row["split"] = split
            raw[split].append(row)
        isolated, isolation_audit = build_pmid_isolated_view(raw)
        name = f"fold_{fold_index}"
        folds[name] = isolated
        split_tails = {
            split: {str(row.get("tail_id", "")) for row in split_rows}
            for split, split_rows in isolated.items()
        }
        audit["folds"][name] = {
            "raw": {split: summarize_benchmark(split_rows) for split, split_rows in raw.items()},
            "pmid_isolation": isolation_audit,
            "tail_counts": {split: len(values) for split, values in split_tails.items()},
            "tail_overlap": {
                "train_vs_dev": len(split_tails["train"] & split_tails["dev"]),
                "train_vs_test": len(split_tails["train"] & split_tails["test"]),
                "dev_vs_test": len(split_tails["dev"] & split_tails["test"]),
            },
            "isolated": {
                split: summarize_benchmark(split_rows) for split, split_rows in isolated.items()
            },
        }
    audit["test_tail_coverage"] = {
        "unique_test_tails": len(set().union(*fold_tails)),
        "tail_assignments": sum(len(values) for values in fold_tails),
    }
    return folds, audit


def source_path(workspace: Path, dataset_name: str, split: str) -> Path:
    base_dir, prefix = SPECS[dataset_name]
    return workspace / base_dir / f"{prefix}.{split}.jsonl"


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = workspace / output_root
    cache_path = output_root / "cache" / "pubmed_postcut_records.json"
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    loaded: dict[tuple[str, str], list[dict[str, Any]]] = {}
    all_pmids: set[str] = set()
    legacy_audit: dict[str, Any] = {}
    for dataset_name in SPECS:
        legacy_audit[dataset_name] = {}
        for split in SPLITS:
            path = source_path(workspace, dataset_name, split)
            rows = read_jsonl(path)
            loaded[(dataset_name, split)] = rows
            legacy_audit[dataset_name][split] = legacy_supervision_audit(rows)
            for row in rows:
                all_pmids.update(parse_pmids(row.get("target", {}).get("future_ctd_post_cut_pmids")))

    expanded_head_pool: dict[str, set[str]] | None = None
    expanded_tail_pool: dict[str, set[str]] | None = None
    expanded_audit: dict[str, Any] | None = None
    if args.ctd_full_path:
        ctd_path = Path(args.ctd_full_path).resolve()
        head_ids = {str(row.get("head_id", "")) for rows in loaded.values() for row in rows}
        tail_ids = {str(row.get("tail_id", "")) for rows in loaded.values() for row in rows}
        expanded_head_pool, expanded_tail_pool, expanded_audit = load_ctd_direct_pmid_pools(
            ctd_path, head_ids, tail_ids
        )
        all_pmids.update(set().union(*expanded_head_pool.values(), *expanded_tail_pool.values()))

    missing = sorted(all_pmids - set(cache))
    if missing and args.no_fetch:
        raise RuntimeError(f"PubMed cache misses {len(missing)} PMIDs while --no-fetch is enabled.")
    if missing:
        checkpoint_size = max(args.batch_size, args.fetch_checkpoint_size)
        for checkpoint in chunks(missing, checkpoint_size):
            cache.update(
                fetch_pubmed_records(
                    checkpoint,
                    email=args.email,
                    batch_size=args.batch_size,
                    workers=args.fetch_workers,
                )
            )
            write_json(cache_path, cache)
            print(
                f"PubMed cache checkpoint: {len(cache)}/{len(all_pmids)} records",
                file=sys.stderr,
                flush=True,
            )

    eligible_postcut_pmids = {
        pmid
        for pmid in all_pmids
        if pmid in cache
        and isinstance(cache[pmid].get("pub_year"), int)
        and cache[pmid]["pub_year"] >= args.minimum_postcut_year
    }
    if expanded_head_pool is not None and expanded_tail_pool is not None:
        expanded_head_pool = {
            key: values & eligible_postcut_pmids for key, values in expanded_head_pool.items()
        }
        expanded_tail_pool = {
            key: values & eligible_postcut_pmids for key, values in expanded_tail_pool.items()
        }

    benchmark_summary: dict[str, Any] = {}
    built_splits: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    split_gold_pmids: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for dataset_name in SPECS:
        benchmark_summary[dataset_name] = {}
        for split in SPLITS:
            rows = loaded[(dataset_name, split)]
            built = build_split(
                rows,
                cache,
                dataset_name=dataset_name,
                split=split,
                seed=args.seed,
                same_tail_negatives=args.same_tail_negatives,
                same_head_negatives=args.same_head_negatives,
                random_negatives=args.random_negatives,
                global_tail_pool=expanded_tail_pool,
                global_head_pool=expanded_head_pool,
                global_postcut_pool=eligible_postcut_pmids if args.ctd_full_path else None,
            )
            write_jsonl(output_root / dataset_name / f"{split}.jsonl", built)
            built_splits[dataset_name][split] = built
            benchmark_summary[dataset_name][split] = summarize_benchmark(built)
            split_gold_pmids[dataset_name][split] = sorted(
                {pmid for row in built for pmid in row["declared_gold_pmids"]}
            )

    isolation_summary: dict[str, Any] = {}
    for dataset_name in SPECS:
        isolated, audit = build_pmid_isolated_view(built_splits[dataset_name])
        isolation_summary[dataset_name] = {"audit": audit, "splits": {}}
        for split in SPLITS:
            write_jsonl(
                output_root / f"{dataset_name}_pmid_isolated" / f"{split}.jsonl",
                isolated[split],
            )
            isolation_summary[dataset_name]["splits"][split] = summarize_benchmark(
                isolated[split]
            )

    cross_validation_audit: dict[str, Any] | None = None
    if args.tail_grouped_folds:
        combined_primary = [
            row for split in SPLITS for row in built_splits["primary_balanced"][split]
        ]
        folds, cross_validation_audit = build_tail_grouped_cross_validation(
            combined_primary, args.tail_grouped_folds, args.seed
        )
        for fold_name, split_rows in folds.items():
            for split in SPLITS:
                write_jsonl(
                    output_root / "tail_grouped_cv_pmid_isolated" / fold_name / f"{split}.jsonl",
                    split_rows[split],
                )

    overlap_audit: dict[str, Any] = {}
    for dataset_name, split_map in split_gold_pmids.items():
        overlap_audit[dataset_name] = {}
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            overlap = sorted(set(split_map[left]) & set(split_map[right]))
            overlap_audit[dataset_name][f"{left}_vs_{right}"] = {
                "count": len(overlap),
                "pmids": overlap,
            }

    manifest = {
        "benchmark": "ctd_publication_provenance",
        "version": "2026-06-23",
        "task": "Rank post-cut CTD-cited publication provenance for each chemical-disease update candidate.",
        "gold_definition": "PMIDs recorded by CTD for the post-cut relation snapshot.",
        "unlabelled_caution": "Non-cited candidate documents are unlabelled, not confirmed biological negatives.",
        "seed": args.seed,
        "negative_budgets": {
            "same_tail": args.same_tail_negatives,
            "same_head": args.same_head_negatives,
            "random_postcut": args.random_negatives,
            "precut_historical": "all retained source documents",
        },
        "pubmed": {
            "declared_unique_pmids": len(all_pmids),
            "fetched_unique_pmids": len(set(all_pmids) & set(cache)),
            "missing_unique_pmids": sorted(all_pmids - set(cache)),
            "cache": str(cache_path.relative_to(workspace)),
            "minimum_postcut_year": args.minimum_postcut_year,
            "eligible_postcut_pmids": len(eligible_postcut_pmids),
        },
        "expanded_ctd_corpus": {
            "enabled": bool(args.ctd_full_path),
            "source_path": str(Path(args.ctd_full_path).resolve()) if args.ctd_full_path else None,
            "audit_before_publication_year_filter": expanded_audit,
        },
        "legacy_sentence_supervision_audit": legacy_audit,
        "splits": benchmark_summary,
        "pmid_isolated_views": isolation_summary,
        "tail_grouped_cross_validation": cross_validation_audit,
        "gold_pmid_overlap": overlap_audit,
    }
    write_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
