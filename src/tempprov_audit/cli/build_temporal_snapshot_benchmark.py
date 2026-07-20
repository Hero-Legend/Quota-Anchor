from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a true 2021-to-2026 CTD version-shift provenance benchmark."
    )
    parser.add_argument("--snapshot-diff", required=True)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--document-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pair_id(chemical_id: str, disease_id: str) -> str:
    return f"{chemical_id}__{disease_id.replace(':', '_')}"


def deterministic_fold(tail_id: str, folds: int) -> int:
    digest = hashlib.sha256(f"tempprov-snapshot|{tail_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def load_fold_test_rows(
    root: Path, folds: int
) -> tuple[dict[str, list[tuple[int, dict[str, Any]]]], list[tuple[int, dict[str, Any]]]]:
    output: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    all_rows: list[tuple[int, dict[str, Any]]] = []
    for fold in range(folds):
        base = root / "tail_grouped_cv_pmid_isolated" / f"fold_{fold}"
        path = base / "test.jsonl"
        if not path.exists():
            path = base / "test.jsonl.gz"
        for row in read_jsonl(path):
            key = str(row["pair_id"])
            prior_folds = {item[0] for item in output.get(key, [])}
            if prior_folds and fold not in prior_folds:
                raise ValueError(f"Pair appears in multiple test folds: {key}")
            output.setdefault(key, []).append((fold, row))
            all_rows.append((fold, row))
    return output, all_rows


def main() -> None:
    args = parse_args()
    diff_path = Path(args.snapshot_diff).resolve()
    benchmark_root = Path(args.benchmark_root).resolve()
    document_path = Path(args.document_json).resolve()
    output_root = Path(args.output_root).resolve()
    years = {int(year) for year in args.years}

    documents = json.loads(document_path.read_text(encoding="utf-8"))
    available_pmids = {str(pmid) for pmid in documents}
    fold_rows, all_fold_rows = load_fold_test_rows(benchmark_root, args.folds)

    all_locked = []
    for fold, source in all_fold_rows:
        if not source.get("gold_document_ids"):
            continue
        all_locked.append(
            {
                key: value
                for key, value in {**source, "fold_index": fold}.items()
                if key != "candidates"
            }
        )
    all_locked.sort(key=lambda row: row["example_id"])

    expanded: list[dict[str, Any]] = []
    strict: list[dict[str, Any]] = []
    exclusions = Counter()
    with diff_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for source_index, source in enumerate(csv.DictReader(handle)):
            if source.get("known_in_pheknowlator_20211101") != "false":
                exclusions["known_in_2021_snapshot"] += 1
                continue
            if not source.get("pheknowlator_20211101_mapped_pairs"):
                exclusions["ontology_projection_unmapped"] += 1
                continue
            if int(source.get("first_pub_year") or 0) not in years:
                exclusions["outside_target_years"] += 1
                continue
            declared_pmids = sorted(filter(None, source.get("post_cut_pmids", "").split("|")))
            gold_pmids = [pmid for pmid in declared_pmids if pmid in available_pmids]
            if not gold_pmids:
                exclusions["no_gold_document_in_store"] += 1
                continue

            key = pair_id(source["chemical_id"], source["disease_id"])
            candidates = fold_rows.get(key, [])
            exact = [
                item
                for item in candidates
                if set(str(value) for value in item[1].get("declared_gold_pmids", []))
                == set(declared_pmids)
            ]
            prior = exact[0] if exact else (candidates[0] if candidates else None)
            fold = prior[0] if prior else deterministic_fold(source["disease_id"], args.folds)
            row = {
                "dataset_name": "ctd_true_version_shift_2021_to_2026",
                "split": "test",
                "example_id": f"ctd_snapshot_shift_{source_index:06d}",
                "pair_id": key,
                "head_id": source["chemical_id"],
                "head_text": source["chemical_name"],
                "tail_id": source["disease_id"],
                "tail_text": source["disease_name"],
                "query_text": f"Chemical: {source['chemical_name']}. Disease: {source['disease_name']}.",
                "gold_document_ids": [f"PMID:{pmid}" for pmid in gold_pmids],
                "declared_gold_pmids": declared_pmids,
                "missing_gold_pmids": sorted(set(declared_pmids) - set(gold_pmids)),
                "fold_index": fold,
                "first_pub_year": int(source["first_pub_year"]),
                "direct_evidence": source["direct_evidence"],
                "snapshot_source": "PheKnowLator 2021-11 integrated graph",
                "snapshot_target": "CTD 2026-04-30 direct chemical-disease report",
                "ontology_projection_verified": True,
                "known_in_source_snapshot": False,
                "strict_existing_pmid_isolation": False,
            }
            expanded.append(row)

            if prior is None:
                exclusions["strict_pair_not_in_existing_test"] += 1
                continue
            prior_gold = set(str(value) for value in prior[1].get("declared_gold_pmids", []))
            if prior_gold != set(declared_pmids) or set(gold_pmids) != set(declared_pmids):
                exclusions["strict_gold_mismatch"] += 1
                continue
            strict_row = {**row, "strict_existing_pmid_isolation": True}
            strict.append(strict_row)

    expanded.sort(key=lambda row: row["example_id"])
    strict.sort(key=lambda row: row["example_id"])
    write_jsonl_gz(output_root / "expanded_zero_shot_queries.jsonl.gz", expanded)
    write_jsonl_gz(output_root / "strict_locked_model_queries.jsonl.gz", strict)
    write_jsonl_gz(output_root / "all_locked_model_queries.jsonl.gz", all_locked)

    def cohort(rows: list[dict[str, Any]]) -> dict[str, Any]:
        summary = {
            "queries": len(rows),
            "gold_documents": sum(len(row["gold_document_ids"]) for row in rows),
            "unique_gold_documents": len({item for row in rows for item in row["gold_document_ids"]}),
            "disease_tails": len({row["tail_id"] for row in rows}),
            "folds": dict(sorted(Counter(str(row["fold_index"]) for row in rows).items())),
        }
        if rows and all("first_pub_year" in row for row in rows):
            summary["years"] = dict(sorted(Counter(str(row["first_pub_year"]) for row in rows).items()))
        if rows and all("direct_evidence" in row for row in rows):
            summary["direct_evidence"] = dict(sorted(Counter(row["direct_evidence"] for row in rows).items()))
        return summary

    summary = {
        "protocol": "true database-version shift; ontology-projected absence in PheKnowLator 2021-11 and direct evidence in CTD 2026-04-30",
        "target_years": sorted(years),
        "document_store_pmids": len(available_pmids),
        "all_locked_model": cohort(all_locked),
        "strict_locked_model": cohort(strict),
        "expanded_zero_shot": cohort(expanded),
        "exclusions": dict(sorted(exclusions.items())),
        "claim_boundary": {
            "strict_locked_model": "Existing fold-matched task-trained checkpoints may be evaluated only on this cohort.",
            "expanded_zero_shot": "Use only methods without CTD task training unless a new temporally isolated model is trained.",
            "absence": "Absence is asserted only after successful ontology projection; unmapped rows are excluded.",
        },
    }
    write_json(output_root / "summary.json", summary)
    write_json(
        output_root / "source_manifest.json",
        {
            "snapshot_diff": {"path": str(diff_path), "sha256": sha256(diff_path)},
            "benchmark_root": str(benchmark_root),
            "document_json": {"path": str(document_path), "sha256": sha256(document_path)},
            "outputs": {
                "strict": "strict_locked_model_queries.jsonl.gz",
                "expanded": "expanded_zero_shot_queries.jsonl.gz",
                "all_locked": "all_locked_model_queries.jsonl.gz",
                "summary": "summary.json",
            },
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
