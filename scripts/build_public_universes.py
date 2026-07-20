from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PMID/year-only universe files without article text."
    )
    parser.add_argument("--ctd-documents", required=True)
    parser.add_argument("--civic-tsv", required=True)
    parser.add_argument(
        "--civic-rankings",
        default="",
        help="Optional locked ranking JSONL(.gz); its candidate union freezes the evaluated universe.",
    )
    parser.add_argument("--output-root", default="data/universes")
    return parser.parse_args()


def iter_ctd_records(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "{":
            payload = json.load(handle)
            records = payload.values() if isinstance(payload, dict) else payload
            yield from records
            return
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_tsv_gz(path: Path, rows: Iterable[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["document_id", "pmid", "pub_year"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)

    ctd_rows = {}
    for record in iter_ctd_records(Path(args.ctd_documents)):
        pmid = str(record.get("pmid") or record.get("document_id", "")).removeprefix("PMID:")
        if pmid:
            ctd_rows[pmid] = str(record.get("pub_year") or "")
    write_tsv_gz(
        output_root / "ctd_55001_pmids.tsv.gz",
        ((f"PMID:{pmid}", pmid, year) for pmid, year in sorted(ctd_rows.items(), key=lambda x: int(x[0]))),
    )

    civic_pmids = set()
    with Path(args.civic_tsv).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            pmid = str(row.get("citation_id", "")).strip()
            if row.get("source_type") == "PubMed" and pmid.isdigit():
                civic_pmids.add(pmid)
    if args.civic_rankings:
        ranking_path = Path(args.civic_rankings)
        opener = gzip.open if ranking_path.suffix == ".gz" else open
        locked_pmids = set()
        with opener(ranking_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                locked_pmids.update(
                    str(item).removeprefix("PMID:")
                    for item in row.get("ranked_document_ids", [])
                )
        civic_pmids = locked_pmids
    write_tsv_gz(
        output_root / "civic_2226_pmids.tsv.gz",
        ((f"PMID:{pmid}", pmid, "") for pmid in sorted(civic_pmids, key=int)),
    )

    print(json.dumps({"ctd_pmids": len(ctd_rows), "civic_pmids": len(civic_pmids)}, indent=2))


if __name__ == "__main__":
    main()
