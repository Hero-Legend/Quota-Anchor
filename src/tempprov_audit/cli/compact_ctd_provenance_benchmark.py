from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


INCLUDED_DIRECTORIES = (
    "primary_balanced_pmid_isolated",
    "tail_disjoint_pmid_isolated",
    "tail_grouped_cv_pmid_isolated",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize and compress the CTD provenance benchmark.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def document_key(candidate: dict[str, Any]) -> str:
    payload = "\0".join(
        (
            str(candidate.get("document_id", "")),
            str(candidate.get("title", "")),
            str(candidate.get("abstract", "")),
            str(candidate.get("text", "")),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compact_candidate(candidate: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        "document_key": key,
        "document_id": candidate.get("document_id"),
        "pmid": candidate.get("pmid"),
        "pub_year": candidate.get("pub_year"),
        "source": candidate.get("source"),
        "is_ctd_cited": candidate.get("is_ctd_cited"),
    }


def document_record(candidate: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        "document_key": key,
        "document_id": candidate.get("document_id"),
        "pmid": candidate.get("pmid"),
        "pub_year": candidate.get("pub_year"),
        "title": candidate.get("title", ""),
        "abstract": candidate.get("abstract", ""),
        "text": candidate.get("text", ""),
    }


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    documents: dict[str, dict[str, Any]] = {}
    split_audit: dict[str, Any] = {}
    source_files = [
        path
        for directory in INCLUDED_DIRECTORIES
        for path in sorted((source_root / directory).rglob("*.jsonl"))
    ]
    for source_path in source_files:
        relative = source_path.relative_to(source_root)
        output_path = output_root / relative.with_suffix(relative.suffix + ".gz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = candidates = 0
        with source_path.open(encoding="utf-8") as source, gzip.open(
            output_path, "wt", encoding="utf-8", compresslevel=6
        ) as target:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                compact_candidates = []
                for candidate in row.get("candidates", []):
                    key = document_key(candidate)
                    documents.setdefault(key, document_record(candidate, key))
                    compact_candidates.append(compact_candidate(candidate, key))
                row["candidates"] = compact_candidates
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
                candidates += len(compact_candidates)
        split_audit[str(relative)] = {
            "rows": rows,
            "candidate_instances": candidates,
            "compressed_bytes": output_path.stat().st_size,
        }

    document_path = output_root / "documents.jsonl.gz"
    document_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(document_path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for key in sorted(documents):
            handle.write(json.dumps(documents[key], ensure_ascii=False) + "\n")
    source_manifest = source_root / "manifest.json"
    if source_manifest.exists():
        (output_root / "source_manifest.json").write_bytes(source_manifest.read_bytes())
    manifest = {
        "format": "normalized_ctd_provenance_jsonl_gzip_v1",
        "source_root": str(source_root),
        "document_store": "documents.jsonl.gz",
        "unique_document_records": len(documents),
        "document_store_compressed_bytes": document_path.stat().st_size,
        "splits": split_audit,
    }
    (output_root / "compact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
