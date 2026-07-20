from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests


EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an independent CIViC therapy-disease provenance benchmark.")
    parser.add_argument("--evidence-tsv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--email")
    return parser.parse_args()


def text(node: Any) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def publication_year(article: ET.Element) -> int:
    for path in (
        ".//JournalIssue/PubDate/Year",
        ".//ArticleDate/Year",
        ".//DateCompleted/Year",
        ".//DateRevised/Year",
    ):
        value = article.findtext(path)
        if value and value.isdigit():
            return int(value)
    medline = article.findtext(".//JournalIssue/PubDate/MedlineDate") or ""
    match = re.search(r"(?:19|20)\d{2}", medline)
    return int(match.group()) if match else 0


def fetch_documents(pmids: list[str], batch_size: int, email: str | None) -> dict[str, dict[str, Any]]:
    output = {}
    session = requests.Session()
    for start in range(0, len(pmids), batch_size):
        batch = pmids[start : start + batch_size]
        params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
        if email:
            params["email"] = email
        for attempt in range(5):
            response = session.get(EFETCH, params=params, timeout=120)
            if response.ok:
                break
            time.sleep(2**attempt)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//MedlineCitation/PMID")
            title = text(article.find(".//Article/ArticleTitle"))
            abstract = " ".join(text(node) for node in article.findall(".//Article/Abstract/AbstractText")).strip()
            if pmid and (title or abstract):
                output[pmid] = {
                    "document_id": f"PMID:{pmid}",
                    "pmid": pmid,
                    "pub_year": publication_year(article),
                    "title": title,
                    "abstract": abstract,
                    "text": f"{title} {abstract}".strip(),
                }
        time.sleep(0.35)
    return output


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    evidence_path = Path(args.evidence_tsv).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("r", encoding="utf-8", newline="") as handle:
        evidence = list(csv.DictReader(handle, delimiter="\t"))

    all_pmids = sorted(
        {
            row["citation_id"]
            for row in evidence
            if row["source_type"] == "PubMed" and row["citation_id"].isdigit()
        },
        key=int,
    )
    document_cache = output_root / "documents.jsonl.gz"
    if document_cache.exists():
        with gzip.open(document_cache, "rt", encoding="utf-8") as handle:
            documents = {row["pmid"]: row for row in map(json.loads, handle)}
    else:
        documents = fetch_documents(all_pmids, args.batch_size, args.email)
        write_jsonl_gz(document_cache, sorted(documents.values(), key=lambda row: int(row["pmid"])))

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    excluded = Counter()
    for row in evidence:
        if row["evidence_type"] != "Predictive":
            excluded["non_predictive"] += 1
            continue
        if not row["therapies"]:
            excluded["missing_therapy"] += 1
            continue
        if "," in row["therapies"]:
            excluded["combination_or_substitute_therapy"] += 1
            continue
        if row["source_type"] != "PubMed" or not row["citation_id"].isdigit():
            excluded["non_pubmed_source"] += 1
            continue
        if row["is_flagged"].lower() == "true":
            excluded["flagged_evidence"] += 1
            continue
        if row["citation_id"] not in documents:
            excluded["missing_pubmed_text"] += 1
            continue
        disease_key = row["doid"] or row["disease"].strip().lower()
        key = (row["therapies"].strip().lower(), disease_key)
        group = groups.setdefault(
            key,
            {
                "therapy": row["therapies"].strip(),
                "disease": row["disease"].strip(),
                "doid": row["doid"].strip(),
                "pmids": set(),
                "evidence_ids": set(),
                "evidence_levels": set(),
                "significances": set(),
            },
        )
        group["pmids"].add(row["citation_id"])
        group["evidence_ids"].add(row["evidence_id"])
        group["evidence_levels"].add(row["evidence_level"])
        group["significances"].add(row["significance"])

    queries = []
    for index, (_, group) in enumerate(sorted(groups.items())):
        queries.append(
            {
                "dataset_name": "civic_predictive_therapy_disease_provenance",
                "split": "external_test",
                "example_id": f"civic_external_{index:05d}",
                "pair_id": f"{group['therapy'].lower()}__{group['doid'] or group['disease'].lower()}",
                "head_id": group["therapy"].lower(),
                "head_text": group["therapy"],
                "tail_id": f"DOID:{group['doid']}" if group["doid"] else group["disease"].lower(),
                "tail_text": group["disease"],
                "query_text": f"Therapy: {group['therapy']}. Disease: {group['disease']}.",
                "gold_document_ids": [f"PMID:{pmid}" for pmid in sorted(group["pmids"], key=int)],
                "evidence_ids": sorted(group["evidence_ids"], key=int),
                "evidence_levels": sorted(group["evidence_levels"]),
                "significances": sorted(group["significances"]),
            }
        )
    write_jsonl_gz(output_root / "queries.jsonl.gz", queries)
    summary = {
        "source": "CIViC accepted evidence nightly release",
        "endpoint": "single-therapy predictive evidence grouped by therapy and Disease Ontology disease",
        "queries": len(queries),
        "documents": len(documents),
        "requested_pubmed_documents": len(all_pmids),
        "gold_documents": len({item for row in queries for item in row["gold_document_ids"]}),
        "evidence_items": sum(len(row["evidence_ids"]) for row in queries),
        "disease_tails": len({row["tail_id"] for row in queries}),
        "exclusions": dict(sorted(excluded.items())),
        "claim_boundary": "Independent precision-oncology provenance retrieval; not a CTD endpoint substitute and not a prospective clinical validation.",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    (output_root / "source_manifest.json").write_text(
        json.dumps({"evidence_tsv": str(evidence_path), "sha256": digest, "pubmed_endpoint": EFETCH}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
