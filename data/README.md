# TempProv-Audit dataset card

## Scope

This release is a publication-provenance ranking benchmark. It asks a model to
rank source publications cited by a frozen biomedical curation resource for a
chemical--disease or therapy--disease query. It does not label uncited documents
as biologically false, and it is not a clinical decision-support dataset.

## Cohorts

| Directory | Queries / endpoint | Purpose |
| --- | ---: | --- |
| `ctd_sampled/tail_grouped_cv_pmid_isolated` | 2,638 rows across five outer folds | Disease-tail-disjoint sampled control |
| `ctd_rolling_origin/test_2024` | 5,597 queries per fold | Forward publication-time audit |
| `ctd_rolling_origin/test_2025` | 5,597 queries per fold | Forward publication-time audit |
| `historical_firewall/strict_locked_model_queries.jsonl.gz` | 188 | Locked 2021--2026 transfer |
| `historical_firewall/expanded_zero_shot_queries.jsonl.gz` | 578 | Zero-shot 2021--2026 transfer |
| `civic/queries.jsonl.gz` | 754 | Independent precision-oncology transfer |

## Common fields

- `example_id`: stable benchmark row identifier.
- `pair_id`: source entity-pair key.
- `head_id`, `tail_id`: normalized entity identifiers.
- `query_text`: serialized model query.
- `gold_document_ids`: cited source PMIDs, prefixed with `PMID:`.
- `candidates`: sampled candidate metadata where applicable.
- `source`: candidate construction stratum.
- `is_ctd_cited`: provenance label, not a biological truth label.

## Leakage controls

The disease-tail-disjoint folds prevent disease tails from crossing the outer
train/dev/test boundary. Positive PMIDs are isolated so a held-out positive
cannot reappear as training gold or as a retained distractor. Historical
firewall rows additionally require ontology-projected absence from the frozen
2021 PheKnowLator state.

## Text and licensing policy

No PubMed article title or abstract is included. `universes/*.tsv.gz` provides
the exact candidate PMID inventory needed to re-fetch metadata from NCBI. See
the repository-level `DATA_LICENSE.md` before reuse.

