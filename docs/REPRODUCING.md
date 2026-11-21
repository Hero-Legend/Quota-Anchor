# Reproducing the legacy TempProv-Audit benchmark

For the current Quota-Anchor method, use [QUOTA_ANCHOR.md](QUOTA_ANCHOR.md).
The protocol below documents the original benchmark release.

## Frozen design

- Disease tails and positive PMIDs are disjoint across train, development, and
  test partitions.
- Task-trained cross-encoders use seeds `13`, `42`, and `3407`.
- Full-corpus retrieval uses Top-1,000 candidates; all rerankers operate on the
  same locked Top-100 membership.
- Primary confidence intervals use 5,000 disease-tail-cluster bootstrap
  resamples.
- Paired comparisons use 100,000 two-sided disease-tail-cluster sign flips and
  Holm correction within each declared comparison family.
- Unretrieved gold PMIDs receive zero rank-aware credit; rerankers may not add
  documents outside the locked candidate pool.

## Data layout

- `data/ctd_sampled`: compact sampled and five-fold disease-tail-disjoint data.
- `data/ctd_rolling_origin`: rolling-origin folds for 2024 and 2025 endpoints.
- `data/historical_firewall`: strict and expanded 2021--2026 firewall queries.
- `data/civic`: the 754-query independent CIViC endpoint and frozen CC0 source.
- `data/universes`: PMID/year inventories used to reconstruct local text stores.

## Models

The public pipeline supports the following model families used in the audit:

- BGE large English v1.5 bi-encoder
- MedCPT query/article encoders and cross-encoder
- BiomedBERT-base task-trained cross-encoder
- ModernBERT task-trained control
- DeepSeek API generative reranker
- Llama-3.1-8B-Instruct local 4-bit control

Model weights are not vendored. Download them from their official providers or
Hugging Face model repositories and record the exact revision locally.

## Recommended execution order

1. Verify the release with `python scripts/verify_release.py`.
2. Obtain the CTD report under CTD terms and reconstruct the PubMed text cache.
3. Run candidate generators with
   `tempprov_audit.cli.run_corpus_retrieval`.
4. Train or evaluate cross-encoders with
   `tempprov_audit.cli.run_ctd_provenance_cross_encoder`.
5. Lock the Top-100 pool and run
   `tempprov_audit.cli.run_full_corpus_retrieve_rerank`.
6. Build the historical and CIViC endpoints with the corresponding `build_*`
   modules before evaluating transfer.

All commands expose their complete arguments through `--help`. Output folders
are deliberately ignored by Git and are not part of the public release.

