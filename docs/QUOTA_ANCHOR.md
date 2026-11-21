# Quota-Anchor: code and reproduction scope

This repository hosts the original TempProv-Audit benchmark release and the
Quota-Anchor method code. The repository URL is retained so existing citations
remain valid. The method implementation lives in
[`src/tempprov_audit/quota_anchor`](../src/tempprov_audit/quota_anchor).

## What is public

- Frozen CTD and CIViC benchmark annotations, candidate PMID inventories and split
  metadata already listed in `data/release_manifest.json`.
- Original retrieval, cross-encoder and benchmark construction programs.
- Quota-grafting, min-max bi-anchor scoring, variance-gated residuals, unlabelled
  drift detection, contextual teacher distillation, candidate-pair calibration,
  and disease-cluster inference routines.
- Frozen scalar/calibration configurations and source provenance hashes.
- A portable interface for candidate construction and cached-score processing.

The addition does **not** publish a new benchmark snapshot or change the original
61 data artifacts. It does not include manuscript files, paper figures, result
tables, model predictions, score caches, teacher responses or trained weights.
It is not a complete archive of every computational asset used in the study.

## Dependencies and installation

Use Python 3.10 or newer. The original project dependencies are declared in
`pyproject.toml`; the cached-score modules use NumPy, SciPy and scikit-learn.

```bash
pip install -e .
python -m tempprov_audit.quota_anchor.cli --help
python -m unittest discover -s tests -p test_quota_anchor.py -v
```

No command runs on import. Commands below act only on explicitly supplied files;
they do not contact a teacher service or submit work to a server scheduler.

## Candidate construction

Supply a JSON array (or JSONL, optionally gzipped) with one object per query:

```text
qid: unique query identifier
bm25: original BM25 Top-1000 document IDs, in rank order
bge_large: dense BGE document IDs, in rank order
medcpt_dual: dense MedCPT-dual document IDs, in rank order
```

All channels must use the same identifier type. The original BM25 list must
contain 1,000 unique IDs. Candidate eligibility is the dense Top-50 intersection
whose BM25 ranks are strictly greater than the budget and at most 1,000.

```bash
python -m tempprov_audit.quota_anchor.cli quota \
  --input retrieval_lists.jsonl --budget 100 --k 95 --output work/pools.json
```

`95` is a command example, **not** a universal selected core size. Select `k`
separately on the complete legal development population using `quota.choose_k`;
the frozen grids are 85/90/95/98 for budget 100 and
175/180/185/190/195 for budget 200. The function expects development query fields
`example_id` and `gold_document_ids` and channel dictionaries keyed by query ID.
Never select `k` using test or external benchmark labels. The returned candidate
membership is fixed before expensive reranking.

## Aligned score records

Scalar-head and distillation commands consume JSON arrays or JSONL records:

```text
qid: unique query identifier
tail: disease-group identifier (required for distillation/grouped inference)
ids: ordered unique candidate document IDs
scores.med: MedCPT cross-encoder scores aligned to ids
scores.bge: BGE reranker scores aligned to ids
scores.lex: lexical BM25 scores aligned to ids
bio_seeds: three Bio seed-score arrays aligned to ids (seeds 13, 42, 3407)
gold: gold document IDs, required for development selection, never for scoring
cohort: optional cohort identity; historical_expanded forbids CTD-selected heads
```

These are **cross-encoder** score channels, distinct from dense retrieval ranks.
Do not replace scores with ranks, fill missing channels with fabricated values,
or pool across candidate sources. Original retrieval and cross-encoder commands
are documented in [REPRODUCING.md](REPRODUCING.md). That guide describes the
historical benchmark protocol; its resampling counts must not be substituted for
the Quota-Anchor settings below.

```bash
python -m tempprov_audit.quota_anchor.cli fit-head \
  --dev own_fold_dev.jsonl --output work/head.json
python -m tempprov_audit.quota_anchor.cli rank \
  --input held_out_scores.jsonl --state work/head.json \
  --variant anchor --output work/ranking.json
```

Fit a separate state using each legal fold's own development records. Scoring
whitelists input fields and does not access gold. `anchor` applies the tuned
bi-anchor directly; `in_domain` and `sharp` expose the distinct residual and
hard-switch configurations. They are distinct scientific comparisons, not
interchangeable names for the proposed method. `fixed-expanded` uses the frozen
expanded-cohort branch without a CTD-selected state:

```bash
python -m tempprov_audit.quota_anchor.cli rank \
  --input expanded_scores.jsonl --variant fixed-expanded \
  --output work/expanded_ranking.json
```

For CIViC transfer, apply the five internal heads to their corresponding
three-seed score arrays, min-max normalize each head's outputs within each query,
then average the five vectors before stable descending sorting. Do not average
raw logits across unrelated heads or fit scalar heads on CIViC gold labels.
The portable `rank` command processes one state at a time; the five-head
aggregation is an explicit caller responsibility.

## Teacher-guided distillation

This is a separate within-CIViC learning protocol on the locked MedCPT Top-100,
not CTD-head zero-shot transfer. Supply the original complete locked pool and an
archived teacher JSONL containing `example_id`, `document_id`, numeric `score`
and `parse_error` (null on successful parsing).

```bash
python -m tempprov_audit.quota_anchor.cli distill-oof \
  --input civic_locked_med_scores.jsonl --teacher archived_teacher.jsonl.gz \
  --output work/distillation.json
```

The latest record for each exact query/document pair must be parse-valid and
finite. Missing pairs are errors; earlier successful retries do not repair later
invalid records. The feature whitelist excludes gold and teacher scores. The
49 contextual features are defined in `listwise.features(extended=False)`.
Disease-group outer five-fold prediction and inner three-fold alpha selection
use teacher MSE only; scaling is fitted within each training partition.

Teacher responses are not redistributed. The retained producer identifies a
`deepseek-v4-pro` service string, temperature 0, 32 output tokens, JSON scores in
0--100 and candidate depth 100. Its prompt requests publication-provenance
support from therapy, disease, title and the first 512 abstract words, excluding
PMIDs and gold evidence labels. An immutable provider release and complete
per-request timestamps are not established by this code release. A new teacher
run is a new computation and must not be presented as the archived responses.

## Risk calibration and uncertainty

`risk.fit_controls(dev)` and `risk.predict_controls(state, query)` implement the
same-fold development-fitted pair controls. Thresholds and calibrators use the
complete same-fold development data; development estimates are optimistic.
The query-normalized guard is empirical and provides no guaranteed test FDR,
nonempty acceptance or clinical safety. Inactive disease clusters have undefined
FDP and must not be filled with zero.

`metrics.clustered_interval` and `risk_statistics.paired_cluster_statistics`
provide disease-cluster uncertainty routines. Quota-Anchor uses 10,000 resamples;
paired comparisons require identical query populations and the original
comparison-family-specific Holm adjustment. Intervals condition on saved
trained/selected outputs and do not measure training-seed variation.

## Third-party data and unavailable assets

Follow `DATA_LICENSE.md` for the original frozen data. Obtain the CTD source
report directly from CTD; reconstruct PubMed text from the released PMID lists
using the documented acquisition route and applicable source terms. PheKnowLator
is an upstream third-party input; a PheKnowLator source snapshot is **not** part
of this repository's frozen data release. No new redistribution permission is
implied by this method-code addition.

Score caches, original trained checkpoints and teacher responses remain outside
this public distribution. The repository enables method inspection and execution
with user-supplied inputs; it does not promise exact regeneration of every paper
number from the public files alone.

## Source provenance

`QUOTA_ANCHOR_SOURCE_MANIFEST.json` records original and published hashes. The
algorithm modules retain their original computations; only package-relative
imports are adjusted. Historical `v2`/`v3` identifiers inside implementation
constants are retained for traceability. They are not manuscript method names.
The portable CLI and engineering tests are new distribution interfaces, not
new scientific experiments. Legacy scheduler clients and server connection
configuration are deliberately excluded from the release.
