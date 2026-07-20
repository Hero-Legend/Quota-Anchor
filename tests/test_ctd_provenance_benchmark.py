from __future__ import annotations

import unittest

from tempprov_audit.cli.build_ctd_provenance_benchmark import (
    build_pmid_isolated_view,
    build_split,
    build_tail_grouped_cross_validation,
    extract_ctd_direct_pmid_pools,
    legacy_supervision_audit,
    parse_pmids,
    parse_pubmed_xml,
)


def synthetic_row(
    example_id: str,
    head_id: str,
    tail_id: str,
    gold_pmids: str,
    relation: str = "future_ctd_update_direct",
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "pair_id": f"{head_id}__{tail_id}",
        "head_id": head_id,
        "head_text": f"chemical-{head_id}",
        "tail_id": tail_id,
        "tail_text": f"disease-{tail_id}",
        "gold_relation": relation,
        "gold_novelty": "background" if relation == "no_future_ctd_update" else "future_update",
        "target": {"future_ctd_post_cut_pmids": gold_pmids},
        "sentences": [
            {
                "doc_id": "D001",
                "pmid": "99",
                "pub_year": 2020,
                "text": "[D001|PMID:99|YEAR:2020] Historical context.",
            }
        ],
        "gold_evidence_sentence_ids": [0],
    }


class CTDProvenanceBenchmarkTests(unittest.TestCase):
    def test_parse_pmids_accepts_mixed_ctd_delimiters(self) -> None:
        self.assertEqual(parse_pmids("123|456; 789,123"), ["123", "456", "789"])

    def test_parse_pubmed_xml_extracts_title_abstract_and_year(self) -> None:
        payload = b"""<?xml version='1.0'?>
        <PubmedArticleSet><PubmedArticle><MedlineCitation>
          <PMID>123</PMID><Article><ArticleTitle>Target title.</ArticleTitle>
          <Abstract><AbstractText Label='BACKGROUND'>First sentence.</AbstractText>
          <AbstractText>Second sentence.</AbstractText></Abstract>
          <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
          </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
        records = parse_pubmed_xml(payload)
        self.assertEqual(records["123"]["title"], "Target title.")
        self.assertEqual(records["123"]["abstract"], "First sentence. Second sentence.")
        self.assertEqual(records["123"]["pub_year"], 2024)

    def test_build_split_constructs_conditional_hard_negatives_and_empty_no_future_gold(self) -> None:
        rows = [
            synthetic_row("a", "h1", "t1", "1"),
            synthetic_row("b", "h2", "t1", "2"),
            synthetic_row("c", "h1", "t2", "3"),
            synthetic_row("d", "h4", "t4", "", relation="no_future_ctd_update"),
        ]
        records = {
            pmid: {
                "pmid": pmid,
                "title": f"title-{pmid}",
                "abstract": f"abstract-{pmid}",
                "text": f"title-{pmid} abstract-{pmid}",
                "pub_year": 2024,
            }
            for pmid in ("1", "2", "3")
        }
        built = build_split(
            rows,
            records,
            dataset_name="synthetic",
            split="test",
            seed=7,
            same_tail_negatives=2,
            same_head_negatives=2,
            random_negatives=2,
        )
        first = next(row for row in built if row["example_id"] == "a")
        self.assertEqual(first["gold_document_ids"], ["PMID:1"])
        sources = {candidate["document_id"]: candidate["source"] for candidate in first["candidates"]}
        self.assertEqual(sources["PMID:1"], "gold_postcut_ctd_citation")
        self.assertEqual(sources["PMID:2"], "postcut_same_tail_distractor")
        self.assertEqual(sources["PMID:3"], "postcut_same_head_distractor")
        self.assertEqual(sources["PMID:99"], "precut_historical_context")
        no_future = next(row for row in built if row["example_id"] == "d")
        self.assertEqual(no_future["gold_document_ids"], [])

        built_with_sparse_global_pools = build_split(
            rows,
            records,
            dataset_name="synthetic",
            split="test",
            seed=7,
            same_tail_negatives=2,
            same_head_negatives=2,
            random_negatives=2,
            global_tail_pool={"t1": {"2"}},
            global_head_pool={},
            global_postcut_pool={"1", "2", "3"},
        )
        self.assertEqual(len(built_with_sparse_global_pools), len(rows))

    def test_legacy_supervision_audit_detects_all_sentence_gold_failure(self) -> None:
        rows = [
            {
                "sentences": [{"text": "a"}, {"text": "b"}],
                "gold_evidence_sentence_ids": [0, 1],
                "gold_relation": "no_future_ctd_update",
            },
            {
                "sentences": [{"text": "c"}],
                "gold_evidence_sentence_ids": [0],
                "gold_relation": "future_ctd_update_direct",
            },
        ]
        audit = legacy_supervision_audit(rows)
        self.assertEqual(audit["all_input_sentences_labelled_gold_rate"], 1.0)
        self.assertEqual(audit["no_future_examples_with_nonempty_gold_evidence"], 1)

    def test_pmid_isolation_removes_later_split_gold_and_distractors(self) -> None:
        def benchmark_row(example_id: str, gold: list[str], candidates: list[str]) -> dict[str, object]:
            return {
                "example_id": example_id,
                "declared_gold_pmids": gold,
                "gold_document_ids": [f"PMID:{pmid}" for pmid in gold],
                "candidates": [
                    {"document_id": f"PMID:{pmid}", "pmid": pmid, "source": "synthetic"}
                    for pmid in candidates
                ],
            }

        isolated, audit = build_pmid_isolated_view(
            {
                "train": [
                    benchmark_row("drop-train", ["2"], ["2", "9"]),
                    benchmark_row("keep-train", ["1"], ["1", "2", "3", "9"]),
                ],
                "dev": [
                    benchmark_row("drop-dev", ["3"], ["3", "8"]),
                    benchmark_row("keep-dev", ["2"], ["2", "3", "8"]),
                ],
                "test": [benchmark_row("test", ["3"], ["3", "7"])],
            }
        )
        self.assertEqual([row["example_id"] for row in isolated["train"]], ["keep-train"])
        self.assertEqual(
            [candidate["pmid"] for candidate in isolated["train"][0]["candidates"]],
            ["1", "9"],
        )
        self.assertEqual([row["example_id"] for row in isolated["dev"]], ["keep-dev"])
        self.assertEqual(
            audit["residual_gold_pmid_overlap"]["train_vs_test"]["count"], 0
        )

    def test_extract_ctd_direct_pmid_pools_filters_inferred_rows(self) -> None:
        lines = [
            "# comment\n",
            "chem\tC1\t\tdis\tD1\tmarker/mechanism\t\t\t\t10|11\n",
            "chem\tC1\t\tdis\tD2\t\tGENE\t4.0\t\t12\n",
            "chem\tC2\t\tdis\tD1\ttherapeutic\t\t\t\t13\n",
        ]
        heads, tails, audit = extract_ctd_direct_pmid_pools(lines, {"C1"}, {"D1"})
        self.assertEqual(heads["C1"], {"10", "11"})
        self.assertEqual(tails["D1"], {"10", "11", "13"})
        self.assertEqual(audit["direct_evidence_rows"], 2)

    def test_tail_grouped_cross_validation_has_complete_disjoint_test_coverage(self) -> None:
        rows = []
        for tail_index in range(6):
            for row_index in range(2):
                pmid = str(100 + tail_index)
                rows.append(
                    {
                        "example_id": f"{tail_index}-{row_index}",
                        "tail_id": f"D{tail_index}",
                        "declared_gold_pmids": [pmid] if row_index == 0 else [],
                        "gold_document_ids": [f"PMID:{pmid}"] if row_index == 0 else [],
                        "missing_gold_pmids": [],
                        "candidates": [
                            {
                                "document_id": f"PMID:{pmid}",
                                "pmid": pmid,
                                "source": "synthetic",
                                "is_ctd_cited": row_index == 0,
                            }
                        ],
                    }
                )
        folds, audit = build_tail_grouped_cross_validation(rows, 3, 7)
        seen_test_tails = []
        for split_rows in folds.values():
            tail_sets = {
                split: {row["tail_id"] for row in values} for split, values in split_rows.items()
            }
            self.assertFalse(tail_sets["train"] & tail_sets["test"])
            self.assertFalse(tail_sets["dev"] & tail_sets["test"])
            seen_test_tails.extend(tail_sets["test"])
        self.assertEqual(sorted(seen_test_tails), [f"D{i}" for i in range(6)])
        self.assertEqual(audit["test_tail_coverage"]["unique_test_tails"], 6)


if __name__ == "__main__":
    unittest.main()
