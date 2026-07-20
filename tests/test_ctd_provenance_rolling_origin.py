from __future__ import annotations

import argparse
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from tempprov_audit.cli.build_ctd_provenance_rolling_origin import (
    assign_tail_groups,
    build_pair_year_example,
    build_year_pools,
    parse_ctd_direct_pairs,
)
from tempprov_audit.cli.run_ctd_provenance_cross_encoder import discard_run_best_model


def document(pmid: str, year: int) -> dict[str, object]:
    return {
        "document_key": f"key-{pmid}",
        "document_id": f"PMID:{pmid}",
        "pmid": pmid,
        "pub_year": year,
        "title": f"Title {pmid}",
        "abstract": "Abstract",
        "text": "Text",
    }


class RollingOriginProvenanceTests(unittest.TestCase):
    def test_parse_ctd_direct_pairs_filters_inferred_rows_and_groups_years(self) -> None:
        documents = {"1": document("1", 2022), "2": document("2", 2024)}
        lines = [
            "Chem\tC1\t-\tDisease\tD1\tmarker/mechanism\t\t\t\t1|2\n",
            "Chem\tC1\t-\tDisease\tD1\t\tGENE\t1\t\t1\n",
            "Other\tC2\t-\tOther disease\tD2\ttherapeutic\t\t\t\t999\n",
        ]
        pairs, audit = parse_ctd_direct_pairs(lines, documents)
        self.assertEqual(set(pairs), {("C1", "D1")})
        self.assertEqual(pairs[("C1", "D1")]["pmids_by_year"][2022], {"1"})
        self.assertEqual(pairs[("C1", "D1")]["pmids_by_year"][2024], {"2"})
        self.assertEqual(audit["ctd_direct_evidence_rows"], 2)

    def test_pair_year_example_has_target_gold_and_only_prior_history(self) -> None:
        documents = {str(i): document(str(i), year) for i, year in [(1, 2022), (2, 2024), (3, 2024), (4, 2024)]}
        record = {
            "head_id": "C1",
            "head_text": "Chem",
            "tail_id": "D1",
            "tail_text": "Disease",
            "pmids_by_year": defaultdict(set, {2022: {"1"}, 2024: {"2"}}),
        }
        pairs = {("C1", "D1"): record}
        head_pool, tail_pool, global_pool = build_year_pools(pairs, (2022, 2024))
        global_pool[2024].update({"3", "4"})
        row = build_pair_year_example(
            record,
            2024,
            documents,
            head_pool,
            tail_pool,
            global_pool,
            seed=7,
            same_tail_negatives=1,
            same_head_negatives=1,
            random_negatives=2,
            minimum_target_year_candidates=3,
            historical_context=2,
        )
        self.assertEqual(row["gold_document_ids"], ["PMID:2"])
        candidates = {item["pmid"]: item for item in row["candidates"]}
        self.assertEqual(candidates["1"]["source"], "prior_pair_ctd_citation_context")
        self.assertTrue(candidates["2"]["is_ctd_cited"])
        self.assertTrue(
            all(item["pub_year"] == 2024 for item in row["candidates"] if item["pmid"] != "1")
        )

    def test_tail_assignment_is_complete_and_disjoint(self) -> None:
        pairs = {}
        for index in range(12):
            pairs[(f"C{index}", f"D{index}")] = {
                "head_id": f"C{index}",
                "tail_id": f"D{index}",
                "pmids_by_year": defaultdict(set, {2024: {str(index)}}),
            }
        folds = assign_tail_groups(pairs, 5, 13)
        self.assertEqual(len(set().union(*folds)), 12)
        self.assertEqual(sum(len(fold) for fold in folds), 12)

    def test_checkpoint_cleanup_is_confined_to_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "seed_13"
            checkpoint = run_dir / "best_model"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.safetensors").write_text("x", encoding="utf-8")
            sibling = run_dir / "summary.json"
            sibling.write_text("{}", encoding="utf-8")
            discard_run_best_model(run_dir)
            self.assertFalse(checkpoint.exists())
            self.assertTrue(sibling.exists())


if __name__ == "__main__":
    unittest.main()
