"""Small synthetic engineering checks; these are not experimental observations."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tempprov_audit.quota_anchor import distill, listwise, monotonic, quota, risk
from tempprov_audit.quota_anchor.cli import main


def record():
    return dict(qid='toy', tail='synthetic', ids=['a', 'b', 'c'], gold=['a'],
                scores=dict(med=[3., 2., 1.], bge=[1., 2., 3.], lex=[0., 1., 0.]),
                bio_seeds=[[3., 1., 0.], [2., 1., 0.], [4., 1., 0.]])


class QuotaAnchorTests(unittest.TestCase):
    def test_quota_budget_core_and_eligibility(self):
        bm = [str(i) for i in range(1000)]
        dense = bm[150:200] + bm[:150] + bm[200:]
        ids, audit = quota.construct(bm, dense, dense, 95, 100)
        self.assertEqual(len(ids), 100)
        self.assertEqual(len(set(ids)), 100)
        self.assertEqual(ids[:95], bm[:95])
        self.assertEqual(ids[95:], bm[150:155])
        self.assertEqual(audit['grafted'], 5)

    def test_quota_fallback_and_rejected_grid(self):
        bm = [str(i) for i in range(1000)]
        ids, audit = quota.construct(bm, bm, bm, 98, 100)
        self.assertEqual(ids, bm[:100])
        self.assertEqual(audit['fallback'], 2)
        with self.assertRaises(ValueError):
            quota.construct(bm, bm, bm, 99, 100)

    def test_minmax_affine_invariance_and_constants(self):
        x = np.array([1., 3., 2., 2.])
        np.testing.assert_allclose(monotonic.unit(x), monotonic.unit(7 * x - 20))
        np.testing.assert_array_equal(monotonic.unit([5., 5.]), [0., 0.])

    def test_features_ignore_gold_and_keep_49_columns(self):
        a = record(); b = copy.deepcopy(a); b['gold'] = ['wrong']
        x, names = listwise.features(a)
        y, _ = listwise.features(b)
        self.assertEqual(x.shape, (3, 49))
        self.assertEqual(len(names), 49)
        np.testing.assert_array_equal(x, y)
        self.assertNotIn('gold', distill._safe_record(a))

    def test_latest_invalid_teacher_is_rejected(self):
        ids = [str(i) for i in range(100)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'teacher.jsonl'
            rows = [dict(example_id='toy', document_id=d, score=1., parse_error=None) for d in ids]
            path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            values, _ = distill.load_teacher_map(path, {'toy': ids})
            self.assertEqual(len(values), 100)
            with path.open('a', encoding='utf-8') as stream:
                stream.write('\n' + json.dumps(dict(example_id='toy', document_id='0',
                                                    score=None, parse_error='invalid')))
            with self.assertRaises(ValueError):
                distill.load_teacher_map(path, {'toy': ids})

    def test_expanded_rejects_ctd_selected_head(self):
        q = record(); q['cohort'] = 'historical_expanded'
        with self.assertRaises(ValueError):
            monotonic.predict({}, q)

    def test_empty_risk_acceptance_stays_explicit(self):
        result = risk.threshold([3., 2., 1.], [0, 0, 0], ['x', 'x', 'x'], 'wilson')
        self.assertIsNone(result['threshold'])
        self.assertTrue(result['empty_acceptance'])
        self.assertFalse(result['guarantee'])

    def test_cli_fits_and_scores_without_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev = root / 'dev.json'; state = root / 'state.json'
            source = root / 'input.json'; output = root / 'ranking.json'
            dev.write_text(json.dumps([record()]), encoding='utf-8')
            main(['fit-head', '--dev', str(dev), '--output', str(state)])
            q = record(); del q['gold']
            source.write_text(json.dumps([q]), encoding='utf-8')
            main(['rank', '--input', str(source), '--state', str(state), '--output', str(output)])
            result = json.loads(output.read_text(encoding='utf-8'))[0]
            self.assertEqual(set(result['ranked_ids']), set(q['ids']))
            self.assertEqual(len(result['scores']), 3)


if __name__ == '__main__':
    unittest.main()
