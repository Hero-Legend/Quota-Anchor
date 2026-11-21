"""Portable entry points for the released Quota-Anchor algorithm cores."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from . import distill, listwise, monotonic, quota


def read_records(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        if path.name.endswith(('.json', '.json.gz')):
            records = json.load(stream)
        else:
            records = [json.loads(line) for line in stream if line.strip()]
    if not isinstance(records, list) or not records:
        raise ValueError('A nonempty list of query records is required')
    if any(not isinstance(q, dict) for q in records):
        raise ValueError('Each query record must be an object')
    qids = [q['qid'] for q in records]
    if len(qids) != len(set(qids)):
        raise ValueError('Duplicate query identifiers')
    return records


def check_scores(records):
    for q in records:
        ids = q['ids']
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Candidate identifiers must be unique and nonempty')
        for key in ('med', 'bge', 'lex'):
            vector = np.asarray(q['scores'][key], dtype=float)
            if vector.shape != (len(ids),) or not np.isfinite(vector).all():
                raise ValueError('Finite, aligned score channel required: ' + key)
        if 'bio_seeds' in q:
            bio = np.asarray(q['bio_seeds'], dtype=float)
            if bio.shape != (3, len(ids)) or not np.isfinite(bio).all():
                raise ValueError('Three aligned Bio seed vectors are required')
    return records


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=convert) + '\n',
                    encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('quota', help='Construct a fixed-budget candidate pool')
    p.add_argument('--input', required=True)
    p.add_argument('--budget', type=int, choices=(100, 200), required=True)
    p.add_argument('--k', type=int, required=True,
                   help='Lexical-core size selected on legal development data')
    p.add_argument('--output', required=True)
    p = commands.add_parser('fit-head', help='Select scalar heads using legal Dev only')
    p.add_argument('--dev', required=True)
    p.add_argument('--config', default=str(Path(__file__).parent / 'config/monotonic.json'))
    p.add_argument('--output', required=True)
    p = commands.add_parser('rank', help='Rank aligned cached scores without gold access')
    p.add_argument('--input', required=True)
    p.add_argument('--state', help='Development-selected state; unused for fixed-expanded')
    p.add_argument('--variant', choices=('anchor', 'in_domain', 'sharp', 'bio', 'fixed-expanded'),
                   default='anchor')
    p.add_argument('--output', required=True)
    p = commands.add_parser('distill-oof', help='Disease-group teacher distillation')
    p.add_argument('--input', required=True)
    p.add_argument('--teacher', required=True, help='Archived teacher JSONL, optionally gzipped')
    p.add_argument('--output', required=True)
    args = parser.parse_args(argv)

    if args.command == 'quota':
        rows = []
        for q in read_records(args.input):
            ids, audit = quota.construct(q['bm25'], q['bge_large'], q['medcpt_dual'],
                                         args.k, args.budget)
            rows.append(dict(qid=q['qid'], ids=ids, audit=audit))
        write_json(args.output, rows)
    elif args.command == 'fit-head':
        config = json.loads(Path(args.config).read_text(encoding='utf-8'))
        monotonic.validate_config(config)
        dev = check_scores(read_records(args.dev))
        write_json(args.output, monotonic.fit_development(dev, config))
    elif args.command == 'rank':
        records = check_scores(read_records(args.input))
        if args.variant != 'fixed-expanded' and not args.state:
            parser.error('--state is required except for fixed-expanded')
        state = None if args.variant == 'fixed-expanded' else json.loads(
            Path(args.state).read_text(encoding='utf-8'))
        output = []
        for q in records:
            # Whitelist the score inputs: scoring never sees gold annotations.
            record = {k: q[k] for k in ('qid', 'ids', 'scores', 'bio_seeds', 'cohort') if k in q}
            scores = (monotonic.fixed_expanded(record) if args.variant == 'fixed-expanded'
                      else monotonic.predict(state, record, args.variant))
            order = np.argsort(-np.asarray(scores), kind='stable')
            output.append(dict(qid=q['qid'], ranked_ids=[q['ids'][i] for i in order],
                               scores=[float(scores[i]) for i in order]))
        write_json(args.output, output)
    else:
        records = check_scores(read_records(args.input))
        teacher, audit = distill.load_teacher_map(
            args.teacher, {q['qid']: q['ids'] for q in records})
        scores, training_audit = distill.fit_predict_distill(records, teacher, listwise.features)
        write_json(args.output, dict(scores=scores, teacher_audit=audit,
                                     training_audit=training_audit))


if __name__ == '__main__':
    main()
