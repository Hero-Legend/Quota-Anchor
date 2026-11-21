"""Independent within-CIViC teacher distillation on the locked MedCPT Top100.

Gold labels never enter feature extraction, training, splitting or selection.
The returned aligned OOF scores are evaluated by the caller in a separate row
from frozen CTD-head zero-shot transfer. No scientific run occurs on import.
"""
from __future__ import annotations
import gzip,hashlib,json,math
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS=(.01,.1,1.,10.,100.)

def load_teacher_map(path,locked_candidates):
    """Latest record wins, THEN parse-valid finite filtering; never fill gaps.

    locked_candidates maps each qid to the original MedCPT Top100 IDs, taken
    from the frozen source retrieval. All expected pairs must have a valid
    *latest* teacher score. An earlier valid retry cannot repair a later error.
    """
    path=Path(path);latest={};rows=0;duplicates=0
    with (gzip.open(path,'rt',encoding='utf8') if path.suffix=='.gz' else path.open(encoding='utf8')) as stream:
        for line in stream:
            if not line.strip():continue
            r=json.loads(line);key=(r['example_id'],r['document_id'])
            duplicates+=int(key in latest);latest[key]=r;rows+=1
    expected=set()
    for qid,ids in locked_candidates.items():
        if len(ids)!=100 or len(set(ids))!=100:raise ValueError('Locked MedCPT pool must contain100 unique IDs per query')
        expected.update((qid,d) for d in ids)
    valid={key:float(r['score']) for key,r in latest.items()
        if r.get('parse_error') is None and type(r.get('score')) in (float,int) and math.isfinite(r['score'])}
    missing=expected-set(valid);extra=set(latest)-expected
    if missing or extra:raise ValueError(f'Teacher/locked-pool mismatch: {len(missing)} missing-or-invalid latest, {len(extra)} extra')
    audit=dict(path=str(path),sha256=_sha(path),rows=rows,duplicate_records=duplicates,
        unique_latest_pairs=len(latest),valid_latest_pairs=len(valid),expected_pairs=len(expected),
        parser='last record per exact query-PMID, then parse_error is None and finite numeric score',
        source_pool='locked MedCPT Top100',missing=0,extra=0,imputed_pairs=0)
    return {key:valid[key] for key in expected},audit

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def _safe_record(record):
    # Explicit feature whitelist prevents even a future feature function from
    # reading labels, source-derived gold fields, teacher predictions or splits.
    allowed=('med','bge','lex','bio','bio_std','modern','huber')
    result={'qid':record['qid'],'tail':record['tail'],'ids':list(record['ids']),
        'scores':{k:np.asarray(v,float).copy() for k,v in record['scores'].items() if k in allowed}}
    if 'bio_seeds' in record:result['bio_seeds']=np.asarray(record['bio_seeds'],float).copy()
    return result

def _aligned_teacher(records,teacher_map):
    keys=[(q['qid'],d) for q in records for d in q['ids']]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate query-PMID pairs')
    if set(keys)!=set(teacher_map):raise ValueError('Records must match complete locked teacher pool exactly; no source replacement or partial filtering')
    values=np.asarray([teacher_map[key] for key in keys],float)
    if not np.isfinite(values).all():raise ValueError('Actual finite teacher scores required; no imputation')
    return values

def fit_predict_distill(records,teacher_map,feature_fn):
    """Return ({qid: aligned OOF scores}, JSON-serializable audit).

    Outer GroupKFold5 splits whole diseases. For each outer training set,
    training-only GroupKFold3 chooses L2 alpha by pooled held-out teacher MSE.
    StandardScaler is fitted inside each inner training partition and again
    on the complete outer training partition. No CIViC gold is accessed.
    """
    records=list(records)
    if len({q['qid'] for q in records})!=len(records):raise ValueError('Duplicate query records')
    if any(len(q['ids'])!=100 or len(set(q['ids']))!=100 for q in records):raise ValueError('Only locked MedCPT Top100 is permitted')
    groups=np.asarray([str(q['tail']) for q in records])
    if len(set(groups))<5:raise ValueError('Five distinct disease groups required for OOF5')
    y=_aligned_teacher(records,teacher_map);arrays=[];feature_names=None
    for q in records:
        x,names=feature_fn(_safe_record(q));x=np.asarray(x,float)
        if x.ndim!=2 or len(x)!=100 or not np.isfinite(x).all():raise ValueError('Finite aligned real contextual features required')
        if feature_names is None:feature_names=list(names)
        if list(names)!=feature_names:raise ValueError('Context feature schema changed')
        arrays.append(x)
    X=np.concatenate(arrays);pair_groups=np.repeat(groups,100)
    query_indices=np.arange(len(records));prediction=np.full(len(y),np.nan);fold_audits=[]
    for fold,(train_q,test_q) in enumerate(GroupKFold(5).split(query_indices,groups=groups)):
        train_groups=set(groups[train_q]);test_groups=set(groups[test_q])
        if train_groups&test_groups:raise ValueError('Disease leakage')
        train=np.concatenate([np.arange(i*100,(i+1)*100) for i in train_q]);test=np.concatenate([np.arange(i*100,(i+1)*100) for i in test_q])
        inner=GroupKFold(min(3,len(train_groups)));trace=[]
        for alpha in ALPHAS:
            squared_error=0.;count=0
            for fit_ix,val_ix in inner.split(train,groups=pair_groups[train]):
                fit=train[fit_ix];val=train[val_ix]
                model=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='cholesky'))
                model.fit(X[fit],y[fit]);p=model.predict(X[val])
                squared_error+=float(np.square(p-y[val]).sum());count+=len(val)
            trace.append(dict(alpha=alpha,training_inner_teacher_MSE=squared_error/count,validation_pairs=count))
        selected=min(trace,key=lambda r:(r['training_inner_teacher_MSE'],-r['alpha']))['alpha']
        model=make_pipeline(StandardScaler(),Ridge(alpha=selected,solver='cholesky'))
        model.fit(X[train],y[train]);prediction[test]=model.predict(X[test])
        scale=model.named_steps['standardscaler'];ridge=model.named_steps['ridge']
        fold_audits.append(dict(fold=fold,train_qids=[records[i]['qid'] for i in train_q],test_qids=[records[i]['qid'] for i in test_q],
            train_diseases=sorted(train_groups),test_diseases=sorted(test_groups),disease_overlap=0,
            selected_alpha=selected,selection_trace=trace,heldout_teacher_MSE=float(np.mean((prediction[test]-y[test])**2)),
            scaler_mean=scale.mean_.tolist(),scaler_scale=scale.scale_.tolist(),coef=ridge.coef_.tolist(),intercept=float(ridge.intercept_)))
    if not np.isfinite(prediction).all():raise ValueError('Incomplete OOF predictions')
    predictions={q['qid']:prediction[i*100:(i+1)*100].tolist() for i,q in enumerate(records)}
    pair_digest=hashlib.sha256(json.dumps([(q['qid'],list(q['ids'])) for q in records],separators=(',',':')).encode()).hexdigest()
    audit=dict(version='civic-disease-group-teacher-distill-v1',status='completed_actual_fit',queries=len(records),pairs=len(y),diseases=len(set(groups)),
        source_pool='locked MedCPT Top100; exact complete teacher pair identity required',pair_identity_sha256=pair_digest,
        features=feature_names,outer_split='GroupKFold5 by disease tail',inner_selection='GroupKFold3 only within outer training; teacher MSE',
        alpha_grid=list(ALPHAS),gold_accessed_for_training=False,gold_accessed_for_hyperparameter_selection=False,
        missing_teacher_imputed=0,external_zero_shot_claim=False,reference_rows='DeepSeek teacher, Llama reference and CTD zero-shot transfer reported separately',folds=fold_audits)
    return predictions,audit

def teacher_reference_predictions(records,teacher_map):
    """Aligned actual teacher/reference scores; no fitting, no missing zeros."""
    records=list(records);values=_aligned_teacher(records,teacher_map);out={};start=0
    for q in records:
        end=start+len(q['ids']);out[q['qid']]=values[start:end].tolist();start=end
    return out
