"""Frozen v3 reciprocal dense-intersection quotas; no hidden cache restriction."""
import numpy as np
from .metrics import ranking_metrics,clustered_interval
GRID={100:[85,90,95,98],200:[175,180,185,190,195]}
POLICY='outside_budget_to1000'
def construct(bm,bge,med,k,budget):
    if budget not in GRID or k not in GRID[budget]:raise ValueError('Frozen K grid required')
    if len(bm)<1000 or len(set(bm[:1000]))!=1000:raise ValueError('1000 unique BM25 ids required')
    core=bm[:k];br={d:i+1 for i,d in enumerate(bm)}
    gr={d:i+1 for i,d in enumerate(bge)};mr={d:i+1 for i,d in enumerate(med)}
    allowed=(set(bge[:50])&set(med[:50]))-set(core)
    allowed={d for d in allowed if budget<br.get(d,1001)<=1000}
    ordered=sorted(allowed,key=lambda d:(-(1/(60+gr[d])+1/(60+mr[d])),br[d],str(d)))
    graft=ordered[:budget-k];ids=list(dict.fromkeys(core+graft+bm))[:budget]
    if len(ids)!=budget or len(set(ids))!=budget:raise ValueError('Quota budget changed')
    return ids,dict(grafted=len(graft),eligible=len(allowed),fallback=budget-k-len(graft))
def choose_k(dev,channels,budget):
    positive=[q for q in dev if q['gold_document_ids']]
    if not positive:raise ValueError('Complete positive development population absent')
    trace=[]
    for k in GRID[budget]:
        cov=[];grafts=0
        for q in positive:
            qid=q['example_id'];ids,a=construct(channels['bm25'][qid],channels['bge_large'][qid],channels['medcpt_dual'][qid],k,budget)
            cov.append(len(set(ids)&set(q['gold_document_ids']))/len(set(q['gold_document_ids'])));grafts+=a['grafted']
        trace.append(dict(kbase=k,development_coverage=float(np.mean(cov)),grafted_pairs=grafts))
    selected=max(trace,key=lambda r:(r['development_coverage'],r['kbase']))
    return selected['kbase'],trace
def holm(rows,family):
    previous=0.
    for i,r in enumerate(sorted(rows,key=lambda x:x['paired_p'])):
        previous=max(previous,min(1.,r['paired_p']*(len(rows)-i)));r.update(holm_p=previous,holm_family=family,holm_family_size=len(rows))
def compare(rows,budget,method,control,metric,repetitions=10000):
    a=sorted([r for r in rows if r['budget']==budget and r['method']==method],key=lambda r:(r['fold'],r['qid']))
    b=sorted([r for r in rows if r['budget']==budget and r['method']==control],key=lambda r:(r['fold'],r['qid']))
    if [(r['fold'],r['qid'],r['tail']) for r in a]!=[(r['fold'],r['qid'],r['tail']) for r in b]:raise ValueError('Unpaired query populations')
    return dict(cohort='full_library',pool='Quota_v3_Top'+str(budget),budget=budget,method=method,control=control,metric=metric,
        **clustered_interval([r[metric] for r in a],[r['tail'] for r in a],[r[metric] for r in b],repetitions=repetitions,seed=20261117))
