"""Full same-fold dev empirical pair controls. No FDR or nonempty guarantee."""
import hashlib
import json
from statistics import NormalDist
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

METHODS=('Raw_Logit','Platt_Logistic','Wilson_Bound','DePro_QueryNormalized_Empirical_v3','DePro_QueryNormalized_v3')
FEATURES=('bio_mean','bio_query_top1_margin','bio_seed_variance','med_percentile','bge_percentile')


def features(q):
    n=len(q['ids']);bio=np.asarray(q['bio_seeds'],float)
    if bio.shape!=(3,n) or not n or not np.isfinite(bio).all():raise ValueError('Three actual aligned seed scores required')
    mean=bio.mean(0);columns=[mean,mean-mean.max(),bio.var(0)]
    for name in ['med','bge']:
        values=np.asarray(q['scores'][name],float)
        if values.shape!=(n,) or not np.isfinite(values).all():raise ValueError('Real aligned anchors required')
        columns.append((rankdata(values,method='average')-.5)/n)
    return np.column_stack(columns)


def labels(q):return np.asarray([str(d) in {str(x) for x in q['gold']} for d in q['ids']],int)


def fit_logistic(X,y,C=1.):
    mean=X.mean(0);scale=np.maximum(X.std(0),1e-8)
    if len(np.unique(y))!=2:raise ValueError('Full-dev pair calibrator requires both actual label classes')
    estimator=LogisticRegression(C=C,max_iter=5000,solver='lbfgs').fit((X-mean)/scale,y)
    return dict(C=C,mean=mean.tolist(),scale=scale.tolist(),coef=estimator.coef_[0].tolist(),
                intercept=float(estimator.intercept_[0]),iterations=int(estimator.n_iter_[0]))


def probability(model,X):
    return expit(((X-np.asarray(model['mean']))/np.asarray(model['scale']))@np.asarray(model['coef'])+model['intercept'])


def wilson(errors,n):
    z=1.959963984540054;p=errors/n
    return (p+z*z/(2*n)+z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def threshold(scores,y,clusters,mode,alpha=.10):
    """Exact distinct-score scan by cumulative counts; no quadratic N x N scan.

    Guard=max(pooled FDP, active-cluster mean + .5 sample SD). A single
    active cluster falls back to its empirical point value, explicitly marked
    degenerate. Inactive clusters have undefined FDP and are never filled zero.
    """
    scores=np.asarray(scores,float);y=np.asarray(y,bool);clusters=np.asarray(clusters)
    if not len(scores) or not np.isfinite(scores).all():raise ValueError('Finite dev scores required')
    order=np.argsort(-scores,kind='stable');s=scores[order];truth=y[order]
    ends=np.flatnonzero(np.r_[s[:-1]!=s[1:],True]);n=ends+1;tp=np.cumsum(truth)[ends];fp=n-tp
    empirical=fp/n;bound=empirical.copy();names=sorted(set(clusters));active=np.zeros(len(ends),int)
    cluster_mean=np.full(len(ends),np.nan);cluster_sd=np.full(len(ends),np.nan)
    if mode=='wilson':bound=wilson(fp,n)
    if mode=='cluster_mean_half_sd':
        values=[]
        for name in names:
            membership=(clusters[order]==name)
            cn=np.cumsum(membership)[ends];ce=np.cumsum(membership & ~truth)[ends]
            values.append(np.divide(ce,cn,out=np.full(len(ends),np.nan),where=cn>0))
        values=np.asarray(values);valid=np.isfinite(values);active=valid.sum(0)
        cluster_mean=np.nansum(values,0)/active
        squared=np.nansum((values-cluster_mean[None,:])**2,0)
        cluster_sd=np.sqrt(np.divide(squared,active-1,out=np.zeros(len(ends)),where=active>1))
        bound=np.maximum(empirical,cluster_mean+.5*cluster_sd)
    eligible=np.flatnonzero(bound<=alpha)
    best=int(max(eligible,key=lambda i:(int(tp[i]),float(tp[i]/n[i]),float(s[ends[i]])))) if len(eligible) else None
    return dict(threshold=None if best is None else float(s[ends[best]]),accepted=0 if best is None else int(n[best]),
        true_accepted=0 if best is None else int(tp[best]),errors=0 if best is None else int(fp[best]),
        dev_fdp=None if best is None else float(empirical[best]),dev_bound=None if best is None else float(bound[best]),
        dev_cluster_mean=None if best is None or mode!='cluster_mean_half_sd' else float(cluster_mean[best]),
        dev_cluster_sample_sd=None if best is None or mode!='cluster_mean_half_sd' else float(cluster_sd[best]),
        active_diseases=None if best is None or mode!='cluster_mean_half_sd' else int(active[best]),
        disease_clusters=len(names),guard_status=('degenerate_full_dev_single_cluster' if len(names)==1 else
            'degenerate_single_active_cluster' if best is not None and mode=='cluster_mean_half_sd' and active[best]==1 else 'empirical_only'),
        mode=mode,alpha=alpha,unique_thresholds=len(ends),empty_acceptance=best is None,
        selection='full-dev fitted and full-dev threshold; max recall then precision then threshold; optimistic empirical development',
        guarantee=False)


def fit_controls(dev,C=1.,alpha=.10):
    X=np.concatenate([features(q) for q in dev]);y=np.concatenate([labels(q) for q in dev])
    clusters=np.concatenate([np.repeat(str(q['tail']),len(q['ids'])) for q in dev])
    platt=fit_logistic(X[:,:1],y,C);qn=fit_logistic(X,y,C)
    ps=probability(platt,X[:,:1]);qs=probability(qn,X)
    controls={}
    for name,s,mode in [(METHODS[0],X[:,0],'empirical'),(METHODS[1],ps,'empirical'),(METHODS[2],ps,'wilson'),
                        (METHODS[3],qs,'empirical'),(METHODS[4],qs,'cluster_mean_half_sd')]:
        controls[name]=threshold(s,y,clusters,mode,alpha)
    return dict(platt=platt,normalized=qn,controls=controls,features=list(FEATURES),full_dev_queries=len(dev),
        full_dev_pairs=len(y),full_dev_diseases=len(set(clusters)),positive_pairs=int(y.sum()),
        fit_qids_sha256=hashlib.sha256(json.dumps(sorted(q['qid'] for q in dev)).encode()).hexdigest())


def predict_controls(model,q):
    X=features(q);ps=probability(model['platt'],X[:,:1]);qs=probability(model['normalized'],X)
    result={}
    for name,s in zip(METHODS,[X[:,0],ps,ps,qs,qs]):
        t=model['controls'][name]['threshold'];result[name]=np.zeros(len(s),bool) if t is None else s>=t
    return result


def count_row(q,accepted,fold):
    y=labels(q).astype(bool);a=np.asarray(accepted,bool)
    if a.shape!=y.shape:raise ValueError('Pair acceptance alignment')
    return dict(qid=str(q['qid']),tail=str(q['tail']),fold=str(fold),accepted=int(a.sum()),errors=int((a&~y).sum()),
        true_accepted=int((a&y).sum()),gold_total=len(set(q['gold'])),gold_in_candidates=int(y.sum()),pairs=len(y),
        candidate_sha256=hashlib.sha256(json.dumps(sorted(q['ids'])).encode()).hexdigest(),
        gold_sha256=hashlib.sha256(json.dumps(sorted(q['gold'])).encode()).hexdigest())
