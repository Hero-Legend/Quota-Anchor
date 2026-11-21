"""CPU-only scalar residual selection and fixed unlabelled sharp switching.

Zero residual restores the base ranking; no held-out performance floor is implied.
"""
import numpy as np
from scipy.stats import rankdata
from .metrics import ranking_metrics

VERSION='depro-monotonic-v3-20261117'
PRIMARY='DePro_Monotonic_v3'
IN_DOMAIN='DePro_InDomain_v3'
GAMMA005='DePro_Sharp_Gamma005_v3'
ANCHOR='BiAnchor_Tuned_v3'
RICH='DePro_Monotonic_Rich_v3'

def validate_config(config):
    expected=dict(omega=[0.,.2,.35,.5,.65,.8],eta=[0.,.1,.25],alpha=[0.,.02,.05,.1,.15,.25],beta=[0.,.02,.05,.1,.15,.25])
    for key,value in expected.items():
        if config.get(key)!=value:raise ValueError('Frozen '+key+' candidate grid changed')
    if config.get('ood_quantile')!=.95 or config.get('gamma_primary')!=0 or config.get('gamma_sensitivity')!=.05:raise ValueError('Fixed unlabelled OOD policy changed')
    if config.get('seed')!=20261117 or config.get('resamples')!=10000 or config.get('max_seconds')!=10800:raise ValueError('Frozen CPU scope changed')
    if config.get('GPU_inference') is not False:raise ValueError('GPU inference is prohibited')
    return True

def unit(values):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('Finite nonempty score vector required')
    width=float(x.max()-x.min())
    if not np.isfinite(width):raise ValueError('Score range overflow')
    return np.zeros_like(x) if width==0 else (x-x.min())/width

def z(values):
    x=np.asarray(values,float)
    if not np.isfinite(x).all():raise ValueError('Finite scores required')
    return (x-x.mean())/max(float(x.std()),1e-6)

def _parts(record,aux_mode='lexical'):
    n=len(record['ids']);scores=record['scores']
    out={}
    for k in ['med','bge','lex']:
        v=np.asarray(scores[k],float)
        if v.shape!=(n,):raise ValueError('Actual aligned '+k+' required')
        out[k]=unit(v)
    bio=np.asarray(record['bio_seeds'],float)
    if bio.shape!=(3,n) or not np.isfinite(bio).all():raise ValueError('Exactly own-fold three actual Bio seeds required')
    raw=bio.mean(0);sd=bio.std(0)
    out.update(raw_bio=raw,bio=unit(raw),uncertainty=np.zeros(n) if not np.any(sd) else (rankdata(sd,method='average')-.5)/n)
    if aux_mode=='lexical':out['aux']=out['lex']
    elif aux_mode=='modern_huber':
        out['aux']=(unit(scores['modern'])+unit(scores['huber']))/2
    else:raise ValueError('Unregistered auxiliary meaning')
    return out

def bi_anchor(parts,omega,eta):
    return ((1-omega)*parts['med']+omega*parts['bge']+eta*parts['lex'])/(1+eta)

def in_domain(parts,anchor,alpha,beta):
    if alpha<0 or beta<0:raise ValueError('Residual grid must be nonnegative')
    return (parts['bio']+alpha*parts['aux']+beta*parts['uncertainty']*anchor)/(1+alpha+beta)

def drift_descriptor(parts,anchor):
    b=parts['raw_bio'];n=len(b);k=min(20,n)
    ib=set(np.argsort(-b,kind='stable')[:k]);ia=set(np.argsort(-anchor,kind='stable')[:k]);ix=np.array(sorted(ib|ia),int)
    rb=rankdata(b,method='average')[ix];ra=rankdata(anchor,method='average')[ix]
    rho=0. if min(float(rb.std()),float(ra.std()))==0 else float(np.corrcoef(rb,ra)[0,1])
    q05,q25,q75,q95=np.quantile(b,[.05,.25,.75,.95]);iqr=float(q75-q25)
    result=np.array([1-len(ib&ia)/k,(1-np.clip(rho,-1,1))/2,float(b.max()),iqr,float(q95-q05)/max(iqr,1e-6)])
    if not np.isfinite(result).all():raise ValueError('Nonfinite unlabelled drift descriptor')
    return result

def _mid_cdf(reference,value):
    x=np.asarray(reference,float)
    return (np.searchsorted(x,value,side='left')+np.searchsorted(x,value,side='right'))/(2*len(x))

def drift_value(reference,descriptor):
    p=np.array([_mid_cdf(r,v) for r,v in zip(reference,descriptor)])
    p[2:]=2*np.abs(p[2:]-.5)
    return float(p.mean())

def fit_ood(parts,anchors,quantile=.95):
    if quantile!=.95:raise ValueError('Fixed unlabelled threshold only')
    x=np.array([drift_descriptor(p,a) for p,a in zip(parts,anchors)])
    reference=[np.sort(x[:,j]).tolist() for j in range(5)]
    values=np.array([drift_value(reference,row) for row in x])
    return dict(reference=reference,threshold=float(np.quantile(values,quantile,method='higher')),
                quantile=quantile,reference_queries=len(x),reference_labels_used=False,
                dev_switched_queries=int(np.sum(values>np.quantile(values,quantile,method='higher'))))

def _map(records,arrays):
    values=[]
    for q,s in zip(records,arrays):
        if q['gold']:
            ids=[q['ids'][i] for i in np.argsort(-s,kind='stable')]
            values.append(ranking_metrics(ids,q['gold'])['MAP'])
    if not values:raise ValueError('Development selection requires a positive query')
    return float(np.mean(values))

def fit_development(records,config,aux_mode='lexical',fixed_anchor=None):
    """Formal callers supply complete legal Dev; this module never loads Test."""
    parts=[_parts(q,aux_mode) for q in records];agrid=[]
    if fixed_anchor is None:
        for w in config['omega']:
            for eta in config['eta']:
                arrays=[bi_anchor(p,w,eta) for p in parts]
                agrid.append(dict(omega=float(w),eta=float(eta),dev_MAP=_map(records,arrays)))
        selected=max(agrid,key=lambda r:(r['dev_MAP'],-r['eta'],-abs(r['omega']-.5),-r['omega']))
        w,eta=selected['omega'],selected['eta']
    else:w,eta=fixed_anchor
    anchors=[bi_anchor(p,w,eta) for p in parts];rgrid=[]
    for alpha in config['alpha']:
        for beta in config['beta']:
            arrays=[in_domain(p,a,alpha,beta) for p,a in zip(parts,anchors)]
            rgrid.append(dict(alpha=float(alpha),beta=float(beta),dev_MAP=_map(records,arrays)))
    selected=max(rgrid,key=lambda r:(r['dev_MAP'],-r['alpha']-r['beta'],-r['alpha'],-r['beta']))
    baseline=_map(records,[p['bio'] for p in parts])
    if selected['dev_MAP']+1e-14<baseline:raise AssertionError('Zero residual candidate not preserved')
    state=dict(version=VERSION,aux_mode=aux_mode,omega=float(w),eta=float(eta),alpha=selected['alpha'],beta=selected['beta'],
        ood=fit_ood(parts,anchors,config['ood_quantile']),gamma_primary=0.,gamma_sensitivity=.05,
        dev_queries=len(records),dev_positive_queries=sum(bool(q['gold']) for q in records),
        dev_diseases=len({q['tail'] for q in records}),dev_positive_diseases=len({q['tail'] for q in records if q['gold']}),
        selection=dict(anchor_grid=agrid,residual_grid=rgrid,selected_in_domain_dev_MAP=selected['dev_MAP'],bio_dev_MAP=baseline,
                       floor_scope='selected in-domain Dev objective only; neither gated Dev nor held-out Test guarantee'))
    return state

def predict(state,record,variant='sharp',gamma=None):
    if record.get('cohort')=='historical_expanded':raise ValueError('Expanded forbids CTD-selected scalar heads')
    p=_parts(record,state['aux_mode']);a=bi_anchor(p,state['omega'],state['eta'])
    if variant=='anchor':return a
    if variant=='bio':return p['bio']
    refined=in_domain(p,a,state['alpha'],state['beta'])
    if variant=='in_domain':return refined
    if variant!='sharp':raise ValueError('Unknown declared branch')
    g=state['gamma_primary'] if gamma is None else float(gamma)
    if g not in [0.,.05]:raise ValueError('Frozen sharp gamma required')
    shifted=drift_value(state['ood']['reference'],drift_descriptor(p,a))>state['ood']['threshold']
    return (1-g)*a+g*p['bio'] if shifted else refined

def fixed_expanded(record):
    s=record['scores'];return (z(s['med'])+z(s['bge'])+.25*z(s['lex']))/2.25

def baseline_scores(record):
    s=record['scores']
    result=dict(MedCPT_CE=np.asarray(s['med']),BGE_Reranker=np.asarray(s['bge']),Lexical_BM25=np.asarray(s['lex']),
                BiAnchor_Equal=(z(s['med'])+z(s['bge']))/2)
    if 'bio_seeds' in record:result['BiomedBERT_ensemble']=np.asarray(record['bio_seeds']).mean(0)
    return result
