"""Query-context RankNet and train/dev-frozen bi-anchor OOD switching."""
import hashlib
import numpy as np
from scipy.stats import rankdata, spearmanr
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

VERSION='listwise-context-bi-anchor-20261117-v2'

def z(x):
    x=np.asarray(x,float)
    if not np.isfinite(x).all():raise ValueError('Finite actual scores required')
    return (x-x.mean())/max(x.std(),1e-6)

def pct(x):return (rankdata(np.asarray(x,float),method='average')-.5)/max(1,len(x))

def normalized_components(record):
    s=record['scores']; n=len(record['ids'])
    for key in ['med','bge','lex']:
        if key not in s or len(s[key])!=n:raise ValueError('Missing genuine '+key+' score')
    bio=np.asarray(record.get('bio_seeds',np.zeros((0,n))),float)
    if bio.size and (bio.ndim!=2 or bio.shape[1]!=n):raise ValueError('Seed-score alignment mismatch')
    return bio,{k:np.asarray(s[k],float) for k in ['med','bge','lex']}

def features(record,extended=False):
    bio,signals=normalized_components(record);n=len(record['ids']);hasbio=bool(bio.size)
    means=bio.mean(0) if hasbio else np.zeros(n)
    std=bio.std(0) if hasbio else np.zeros(n)
    signals={'bio':means,**signals};columns=[];names=[]
    for name,values in signals.items():
        standardized=z(values);percentile=pct(values);prob=softmax(standardized)
        ordered=np.sort(values)
        entries={'raw':values,'z':standardized,'pct':percentile,'softmax':prob,
                 'top1_margin':values-values.max(),
                 'top5_margin':values-ordered[-min(5,n):].mean(),
                 'top10_margin':values-ordered[-min(10,n):].mean(),
                 'entropy_x_z':(-np.sum(prob*np.log(prob+1e-12)))*standardized}
        for suffix,array in entries.items():columns.append(array);names.append(name+'_'+suffix)
    columns += [std,z(std),np.full(n,float(hasbio))]
    names += ['bio_seed_std','bio_seed_std_z','bio_available']
    for key in ['med','bge','lex']:
        a=pct(means);b=pct(signals[key]);difference=b-a
        columns += [difference,np.abs(difference),z(std)*difference,z(means)*z(signals[key])]
        names += [key+'_bio_rank_difference',key+'_bio_rank_disagreement',
                  key+'_variance_rank_interaction',key+'_bio_score_product']
    columns += [pct(signals['med'])-pct(signals['bge']),z(signals['med'])*z(signals['bge'])]
    names += ['anchor_rank_difference','anchor_score_product']
    if extended:
        for key in ['modern','huber']:
            if key not in record['scores']:raise ValueError('Internal rich model requires actual '+key)
            v=np.asarray(record['scores'][key]);u=z(v);p=pct(v)
            columns += [u,p,p-pct(means),u*z(means),u*z(signals['med'])]
            names += [key+'_z',key+'_pct',key+'_bio_rank_difference',key+'_bio_product',key+'_med_product']
    x=np.column_stack(columns)
    if not np.isfinite(x).all():raise ValueError('Nonfinite contextual features')
    return x,names

def anchor(record,weights):
    _,s=normalized_components(record)
    return sum(float(w)*z(s[k]) for k,w in zip(['med','bge','lex'],weights))

def shift(record,weights):
    bio,_=normalized_components(record)
    if not bio.size:return np.ones(2)
    mean=bio.mean(0);a=anchor(record,weights)
    top=set(np.argsort(mean)[-min(20,len(mean)):])|set(np.argsort(a)[-min(20,len(a)):])
    ix=np.asarray(sorted(top),int)
    r1=rankdata(mean)[ix];r2=rankdata(a)[ix]
    correlation=0. if min(r1.std(),r2.std())<1e-8 else float(spearmanr(r1,r2).statistic)
    variance=float(np.mean(np.var(bio,axis=0))/max(np.var(mean),1e-6))
    return np.asarray([1-correlation,variance])

def ap(record,score):
    gold=set(record['gold'])
    if not gold:return None
    order=np.argsort(-np.asarray(score),kind='stable');hit=0;total=0.
    for rank,i in enumerate(order,1):
        if record['ids'][i] in gold:hit+=1;total+=hit/rank
    return total/len(gold)

def objective(records,score):
    values=[ap(q,score(q)) for q in records if q['gold']]
    return float(np.mean(values)) if values else float('-inf')

def fit_linear(records,C,*,pairwise=True,extended=False):
    arrays=[];labels=[];weights=[]
    candidates=np.concatenate([features(q,extended)[0] for q in records]);scale=StandardScaler().fit(candidates)
    names=features(records[0],extended)[1]
    for q in records:
        x=scale.transform(features(q,extended)[0]);y=np.asarray([d in q['gold'] for d in q['ids']],bool)
        if pairwise:
            pos=np.flatnonzero(y);negative=np.flatnonzero(~y)
            if not len(pos) or not len(negative):continue
            seed=int(hashlib.sha256(q['qid'].encode()).hexdigest()[:8],16)
            rng=np.random.default_rng(seed)
            hard=set()
            for s in [q['scores'].get('bio',np.zeros(len(y))),q['scores']['med'],q['scores']['bge']]:
                hard.update(sorted(negative,key=lambda i:-s[i])[:10])
            hard.update(rng.choice(negative,min(10,len(negative)),replace=False).tolist())
            pairs=[(p,n) for p in pos for n in sorted(hard)]
            # Each positive query contributes equal total mass to RankNet loss.
            d=np.asarray([x[p]-x[n] for p,n in pairs]);arrays.extend([d,-d])
            labels.extend([np.ones(len(d)),np.zeros(len(d))]);weights.extend([np.full(len(d),1/(2*len(d)))]*2)
        else:
            arrays.append(x);labels.append(y);weights.append(np.full(len(y),1/max(1,len(y))))
    if not arrays:raise ValueError('No valid development ranking pairs')
    X=np.concatenate(arrays);Y=np.concatenate(labels);W=np.concatenate(weights)
    # Normalize sample weights to preserve meaningful C across record sizes.
    W*=len(W)/W.sum()
    model=LogisticRegression(C=C,fit_intercept=not pairwise,solver='lbfgs',max_iter=1000,
                             random_state=20261117).fit(X,Y,sample_weight=W)
    return dict(version=VERSION,features=names,mean=scale.mean_.tolist(),scale=scale.scale_.tolist(),
                coef=model.coef_[0].tolist(),intercept=float(model.intercept_[0]),C=C,pairwise=pairwise,
                extended=extended,fit_qids=[q['qid'] for q in records],training_pairs=len(Y),converged=bool(model.n_iter_[0]<1000))

def linear_score(state,record):
    x,names=features(record,state.get('extended',False))
    if names!=state['features']:raise ValueError('Feature schema mismatch')
    return ((x-np.asarray(state['mean']))/np.asarray(state['scale']))@np.asarray(state['coef'])+state['intercept']

def gate_value(state,record):
    bio,_=normalized_components(record)
    if not bio.size:return 1.
    u=shift(record,state['anchor_weights'])
    quantiles=[np.searchsorted(np.asarray(c),v,side='right')/len(c) for c,v in zip(state['shift_reference'],u)]
    D=float(np.mean(quantiles))
    return float(expit((D-state['gate_threshold'])/state['gate_temperature']))

def predict(state,record,variant='full'):
    base=anchor(record,state['anchor_weights'])
    if variant=='anchor' or not np.asarray(record.get('bio_seeds',[])).size:return base
    local=z(linear_score(state['ranker'],record))
    if variant=='refiner':return local
    if variant=='pointwise':return linear_score(state['pointwise'],record)
    if variant=='static_half':return .5*local+.5*base
    g=gate_value(state,record)
    return (1-g)*local+g*base

def fit_development(fit,selection):
    if set(q['qid'] for q in fit)&set(q['qid'] for q in selection):raise ValueError('Development split overlap')
    grid=[.01,.1,1.,10.];trace=[]
    heads={}
    for kind in ['ranker','pointwise']:
        options=[]
        for c in grid:
            h=fit_linear(fit,c,pairwise=kind=='ranker')
            value=objective(selection,lambda q:linear_score(h,q));trace.append(dict(kind=kind,C=c,selection_MAP=value))
            options.append((value,h))
        heads[kind]=max(options,key=lambda x:x[0])[1]
    weights_grid=[[1.,0,0],[0,1.,0],[0,0,1.],[.8,.2,0],[.5,.5,0],[.6,.3,.1],[.5,.25,.25]]
    weights=max(weights_grid,key=lambda w:objective(selection,lambda q:anchor(q,w)))
    reference=np.asarray([shift(q,weights) for q in fit]);cdf=[sorted(column.tolist()) for column in reference.T]
    state=dict(version=VERSION,**heads,anchor_weights=weights,shift_reference=cdf,
               gate_threshold=.95,gate_temperature=.025,fit_qids=[q['qid'] for q in fit],selection_qids=[q['qid'] for q in selection])
    options=[]
    for threshold in [.75,.9,.95]:
        for temperature in [.025,.1]:
            candidate={**state,'gate_threshold':threshold,'gate_temperature':temperature}
            value=objective(selection,lambda q:predict(candidate,q))
            trace.append(dict(kind='OOD_gate',threshold=threshold,temperature=temperature,selection_MAP=value))
            options.append((value,candidate))
    state=max(options,key=lambda x:x[0])[1];state['selection_trace']=trace
    return state
