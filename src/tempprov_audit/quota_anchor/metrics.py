"""Corrected unique-gold ranking metrics and disease-cluster paired inference."""
import numpy as np

def ranking_metrics(ranked, gold):
    gold = set(gold)
    ranked = list(dict.fromkeys(ranked))
    if not gold: return None
    hit = np.asarray([d in gold for d in ranked], dtype=float)
    precision = np.cumsum(hit) / np.arange(1, len(hit)+1)
    n = min(10, len(hit))
    dcg = float(np.sum(hit[:n] / np.log2(np.arange(2, n+2))))
    ideal = float(np.sum(1 / np.log2(np.arange(2, min(10, len(gold))+2))))
    return dict(MAP=float(np.sum(precision * hit)/len(gold)),
                Recall10=float(hit[:10].sum()/len(gold)),
                Recall50=float(hit[:50].sum()/len(gold)),
                Coverage100=float(hit[:100].sum()/len(gold)),
                Coverage1000=float(hit[:1000].sum()/len(gold)),
                NDCG10=dcg/ideal)

def clustered_interval(values, clusters, other=None, repetitions=10000, seed=20261116):
    """Query-weighted mean, resampling whole diseases. Sign flip by disease."""
    values = np.asarray(values, dtype=float)
    unique = sorted(set(clusters))
    indices = [[i for i,c in enumerate(clusters) if c==g] for g in unique]
    sums = np.asarray([values[ix].sum() for ix in indices])
    counts = np.asarray([len(ix) for ix in indices])
    rng = np.random.default_rng(seed)
    samples=[]; extreme=0
    delta = None if other is None else values-np.asarray(other, dtype=float)
    dsums = None if delta is None else np.asarray([delta[ix].sum() for ix in indices])
    for start in range(0,repetitions,128):
        n=min(128,repetitions-start)
        ix=rng.integers(0,len(unique),size=(n,len(unique)))
        samples.extend(((sums[ix].sum(1))/(counts[ix].sum(1))).tolist())
        if delta is not None:
            signs=rng.choice([-1,1],size=(n,len(unique)))
            extreme+=int((np.abs((signs*dsums).sum(1)) >= abs(delta.sum())-1e-12).sum())
    ans=dict(mean=float(values.mean()),ci_low=float(np.quantile(samples,.025)),
             ci_high=float(np.quantile(samples,.975)),clusters=len(unique),resamples=repetitions)
    if delta is not None:
        diff=clustered_interval(delta,clusters,repetitions=repetitions,seed=seed+1)
        ans.update(delta=diff['mean'],delta_ci_low=diff['ci_low'],delta_ci_high=diff['ci_high'],
                   paired_p=(extreme+1)/(repetitions+1))
    return ans
