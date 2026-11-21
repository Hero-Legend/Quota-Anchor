"""Inherited count-based disease-cluster statistics; v3 protocol labels remain explicit."""

import numpy as np

STAT_METRICS = ("PooledFDP", "WorstFoldFDP", "FoldFDPStd", "FoldFDPVariance", "AcceptedRecall", "AcceptedCount")

def summarize_risk_rows(rows):
    def counts(part):
        a = sum(r["accepted"] for r in part); e = sum(r["errors"] for r in part)
        t = sum(r["true_accepted"] for r in part); g = sum(r["gold_total"] for r in part)
        gc = sum(r["gold_in_candidates"] for r in part)
        return {"AcceptedCount": int(a), "ErrorCount": int(e), "TrueAcceptedCount": int(t),
                "GoldCount": int(g), "GoldInCandidates": int(gc),
                "PooledFDP": float(e/a) if a else None,
                "AcceptedRecall": float(t/g) if g else None,
                "CandidateConditionalAcceptedRecall": float(t/gc) if gc else None,
                "EmptyAcceptanceQueries": sum(r["accepted"] == 0 for r in part),
                "PairCount": sum(r["pairs"] for r in part)}
    summary = counts(rows)
    per_fold = [{"fold": f, **counts([r for r in rows if r["fold"] == f])}
                for f in sorted({r["fold"] for r in rows})]
    fdps = [r["PooledFDP"] for r in per_fold if r["PooledFDP"] is not None]
    summary.update({"WorstFoldFDP": max(fdps) if fdps else None,
                    "FoldFDPStd": float(np.std(fdps)) if fdps else None,
                    "FoldFDPVariance": float(np.var(fdps)) if fdps else None,
                    "FoldFDPStd_ddof": 0, "EmptyAcceptanceFolds": sum(r["AcceptedCount"] == 0 for r in per_fold),
                    "NonemptyFolds": len(fdps), "folds": per_fold})
    return summary

def _arrays(rows):
    clusters = sorted({r["tail"] for r in rows}); folds = sorted({r["fold"] for r in rows})
    arr = np.zeros((len(clusters), len(folds), 4))
    ci = {c: i for i, c in enumerate(clusters)}; fi = {f: i for i, f in enumerate(folds)}
    for r in rows:
        arr[ci[r["tail"]], fi[r["fold"]]] += [r["accepted"], r["errors"], r["true_accepted"], r["gold_total"]]
    return clusters, folds, arr

def _batch_metrics(a):
    # a shape B x F x (accepted,error,trueaccepted,gold)
    sums = a.sum(axis=1); n = sums[:, 0]
    pooled = np.divide(sums[:, 1], n, out=np.full(len(a), np.nan), where=n > 0)
    recall = np.divide(sums[:, 2], sums[:, 3], out=np.full(len(a), np.nan), where=sums[:, 3] > 0)
    per = np.divide(a[:, :, 1], a[:, :, 0], out=np.full(a.shape[:2], np.nan), where=a[:, :, 0] > 0)
    valid = np.isfinite(per); k = valid.sum(axis=1)
    mean = np.divide(np.nansum(per, axis=1), k, out=np.zeros(len(a)), where=k > 0)
    variance = np.divide(np.nansum((per-mean[:, None])**2, axis=1), k,
                         out=np.full(len(a), np.nan), where=k > 0)
    worst = np.max(np.where(valid, per, -np.inf), axis=1); worst[k == 0] = np.nan
    return np.column_stack((pooled, worst, np.sqrt(variance), variance, recall, n))

def paired_cluster_statistics(rows, reference_rows=None, *, n_bootstrap=10000, seed=20261117):
    """Cluster bootstrap CIs + paired cluster method-swap randomization p.

    Entire diseases, across folds, are resampled together. For nonlinear FDP and
    worst-fold summaries, swaps recompute the full statistic (not query FDP means).
    Empty-denominator draws remain undefined and their count is reported.
    """
    clusters, folds, a = _arrays(rows)
    if not clusters or n_bootstrap < 1:
        raise ValueError("Nonempty rows and positive repetitions required")
    observed = _batch_metrics(a.sum(axis=0)[None, :])[0]
    b = None
    if reference_rows is not None:
        keys = lambda rs: {(r["fold"], r["qid"]): (r["tail"], r["pairs"], r["gold_total"],
                                                   r.get("candidate_sha256"), r.get("gold_sha256")) for r in rs}
        if len(keys(rows)) != len(rows) or keys(rows) != keys(reference_rows):
            raise ValueError("Paired statistics require identical query/fold/candidate/gold coverage")
        cb, fb, b = _arrays(reference_rows)
        if clusters != cb or folds != fb or not np.array_equal(a[:, :, 3], b[:, :, 3]):
            raise ValueError("Reference cluster/fold/gold mismatch")
    rng = np.random.default_rng(seed); boot = []; boot_delta = []; perm_delta = []
    for start in range(0, n_bootstrap, 128):
        n = min(128, n_bootstrap-start)
        weights = rng.multinomial(len(clusters), np.full(len(clusters), 1/len(clusters)), size=n)
        va = _batch_metrics(np.einsum("bc,cfk->bfk", weights, a)); boot.append(va)
        if b is not None:
            vb = _batch_metrics(np.einsum("bc,cfk->bfk", weights, b)); boot_delta.append(va-vb)
            swaps = rng.integers(0, 2, size=(n, len(clusters)))
            aa = np.einsum("bc,cfk->bfk", swaps, a) + np.einsum("bc,cfk->bfk", 1-swaps, b)
            bb = np.einsum("bc,cfk->bfk", swaps, b) + np.einsum("bc,cfk->bfk", 1-swaps, a)
            perm_delta.append(_batch_metrics(aa)-_batch_metrics(bb))
    bootstrap = np.concatenate(boot)
    reference = _batch_metrics(b.sum(axis=0)[None, :])[0] if b is not None else None
    delta = np.concatenate(boot_delta) if b is not None else None
    perm = np.concatenate(perm_delta) if b is not None else None
    def interval(values):
        finite = values[np.isfinite(values)]
        return {"ci_low": float(np.quantile(finite, .025)) if len(finite) else None,
                "ci_high": float(np.quantile(finite, .975)) if len(finite) else None,
                "defined_repetitions": int(len(finite)), "undefined_repetitions": int(len(values)-len(finite))}
    result = {"repetitions": n_bootstrap, "clusters": len(clusters), "seed": seed,
              "resampling_unit": "whole disease cluster across folds",
              "status": "degenerate_single_disease_cluster" if len(clusters) == 1 else "empirical_cluster_bootstrap",
              "metrics": {}}
    for i, metric in enumerate(STAT_METRICS):
        entry = {"estimate": float(observed[i]) if np.isfinite(observed[i]) else None,
                 **interval(bootstrap[:, i])}
        if b is not None:
            diff = observed[i]-reference[i]; null = perm[:, i]; finite = np.isfinite(null)
            entry.update({"reference_estimate": float(reference[i]) if np.isfinite(reference[i]) else None,
                          "delta": float(diff) if np.isfinite(diff) else None,
                          "delta_interval": interval(delta[:, i]),
                          "paired_p": float((1+np.sum(np.abs(null[finite]) >= abs(diff)-1e-14))/(1+finite.sum()))
                                      if np.isfinite(diff) and finite.any() else None,
                          "paired_p_defined_repetitions": int(finite.sum()),
                          "paired_test": "two-sided disease-cluster whole-method swap"})
        result["metrics"][metric] = entry
    return result
