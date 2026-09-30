"""Task-level metrics. Never bootstrap correlated candidates as independent targets."""
import numpy as np


def ca_lddt(pred, true, mask=None, cutoff=15.0):
    pred,true = np.asarray(pred,float),np.asarray(true,float)
    if pred.shape!=true.shape or pred.ndim!=2 or pred.shape[-1]!=3:
        raise ValueError('Expected matching [residues,3] coordinates')
    if not np.isfinite(pred).all(): return 0.0
    resolved = np.isfinite(true).all(-1)
    if mask is not None: resolved &= np.asarray(mask,bool)
    pd = np.linalg.norm(pred[:,None]-pred[None,:],axis=-1)
    td = np.linalg.norm(true[:,None]-true[None,:],axis=-1)
    pairs = (td<cutoff)&(~np.eye(len(true),dtype=bool))&resolved[:,None]&resolved[None,:]
    if not pairs.any(): raise ValueError('No evaluable reference neighbors')
    delta = np.abs(pd-td)
    score = sum((delta<t).astype(float) for t in (0.5,1,2,4))/4
    count = pairs.sum(-1)
    per_residue = (score*pairs).sum(-1)/np.maximum(count,1)
    return float(per_residue[count>0].mean())


def paired_summary(native, omni, seed=1729, replicates=10000, groups=None):
    native,omni = np.asarray(native,float),np.asarray(omni,float)
    if native.shape!=omni.shape or native.ndim!=1 or not len(native):
        raise ValueError('Require matched, nonempty per-target values')
    if not (np.isfinite(native).all() and np.isfinite(omni).all()):
        raise ValueError('Resolve/report failures before aggregation; no silent NaN exclusion')
    delta = omni-native
    rng = np.random.default_rng(seed)
    if groups is None: groups=list(range(len(delta)))
    if len(groups)!=len(delta): raise ValueError('One independent-cluster label per paired row required')
    unique=list(dict.fromkeys(groups))
    sums=np.array([delta[np.array([g==key for g in groups])].sum() for key in unique])
    sizes=np.array([sum(g==key for g in groups) for key in unique])
    choices=rng.integers(len(unique),size=(replicates,len(unique)))
    boot=sums[choices].sum(-1)/sizes[choices].sum(-1)
    lo,hi = np.quantile(boot,[0.025,0.975])
    return dict(n_targets=len(delta),n_independent_groups=len(unique),native_mean=float(native.mean()),omni_mean=float(omni.mean()),
                difference=float(delta.mean()),ci95=[float(lo),float(hi)],
                positive=bool(lo>0),degenerate_interval=bool(lo==hi))
