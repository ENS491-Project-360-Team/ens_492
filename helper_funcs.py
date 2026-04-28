import numpy as np


def normalize(
    X,
    means1=None,
    std1=None,
    means2=None,
    std2=None,
    feat_filt=None,
    norm='tanh_norm',
    mins=None,
    maxs=None,
):
    """Feature scaling. Supports minmax plus legacy tanh / tanh_norm / norm paths."""
    if norm == 'minmax':
        X = np.asarray(X, dtype=np.float32)
        if mins is None:
            mins = np.nanmin(X, axis=0)
        if maxs is None:
            maxs = np.nanmax(X, axis=0)
        if feat_filt is None:
            feat_filt = (maxs - mins) != 0

        X = X[:, feat_filt]
        mins_f = mins[feat_filt]
        maxs_f = maxs[feat_filt]

        denom = maxs_f - mins_f
        denom[denom == 0] = 1.0

        X = (X - mins_f) / denom
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        X = np.clip(X, 0.0, 1.0)

        return X, mins, maxs, feat_filt

    # Legacy tanh / tanh_norm / norm (float64, clean non-finite inputs first)
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if std1 is None:
        std1 = np.nanstd(X, axis=0)
    std1 = np.nan_to_num(std1, nan=0.0, posinf=0.0, neginf=0.0)
    if feat_filt is None:
        feat_filt = std1 > 0
    X = X[:, feat_filt]
    X = np.ascontiguousarray(X)
    if means1 is None:
        means1 = np.mean(X, axis=0)
    denom1 = std1[feat_filt]
    denom1 = np.where(denom1 > 0, denom1, 1.0)
    X = (X - means1) / denom1
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if norm == 'norm':
        return (X, means1, std1, feat_filt)
    elif norm == 'tanh':
        return (np.tanh(X), means1, std1, feat_filt)
    elif norm == 'tanh_norm':
        X = np.tanh(X)
        if means2 is None:
            means2 = np.mean(X, axis=0)
        if std2 is None:
            std2 = np.std(X, axis=0)
        std2 = np.nan_to_num(std2, nan=0.0, posinf=0.0, neginf=0.0)
        denom2 = np.where(std2 > 0, std2, 1.0)
        X = (X - means2) / denom2
        X[:, std2 == 0] = 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return (X, means1, std1, means2, std2, feat_filt)

    raise ValueError("Unknown norm mode: {}".format(norm))
