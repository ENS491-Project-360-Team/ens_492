import numpy as np

def normalize(X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm='tanh_norm', mins=None, maxs=None):
    X = np.asarray(X, dtype=np.float32)

    if norm == 'minmax':
        # train-time fit
        if mins is None:
            mins = np.nanmin(X, axis=0)
        if maxs is None:
            maxs = np.nanmax(X, axis=0)

        # remove constant columns using train statistics only
        if feat_filt is None:
            feat_filt = (maxs - mins) != 0

        X = X[:, feat_filt]
        mins_f = mins[feat_filt]
        maxs_f = maxs[feat_filt]

        denom = maxs_f - mins_f
        denom[denom == 0] = 1.0

        X = (X - mins_f) / denom

        # numerical safety
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        X = np.clip(X, 0.0, 1.0)

        return X, mins, maxs, feat_filt

    # ===== existing old modes =====
    if std1 is None:
        std1 = np.nanstd(X, axis=0)
    if feat_filt is None:
        feat_filt = std1 != 0

    X = X[:, feat_filt]
    X = np.ascontiguousarray(X)

    if means1 is None:
        means1 = np.nanmean(X, axis=0)

    safe_std1 = std1[feat_filt].copy()
    safe_std1[safe_std1 == 0] = 1.0

    X = (X - means1) / safe_std1

    if norm == 'norm':
        return X, means1, std1, feat_filt

    elif norm == 'tanh':
        return np.tanh(X), means1, std1, feat_filt

    elif norm == 'tanh_norm':
        X = np.tanh(X)

        if means2 is None:
            means2 = np.nanmean(X, axis=0)
        if std2 is None:
            std2 = np.nanstd(X, axis=0)

        safe_std2 = std2.copy()
        safe_std2[safe_std2 == 0] = 1.0

        X = (X - means2) / safe_std2
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X[:, std2 == 0] = 0

        return X, means1, std1, means2, std2, feat_filt