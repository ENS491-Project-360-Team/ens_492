import numpy as np
import sys

def normalize(X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm='tanh_norm'):
    X = np.asarray(X, dtype=np.float64)
    # Replace non-finite inputs early; downstream divisions otherwise create NaNs.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if std1 is None:
        std1 = np.nanstd(X, axis=0)
    std1 = np.nan_to_num(std1, nan=0.0, posinf=0.0, neginf=0.0)
    if feat_filt is None:
        feat_filt = std1 > 0
    X = X[:,feat_filt]
    X = np.ascontiguousarray(X)
    if means1 is None:
        means1 = np.mean(X, axis=0)
    denom1 = std1[feat_filt]
    denom1 = np.where(denom1 > 0, denom1, 1.0)
    X = (X-means1)/denom1
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if norm == 'norm':
        return(X, means1, std1, feat_filt)
    elif norm == 'tanh':
        return(np.tanh(X), means1, std1, feat_filt)
    elif norm == 'tanh_norm':
        X = np.tanh(X)
        if means2 is None:
            means2 = np.mean(X, axis=0)
        if std2 is None:
            std2 = np.std(X, axis=0)
        std2 = np.nan_to_num(std2, nan=0.0, posinf=0.0, neginf=0.0)
        denom2 = np.where(std2 > 0, std2, 1.0)
        X = (X-means2)/denom2
        X[:,std2==0]=0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return(X, means1, std1, means2, std2, feat_filt)
