from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import numpy as np


def pearson(y, pred):
    pear = stats.pearsonr(y, pred)
    pear_value = pear[0]
    pear_p_val = pear[1]
    print("Pearson correlation is {} and related p_value is {}".format(pear_value, pear_p_val))
    return pear_value

def spearman(y, pred):
    spear = stats.spearmanr(y, pred)
    spear_value = spear[0]
    spear_p_val = spear[1]
    print("Spearman correlation is {} and related p_value is {}".format(spear_value, spear_p_val))
    return spear_value

def mse(y, pred):
    err = mean_squared_error(y, pred)
    print("Mean squared error is {}".format(err))
    return err

def squared_error(y,pred):
    errs = []
    for i in range(y.shape[0]):
        err = (y[i]-pred[i]) * (y[i]-pred[i])
        errs.append(err)
    return np.asarray(errs)


def classification_metrics(y_true, y_score, threshold=0.0):
    """
    Compute threshold-based classification metrics from continuous predictions.
    y_true: binary labels in {0,1}
    y_score: continuous model outputs
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= float(threshold)).astype(int)

    auc = float("nan")
    auprc = float("nan")
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)

    f1 = f1_score(y_true, y_pred, zero_division=0)
    print("Classification AUC={} AUPRC={} F1={} at threshold={}".format(auc, auprc, f1, threshold))
    return {
        "auc": float(auc),
        "auprc": float(auprc),
        "f1": float(f1),
    }

