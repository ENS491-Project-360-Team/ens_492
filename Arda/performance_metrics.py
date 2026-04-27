from scipy import stats
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix

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


def classification_metrics(y_true, y_pred, threshold=0.0):
    y_true_bin = (y_true > threshold).astype(int)
    y_pred_bin = (y_pred > threshold).astype(int)

    acc = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)

    try:
        auc = roc_auc_score(y_true_bin, y_pred)
    except:
        auc = np.nan

    cm = confusion_matrix(y_true_bin, y_pred_bin)

    print(f"Threshold = {threshold}")
    print(f"Accuracy = {acc}")
    print(f"F1 = {f1}")
    print(f"Precision = {prec}")
    print(f"Recall = {rec}")
    print(f"AUC = {auc}")
    print("Confusion matrix:")
    print(cm)

    return {
        "threshold": threshold,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auc": auc
    }


def classification_metrics_filtered(y_true, y_pred, low_thr, high_thr):
    mask = (y_true < low_thr) | (y_true > high_thr)

    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    y_true_bin = (y_true_f > high_thr).astype(int)
    y_pred_bin = (y_pred_f > high_thr).astype(int)

    acc = accuracy_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)

    try:
        auc = roc_auc_score(y_true_bin, y_pred_f)
    except:
        auc = np.nan

    cm = confusion_matrix(y_true_bin, y_pred_bin)

    print(f"Filtered thresholds: low={low_thr}, high={high_thr}")
    print(f"N kept = {len(y_true_f)}")
    print(f"Accuracy = {acc}")
    print(f"F1 = {f1}")
    print(f"Precision = {prec}")
    print(f"Recall = {rec}")
    print(f"AUC = {auc}")
    print("Confusion matrix:")
    print(cm)

    return {
        "low_thr": low_thr,
        "high_thr": high_thr,
        "n_kept": len(y_true_f),
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auc": auc
    }

