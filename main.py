import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

import MatchMaker as MatchMaker
import performance_metrics

from sklearn.model_selection import KFold

# -------------------- Args --------------------
parser = argparse.ArgumentParser(description='MatchMaker training / evaluation')

parser.add_argument('--comb-data-name', default='data/DrugCombinationData.tsv')
parser.add_argument('--label-column', default='synergy_loewe',
                    help='Column from comb-data-name to use as regression target')
parser.add_argument('--cell_line-gex', default='data/cell_line_gex.csv')
parser.add_argument('--drug1-chemicals', default='data/drug1_chem.csv')
parser.add_argument('--drug2-chemicals', default='data/drug2_chem.csv')

parser.add_argument('--gpu-devices', default='0', type=str)
parser.add_argument('--gpu-support', default=True, type=bool)

parser.add_argument('--train-test-mode', default=1, type=int, help="0: test only, 1: train+test")

parser.add_argument('--train-ind', default='data/train_inds.txt')
parser.add_argument('--val-ind', default='data/val_inds.txt')
parser.add_argument('--test-ind', default='data/test_inds.txt')

parser.add_argument('--arch', default='architecture.txt')
parser.add_argument('--saved-model-name', default="matchmaker.h5")

# NEW: splitting controls
parser.add_argument('--split-mode', default='files', choices=['files', 'random', 'kfold'],
                    help="files: use *_inds.txt, random: make one split, kfold: cross-validation")
parser.add_argument('--split', nargs=3, type=float, default=[0.6, 0.2, 0.2],
                    help="train/val/test fractions for split-mode=random (e.g., 0.6 0.2 0.2)")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--kfolds', type=int, default=10)

# NEW: optional final fit (train on train+val) for random split
parser.add_argument('--final-fit', action='store_true',
                    help="After choosing hyperparams, fit on train+val and evaluate on test (random mode)")

parser.add_argument('--use-trainval', action='store_true',
                    help="Train on train_inds + val_inds, keep val set for early stopping from a small subset of trainval.")
parser.add_argument('--tiny-val-frac', type=float, default=0.1,
                    help="Fraction of trainval to hold out as tiny validation when --use-trainval is set.")

# NEW: where to write results
parser.add_argument('--outdir', default='results')
parser.add_argument('--classification-label-column', default='',
                    help='Optional binary label column for classification metrics (e.g. synergy_binary)')
parser.add_argument('--classification-threshold', type=float, default=0.0,
                    help='Threshold applied to regression prediction for F1. AUC/AUPRC use score directly.')

args = parser.parse_args()


# -------------------- TF/GPU config --------------------
num_cores = 8
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices

if args.gpu_support:
    num_GPU = 1
    num_CPU = 1
else:
    num_CPU = 2
    num_GPU = 0

tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=num_cores,
    inter_op_parallelism_threads=num_cores,
    allow_soft_placement=True,
    device_count={'CPU': num_CPU, 'GPU': num_GPU}
)

os.makedirs(args.outdir, exist_ok=True)

# -------------------- Load data --------------------
print("File reading ...")
chem1, chem2, cell_line, synergies = MatchMaker.data_loader(
    args.drug1_chemicals, args.drug2_chemicals, args.cell_line_gex, args.comb_data_name, args.label_column
)
synergies = np.asarray(synergies, dtype=np.float64)
if np.any(~np.isfinite(synergies)):
    med = float(np.nanmedian(synergies[np.isfinite(synergies)])) if np.any(np.isfinite(synergies)) else 0.0
    synergies = np.nan_to_num(synergies, nan=med, posinf=med, neginf=med)

comb_df = pd.read_csv(args.comb_data_name, sep="\t")
class_labels = None
if args.classification_label_column:
    if args.classification_label_column not in comb_df.columns:
        raise ValueError("classification-label-column '{}' not found in {}".format(
            args.classification_label_column, args.comb_data_name
        ))
    class_labels = np.array(comb_df[args.classification_label_column])

N = len(synergies)
all_idx = np.arange(N)

# -------------------- Load architecture --------------------
architecture = pd.read_csv(args.arch)
layers = {
    'DSN_1': architecture['DSN_1'][0],
    'DSN_2': architecture['DSN_2'][0],
    'SPN': architecture['SPN'][0],
}

# -------------------- Constant hyperparams (you can tune later) --------------------
l_rate = 0.0001
inDrop = 0.2
drop = 0.5
max_epoch = 1000
batch_size = 128
earlyStop_patience = 100

norm = 'tanh_norm'

def make_loss_weight(train_y: np.ndarray) -> np.ndarray:
    """Sample weights for MSE; must be finite (NaN here -> nan loss)."""
    y = np.asarray(train_y, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(y)):
        y = np.nan_to_num(y, nan=np.nanmedian(y[np.isfinite(y)]) if np.any(np.isfinite(y)) else 0.0)
    min_s = float(np.min(y))
    w = np.log(np.maximum(y - min_s, 0.0) + np.e)
    w = np.where(np.isfinite(w), w, 1.0)
    return w

def evaluate_model(model, test_data, test_idx=None, tag=""):
    # Predict in Drug1, Drug2 order
    pred1 = MatchMaker.predict(model, [test_data['drug1'], test_data['drug2']])
    # Predict in Drug2, Drug1 order
    pred2 = MatchMaker.predict(model, [test_data['drug2'], test_data['drug1']])
    pred = (pred1 + pred2) / 2

    mse_value = performance_metrics.mse(test_data['y'], pred)
    spearman_value = performance_metrics.spearman(test_data['y'], pred)
    pearson_value = performance_metrics.pearson(test_data['y'], pred)

    out = {
        "tag": tag,
        "mse": float(mse_value),
        "spearman": float(spearman_value[0] if isinstance(spearman_value, (list, tuple)) else spearman_value),
        "pearson": float(pearson_value[0] if isinstance(pearson_value, (list, tuple)) else pearson_value),
    }
    if class_labels is not None and test_idx is not None:
        y_cls = class_labels[np.asarray(test_idx, dtype=int)]
        cls = performance_metrics.classification_metrics(y_cls, pred, threshold=args.classification_threshold)
        out["auc"] = cls["auc"]
        out["auprc"] = cls["auprc"]
        out["f1"] = cls["f1"]
    return out

def run_one_split(train_idx, val_idx, test_idx, run_tag):
    print("Data normalization and preparation of train/validation/test data")
    # NOTE: prepare_data takes filenames in your current MatchMaker.py.
    # We will write temporary index files per run/fold to avoid editing MatchMaker.py.
    tmp_train = os.path.join(args.outdir, f"train_{run_tag}.txt")
    tmp_val   = os.path.join(args.outdir, f"val_{run_tag}.txt")
    tmp_test  = os.path.join(args.outdir, f"test_{run_tag}.txt")

    np.savetxt(tmp_train, train_idx, fmt="%d")
    np.savetxt(tmp_val,   val_idx,   fmt="%d")
    np.savetxt(tmp_test,  test_idx,  fmt="%d")

    

    train_data, val_data, test_data = MatchMaker.prepare_data(
        chem1, chem2, cell_line, synergies, norm, tmp_train, tmp_val, tmp_test
    )

    loss_weight = make_loss_weight(train_data['y'])

    model = MatchMaker.generate_network(train_data, layers, inDrop, drop)

    if args.train_test_mode == 1:
        model = MatchMaker.trainer(
            model, l_rate, train_data, val_data,
            max_epoch, batch_size, earlyStop_patience,
            args.saved_model_name, loss_weight
        )

    # Always load best weights before evaluation
    model.load_weights(args.saved_model_name)

    metrics = evaluate_model(model, test_data, test_idx=test_idx, tag=run_tag)
    pred_path = os.path.join(args.outdir, "pred_{}.csv".format(run_tag))
    pd.DataFrame({
        "idx": np.asarray(test_idx, dtype=int),
        "y_true": test_data["y"],
        "y_pred": (MatchMaker.predict(model, [test_data['drug1'], test_data['drug2']]) + MatchMaker.predict(model, [test_data['drug2'], test_data['drug1']])) / 2,
    }).to_csv(pred_path, index=False)
    print("Run:", run_tag, "=>", metrics)
    return metrics

results = []

# -------------------- Split modes --------------------
if args.split_mode == "files":
    # Use existing files exactly (your current behavior)


    if args.use_trainval:
        train_idx = np.loadtxt(args.train_ind, dtype=int)
        val_idx   = np.loadtxt(args.val_ind, dtype=int)
        test_idx  = np.loadtxt(args.test_ind, dtype=int)

        trainval_idx = np.unique(np.concatenate([train_idx, val_idx]))

        # create a tiny validation split from trainval for early stopping
        rng = np.random.default_rng(42)
        tv = trainval_idx.copy()
        rng.shuffle(tv)

        n_tiny = max(1, int(args.tiny_val_frac * len(tv)))
        tiny_val_idx = tv[:n_tiny]
        final_train_idx = tv[n_tiny:]

        # write temp index files (because prepare_data expects filenames)
        tmp_train = os.path.join(args.outdir, "train_final.txt")
        tmp_val   = os.path.join(args.outdir, "val_tiny.txt")
        tmp_test  = os.path.join(args.outdir, "test_final.txt")

        np.savetxt(tmp_train, final_train_idx, fmt="%d")
        np.savetxt(tmp_val,   tiny_val_idx,    fmt="%d")
        np.savetxt(tmp_test,  test_idx,        fmt="%d")

        train_data, val_data, test_data = MatchMaker.prepare_data(
            chem1, chem2, cell_line, synergies, norm,
            tmp_train, tmp_val, tmp_test
        )
    else:
        train_data, val_data, test_data = MatchMaker.prepare_data(
            chem1, chem2, cell_line, synergies, norm,
            args.train_ind, args.val_ind, args.test_ind
        )

    loss_weight = make_loss_weight(train_data['y'])
    model = MatchMaker.generate_network(train_data, layers, inDrop, drop)

    if args.train_test_mode == 1:
        model = MatchMaker.trainer(
            model, l_rate, train_data, val_data,
            max_epoch, batch_size, earlyStop_patience,
            args.saved_model_name, loss_weight
        )

    model.load_weights(args.saved_model_name)
    test_idx = np.loadtxt(args.test_ind, dtype=int)
    metrics = evaluate_model(model, test_data, test_idx=test_idx, tag="files_split")
    results.append(metrics)

elif args.split_mode == "random":
    tr, va, te = args.split
    if not np.isclose(tr + va + te, 1.0):
        raise ValueError("--split must sum to 1.0, got: " + str(args.split))

    rng = np.random.default_rng(args.seed)
    idx = all_idx.copy()
    rng.shuffle(idx)

    n_tr = int(tr * N)
    n_va = int(va * N)

    train_idx = idx[:n_tr]
    val_idx   = idx[n_tr:n_tr + n_va]
    test_idx  = idx[n_tr + n_va:]

    results.append(run_one_split(train_idx, val_idx, test_idx, run_tag=f"random_seed{args.seed}"))

    # Optional: final fit on train+val (after you decide hyperparams)
    if args.final_fit:
        trainval_idx = np.concatenate([train_idx, val_idx])
        # keep a tiny val set for early stopping OR reuse val (here: reuse val as val, but train on train+val is not possible)
        # simplest: set val to train_idx (not ideal) OR do no-early-stopping.
        # We'll do: val_idx = test_idx[:max(1, len(test_idx)//10)] as a tiny validation subset.
        tiny_val = test_idx[:max(1, len(test_idx)//10)]
        results.append(run_one_split(trainval_idx, tiny_val, test_idx, run_tag=f"finalfit_seed{args.seed}"))

elif args.split_mode == "kfold":
    kf = KFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)

    fold = 0
    for trainval_idx, test_idx in kf.split(all_idx):
        fold += 1

        # Split trainval into train/val (20% of trainval as val)
        rng = np.random.default_rng(args.seed + fold)
        tv = trainval_idx.copy()
        rng.shuffle(tv)

        n_val = max(1, int(0.2 * len(tv)))
        val_idx = tv[:n_val]
        train_idx = tv[n_val:]

        run_tag = f"kfold{args.kfolds}_fold{fold}"
        results.append(run_one_split(train_idx, val_idx, test_idx, run_tag=run_tag))

else:
    raise ValueError("Unknown split mode.")

# -------------------- Save results distribution --------------------
df_res = pd.DataFrame(results)
out_csv = os.path.join(args.outdir, "results.csv")
df_res.to_csv(out_csv, index=False)
print("Saved results to:", out_csv)
print(df_res)