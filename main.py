import os
import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf

import MatchMaker as MatchMaker
import performance_metrics

from sklearn.model_selection import KFold

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _log(msg: str) -> None:
    """Timestamped, flushed log line for notebooks / non-TTY subprocess output."""
    from datetime import datetime

    print("[matchmaker %s] %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def impute_synergies_from_train(synergies_arr, train_idx):
    """Replace non-finite regression labels using the median of *training* finite values only (no test/val leakage)."""
    synergies_arr = np.asarray(synergies_arr, dtype=np.float64).copy()
    train_idx = np.asarray(train_idx, dtype=int).ravel()
    tr_vals = synergies_arr[train_idx]
    fin = tr_vals[np.isfinite(tr_vals)]
    fill = float(np.median(fin)) if fin.size else 0.0
    synergies_arr[~np.isfinite(synergies_arr)] = fill
    return synergies_arr

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

parser.add_argument('--split-mode', default='files', choices=['files', 'random', 'kfold'],
                    help="files: use *_inds.txt, random: make one split, kfold: cross-validation")
parser.add_argument('--split', nargs=3, type=float, default=[0.6, 0.2, 0.2],
                    help="train/val/test fractions for split-mode=random (e.g., 0.6 0.2 0.2)")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--kfolds', type=int, default=10)

parser.add_argument('--final-fit', action='store_true',
                    help="After choosing hyperparams, fit on train+val and evaluate on test (random mode)")

parser.add_argument('--use-trainval', action='store_true',
                    help="Train on train_inds + val_inds, keep val set for early stopping from a small subset of trainval.")
parser.add_argument('--tiny-val-frac', type=float, default=0.1,
                    help="Fraction of trainval to hold out as tiny validation when --use-trainval is set.")

parser.add_argument('--outdir', default='results')
parser.add_argument('--classification-label-column', default='',
                    help='Optional binary label column for classification metrics (e.g. synergy_binary)')
parser.add_argument('--classification-threshold', type=float, default=0.0,
                    help='Threshold applied to regression prediction for F1. AUC/AUPRC use score directly.')

parser.add_argument('--norm', default='minmax', choices=['minmax', 'tanh_norm', 'norm', 'tanh'],
                    help='Feature scaling: minmax (per-block) or legacy tanh_norm / norm / tanh.')
parser.add_argument('--weight-mode', default='uniform', choices=['uniform', 'q3_upweight', 'log'],
                    help='MSE sample weights: uniform, q3 upweight (Arda-style), or log (legacy).')
parser.add_argument('--weight-alpha', type=float, default=3.0,
                    help='Strength for q3_upweight (max weight ≈ 1 + alpha).')

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--input-dropout', type=float, default=0.2)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--max-epoch', type=int, default=1000)
parser.add_argument('--earlystop', type=int, default=100)

parser.add_argument('--use-wandb', action='store_true')
parser.add_argument('--wandb-project', default='matchmaker-pdac')
parser.add_argument('--wandb-run-name', default=None)

args = parser.parse_args()

_log("start main.py — comb_data=%r outdir=%r seed=%s" % (args.comb_data_name, args.outdir, args.seed))

os.environ["PYTHONHASHSEED"] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)


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

_log(
    "TensorFlow %s | CUDA_VISIBLE_DEVICES=%s | gpu_support=%s"
    % (tf.__version__, os.environ.get("CUDA_VISIBLE_DEVICES", ""), args.gpu_support)
)

os.makedirs(args.outdir, exist_ok=True)

# -------------------- Load data --------------------
_log("loading drug/cell features + combination table (I/O can take minutes on large files) …")
print("File reading ...")
chem1, chem2, cell_line, synergies = MatchMaker.data_loader(
    args.drug1_chemicals, args.drug2_chemicals, args.cell_line_gex, args.comb_data_name, args.label_column
)
synergies = np.asarray(synergies, dtype=np.float64)
# Missing / inf labels are imputed once per split using training-set median only (see split branches / run_one_split).

comb_df = pd.read_csv(args.comb_data_name, sep="\t")
_log(
    "arrays ready: synergies N=%d chem1=%s chem2=%s cell=%s"
    % (len(synergies), chem1.shape, chem2.shape, cell_line.shape)
)
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

_log("architecture from %r: DSN_1=%s DSN_2=%s SPN=%s" % (args.arch, layers["DSN_1"], layers["DSN_2"], layers["SPN"]))

l_rate = args.lr
inDrop = args.input_dropout
drop = args.dropout
max_epoch = args.max_epoch
batch_size = args.batch_size
earlyStop_patience = args.earlystop
norm = args.norm

use_wandb = args.use_wandb and WANDB_AVAILABLE
if args.use_wandb and not WANDB_AVAILABLE:
    print("W&B requested but not installed; continuing without W&B.")

if use_wandb:
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "seed": args.seed,
            "norm": norm,
            "split_mode": args.split_mode,
            "split": args.split,
            "kfolds": args.kfolds,
            "lr": l_rate,
            "weight_mode": args.weight_mode,
            "weight_alpha": args.weight_alpha,
            "input_dropout": inDrop,
            "dropout": drop,
            "batch_size": batch_size,
            "max_epoch": max_epoch,
            "earlystop": earlyStop_patience,
            "label_column": args.label_column,
            "comb_data_name": args.comb_data_name,
            "arch_DSN_1": layers["DSN_1"],
            "arch_DSN_2": layers["DSN_2"],
            "arch_SPN": layers["SPN"],
            "n_samples": int(N),
        },
    )


def make_loss_weight(train_y: np.ndarray, mode="uniform", alpha=3.0):
    train_y = np.asarray(train_y, dtype=np.float64).reshape(-1)

    if mode == "uniform":
        return np.ones_like(train_y, dtype=np.float64)

    if mode == "log":
        y = train_y.copy()
        if np.any(~np.isfinite(y)):
            y = np.nan_to_num(y, nan=np.nanmedian(y[np.isfinite(y)]) if np.any(np.isfinite(y)) else 0.0)
        min_s = float(np.min(y))
        w = np.log(np.maximum(y - min_s, 0.0) + np.e)
        w = np.where(np.isfinite(w), w, 1.0)
        return w

    if mode == "q3_upweight":
        y = np.nan_to_num(train_y, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        q3 = np.quantile(y, 0.75)
        max_y = np.max(y)
        denom = max(max_y - q3, 1e-8)
        weights = np.ones_like(y, dtype=np.float64)
        mask = y > q3
        weights[mask] = 1.0 + alpha * ((y[mask] - q3) / denom)
        weights = np.nan_to_num(weights, nan=1.0, posinf=10.0, neginf=1.0)
        weights = np.clip(weights, 1.0, 1.0 + alpha)
        print("[Weighting] mode=q3_upweight, q3={:.6f}, alpha={}".format(q3, alpha))
        print("[Weighting] min={:.4f}, mean={:.4f}, max={:.4f}".format(
            weights.min(), weights.mean(), weights.max()))
        return weights

    raise ValueError("Unknown weight-mode: {}".format(mode))

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
    synergies_imp = impute_synergies_from_train(synergies, train_idx)
    # NOTE: prepare_data takes filenames in your current MatchMaker.py.
    # We will write temporary index files per run/fold to avoid editing MatchMaker.py.
    tmp_train = os.path.join(args.outdir, f"train_{run_tag}.txt")
    tmp_val   = os.path.join(args.outdir, f"val_{run_tag}.txt")
    tmp_test  = os.path.join(args.outdir, f"test_{run_tag}.txt")

    np.savetxt(tmp_train, train_idx, fmt="%d")
    np.savetxt(tmp_val,   val_idx,   fmt="%d")
    np.savetxt(tmp_test,  test_idx,  fmt="%d")

    

    train_data, val_data, test_data = MatchMaker.prepare_data(
        chem1, chem2, cell_line, synergies_imp, norm, tmp_train, tmp_val, tmp_test
    )

    loss_weight = make_loss_weight(
        train_data['y'], mode=args.weight_mode, alpha=args.weight_alpha)

    model = MatchMaker.generate_network(train_data, layers, inDrop, drop)
    model_path = os.path.join(args.outdir, "{}_{}".format(run_tag, args.saved_model_name))

    if args.train_test_mode == 1:
        _log("training run_tag=%s (Keras fit) …" % run_tag)
        model = MatchMaker.trainer(
            model, l_rate, train_data, val_data,
            max_epoch, batch_size, earlyStop_patience,
            model_path, loss_weight, use_wandb)

    model.load_weights(model_path)

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
        rng = np.random.default_rng(args.seed)
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

        synergies_imp = impute_synergies_from_train(synergies, final_train_idx)
        train_data, val_data, test_data = MatchMaker.prepare_data(
            chem1, chem2, cell_line, synergies_imp, norm,
            tmp_train, tmp_val, tmp_test
        )
    else:
        _log("normalizing features (prepare_data from split index files) …")
        train_idx_files = np.loadtxt(args.train_ind, dtype=int)
        synergies_imp = impute_synergies_from_train(synergies, train_idx_files)
        train_data, val_data, test_data = MatchMaker.prepare_data(
            chem1, chem2, cell_line, synergies_imp, norm,
            args.train_ind, args.val_ind, args.test_ind
        )

    loss_weight = make_loss_weight(
        train_data['y'], mode=args.weight_mode, alpha=args.weight_alpha)
    _log("building Keras model …")
    model = MatchMaker.generate_network(train_data, layers, inDrop, drop)
    model_path = os.path.join(args.outdir, args.saved_model_name)

    if args.train_test_mode == 1:
        _log("training (Keras fit — first epoch may compile the graph / warm up GPU) …")
        model = MatchMaker.trainer(
            model, l_rate, train_data, val_data,
            max_epoch, batch_size, earlyStop_patience,
            model_path, loss_weight, use_wandb)

    model.load_weights(model_path)
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

if use_wandb:
    wandb.finish()