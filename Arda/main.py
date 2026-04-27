import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import random

import MatchMaker as MatchMaker
import performance_metrics

from sklearn.model_selection import KFold

# Optional W&B
try:
    import wandb
    from wandb.integration.keras import WandbMetricsLogger
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# -------------------- Args --------------------
parser = argparse.ArgumentParser(description='MatchMaker training / evaluation')

parser.add_argument('--comb-data-name', default='data/DrugCombinationData.tsv')
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

parser.add_argument('--split-mode', default='random', choices=['files', 'random', 'kfold'])
parser.add_argument('--split', nargs=3, type=float, default=[0.8, 0.1, 0.1])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--kfolds', type=int, default=10)

parser.add_argument('--final-fit', action='store_true')
parser.add_argument('--use-trainval', action='store_true')
parser.add_argument('--tiny-val-frac', type=float, default=0.1)

parser.add_argument('--weight-mode', default='uniform', choices=['uniform', 'q3_upweight'])
parser.add_argument('--weight-alpha', type=float, default=3.0)

parser.add_argument('--outdir', default='results')

# Hyperparameters
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--input-dropout', type=float, default=0.2)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--max-epoch', type=int, default=1000)
parser.add_argument('--earlystop', type=int, default=100)

# W&B
parser.add_argument('--use-wandb', action='store_true')
parser.add_argument('--wandb-project', default='matchmaker-pdac')
parser.add_argument('--wandb-run-name', default=None)



parser.add_argument('--norm', default='minmax', choices=['minmax', 'tanh_norm', 'norm', 'tanh'])

args = parser.parse_args()

# -------------------- Reproducibility --------------------
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

os.makedirs(args.outdir, exist_ok=True)

# -------------------- Load data --------------------
print("Loading data...")
chem1, chem2, cell_line, synergies = MatchMaker.data_loader(
    args.drug1_chemicals, args.drug2_chemicals, args.cell_line_gex, args.comb_data_name
)

N = len(synergies)
all_idx = np.arange(N)

print(f"Total samples: {N}")
print(f"drug1_chem shape: {chem1.shape}")
print(f"drug2_chem shape: {chem2.shape}")
print(f"cell_line_gex shape: {cell_line.shape}")

# -------------------- Load architecture --------------------
architecture = pd.read_csv(args.arch)
layers = {
    'DSN_1': architecture['DSN_1'][0],
    'DSN_2': architecture['DSN_2'][0],
    'SPN': architecture['SPN'][0],
}

# -------------------- Hyperparams --------------------
l_rate = args.lr
inDrop = args.input_dropout
drop = args.dropout
max_epoch = args.max_epoch
batch_size = args.batch_size
earlyStop_patience = args.earlystop
norm = args.norm

# -------------------- W&B init --------------------
use_wandb = args.use_wandb and WANDB_AVAILABLE
if args.use_wandb and not WANDB_AVAILABLE:
    print("W&B requested but not installed. Continuing without W&B.")

if use_wandb:
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "seed": args.seed,
            "split_mode": args.split_mode,
            "split": args.split,
            "kfolds": args.kfolds,
            "lr": l_rate,
            "input_dropout": inDrop,
            "dropout": drop,
            "batch_size": batch_size,
            "max_epoch": max_epoch,
            "earlystop": earlyStop_patience,
            "arch_DSN_1": layers["DSN_1"],
            "arch_DSN_2": layers["DSN_2"],
            "arch_SPN": layers["SPN"],
            "comb_data_name": args.comb_data_name,
            "drug1_chemicals": args.drug1_chemicals,
            "drug2_chemicals": args.drug2_chemicals,
            "cell_line_gex": args.cell_line_gex,
            "n_samples": int(N),
        }
    )

def make_loss_weight(train_y: np.ndarray, mode="q3_upweight", q3=None, alpha=3.0):
    """
    mode:
        - "uniform": all weights = 1
        - "q3_upweight": emphasize samples above Q3
    q3:
        - if None, computed from training labels only
    alpha:
        - strength of upweighting for y > Q3
    """
    train_y = np.nan_to_num(train_y, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    if mode == "uniform":
        return np.ones_like(train_y, dtype=np.float32)

    if mode == "q3_upweight":
        if q3 is None:
            q3 = np.quantile(train_y, 0.75)

        max_y = np.max(train_y)
        denom = max(max_y - q3, 1e-8)

        weights = np.ones_like(train_y, dtype=np.float32)
        mask = train_y > q3

        # smoothly increase weights above Q3
        weights[mask] = 1.0 + alpha * ((train_y[mask] - q3) / denom)

        # numerical safety
        weights = np.nan_to_num(weights, nan=1.0, posinf=10.0, neginf=1.0)
        weights = np.clip(weights, 1.0, 1.0 + alpha)

        print(f"[Weighting] mode=q3_upweight, q3={q3:.6f}, alpha={alpha}")
        print(f"[Weighting] min={weights.min():.4f}, mean={weights.mean():.4f}, max={weights.max():.4f}")
        print(f"[Weighting] n_upweighted={(mask).sum()} / {len(train_y)}")

        return weights.astype(np.float32)

    raise ValueError(f"Unknown weighting mode: {mode}")

def evaluate_model(model, test_data, tag=""):
    pred1 = MatchMaker.predict(model, [test_data['drug1'], test_data['drug2']])
    pred2 = MatchMaker.predict(model, [test_data['drug2'], test_data['drug1']])
    pred = (pred1 + pred2) / 2

    mse_value = performance_metrics.mse(test_data['y'], pred)
    spearman_value = performance_metrics.spearman(test_data['y'], pred)
    pearson_value = performance_metrics.pearson(test_data['y'], pred)

    metrics = {
        "tag": tag,
        "mse": float(mse_value),
        "spearman": float(spearman_value[0] if isinstance(spearman_value, (list, tuple)) else spearman_value),
        "pearson": float(pearson_value[0] if isinstance(pearson_value, (list, tuple)) else pearson_value),
    }

    if use_wandb:
        wandb.log({
            f"{tag}/mse": metrics["mse"],
            f"{tag}/spearman": metrics["spearman"],
            f"{tag}/pearson": metrics["pearson"],
        })

    return metrics

def run_one_split(train_idx, val_idx, test_idx, run_tag):
    print(f"\n===== Running split: {run_tag} =====")
    print(f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    tmp_train = os.path.join(args.outdir, f"train_{run_tag}.txt")
    tmp_val   = os.path.join(args.outdir, f"val_{run_tag}.txt")
    tmp_test  = os.path.join(args.outdir, f"test_{run_tag}.txt")

    np.savetxt(tmp_train, train_idx, fmt="%d")
    np.savetxt(tmp_val,   val_idx,   fmt="%d")
    np.savetxt(tmp_test,  test_idx,  fmt="%d")

    train_data, val_data, test_data = MatchMaker.prepare_data(
        chem1, chem2, cell_line, synergies, norm, tmp_train, tmp_val, tmp_test
    )

    loss_weight = make_loss_weight(
    train_data['y'],
    mode=args.weight_mode,
    alpha=args.weight_alpha
)

    model = MatchMaker.generate_network(train_data, layers, inDrop, drop)

    model_path = os.path.join(args.outdir, f"{run_tag}_{args.saved_model_name}")

    if args.train_test_mode == 1:
        model = MatchMaker.trainer(
            model=model,
            l_rate=l_rate,
            train=train_data,
            val=val_data,
            epo=max_epoch,
            batch_size=batch_size,
            earlyStop=earlyStop_patience,
            modelName=model_path,
            weights=loss_weight,
            use_wandb=use_wandb
        )

    model.load_weights(model_path)
    metrics = evaluate_model(model, test_data, tag=run_tag)
    print("Run:", run_tag, "=>", metrics)
    return metrics

results = []

# -------------------- Split modes --------------------
if args.split_mode == "files":
    if args.use_trainval:
        train_idx = np.loadtxt(args.train_ind, dtype=int)
        val_idx   = np.loadtxt(args.val_ind, dtype=int)
        test_idx  = np.loadtxt(args.test_ind, dtype=int)

        trainval_idx = np.unique(np.concatenate([train_idx, val_idx]))
        rng = np.random.default_rng(args.seed)
        tv = trainval_idx.copy()
        rng.shuffle(tv)

        n_tiny = max(1, int(args.tiny_val_frac * len(tv)))
        val_idx = tv[:n_tiny]
        train_idx = tv[n_tiny:]
    else:
        train_idx = np.loadtxt(args.train_ind, dtype=int)
        val_idx   = np.loadtxt(args.val_ind, dtype=int)
        test_idx  = np.loadtxt(args.test_ind, dtype=int)

    results.append(run_one_split(train_idx, val_idx, test_idx, run_tag="files_split"))

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

elif args.split_mode == "kfold":
    kf = KFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)

    fold = 0
    for trainval_idx, test_idx in kf.split(all_idx):
        fold += 1
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

# -------------------- Save results --------------------
df_res = pd.DataFrame(results)
out_csv = os.path.join(args.outdir, "results.csv")
df_res.to_csv(out_csv, index=False)

print("\nSaved results to:", out_csv)
print(df_res)

if use_wandb:
    for _, row in df_res.iterrows():
        wandb.log({
            "final/mse": row["mse"],
            "final/spearman": row["spearman"],
            "final/pearson": row["pearson"],
        })
    wandb.finish()