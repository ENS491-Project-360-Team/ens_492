import argparse
import json
import os
import numpy as np
import pandas as pd


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_split(base_dir, split_name, seed, train_idx, val_idx, test_idx):
    out = os.path.join(base_dir, split_name, "seed_{}".format(seed))
    ensure_dir(out)
    np.savetxt(os.path.join(out, "train_inds.txt"), np.asarray(train_idx, dtype=int), fmt="%d")
    np.savetxt(os.path.join(out, "val_inds.txt"), np.asarray(val_idx, dtype=int), fmt="%d")
    np.savetxt(os.path.join(out, "test_inds.txt"), np.asarray(test_idx, dtype=int), fmt="%d")
    return out


def split_train_val(rng, idx, val_frac=0.2):
    idx = np.asarray(idx, dtype=int).copy()
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_frac))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def random_lto(df, seed, test_frac=0.2, val_frac=0.2):
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(df))
    rng.shuffle(all_idx)
    n_test = max(1, int(len(all_idx) * test_frac))
    test_idx = all_idx[:n_test]
    rest = all_idx[n_test:]
    train_idx, val_idx = split_train_val(rng, rest, val_frac=val_frac)
    return train_idx, val_idx, test_idx


def lpo(df, seed, test_frac=0.2, val_frac=0.2):
    rng = np.random.default_rng(seed)
    pairs = df[["drug1_id", "drug2_id"]].astype(str).agg("|".join, axis=1).values
    uniq = np.unique(pairs)
    rng.shuffle(uniq)
    n_test_pairs = max(1, int(len(uniq) * test_frac))
    test_pairs = set(uniq[:n_test_pairs])
    test_mask = np.array([p in test_pairs for p in pairs], dtype=bool)
    test_idx = np.where(test_mask)[0]
    rest = np.where(~test_mask)[0]
    train_idx, val_idx = split_train_val(rng, rest, val_frac=val_frac)
    return train_idx, val_idx, test_idx


def lco(df, seed, test_frac=0.2, val_frac=0.2):
    rng = np.random.default_rng(seed)
    cells = df["cell_line"].astype(str).values
    uniq = np.unique(cells)
    rng.shuffle(uniq)
    n_test_cells = max(1, int(len(uniq) * test_frac))
    test_cells = set(uniq[:n_test_cells])
    test_mask = np.array([c in test_cells for c in cells], dtype=bool)
    test_idx = np.where(test_mask)[0]
    rest = np.where(~test_mask)[0]
    train_idx, val_idx = split_train_val(rng, rest, val_frac=val_frac)
    return train_idx, val_idx, test_idx


def lodo(df, seed, test_frac=0.2, val_frac=0.2):
    rng = np.random.default_rng(seed)
    d1 = df["drug1_id"].astype(str).values
    d2 = df["drug2_id"].astype(str).values
    drugs = np.unique(np.concatenate([d1, d2]))
    rng.shuffle(drugs)
    n_test_drugs = max(1, int(len(drugs) * test_frac))
    test_drugs = set(drugs[:n_test_drugs])
    # strict LODO: test has exactly one unseen drug; train has no unseen drugs
    in_a = np.array([a in test_drugs for a in d1], dtype=bool)
    in_b = np.array([b in test_drugs for b in d2], dtype=bool)
    test_mask = np.logical_xor(in_a, in_b)
    train_mask = np.logical_not(np.logical_or(in_a, in_b))
    test_idx = np.where(test_mask)[0]
    rest = np.where(train_mask)[0]
    train_idx, val_idx = split_train_val(rng, rest, val_frac=val_frac)
    return train_idx, val_idx, test_idx


def ldo(df, seed, test_frac=0.2, val_frac=0.2):
    rng = np.random.default_rng(seed)
    d1 = df["drug1_id"].astype(str).values
    d2 = df["drug2_id"].astype(str).values
    drugs = np.unique(np.concatenate([d1, d2]))
    rng.shuffle(drugs)
    n_test_drugs = max(1, int(len(drugs) * test_frac))
    test_drugs = set(drugs[:n_test_drugs])
    # strict LDO: both drugs unseen in test; train has no unseen drugs at all
    in_a = np.array([a in test_drugs for a in d1], dtype=bool)
    in_b = np.array([b in test_drugs for b in d2], dtype=bool)
    test_mask = np.logical_and(in_a, in_b)
    train_mask = np.logical_not(np.logical_or(in_a, in_b))
    test_idx = np.where(test_mask)[0]
    rest = np.where(train_mask)[0]
    train_idx, val_idx = split_train_val(rng, rest, val_frac=val_frac)
    return train_idx, val_idx, test_idx


def leakage_report(df, train_idx, test_idx):
    tr = df.iloc[train_idx].copy()
    te = df.iloc[test_idx].copy()
    tr_pairs = set(tr[["drug1_id", "drug2_id"]].astype(str).agg("|".join, axis=1).tolist())
    te_pairs = set(te[["drug1_id", "drug2_id"]].astype(str).agg("|".join, axis=1).tolist())
    tr_drugs = set(pd.concat([tr["drug1_id"].astype(str), tr["drug2_id"].astype(str)]).tolist())
    te_drugs = set(pd.concat([te["drug1_id"].astype(str), te["drug2_id"].astype(str)]).tolist())
    tr_cells = set(tr["cell_line"].astype(str).tolist())
    te_cells = set(te["cell_line"].astype(str).tolist())
    return {
        "pair_overlap": len(tr_pairs.intersection(te_pairs)),
        "drug_overlap": len(tr_drugs.intersection(te_drugs)),
        "cell_overlap": len(tr_cells.intersection(te_cells)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", default="data/processed/pancreatic_disagreement_filtered.tsv")
    parser.add_argument("--outdir", default="splits")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(args.input_tsv, sep="\t")
    methods = {
        "lto": random_lto,
        "lpo": lpo,
        "lco": lco,
        "lodo": lodo,
        "ldo": ldo,
    }

    all_reports = []
    for seed in args.seeds:
        for name, fn in methods.items():
            train_idx, val_idx, test_idx = fn(df, seed=seed, test_frac=args.test_frac, val_frac=args.val_frac)
            folder = save_split(args.outdir, name, seed, train_idx, val_idx, test_idx)
            rep = leakage_report(df, train_idx, test_idx)
            rep.update({"split": name, "seed": seed, "folder": folder})
            all_reports.append(rep)
            print("Saved", name, "seed", seed, "->", folder)

    report_path = os.path.join(args.outdir, "leakage_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)
    print("Wrote leakage report:", report_path)


if __name__ == "__main__":
    main()
