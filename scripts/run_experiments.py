import argparse
import os
import subprocess
import pandas as pd


def run_cmd(cmd, cwd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--splits-root", default="splits")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--out-root", default="results/runs")
    parser.add_argument("--gpu-devices", default="0")
    parser.add_argument("--train-test-mode", type=int, default=1)
    parser.add_argument("--classification-label-column", default="synergy_binary")
    parser.add_argument("--classification-threshold", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = {
        "unfiltered": os.path.join(args.processed_root, "pancreatic_unfiltered.tsv"),
        "filtered": os.path.join(args.processed_root, "pancreatic_disagreement_filtered.tsv"),
    }
    split_names = ["lto", "lpo", "lco", "lodo", "ldo"]
    rows = []

    for dname, dpath in datasets.items():
        for split_name in split_names:
            for seed in args.seeds:
                split_dir = os.path.join(args.splits_root, split_name, "seed_{}".format(seed))
                outdir = os.path.join(args.out_root, dname, split_name, "seed_{}".format(seed))
                os.makedirs(outdir, exist_ok=True)
                saved_model = os.path.join(outdir, "matchmaker.h5")

                cmd = [
                    "python3", "main.py",
                    "--comb-data-name", dpath,
                    "--label-column", "synergy_loewe",
                    "--classification-label-column", args.classification_label_column,
                    "--classification-threshold", str(args.classification_threshold),
                    "--train-ind", os.path.join(split_dir, "train_inds.txt"),
                    "--val-ind", os.path.join(split_dir, "val_inds.txt"),
                    "--test-ind", os.path.join(split_dir, "test_inds.txt"),
                    "--split-mode", "files",
                    "--saved-model-name", saved_model,
                    "--outdir", outdir,
                    "--gpu-devices", args.gpu_devices,
                    "--train-test-mode", str(args.train_test_mode),
                ]
                if args.dry_run:
                    print("[DRY RUN]", " ".join(cmd))
                    continue
                run_cmd(cmd, cwd=args.project_root)

                res_csv = os.path.join(outdir, "results.csv")
                df = pd.read_csv(res_csv)
                rec = df.iloc[0].to_dict()
                rec.update({"dataset": dname, "split": split_name, "seed": seed})
                rows.append(rec)

    if args.dry_run:
        print("Dry-run complete. No training executed.")
        return

    all_df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    all_path = os.path.join("results", "per_split_seed_metrics.csv")
    all_df.to_csv(all_path, index=False)
    numeric_cols = all_df.select_dtypes(include="number").columns.tolist()
    summary = all_df.groupby(["dataset", "split"], as_index=False)[numeric_cols].agg(["mean", "std"])
    summary.columns = ["{}_{}".format(a, b) if b else str(a) for a, b in summary.columns]
    summary_path = os.path.join("results", "summary_metrics.csv")
    summary.to_csv(summary_path, index=False)
    print("Wrote:", all_path)
    print("Wrote:", summary_path)


if __name__ == "__main__":
    main()
