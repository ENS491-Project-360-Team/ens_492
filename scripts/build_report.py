import argparse
import os
import pandas as pd


def safe_pick(df, col_candidates):
    for c in col_candidates:
        if c in df.columns:
            return c
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", default="results/summary_metrics.csv")
    parser.add_argument("--per-seed-csv", default="results/per_split_seed_metrics.csv")
    parser.add_argument("--out-md", default="results/report.md")
    args = parser.parse_args()

    if not os.path.isfile(args.summary_csv) or not os.path.isfile(args.per_seed_csv):
        out_dir = os.path.dirname(args.out_md)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        msg = (
            "# Pancreatic MatchMaker report\n\n"
            "## Error\n\n"
            "Missing input CSV(s). Training likely failed or did not finish.\n\n"
            "- Expected: `{}`\n"
            "- Expected: `{}`\n\n"
            "Fix NaN labels / rerun training, then rebuild report.\n"
        ).format(args.summary_csv, args.per_seed_csv)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(msg)
        print("Wrote stub report:", args.out_md)
        print("Missing:", args.summary_csv, args.per_seed_csv)
        return

    summary = pd.read_csv(args.summary_csv)
    per_seed = pd.read_csv(args.per_seed_csv)

    mse_col = safe_pick(summary, ["mse_mean", "mse"])
    pear_col = safe_pick(summary, ["pearson_mean", "pearson"])
    f1_col = safe_pick(summary, ["f1_mean", "f1"])
    auc_col = safe_pick(summary, ["auc_mean", "auc"])
    auprc_col = safe_pick(summary, ["auprc_mean", "auprc"])

    lines = []
    lines.append("# Pancreatic MatchMaker report\n")
    lines.append("## Inputs\n")
    lines.append("- Summary metrics: `{}`".format(args.summary_csv))
    lines.append("- Per-seed metrics: `{}`".format(args.per_seed_csv))
    lines.append("")

    lines.append("## Best stable scenario\n")
    if mse_col is not None and pear_col is not None:
        subset = summary.sort_values([mse_col, pear_col], ascending=[True, False]).head(1)
        row = subset.iloc[0]
        lines.append(
            "- Regression winner: dataset=`{}`, split=`{}`, MSE=`{:.4f}`, Pearson=`{:.4f}`".format(
                row.get("dataset_"), row.get("split_"), row[mse_col], row[pear_col]
            )
        )
    if f1_col is not None:
        subset = summary.sort_values([f1_col], ascending=[False]).head(1)
        row = subset.iloc[0]
        lines.append(
            "- Classification winner: dataset=`{}`, split=`{}`, F1=`{:.4f}`".format(
                row.get("dataset_"), row.get("split_"), row[f1_col]
            )
        )
    if auc_col is not None and auprc_col is not None:
        subset = summary.sort_values([auc_col, auprc_col], ascending=[False, False]).head(1)
        row = subset.iloc[0]
        lines.append(
            "- Ranking winner (AUC/AUPRC): dataset=`{}`, split=`{}`, AUC=`{:.4f}`, AUPRC=`{:.4f}`".format(
                row.get("dataset_"), row.get("split_"), row[auc_col], row[auprc_col]
            )
        )
    lines.append("")

    lines.append("## Split difficulty trend\n")
    split_order = ["lto", "lpo", "lco", "lodo", "ldo"]
    if mse_col is not None:
        by_split = per_seed.groupby("split", as_index=False)["mse"].mean()
        by_split["order"] = by_split["split"].map({s: i for i, s in enumerate(split_order)})
        by_split = by_split.sort_values("order")
        lines.append("- Mean MSE by split:")
        for _, r in by_split.iterrows():
            lines.append("  - {}: {:.4f}".format(r["split"], r["mse"]))
    lines.append("")

    lines.append("## Disagreement filter impact\n")
    if "dataset" in per_seed.columns:
        for split in sorted(per_seed["split"].unique()):
            a = per_seed[(per_seed["split"] == split) & (per_seed["dataset"] == "unfiltered")]
            b = per_seed[(per_seed["split"] == split) & (per_seed["dataset"] == "filtered")]
            if len(a) == 0 or len(b) == 0:
                continue
            msg = "- `{}`: ".format(split)
            if "mse" in a.columns:
                msg += "MSE delta(filtered-unfiltered)={:.4f}; ".format(b["mse"].mean() - a["mse"].mean())
            if "f1" in a.columns:
                msg += "F1 delta(filtered-unfiltered)={:.4f}".format(b["f1"].mean() - a["f1"].mean())
            lines.append(msg)

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Wrote:", args.out_md)


if __name__ == "__main__":
    main()
