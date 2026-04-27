import argparse
import os
import pandas as pd


def canonical_pair(df):
    d1 = df["drug1_id"].astype(str)
    d2 = df["drug2_id"].astype(str)
    left = d1.where(d1 <= d2, d2)
    right = d2.where(d1 <= d2, d1)
    return left, right


def build_processed(raw_csv, out_dir):
    df = pd.read_csv(raw_csv)
    req = ["drug1_id", "drug2_id", "cell_line", "bliss", "synergy_binary"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: {}".format(missing))

    p1, p2 = canonical_pair(df)
    df["pair_id_1"] = p1
    df["pair_id_2"] = p2
    key_cols = ["pair_id_1", "pair_id_2", "cell_line"]

    rows = []
    for key, group in df.groupby(key_cols, dropna=False):
        rep_n = len(group)
        binary_vals = group["synergy_binary"].dropna().astype(int)
        if len(binary_vals) == 0:
            maj = 0
            agree = 0.0
            dis = 0
        else:
            c0 = int((binary_vals == 0).sum())
            c1 = int((binary_vals == 1).sum())
            maj = 1 if c1 >= c0 else 0
            agree = float(max(c0, c1) / (c0 + c1))
            dis = 1 if (c0 > 0 and c1 > 0) else 0
        bliss_vals = group["bliss"].dropna()
        # Empty bliss -> NaN; fixed after building full table (fillna) so row count matches features
        bliss_mean = float(bliss_vals.mean()) if len(bliss_vals) > 0 else float("nan")
        source = str(group["source"].iloc[0]) if "source" in group.columns else "unknown"
        rows.append(
            {
                "drug1_id": key[0],
                "drug2_id": key[1],
                "cell_line": key[2],
                "bliss": bliss_mean,
                "synergy_binary": int(maj),
                "n_replicates": int(rep_n),
                "agreement_rate": float(agree),
                "disagreement_flag": int(dis),
                "source": source,
            }
        )

    agg = pd.DataFrame(rows)
    agg = agg.rename(columns={"bliss": "synergy_loewe"})

    # Never train on NaN labels (would give loss: nan). Impute missing bliss with global median.
    med = agg["synergy_loewe"].median()
    if pd.isna(med):
        med = 0.0
    agg["synergy_loewe"] = agg["synergy_loewe"].fillna(med)

    os.makedirs(out_dir, exist_ok=True)
    all_path = os.path.join(out_dir, "pancreatic_unfiltered.tsv")
    filt_path = os.path.join(out_dir, "pancreatic_disagreement_filtered.tsv")
    card_path = os.path.join(out_dir, "data_card.md")

    agg.to_csv(all_path, sep="\t", index=False)
    agg[agg["disagreement_flag"] == 0].to_csv(filt_path, sep="\t", index=False)

    total = len(agg)
    filtered = int((agg["disagreement_flag"] == 0).sum())
    dis = int((agg["disagreement_flag"] == 1).sum())
    rep = int((agg["n_replicates"] > 1).sum())
    with open(card_path, "w", encoding="utf-8") as f:
        f.write("# Pancreatic processed data card\n\n")
        f.write("- Raw input: `{}`\n".format(raw_csv))
        f.write("- Aggregation key: `(sorted(drug1_id, drug2_id), cell_line)`\n")
        f.write("- Rows (unfiltered): `{}`\n".format(total))
        f.write("- Rows (disagreement filtered): `{}`\n".format(filtered))
        f.write("- Triplets with disagreement: `{}`\n".format(dis))
        f.write("- Triplets with replicate count > 1: `{}`\n".format(rep))
        f.write("- Regression target: `synergy_loewe` (mean bliss; missing filled with global median)\n")
        f.write("- Classification target: `synergy_binary` (majority vote)\n")

    print("Wrote:", all_path)
    print("Wrote:", filt_path)
    print("Wrote:", card_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-csv",
        default="data/synergy - comb - Combination data.csv",
        help="Raw pancreatic combination CSV",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help="Output directory for processed files",
    )
    args = parser.parse_args()
    build_processed(args.raw_csv, args.out_dir)


if __name__ == "__main__":
    main()
