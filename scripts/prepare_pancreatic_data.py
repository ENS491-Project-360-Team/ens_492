import argparse
import os

import pandas as pd


def canonical_pair(df):
    d1 = df["drug1_id"].astype(str)
    d2 = df["drug2_id"].astype(str)
    left = d1.where(d1 <= d2, d2)
    right = d2.where(d1 <= d2, d1)
    return left, right


def bliss_variance(bliss_series):
    """Sample variance across replicates (ddof=1); single replicate → 0.0 (same idea as notebook)."""
    vals = bliss_series.dropna()
    if len(vals) <= 1:
        return 0.0
    return float(vals.var(ddof=1))


def _normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def build_processed(raw_csv, out_dir, infer_synergy_binary_threshold=None):
    if not os.path.isfile(raw_csv):
        raise FileNotFoundError(
            (
                "Raw CSV not found: {!r}. If you cloned from Git: `data/` is gitignored, "
                "so upload your combination CSV to Colab under that path "
                "(e.g. Files sidebar → upload into the session `data/` folder)."
            ).format(raw_csv)
        )

    df = pd.read_csv(raw_csv, encoding="utf-8-sig")
    df = _normalize_columns(df)
    req = ["drug1_id", "drug2_id", "cell_line", "bliss"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: {}".format(missing))
    if "synergy_binary" not in df.columns:
        if infer_synergy_binary_threshold is None:
            raise ValueError(
                "Missing column 'synergy_binary'. Add it to the CSV, or pass "
                "--infer-synergy-binary-bliss-threshold (e.g. 0.0) to derive binary labels from `bliss`."
            )
        t = infer_synergy_binary_threshold
        df["synergy_binary"] = (df["bliss"].astype(float) > t).astype(int)

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
        bliss_mean = float(bliss_vals.mean()) if len(bliss_vals) > 0 else float("nan")
        b_var = bliss_variance(bliss_vals)
        source = str(group["source"].iloc[0]) if "source" in group.columns else "unknown"
        rows.append(
            {
                "drug1_id": key[0],
                "drug2_id": key[1],
                "cell_line": key[2],
                "bliss": bliss_mean,
                "bliss_variance": b_var,
                "synergy_binary": int(maj),
                "n_replicates": int(rep_n),
                "agreement_rate": float(agree),
                "disagreement_flag": int(dis),
                "source": source,
            }
        )

    agg = pd.DataFrame(rows)
    agg = agg.rename(columns={"bliss": "synergy_loewe"})

    med = agg["synergy_loewe"].median()
    if pd.isna(med):
        med = 0.0
    agg["synergy_loewe"] = agg["synergy_loewe"].fillna(med)

    # IQR rule on bliss replicate variance across triplets (meeting4_dataprocessing-style; no quartile class filter).
    vv = agg["bliss_variance"]
    var_q1 = float(vv.quantile(0.25))
    var_q3 = float(vv.quantile(0.75))
    var_iqr = var_q3 - var_q1
    variance_threshold = float(var_q3 + 1.5 * var_iqr)
    agg["high_variance_iqr"] = (vv > variance_threshold).astype(int)

    os.makedirs(out_dir, exist_ok=True)
    all_path = os.path.join(out_dir, "pancreatic_unfiltered.tsv")
    var_filt_path = os.path.join(out_dir, "pancreatic_variance_filtered.tsv")
    card_path = os.path.join(out_dir, "data_card.md")

    agg_var_ok = agg[agg["high_variance_iqr"] == 0].copy()

    agg.to_csv(all_path, sep="\t", index=False)
    agg_var_ok.to_csv(var_filt_path, sep="\t", index=False)

    total = len(agg)
    n_var_kept = len(agg_var_ok)
    n_high_var = int((agg["high_variance_iqr"] == 1).sum())
    dis = int((agg["disagreement_flag"] == 1).sum())
    rep = int((agg["n_replicates"] > 1).sum())

    with open(card_path, "w", encoding="utf-8") as f:
        f.write("# Pancreatic processed data card\n\n")
        f.write("- Raw input: `{}`\n".format(raw_csv))
        f.write("- Aggregation key: `(sorted(drug1_id, drug2_id), cell_line)`\n")
        f.write("- Rows (unfiltered): `{}`\n".format(total))
        f.write("- **Filtered file** `{}`: rows **not** flagged high Bliss variance (IQR rule).\n".format(os.path.basename(var_filt_path)))
        f.write("  - Bliss variance Q1={}, Q3={}, IQR={}\n".format(var_q1, var_q3, var_iqr))
        f.write("  - Threshold: Q3 + 1.5×IQR = **{}**\n".format(variance_threshold))
        f.write("  - Triplets dropped (high variance): `{}`\n".format(n_high_var))
        f.write("  - Rows after variance filter: `{}`\n".format(n_var_kept))
        f.write("- Triplets with binary synergy label disagreement: `{}` (stored in column `disagreement_flag`; not used to drop rows).\n".format(dis))
        f.write("- Triplets with replicate count > 1: `{}`\n".format(rep))
        f.write("- Regression target: `synergy_loewe` (mean bliss; missing filled with global median)\n")
        f.write("- Extra columns: `bliss_variance`, `high_variance_iqr` (0/1)\n")
        f.write("- Classification target: `synergy_binary` (majority vote)\n")

    print("Wrote:", all_path)
    print("Wrote:", var_filt_path)
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
    parser.add_argument(
        "--infer-synergy-binary-bliss-threshold",
        type=float,
        default=None,
        metavar="T",
        help="If synergy_binary column is absent, create it as (bliss > T). Omit to require synergy_binary.",
    )
    args = parser.parse_args()
    build_processed(
        args.raw_csv,
        args.out_dir,
        infer_synergy_binary_threshold=args.infer_synergy_binary_bliss_threshold,
    )


if __name__ == "__main__":
    main()
