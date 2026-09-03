import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

# =========================
# Paths
# =========================
BATCH1_DIR  = r"results\features_analysis"
BATCH2_DIR  = r"results\features_analysis\runs_6_15"
OUTPUT_DIR  = r"results\features_analysis\combined_15runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Settings
# =========================
MODELS      = ["LDA", "Decision_Tree", "Random_Forest", "XGBoost"]
EXPERIMENTS = ["C1", "C2"]
TOP_K       = 10
BATCH1_N    = 5
BATCH2_N    = 10
TOTAL_N     = BATCH1_N + BATCH2_N


def load_rankings(folder, exp, model):
    path = os.path.join(folder, f"shap_runs_{exp}_{model}.csv")
    if not os.path.exists(path):
        print(f"  WARNING: File not found — {path}")
        return None
    return pd.read_csv(path)


def load_mean_std(folder, exp, model):
    path = os.path.join(folder, f"shap_mean_std_{exp}_{model}.csv")
    if not os.path.exists(path):
        print(f"  WARNING: File not found — {path}")
        return None
    return pd.read_csv(path)


def get_top_features_from_rankings(df):
    rank_cols = [c for c in df.columns if c.startswith("Rank_")]
    all_features = []
    for _, row in df.iterrows():
        for col in rank_cols:
            val = row[col]
            if pd.notna(val):
                all_features.append(val)
    return all_features


def build_combined_stability(batch1_df, batch2_df, exp, model):
    all_features_b1 = get_top_features_from_rankings(batch1_df)
    all_features_b2 = get_top_features_from_rankings(batch2_df)

    counts_b1 = Counter(all_features_b1)
    counts_b2 = Counter(all_features_b2)
    counts_all = Counter(all_features_b1 + all_features_b2)

    all_feats = sorted(set(counts_all.keys()))

    rows = []
    for feat in all_feats:
        c1 = counts_b1.get(feat, 0)
        c2 = counts_b2.get(feat, 0)
        ct = counts_all.get(feat, 0)
        rows.append({
            "Feature":              feat,
            "Runs_1_5_Appearances":  c1,
            "Runs_6_15_Appearances": c2,
            "Total_15_Appearances":  ct,
            "Stability_Pct_15runs":  round(ct / TOTAL_N * 100, 1),
            "Stable_5runs":          c1 == BATCH1_N,
            "Stable_15runs":         ct == TOTAL_N,
        })

    df_out = (
        pd.DataFrame(rows)
        .sort_values("Total_15_Appearances", ascending=False)
        .reset_index(drop=True)
    )
    return df_out


def build_mean_std_combined(df1, df2):
    run_cols1 = [c for c in df1.columns if c.startswith("Run")]
    run_cols2 = [c for c in df2.columns if c.startswith("Run")]

    # Rename columns to avoid clashes
    rename1 = {c: f"B1_{c}" for c in run_cols1}
    rename2 = {c: f"B2_{c}" for c in run_cols2}

    d1 = df1[["Feature"] + run_cols1].rename(columns=rename1)
    d2 = df2[["Feature"] + run_cols2].rename(columns=rename2)

    merged = d1.merge(d2, on="Feature", how="outer")
    all_run_cols = [c for c in merged.columns if c.startswith("B1_") or c.startswith("B2_")]

    merged["Mean_SHAP_15runs"] = merged[all_run_cols].mean(axis=1)
    merged["Std_SHAP_15runs"]  = merged[all_run_cols].std(axis=1)
    merged = merged.sort_values("Mean_SHAP_15runs", ascending=False).reset_index(drop=True)
    merged["Mean_Rank_15runs"] = merged.index + 1

    return merged[["Feature", "Mean_SHAP_15runs", "Std_SHAP_15runs", "Mean_Rank_15runs"]]


def plot_stability_bar(stability_df, mean_std_df, exp, model, output_dir):
    top20_feats = mean_std_df.head(20)["Feature"].tolist()

    plot_df = mean_std_df[mean_std_df["Feature"].isin(top20_feats)].head(20)
    stab_map = dict(zip(stability_df["Feature"], stability_df["Total_15_Appearances"]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: mean SHAP ± std
    ax = axes[0]
    x = np.arange(len(plot_df))
    ax.barh(
        x, plot_df["Mean_SHAP_15runs"],
        xerr=plot_df["Std_SHAP_15runs"],
        align="center", color="steelblue", alpha=0.85,
        error_kw=dict(ecolor="black", capsize=3, linewidth=1)
    )
    ax.set_yticks(x)
    ax.set_yticklabels(plot_df["Feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| across 15 runs (± std)")
    ax.set_title(f"Mean SHAP Importance\n{exp} | {model} (15 runs)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Right: appearance count out of 15
    ax2 = axes[1]
    counts = [stab_map.get(f, 0) for f in plot_df["Feature"]]
    colors = ["seagreen" if c == TOTAL_N else "steelblue" if c >= 10 else "darkorange" for c in counts]
    ax2.barh(x, counts, align="center", color=colors, alpha=0.85)
    ax2.set_yticks(x)
    ax2.set_yticklabels(plot_df["Feature"], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel(f"Times in Top-{TOP_K} (out of {TOTAL_N} runs)")
    ax2.set_title(f"Feature Stability Count\n{exp} | {model} (15 runs)")
    ax2.axvline(TOTAL_N, color="red", linestyle="--", linewidth=1, label=f"Perfect ({TOTAL_N}/15)")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, TOTAL_N + 1)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    plt.suptitle(
        f"SHAP Stability Analysis — {exp} | {model} | 15 Runs",
        fontsize=14, y=1.01
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"shap_stability_15runs_{exp}_{model}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved chart: {out_path}")


def main():
    print("Stage — SHAP Stability Merge (15 Runs Combined)")
    print("=" * 60)

    all_combined_rows  = []
    comparison_rows    = []

    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp}")
        print(f"{'='*60}")

        for model in MODELS:
            print(f"\n  {exp} | {model}")

            b1_rank = load_rankings(BATCH1_DIR, exp, model)
            b2_rank = load_rankings(BATCH2_DIR, exp, model)
            b1_mean = load_mean_std(BATCH1_DIR, exp, model)
            b2_mean = load_mean_std(BATCH2_DIR, exp, model)

            if b1_rank is None or b2_rank is None:
                print(f"    Skipping — missing ranking file.")
                continue
            if b1_mean is None or b2_mean is None:
                print(f"    Skipping — missing mean/std file.")
                continue

            # Build combined stability
            stab_df = build_combined_stability(b1_rank, b2_rank, exp, model)
            stab_csv = os.path.join(OUTPUT_DIR, f"shap_stability_15runs_{exp}_{model}.csv")
            stab_df.to_csv(stab_csv, index=False)
            print(f"    Saved stability: {stab_csv}")

            # Build combined mean/std
            mean_std_df = build_mean_std_combined(b1_mean, b2_mean)
            mean_csv = os.path.join(OUTPUT_DIR, f"shap_mean_std_15runs_{exp}_{model}.csv")
            mean_std_df.to_csv(mean_csv, index=False)
            print(f"    Saved mean/std:  {mean_csv}")

            # Plot
            plot_stability_bar(stab_df, mean_std_df, exp, model, OUTPUT_DIR)

            # Summary rows
            always_stable_5  = stab_df[stab_df["Stable_5runs"]  == True]["Feature"].tolist()
            always_stable_15 = stab_df[stab_df["Stable_15runs"] == True]["Feature"].tolist()

            all_combined_rows.append({
                "Experiment":                exp,
                "Model":                     model,
                "Always_Top10_in_5runs":     len(always_stable_5),
                "Always_Top10_in_15runs":    len(always_stable_15),
                "Features_stable_5runs":     ", ".join(always_stable_5),
                "Features_stable_15runs":    ", ".join(always_stable_15),
            })

            # Comparison: 5 vs 15 runs per feature
            top10_5runs  = set(stab_df[stab_df["Stable_5runs"]  == True]["Feature"].tolist())
            top10_15runs = set(stab_df[stab_df["Stable_15runs"] == True]["Feature"].tolist())
            consistent   = top10_5runs & top10_15runs

            comparison_rows.append({
                "Experiment":                exp,
                "Model":                     model,
                "Stable_features_5runs":     len(top10_5runs),
                "Stable_features_15runs":    len(top10_15runs),
                "Consistent_both":           len(consistent),
                "Consistent_features":       ", ".join(sorted(consistent)),
            })

    # Save combined summary
    if all_combined_rows:
        combined_df = pd.DataFrame(all_combined_rows)
        combined_csv = os.path.join(OUTPUT_DIR, "shap_stability_combined_summary_15runs.csv")
        combined_df.to_csv(combined_csv, index=False)
        print(f"\n✅ Combined summary saved:\n{combined_csv}")
        print("\nCombined Summary:")
        print(combined_df[["Experiment", "Model",
                            "Always_Top10_in_5runs",
                            "Always_Top10_in_15runs"]].to_string(index=False))

    # Save 5 vs 15 comparison
    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        comp_csv = os.path.join(OUTPUT_DIR, "shap_stability_5vs15_comparison.csv")
        comp_df.to_csv(comp_csv, index=False)
        print(f"\n✅ 5 vs 15 comparison saved:\n{comp_csv}")
        print("\n5 vs 15 Runs Comparison:")
        print(comp_df[["Experiment", "Model",
                        "Stable_features_5runs",
                        "Stable_features_15runs",
                        "Consistent_both"]].to_string(index=False))

    print(f"\n✅ All outputs saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()