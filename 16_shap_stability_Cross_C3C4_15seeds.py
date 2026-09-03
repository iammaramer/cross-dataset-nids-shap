import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# =========================
# Paths
# =========================
LYCOS17    = r"data\LycoS-IDS2017_FINAL.csv"
LYCOS18    = r"data\LycoS-Unicas-IDS2018_CLEANED.csv"
MODELS_DIR = r"models"
OUTPUT_DIR = r"results\features_analysis\C3_C4_combined_15runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Settings
# =========================
RANDOM_STATE = 42
SAMPLE_18    = 500_000
SHAP_SAMPLE  = 2000
TOP_K        = 10
N_RUNS       = 15
SEEDS        = [42, 123, 456, 789, 1000, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1234]

MODELS = ["LDA", "Decision_Tree", "Random_Forest", "XGBoost"]

# C3: model trained on C1 (LycoS17), tested on LycoS18
# C4: model trained on C2 (LycoS18), tested on LycoS17
EXPERIMENTS = {
    "C3": {"model_src": "C1", "test_data": "LycoS18"},
    "C4": {"model_src": "C2", "test_data": "LycoS17"},
}


def load_dataset(path, name, sample=None):
    print(f"  Loading {name}...")
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")

    if sample and len(df) > sample:
        parts = []
        total = len(df)
        for _, group in df.groupby("label"):
            frac = len(group) / total
            n = min(int(round(sample * frac)), len(group))
            if n > 0:
                parts.append(group.sample(n=n, random_state=RANDOM_STATE))
        df = (pd.concat(parts, ignore_index=True)
                .sample(frac=1, random_state=RANDOM_STATE)
                .reset_index(drop=True))

    print(f"  {name} shape: {df.shape}")
    return df


def get_test_split(df):
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    return scaler.transform(X_test), y_test


def load_model(model_src, model_name):
    fname = os.path.join(MODELS_DIR, f"{model_src}_{model_name}.pkl")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Model file not found: {fname}")
    with open(fname, "rb") as f:
        return pickle.load(f)


def compute_shap_importance(model, X_sample, feature_names, model_name):
    try:
        if model_name == "LDA":
            explainer = shap.LinearExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_arr = np.mean(
                np.stack([np.abs(sv) for sv in shap_values], axis=0), axis=0
            )
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_arr = np.abs(shap_values).mean(axis=2)
        else:
            shap_arr = np.abs(shap_values)

        mean_shap = shap_arr.mean(axis=0)

        importance_df = (
            pd.DataFrame({"Feature": feature_names, "MeanSHAP": mean_shap})
            .sort_values("MeanSHAP", ascending=False)
            .reset_index(drop=True)
        )
        importance_df["Rank"] = importance_df.index + 1
        return importance_df

    except Exception as e:
        print(f"    SHAP failed for {model_name}: {e}")
        return None


def run_all_15(exp, model_name, model_src, X_test, feature_names):
    print(f"\n  {exp} | {model_name} — running {N_RUNS} SHAP runs...")

    try:
        model = load_model(model_src, model_name)
    except FileNotFoundError as e:
        print(f"    Skipping: {e}")
        return None

    all_runs = []

    for i, seed in enumerate(SEEDS):
        np.random.seed(seed)
        idx = np.random.choice(len(X_test), min(SHAP_SAMPLE, len(X_test)), replace=False)
        X_sample = X_test[idx]

        imp_df = compute_shap_importance(model, X_sample, feature_names, model_name)
        if imp_df is None:
            continue

        top_k = imp_df.head(TOP_K)["Feature"].tolist()
        run_record = {"Run": i + 1, "Seed": seed}
        for rank, feat in enumerate(top_k, 1):
            run_record[f"Rank_{rank}"] = feat

        all_runs.append((run_record, imp_df))
        print(f"    Run {i+1} (seed={seed}) — Top 3: {top_k[:3]}")

    return all_runs if all_runs else None


def build_stability(all_runs):
    run_records = [r[0] for r in all_runs]
    imp_dfs     = [r[1] for r in all_runs]

    rankings_df = pd.DataFrame(run_records)

    # Count top-K appearances
    all_top_features = []
    for r in run_records:
        for k in range(1, TOP_K + 1):
            col = f"Rank_{k}"
            if col in r and pd.notna(r[col]):
                all_top_features.append(r[col])

    counts = Counter(all_top_features)
    stability_df = (
        pd.DataFrame(
            [{"Feature": f, "Top10_Appearances": c} for f, c in counts.items()]
        )
        .sort_values("Top10_Appearances", ascending=False)
        .reset_index(drop=True)
    )
    stability_df["Stability_Pct"] = (
        stability_df["Top10_Appearances"] / N_RUNS * 100
    ).round(1)
    stability_df["Stable_15runs"] = stability_df["Top10_Appearances"] == N_RUNS

    # Mean and std SHAP across all 15 runs
    merged = imp_dfs[0][["Feature", "MeanSHAP"]].rename(columns={"MeanSHAP": "Run1"})
    for i, df in enumerate(imp_dfs[1:], 2):
        merged = merged.merge(
            df[["Feature", "MeanSHAP"]].rename(columns={"MeanSHAP": f"Run{i}"}),
            on="Feature"
        )

    run_cols = [c for c in merged.columns if c.startswith("Run")]
    merged["Mean_SHAP"] = merged[run_cols].mean(axis=1)
    merged["Std_SHAP"]  = merged[run_cols].std(axis=1)
    merged = merged.sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)
    merged["Mean_Rank"] = merged.index + 1

    return rankings_df, stability_df, merged


def plot_stability_bar(merged_df, stability_df, exp, model_name, output_dir):
    top20    = merged_df.head(20)
    stab_map = dict(zip(stability_df["Feature"], stability_df["Top10_Appearances"]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: mean SHAP ± std
    ax = axes[0]
    x = np.arange(len(top20))
    ax.barh(
        x, top20["Mean_SHAP"],
        xerr=top20["Std_SHAP"],
        align="center", color="tomato", alpha=0.85,
        error_kw=dict(ecolor="black", capsize=3, linewidth=1)
    )
    ax.set_yticks(x)
    ax.set_yticklabels(top20["Feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"Mean |SHAP| across {N_RUNS} runs (± std)")
    ax.set_title(f"Mean SHAP Importance\n{exp} | {model_name} ({N_RUNS} runs)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Right: appearance count
    ax2 = axes[1]
    counts = [stab_map.get(f, 0) for f in top20["Feature"]]
    colors = [
        "seagreen"   if c == N_RUNS else
        "steelblue"  if c >= int(N_RUNS * 0.7) else
        "darkorange"
        for c in counts
    ]
    ax2.barh(x, counts, align="center", color=colors, alpha=0.85)
    ax2.set_yticks(x)
    ax2.set_yticklabels(top20["Feature"], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel(f"Times in Top-{TOP_K} (out of {N_RUNS} runs)")
    ax2.set_title(f"Feature Stability Count\n{exp} | {model_name} ({N_RUNS} runs)")
    ax2.axvline(N_RUNS, color="red", linestyle="--", linewidth=1,
                label=f"Perfect ({N_RUNS}/{N_RUNS})")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, N_RUNS + 1)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    plt.suptitle(
        f"SHAP Stability — {exp} | {model_name} | {N_RUNS} Runs (Cross-Dataset)",
        fontsize=13, y=1.01
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"shap_stability_15runs_{exp}_{model_name}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved chart: {out_path}")


def main():
    print("Stage — SHAP Stability: C3 & C4 (15 Runs Combined)")
    print("=" * 60)

    print("\nLoading datasets...")
    df17 = load_dataset(LYCOS17, "LycoS17")
    df18 = load_dataset(LYCOS18, "LycoS18", sample=SAMPLE_18)

    feature_names = [c for c in df17.columns if c != "label"]

    print("\nPreparing test splits...")
    X17_test, _ = get_test_split(df17)
    X18_test, _ = get_test_split(df18)

    test_data_map = {
        "LycoS17": X17_test,
        "LycoS18": X18_test,
    }

    all_summary_rows = []
    comparison_rows  = []

    for exp, cfg in EXPERIMENTS.items():
        model_src = cfg["model_src"]
        X_test    = test_data_map[cfg["test_data"]]

        print(f"\n{'='*60}")
        print(f"Experiment: {exp} — model from {model_src}, tested on {cfg['test_data']}")
        print(f"{'='*60}")

        for model_name in MODELS:
            all_runs = run_all_15(exp, model_name, model_src, X_test, feature_names)

            if all_runs is None:
                continue

            rankings_df, stability_df, merged_df = build_stability(all_runs)

            # Save rankings
            rankings_csv = os.path.join(OUTPUT_DIR, f"shap_runs_{exp}_{model_name}.csv")
            rankings_df.to_csv(rankings_csv, index=False)
            print(f"    Saved rankings:  {rankings_csv}")

            # Save stability
            stability_csv = os.path.join(OUTPUT_DIR, f"shap_stability_summary_{exp}_{model_name}.csv")
            stability_df.to_csv(stability_csv, index=False)
            print(f"    Saved stability: {stability_csv}")

            # Save mean/std
            merged_csv = os.path.join(OUTPUT_DIR, f"shap_mean_std_{exp}_{model_name}.csv")
            merged_df.to_csv(merged_csv, index=False)
            print(f"    Saved mean/std:  {merged_csv}")

            # Plot
            plot_stability_bar(merged_df, stability_df, exp, model_name, OUTPUT_DIR)

            # Summary
            always_15 = stability_df[stability_df["Stable_15runs"] == True]["Feature"].tolist()

            all_summary_rows.append({
                "Experiment":             exp,
                "Model":                  model_name,
                "Always_Top10_in_15runs": len(always_15),
                "Features_stable_15runs": ", ".join(always_15),
            })

            comparison_rows.append({
                "Experiment":      exp,
                "Model":           model_name,
                "Stable_15runs":   len(always_15),
                "Stable_features": ", ".join(always_15),
            })

    # Save combined summary
    if all_summary_rows:
        summary_df  = pd.DataFrame(all_summary_rows)
        summary_csv = os.path.join(OUTPUT_DIR, "shap_stability_combined_summary_15runs.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"\n✅ Combined summary saved:\n{summary_csv}")
        print("\nCombined Summary:")
        print(summary_df[["Experiment", "Model",
                           "Always_Top10_in_15runs"]].to_string(index=False))

    # Save comparison
    if comparison_rows:
        comp_df  = pd.DataFrame(comparison_rows)
        comp_csv = os.path.join(OUTPUT_DIR, "shap_stability_comparison_C3_C4.csv")
        comp_df.to_csv(comp_csv, index=False)
        print(f"\n✅ Comparison saved:\n{comp_csv}")
        print("\nStability Comparison C3 & C4:")
        print(comp_df[["Experiment", "Model", "Stable_15runs"]].to_string(index=False))

    print(f"\n✅ All outputs saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()