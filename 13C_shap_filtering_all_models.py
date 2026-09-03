import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = r"D:\thesis_implementation\results"

RF_XGB_RESULTS = os.path.join(RESULTS_DIR, "shap_filtered_results.csv")
RF_XGB_COMP    = os.path.join(RESULTS_DIR, "shap_filtering_comparison.csv")

LDA_DT_RESULTS = os.path.join(RESULTS_DIR, "shap_filtered_LDA_DT_results.csv")
LDA_DT_COMP    = os.path.join(RESULTS_DIR, "shap_filtering_LDA_DT_comparison.csv")

ALL_RESULTS_CSV = os.path.join(RESULTS_DIR, "shap_filtered_results_all_models.csv")
ALL_COMP_CSV    = os.path.join(RESULTS_DIR, "shap_filtering_comparison_all_models.csv")
ALL_BAR_PNG     = os.path.join(RESULTS_DIR, "shap_filtering_all_models_mcc_bar.png")


def load_csv(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found:\n{path}")
    df = pd.read_csv(path)
    print(f"Loaded {name}: {df.shape}")
    return df


def normalize_results_columns(df):
    df = df.copy()

    if "Feature_Set" not in df.columns:
        df["Feature_Set"] = "SHAP_stable"

    preferred_order = [
        "Experiment", "Model", "Feature_Set",
        "Accuracy", "Precision", "Recall", "F1", "MCC"
    ]

    for col in preferred_order:
        if col not in df.columns:
            df[col] = np.nan

    extra_cols = [c for c in df.columns if c not in preferred_order]
    df = df[preferred_order + extra_cols]
    return df


def normalize_comparison_columns(df):
    df = df.copy()

    preferred_order = [
        "Experiment", "Model", "MCC_original", "MCC_filtered", "MCC_delta"
    ]

    for col in preferred_order:
        if col not in df.columns:
            df[col] = np.nan

    extra_cols = [c for c in df.columns if c not in preferred_order]
    df = df[preferred_order + extra_cols]
    return df


def merge_results():
    rf_xgb_res = load_csv(RF_XGB_RESULTS, "RF/XGB filtered results")
    lda_dt_res = load_csv(LDA_DT_RESULTS, "LDA/DT filtered results")

    rf_xgb_res = normalize_results_columns(rf_xgb_res)
    lda_dt_res = normalize_results_columns(lda_dt_res)

    all_results = pd.concat([rf_xgb_res, lda_dt_res], ignore_index=True)

    model_order = ["LDA", "Decision Tree", "Random Forest", "XGBoost"]
    exp_order = ["C3", "C4"]

    all_results["Model"] = pd.Categorical(all_results["Model"], categories=model_order, ordered=True)
    all_results["Experiment"] = pd.Categorical(all_results["Experiment"], categories=exp_order, ordered=True)

    all_results = all_results.sort_values(["Experiment", "Model", "Feature_Set"]).reset_index(drop=True)
    all_results.to_csv(ALL_RESULTS_CSV, index=False)

    print(f"\nSaved combined filtered results:\n{ALL_RESULTS_CSV}")
    return all_results


def merge_comparison():
    rf_xgb_comp = load_csv(RF_XGB_COMP, "RF/XGB comparison")
    lda_dt_comp = load_csv(LDA_DT_COMP, "LDA/DT comparison")

    rf_xgb_comp = normalize_comparison_columns(rf_xgb_comp)
    lda_dt_comp = normalize_comparison_columns(lda_dt_comp)

    all_comp = pd.concat([rf_xgb_comp, lda_dt_comp], ignore_index=True)

    model_order = ["LDA", "Decision Tree", "Random Forest", "XGBoost"]
    exp_order = ["C3", "C4"]

    all_comp["Model"] = pd.Categorical(all_comp["Model"], categories=model_order, ordered=True)
    all_comp["Experiment"] = pd.Categorical(all_comp["Experiment"], categories=exp_order, ordered=True)

    all_comp = all_comp.sort_values(["Experiment", "Model"]).reset_index(drop=True)
    all_comp.to_csv(ALL_COMP_CSV, index=False)

    print(f"\nSaved combined comparison:\n{ALL_COMP_CSV}")
    print("\nCombined comparison table:")
    print(all_comp.to_string(index=False))

    return all_comp


def plot_combined_bar_chart(all_comp):
    model_order = ["LDA", "Decision Tree", "Random Forest", "XGBoost"]
    exp_order = ["C3", "C4"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    for ax, exp in zip(axes, exp_order):
        sub = all_comp[all_comp["Experiment"] == exp].copy()
        sub["Model"] = pd.Categorical(sub["Model"], categories=model_order, ordered=True)
        sub = sub.sort_values("Model")

        x = np.arange(len(sub))
        width = 0.36

        bars1 = ax.bar(
            x - width / 2,
            sub["MCC_original"],
            width,
            label="Original tuned",
            color="steelblue"
        )
        bars2 = ax.bar(
            x + width / 2,
            sub["MCC_filtered"],
            width,
            label="SHAP-stable",
            color="seagreen"
        )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["Model"], rotation=20)
        ax.set_ylabel("MCC")
        ax.set_title(f"{exp}: Original vs SHAP-stable")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            x_pos = bar.get_x() + bar.get_width() / 2

            if h >= 0:
                ax.text(x_pos, h + 0.012, f"{h:.3f}", ha="center", va="bottom", fontsize=9, rotation=90)
            else:
                ax.text(x_pos, h + 0.012, f"{h:.3f}", ha="center", va="bottom", fontsize=9, rotation=90)

    fig.suptitle(
        "SHAP-guided Feature Filtering: MCC Comparison Across All Models",
        fontsize=18,
        y=0.98
    )

    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False,
        fontsize=11
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(ALL_BAR_PNG, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved combined bar chart:\n{ALL_BAR_PNG}")


def main():
    print("=== Stage 6C: Merge SHAP-filtering outputs for all models ===\n")
    print("Results directory:", RESULTS_DIR)

    merge_results()
    all_comp = merge_comparison()
    plot_combined_bar_chart(all_comp)

    print("\n=== Done ===")
    print("Files created:")
    print("1.", ALL_RESULTS_CSV)
    print("2.", ALL_COMP_CSV)
    print("3.", ALL_BAR_PNG)


if __name__ == "__main__":
    main()