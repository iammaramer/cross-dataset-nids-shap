

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

RESULTS_DIR = r"D:\thesis_implementation\results\shap"

spearman_files = {
    "LDA": "shap_spearman_LDA_C1C4.csv",
    "Decision Tree": "shap_spearman_Decision_Tree_C1C4.csv",
    "Random Forest": "shap_spearman_RF_C1C4.csv",
    "XGBoost": "shap_spearman_XGB_C1C4.csv",
}

jaccard_files = {
    "LDA": "shap_jaccard_LDA_C1C4.csv",
    "Decision Tree": "shap_jaccard_Decision_Tree_C1C4.csv",
    "Random Forest": "shap_jaccard_RF_C1C4.csv",
    "XGBoost": "shap_jaccard_XGB_C1C4.csv",
}

exp_labels = ["C1", "C2", "C3", "C4"]


def load_spearman_matrix(csv_name):
    path = os.path.join(RESULTS_DIR, csv_name)
    df = pd.read_csv(path, index_col=0)
    df = df.loc[[f"Rank_{c}" for c in exp_labels], [f"Rank_{c}" for c in exp_labels]]
    return df


def load_jaccard_k20(csv_name):
    path = os.path.join(RESULTS_DIR, csv_name)
    df = pd.read_csv(path)
    sub = df[df["K"] == 20].copy()

    c1c3 = sub[(sub["Exp_A"] == "C1") & (sub["Exp_B"] == "C3")]["Jaccard"].iloc[0]
    c2c4 = sub[(sub["Exp_A"] == "C2") & (sub["Exp_B"] == "C4")]["Jaccard"].iloc[0]
    return c1c3, c2c4


def highlight_key_cells(ax):
    cells = [(0, 2), (2, 0), (1, 3), (3, 1)]
    for r, c in cells:
        rect = Rectangle(
            (c - 0.5, r - 0.5), 1, 1,
            fill=False, edgecolor="black", linewidth=1.8
        )
        ax.add_patch(rect)


def plot_combined_spearman():
    fig = plt.figure(figsize=(12, 8.6))
    gs = GridSpec(
        2, 3,
        width_ratios=[1, 1, 0.06],
        height_ratios=[1, 1],
        wspace=0.28,
        hspace=0.20,
        figure=fig
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    cax = fig.add_subplot(gs[:, 2])

    cmap = "coolwarm"
    vmin, vmax = 0.5, 1.0
    im = None

    for ax, (model_name, csv_name) in zip(axes, spearman_files.items()):
        sp = load_spearman_matrix(csv_name)
        vals = sp.values.astype(float)

        im = ax.imshow(vals, vmin=vmin, vmax=vmax, cmap=cmap)

        ax.set_xticks(np.arange(len(exp_labels)))
        ax.set_yticks(np.arange(len(exp_labels)))
        ax.set_xticklabels(exp_labels, fontsize=10)
        ax.set_yticklabels(exp_labels, fontsize=10)
        ax.set_title(model_name, fontsize=13, fontweight="bold", pad=8)

        for i in range(len(exp_labels)):
            for j in range(len(exp_labels)):
                ax.text(
                    j, i, f"{vals[i, j]:.2f}",
                    ha="center", va="center",
                    fontsize=8.5, color="black"
                )

        highlight_key_cells(ax)

        ax.set_xticks(np.arange(-0.5, len(exp_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(exp_labels), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Spearman ρ", fontsize=11, rotation=90, labelpad=10)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        "Combined SHAP Stability Heatmaps Across Models",
        fontsize=15,
        fontweight="bold",
        y=0.97
    )

    out_path = os.path.join(RESULTS_DIR, "combined_spearman_stability_all_models_fixed.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Fixed Spearman heatmap saved → {out_path}")


def plot_combined_jaccard():
    model_names = list(jaccard_files.keys())

    j_c1c3 = []
    j_c2c4 = []

    for model_name in model_names:
        c1c3, c2c4 = load_jaccard_k20(jaccard_files[model_name])
        j_c1c3.append(c1c3)
        j_c2c4.append(c2c4)

    x = np.arange(len(model_names))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width/2, j_c1c3, width, label="C1↔C3 (train fixed)", color="#1f77b4")
    bars2 = ax.bar(x + width/2, j_c2c4, width, label="C2↔C4 (train fixed)", color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Jaccard similarity (K=20)", fontsize=12)
    ax.set_title("Top-20 SHAP Feature Overlap Across Years", fontsize=16, fontweight="bold", pad=12)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for b in bars1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", fontsize=10)

    for b in bars2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.02, f"{h:.2f}",
                ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "combined_jaccard_k20_all_models_fixed.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Fixed Jaccard bar chart saved → {out_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("Generating combined stability figures...")
    print("=" * 70)

    plot_combined_spearman()
    plot_combined_jaccard()

    print("\nAll figures generated successfully.")