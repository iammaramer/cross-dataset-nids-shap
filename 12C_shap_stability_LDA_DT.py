
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = r"results\shap"

rank_cols  = ['Rank_C1', 'Rank_C2', 'Rank_C3', 'Rank_C4']
exp_labels = ['C1', 'C2', 'C3', 'C4']

pairs = [
    ('C1', 'C2'),
    ('C1', 'C3'),
    ('C1', 'C4'),
    ('C2', 'C3'),
    ('C2', 'C4'),
    ('C3', 'C4'),
]
K_values = [10, 20, 30]


def jaccard_for_pair(df, exp_a, exp_b, k):
    ra = df[f"Rank_{exp_a}"]
    rb = df[f"Rank_{exp_b}"]
    set_a = set(df.loc[ra <= k, 'Feature'])
    set_b = set(df.loc[rb <= k, 'Feature'])
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    jac   = inter / union if union > 0 else np.nan
    return jac, inter, union


def run_stability(model_tag, csv_filename):
    print(f"\n{'='*60}")
    print(f"  Stability metrics — {model_tag}")
    print(f"{'='*60}")

    ranking_csv = os.path.join(RESULTS_DIR, csv_filename)
    print(f"Loading ranking table: {csv_filename}...")
    df = pd.read_csv(ranking_csv)
    print(f"  Loaded {df.shape[0]} features with ranks for C1–C4.")

    # ── Spearman rank correlation ──────────────────────────────────────────
    print("\nComputing Spearman rank correlation matrix (C1–C4)...")
    ranks_df = df[rank_cols].copy()
    spearman = ranks_df.corr(method='spearman')

    spearman_csv = os.path.join(RESULTS_DIR, f"shap_spearman_{model_tag}_C1C4.csv")
    spearman.to_csv(spearman_csv)
    print(f"  ✅ Spearman matrix saved → {spearman_csv}")

    print(f"\nSpearman rank correlation matrix ({model_tag} SHAP ranks):")
    print(spearman.to_string(float_format=lambda x: f"{x: .3f}"))

    # ── Spearman heatmap ───────────────────────────────────────────────────
    print("\nCreating Spearman stability heatmap...")
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(spearman.values, vmin=-1, vmax=1, cmap='coolwarm')

    ax.set_xticks(np.arange(len(exp_labels)))
    ax.set_yticks(np.arange(len(exp_labels)))
    ax.set_xticklabels(exp_labels)
    ax.set_yticklabels(exp_labels)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    for i in range(len(exp_labels)):
        for j in range(len(exp_labels)):
            val = spearman.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

    ax.set_title(f"Spearman rank correlation of {model_tag} SHAP feature rankings (C1–C4)")
    fig.colorbar(cax, ax=ax, label="Spearman ρ")

    plt.tight_layout()
    heatmap_path = os.path.join(RESULTS_DIR, f"shap_spearman_matrix_{model_tag}_C1C4.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"  ✅ Heatmap saved → {heatmap_path}")

    # ── Jaccard similarity ─────────────────────────────────────────────────
    print("\nComputing Jaccard similarity for top-K feature sets...")
    rows = []
    for a, b in pairs:
        for K in K_values:
            jac, inter, union = jaccard_for_pair(df, a, b, K)
            rows.append({
                'Exp_A':        a,
                'Exp_B':        b,
                'K':            K,
                'Intersection': inter,
                'Union':        union,
                'Jaccard':      jac,
            })

    jac_df = pd.DataFrame(rows)

    jac_txt = os.path.join(RESULTS_DIR, f"shap_jaccard_{model_tag}_C1C4.txt")
    with open(jac_txt, "w") as f:
        f.write(f"Jaccard similarity of top-K {model_tag} SHAP features (C1–C4)\n")
        f.write("===================================================\n\n")
        for (a, b) in pairs:
            sub = jac_df[(jac_df['Exp_A'] == a) & (jac_df['Exp_B'] == b)]
            f.write(f"{a} vs {b}:\n")
            for _, row in sub.iterrows():
                f.write(
                    f"  K={int(row['K']):2d}: "
                    f"Jaccard={row['Jaccard']:.3f} "
                    f"(intersection={int(row['Intersection'])}, union={int(row['Union'])})\n"
                )
            f.write("\n")

    jac_csv = os.path.join(RESULTS_DIR, f"shap_jaccard_{model_tag}_C1C4.csv")
    jac_df.to_csv(jac_csv, index=False)
    print(f"  ✅ Jaccard CSV saved  → {jac_csv}")
    print(f"  ✅ Jaccard TXT saved  → {jac_txt}")

    # ── Print key pairs (for thesis text) ──────────────────────────────────
    for a, b in [('C1', 'C3'), ('C2', 'C4')]:
        sub = jac_df[(jac_df['Exp_A'] == a) & (jac_df['Exp_B'] == b)]
        print(f"\n  {a} vs {b} Jaccard (top-K):")
        for _, row in sub.iterrows():
            print(
                f"    K={int(row['K']):2d}: "
                f"Jaccard={row['Jaccard']:.3f} "
                f"(intersection={int(row['Intersection'])}, union={int(row['Union'])})"
            )

    print(f"\n  Stage 5 — SHAP stability metrics ({model_tag}) complete.")
    print(f"  Outputs in: {RESULTS_DIR}")


# ── Run for LDA ───────────────────────────────────────────────────────────────
run_stability(
    model_tag    = "LDA",
    csv_filename = "shap_feature_ranking_LDA_C1C4.csv"
)

# ── Run for Decision Tree ─────────────────────────────────────────────────────
run_stability(
    model_tag    = "Decision_Tree",
    csv_filename = "shap_feature_ranking_Decision_Tree_C1C4.csv"
)

print(f"\n{'='*60}")
print("  All stability metrics for LDA and Decision Tree complete!")
print(f"{'='*60}")