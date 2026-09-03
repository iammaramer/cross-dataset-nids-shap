import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
RESULTS_DIR = r"D:\thesis_implementation\results\shap"
RANKING_CSV = os.path.join(RESULTS_DIR, "shap_feature_ranking_RF_C1C4.csv")

print("Loading global RF SHAP ranking table...")
df = pd.read_csv(RANKING_CSV)
print(f"  Loaded {df.shape[0]} features with ranks for C1–C4.")

# Columns with ranks
rank_cols = ['Rank_C1', 'Rank_C2', 'Rank_C3', 'Rank_C4']
exp_labels = ['C1', 'C2', 'C3', 'C4']

# 1) Spearman rank correlation matrix ----------------------------------------
print("\nComputing Spearman rank correlation matrix (C1–C4)...")

ranks_df = df[rank_cols].copy()
spearman = ranks_df.corr(method='spearman')

# Save matrix as CSV
spearman_csv = os.path.join(RESULTS_DIR, "shap_spearman_RF_C1C4.csv")
spearman.to_csv(spearman_csv)
print(f"  ✅ Spearman matrix saved → {spearman_csv}")

print("\nSpearman rank correlation matrix (Random Forest SHAP ranks):")
print(spearman.to_string(float_format=lambda x: f"{x: .3f}"))

# 2) Stability heatmap figure -------------------------------------------------
print("\nCreating Spearman stability heatmap...")

fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.imshow(spearman.values, vmin=-1, vmax=1, cmap='coolwarm')

ax.set_xticks(np.arange(len(exp_labels)))
ax.set_yticks(np.arange(len(exp_labels)))
ax.set_xticklabels(exp_labels)
ax.set_yticklabels(exp_labels)

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

for i in range(len(exp_labels)):
    for j in range(len(exp_labels)):
        val = spearman.values[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")

ax.set_title("Spearman rank correlation of RF SHAP feature rankings (C1–C4)")
fig.colorbar(cax, ax=ax, label="Spearman ρ")

plt.tight_layout()
heatmap_path = os.path.join(RESULTS_DIR, "shap_spearman_matrix_RF_C1C4.png")
plt.savefig(heatmap_path, dpi=150)
plt.close()
print(f"  ✅ Heatmap saved → {heatmap_path}")

# 3) Jaccard similarity of top-K features ------------------------------------
def jaccard_for_pair(exp_a, exp_b, k):
    """
    exp_a, exp_b: 'C1', 'C2', 'C3', 'C4'
    k: top-K threshold
    """
    ra = df[f"Rank_{exp_a}"]
    rb = df[f"Rank_{exp_b}"]
    set_a = set(df.loc[ra <= k, 'Feature'])
    set_b = set(df.loc[rb <= k, 'Feature'])
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    jac = inter / union if union > 0 else np.nan
    return jac, inter, union

pairs = [
    ('C1', 'C2'),
    ('C1', 'C3'),
    ('C1', 'C4'),
    ('C2', 'C3'),
    ('C2', 'C4'),
    ('C3', 'C4'),
]
K_values = [10, 20, 30]

print("\nComputing Jaccard similarity for top-K feature sets...")
rows = []
for a, b in pairs:
    for K in K_values:
        jac, inter, union = jaccard_for_pair(a, b, K)
        rows.append({
            'Exp_A': a,
            'Exp_B': b,
            'K': K,
            'Intersection': inter,
            'Union': union,
            'Jaccard': jac,
        })

jac_df = pd.DataFrame(rows)
jac_txt = os.path.join(RESULTS_DIR, "shap_jaccard_RF_C1C4.txt")
with open(jac_txt, "w") as f:
    f.write("Jaccard similarity of top-K RF SHAP features (C1–C4)\n")
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

jac_csv = os.path.join(RESULTS_DIR, "shap_jaccard_RF_C1C4.csv")
jac_df.to_csv(jac_csv, index=False)
print(f"  ✅ Jaccard results saved → {jac_csv}")
print(f"  ✅ Text report saved     → {jac_txt}")

# Highlight key pairs (for thesis text)
def print_pair(a, b):
    sub = jac_df[(jac_df['Exp_A'] == a) & (jac_df['Exp_B'] == b)]
    print(f"\n{a} vs {b} Jaccard (top-K):")
    for _, row in sub.iterrows():
        print(
            f"  K={int(row['K']):2d}: "
            f"Jaccard={row['Jaccard']:.3f} "
            f"(intersection={int(row['Intersection'])}, union={int(row['Union'])})"
        )

print_pair('C1', 'C3')
print_pair('C2', 'C4')

print("\nStage 5 — SHAP stability metrics (RF) complete.")
print(f"Outputs in: {RESULTS_DIR}")