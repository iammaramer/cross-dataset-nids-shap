import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
LYCOS17     = r"D:\thesis_implementation\LycoS-IDS2017_FINAL.csv"
LYCOS18     = r"D:\thesis_implementation\LycoS-Unicas-IDS2018_CLEANED.csv"
MODELS_DIR  = r"D:\thesis_implementation\results\models"
RESULTS_DIR = r"D:\thesis_implementation\results\shap"
SAMPLE_18   = 500_000
SHAP_SAMPLE = 2_000

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load datasets ─────────────────────────────────────────────────────────────
print("Loading datasets...")
df17 = pd.read_csv(LYCOS17)
print(f"  LycoS17 : {df17.shape}")

df18_full = pd.read_csv(LYCOS18)
parts = []
total = len(df18_full)
for label_val, group in df18_full.groupby('label'):
    frac = len(group) / total
    n = min(int(SAMPLE_18 * frac), len(group))
    parts.append(group.sample(n=n, random_state=42))
df18 = pd.concat(parts, ignore_index=True)
del df18_full
print(f"  LycoS18 : {df18.shape}\n")

feature_names = [c for c in df17.columns if c != 'label']

# ── Replicate exact splits + scalers from 07_tuned_experiments.py ─────────────
print("Preparing splits and scalers (replicating 07_tuned_experiments.py)...")

X17 = df17.drop(columns=['label']).values
y17 = df17['label'].values
X17_train, X17_test, y17_train, y17_test = train_test_split(
    X17, y17, test_size=0.2, random_state=42, stratify=y17)

X18 = df18.drop(columns=['label']).values
y18 = df18['label'].values
X18_train, X18_test, y18_train, y18_test = train_test_split(
    X18, y18, test_size=0.2, random_state=42, stratify=y18)

# scaler17 → fit on X17_train (used for C3 test set)
scaler17 = MinMaxScaler()
scaler17.fit(X17_train)

# scaler18 → fit on X18_train (used for C4 test set)
scaler18 = MinMaxScaler()
scaler18.fit(X18_train)

# C3 test: LycoS18 20% split rows, scaled with scaler17
X18_test_scaled17 = scaler17.transform(
    df18.drop(columns=['label']).values[
        train_test_split(range(len(df18)), test_size=0.2, random_state=42,
                         stratify=y18)[1]
    ]
)

# C4 test: LycoS17 20% split rows, scaled with scaler18
X17_test_scaled18 = scaler18.transform(
    df17.drop(columns=['label']).values[
        train_test_split(range(len(df17)), test_size=0.2, random_state=42,
                         stratify=y17)[1]
    ]
)

print("  ✅ Splits and scalers ready.\n")

# ── Helper: load model ────────────────────────────────────────────────────────
def load_model(exp, model_name):
    path = os.path.join(MODELS_DIR, f"{exp}_{model_name}.pkl")
    with open(path, 'rb') as f:
        return pickle.load(f)

# ── Helper: run SHAP (same pattern as 08_shap_analysis.py) ───────────────────
def run_shap(model, X_test, model_name, exp_name, ds_name):
    print(f"\n  [SHAP] {model_name} | {exp_name} | {ds_name}")

    idx      = np.random.choice(len(X_test), min(SHAP_SAMPLE, len(X_test)), replace=False)
    X_sample = X_test[idx]
    X_df     = pd.DataFrame(X_sample, columns=feature_names)

    print(f"    Computing SHAP values on {len(X_df)} samples...")

    explainer = shap.TreeExplainer(model)
    shap_exp  = explainer(X_df)

    vals = shap_exp.values
    print(f"    SHAP values shape: {vals.shape}")

    if vals.ndim == 3:
        vals = np.abs(vals).mean(axis=2)
    else:
        vals = np.abs(vals)

    importance = vals.mean(axis=0)

    global_importance = pd.DataFrame({
        'Feature'   : feature_names,
        'Mean_SHAP' : importance
    }).sort_values('Mean_SHAP', ascending=False).reset_index(drop=True)
    global_importance['Rank'] = global_importance.index + 1

    csv_name = f"shap_importance_{exp_name}_{model_name}.csv"
    global_importance.to_csv(os.path.join(RESULTS_DIR, csv_name), index=False)
    print(f"    ✅ Importance saved  → {csv_name}")

    # ── Bar plot ───────────────────────────────────────────────────────────
    top20 = global_importance.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top20['Feature'][::-1], top20['Mean_SHAP'][::-1], color='darkorange')
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title(f'Top 20 Features — {model_name} | {exp_name} ({ds_name})')
    plt.tight_layout()
    bar_name = f"shap_bar_{exp_name}_{model_name}.png"
    plt.savefig(os.path.join(RESULTS_DIR, bar_name), dpi=150)
    plt.close()
    print(f"    ✅ Bar plot saved    → {bar_name}")

    # ── Beeswarm plot ──────────────────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    if shap_exp.values.ndim == 3:
        plot_exp = shap.Explanation(
            values        = shap_exp.values[:, :, 0],
            base_values   = shap_exp.base_values[:, 0] if shap_exp.base_values.ndim > 1 else shap_exp.base_values,
            data          = shap_exp.data,
            feature_names = feature_names
        )
    else:
        plot_exp = shap_exp
    shap.plots.beeswarm(plot_exp, max_display=20, show=False)
    plt.title(f'SHAP Summary — {model_name} | {exp_name} ({ds_name})')
    plt.tight_layout()
    bee_name = f"shap_beeswarm_{exp_name}_{model_name}.png"
    plt.savefig(os.path.join(RESULTS_DIR, bee_name), dpi=150)
    plt.close()
    print(f"    ✅ Beeswarm saved    → {bee_name}")

    print(f"\n    Top 10 features:")
    for _, row in global_importance.head(10).iterrows():
        print(f"      #{int(row['Rank']):2d}  {row['Feature']:35s}  SHAP={row['Mean_SHAP']:.5f}")

    return global_importance

# ── C3: LycoS17 → LycoS18 (cross-dataset) ────────────────────────────────────
print(f"\n{'='*60}")
print("  SHAP — C3: Train=LycoS17 | Test=LycoS18  (cross-dataset)")
print(f"{'='*60}")

shap_c3_rf  = run_shap(load_model('C3', 'Random_Forest'), X18_test_scaled17, 'Random_Forest', 'C3', 'LycoS17→LycoS18')
shap_c3_xgb = run_shap(load_model('C3', 'XGBoost'),       X18_test_scaled17, 'XGBoost',       'C3', 'LycoS17→LycoS18')

# ── C4: LycoS18 → LycoS17 (cross-dataset) ────────────────────────────────────
print(f"\n{'='*60}")
print("  SHAP — C4: Train=LycoS18 | Test=LycoS17  (cross-dataset)")
print(f"{'='*60}")

shap_c4_rf  = run_shap(load_model('C4', 'Random_Forest'), X17_test_scaled18, 'Random_Forest', 'C4', 'LycoS18→LycoS17')
shap_c4_xgb = run_shap(load_model('C4', 'XGBoost'),       X17_test_scaled18, 'XGBoost',       'C4', 'LycoS18→LycoS17')

# ── Cross-dataset feature stability comparison ────────────────────────────────
print(f"\n{'='*60}")
print("  CROSS-DATASET STABILITY — C3 vs C4 (Random Forest)")
print(f"{'='*60}")

top_rf_c3 = set(shap_c3_rf.head(20)['Feature'].tolist())
top_rf_c4 = set(shap_c4_rf.head(20)['Feature'].tolist())
stable    = top_rf_c3 & top_rf_c4
only_c3   = top_rf_c3 - top_rf_c4
only_c4   = top_rf_c4 - top_rf_c3

print(f"\n  Stable features (in BOTH C3 and C4 top 20) : {len(stable)}")
for f in sorted(stable):
    print(f"    ✅ {f}")
print(f"\n  Only in C3 top 20 : {len(only_c3)}")
for f in sorted(only_c3):
    print(f"    📌 {f}")
print(f"\n  Only in C4 top 20 : {len(only_c4)}")
for f in sorted(only_c4):
    print(f"    📌 {f}")

comp_df = pd.DataFrame({
    'Feature'         : sorted(top_rf_c3 | top_rf_c4),
    'In_C3_Top20'     : [f in top_rf_c3 for f in sorted(top_rf_c3 | top_rf_c4)],
    'In_C4_Top20'     : [f in top_rf_c4 for f in sorted(top_rf_c3 | top_rf_c4)],
    'Stable_C3_C4'    : [f in stable    for f in sorted(top_rf_c3 | top_rf_c4)],
})
comp_df.to_csv(os.path.join(RESULTS_DIR, 'shap_cross_dataset_comparison_C3_C4_RF.csv'), index=False)
print(f"\n✅ C3 vs C4 comparison saved → shap_cross_dataset_comparison_C3_C4_RF.csv")

print(f"\n{'='*60}")
print("  Stage 4.2 Complete!")
print(f"{'='*60}")
print(f"  All outputs in: {RESULTS_DIR}")