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
LYCOS17     = r"data\LycoS-IDS2017_FINAL.csv"
LYCOS18     = r"data\LycoS-Unicas-IDS2018_CLEANED.csv"
MODELS_DIR  = r"models"
RESULTS_DIR = r"results\shap"
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

# ── Helper: prepare test split ────────────────────────────────────────────────
def get_test_split(df):
    X = df.drop(columns=['label']).values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    return scaler.transform(X_test), y_test

# ── Helper: load model ────────────────────────────────────────────────────────
def load_model(exp, model_name):
    path = os.path.join(MODELS_DIR, f"{exp}_{model_name}.pkl")
    with open(path, 'rb') as f:
        return pickle.load(f)

# ── Helper: run SHAP ──────────────────────────────────────────────────────────
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
    print(f"    Importance vector shape: {importance.shape}")
    print(f"    len(feature_names): {len(feature_names)}")

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
    ax.barh(top20['Feature'][::-1], top20['Mean_SHAP'][::-1], color='steelblue')
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
            values      = shap_exp.values[:, :, 0],
            base_values = shap_exp.base_values[:, 0] if shap_exp.base_values.ndim > 1 else shap_exp.base_values,
            data        = shap_exp.data,
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

# ── C1: RF and XGBoost on LycoS17 ────────────────────────────────────────────
print(f"\n{'='*60}")
print("  SHAP — C1: LycoS17 → LycoS17")
print(f"{'='*60}")
X17_test, y17_test = get_test_split(df17)

shap_c1_rf  = run_shap(load_model('C1', 'Random_Forest'), X17_test, 'Random_Forest', 'C1', 'LycoS17')
shap_c1_xgb = run_shap(load_model('C1', 'XGBoost'),       X17_test, 'XGBoost',       'C1', 'LycoS17')

# ── C2: RF and XGBoost on LycoS18 ────────────────────────────────────────────
print(f"\n{'='*60}")
print("  SHAP — C2: LycoS18 → LycoS18")
print(f"{'='*60}")
X18_test, y18_test = get_test_split(df18)

shap_c2_rf  = run_shap(load_model('C2', 'Random_Forest'), X18_test, 'Random_Forest', 'C2', 'LycoS18')
shap_c2_xgb = run_shap(load_model('C2', 'XGBoost'),       X18_test, 'XGBoost',       'C2', 'LycoS18')

# ── Cross-dataset comparison ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  CROSS-DATASET SHAP COMPARISON — Random Forest")
print(f"{'='*60}")

top_rf_17 = set(shap_c1_rf.head(20)['Feature'].tolist())
top_rf_18 = set(shap_c2_rf.head(20)['Feature'].tolist())
stable    = top_rf_17 & top_rf_18
only_17   = top_rf_17 - top_rf_18
only_18   = top_rf_18 - top_rf_17

print(f"\n  Stable features (in BOTH top 20)   : {len(stable)}")
for f in sorted(stable):
    print(f"    ✅ {f}")
print(f"\n  Only in LycoS17 top 20             : {len(only_17)}")
for f in sorted(only_17):
    print(f"    📌 {f}")
print(f"\n  Only in LycoS18 top 20             : {len(only_18)}")
for f in sorted(only_18):
    print(f"    📌 {f}")

comp_df = pd.DataFrame({
    'Feature'          : sorted(top_rf_17 | top_rf_18),
    'In_LycoS17_Top20' : [f in top_rf_17 for f in sorted(top_rf_17 | top_rf_18)],
    'In_LycoS18_Top20' : [f in top_rf_18 for f in sorted(top_rf_17 | top_rf_18)],
    'Stable'           : [f in stable    for f in sorted(top_rf_17 | top_rf_18)],
})
comp_df.to_csv(os.path.join(RESULTS_DIR, 'shap_cross_dataset_comparison_RF.csv'), index=False)
print(f"\n✅ Comparison saved → shap_cross_dataset_comparison_RF.csv")

print(f"\n{'='*60}")
print("  Stage 4.1 Complete!")
print(f"{'='*60}")
print(f"  All outputs in: {RESULTS_DIR}")