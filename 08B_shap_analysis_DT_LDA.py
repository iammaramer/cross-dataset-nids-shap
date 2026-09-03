
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
LYCOS17    = r"D:\thesis_implementation\LycoS-IDS2017_FINAL.csv"
LYCOS18    = r"D:\thesis_implementation\LycoS-Unicas-IDS2018_CLEANED.csv"
MODELS_DIR = r"D:\thesis_implementation\results\models"
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

# ── Helper: prepare within-dataset test split ────────────────────────────────
def get_test_split(df):
    X = df.drop(columns=['label']).values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    return scaler.transform(X_test), y_test

# ── Helper: recreate cross-dataset test split ────────────────────────────────
def get_cross_test_splits(df17, df18):
    X17 = df17.drop(columns=['label']).values
    y17 = df17['label'].values
    X18 = df18.drop(columns=['label']).values
    y18 = df18['label'].values

    X17_train, X17_test, y17_train, y17_test = train_test_split(
        X17, y17, test_size=0.2, random_state=42, stratify=y17
    )
    X18_train, X18_test, y18_train, y18_test = train_test_split(
        X18, y18, test_size=0.2, random_state=42, stratify=y18
    )

    scaler17 = MinMaxScaler()
    scaler18 = MinMaxScaler()

    scaler17.fit(X17_train)
    scaler18.fit(X18_train)

    X18_test_s17 = scaler17.transform(X18_test)  # C3: train on 17, test on 18
    X17_test_s18 = scaler18.transform(X17_test)  # C4: train on 18, test on 17

    return X18_test_s17, y18_test, X17_test_s18, y17_test, X17_train, X18_train

# ── Helper: load model ────────────────────────────────────────────────────────
def load_model(exp, model_name):
    path = os.path.join(MODELS_DIR, f"{exp}_{model_name}.pkl")
    with open(path, 'rb') as f:
        return pickle.load(f)

# ── Helper: run SHAP ──────────────────────────────────────────────────────────
def run_shap(model, X_test, X_background, model_name, exp_name, ds_name, explainer_type):
    print(f"\n  [SHAP] {model_name} | {exp_name} | {ds_name}")

    idx = np.random.choice(len(X_test), min(SHAP_SAMPLE, len(X_test)), replace=False)
    X_sample = X_test[idx]
    X_df = pd.DataFrame(X_sample, columns=feature_names)

    print(f"    Computing SHAP values on {len(X_df)} samples...")

    if explainer_type == 'linear':
        bg_idx = np.random.choice(len(X_background), min(SHAP_SAMPLE, len(X_background)), replace=False)
        X_bg = X_background[bg_idx]
        explainer = shap.LinearExplainer(model, X_bg)
        shap_exp = explainer(X_df)
    elif explainer_type == 'tree':
        explainer = shap.TreeExplainer(model)
        shap_exp = explainer(X_df)
    else:
        raise ValueError(f"Unknown explainer_type: {explainer_type}")

    vals = shap_exp.values
    print(f"    SHAP values shape: {vals.shape}")

    vals = np.array(vals)

    if vals.ndim == 3:
        vals = np.abs(vals).mean(axis=2)
    else:
        vals = np.abs(vals)

    importance = vals.mean(axis=0)
    print(f"    Importance vector shape: {importance.shape}")
    print(f"    len(feature_names): {len(feature_names)}")

    global_importance = pd.DataFrame({
        'Feature': feature_names,
        'Mean_SHAP': importance
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
            values=shap_exp.values[:, :, 0],
            base_values=shap_exp.base_values[:, 0] if np.ndim(shap_exp.base_values) > 1 else shap_exp.base_values,
            data=shap_exp.data,
            feature_names=feature_names
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

# ── Prepare splits ────────────────────────────────────────────────────────────
X17_test, y17_test = get_test_split(df17)
X18_test, y18_test = get_test_split(df18)
X18_test_s17, y18_test_cross, X17_test_s18, y17_test_cross, X17_train_raw, X18_train_raw = get_cross_test_splits(df17, df18)

# Scale train backgrounds for explainers
scaler17_bg = MinMaxScaler()
scaler18_bg = MinMaxScaler()

X17 = df17.drop(columns=['label']).values
y17 = df17['label'].values
X18 = df18.drop(columns=['label']).values
y18 = df18['label'].values

X17_train_full, _, _, _ = train_test_split(
    X17, y17, test_size=0.2, random_state=42, stratify=y17
)
X18_train_full, _, _, _ = train_test_split(
    X18, y18, test_size=0.2, random_state=42, stratify=y18
)

X17_train_scaled = scaler17_bg.fit_transform(X17_train_full)
X18_train_scaled = scaler18_bg.fit_transform(X18_train_full)

# ── C1: LDA and DT on LycoS17 ────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  SHAP — C1: LycoS17 → LycoS17")
print(f"{'='*60}")

shap_c1_lda = run_shap(load_model('C1', 'LDA'), X17_test, X17_train_scaled, 'LDA', 'C1', 'LycoS17', 'linear')
shap_c1_dt  = run_shap(load_model('C1', 'Decision_Tree'), X17_test, X17_train_scaled, 'Decision_Tree', 'C1', 'LycoS17', 'tree')

# ── C2: LDA and DT on LycoS18 ────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  SHAP — C2: LycoS18 → LycoS18")
print(f"{'='*60}")

shap_c2_lda = run_shap(load_model('C2', 'LDA'), X18_test, X18_train_scaled, 'LDA', 'C2', 'LycoS18', 'linear')
shap_c2_dt  = run_shap(load_model('C2', 'Decision_Tree'), X18_test, X18_train_scaled, 'Decision_Tree', 'C2', 'LycoS18', 'tree')

# ── C3: LDA and DT on LycoS17 → LycoS18 ──────────────────────────────────────
print(f"\n{'='*60}")
print("  SHAP — C3: LycoS17 → LycoS18")
print(f"{'='*60}")

shap_c3_lda = run_shap(load_model('C3', 'LDA'), X18_test_s17, X17_train_scaled, 'LDA', 'C3', 'LycoS17→LycoS18', 'linear')
shap_c3_dt  = run_shap(load_model('C3', 'Decision_Tree'), X18_test_s17, X17_train_scaled, 'Decision_Tree', 'C3', 'LycoS17→LycoS18', 'tree')

# ── C4: LDA and DT on LycoS18 → LycoS17 ──────────────────────────────────────
print(f"\n{'='*60}")
print("  SHAP — C4: LycoS18 → LycoS17")
print(f"{'='*60}")

shap_c4_lda = run_shap(load_model('C4', 'LDA'), X17_test_s18, X18_train_scaled, 'LDA', 'C4', 'LycoS18→LycoS17', 'linear')
shap_c4_dt  = run_shap(load_model('C4', 'Decision_Tree'), X17_test_s18, X18_train_scaled, 'Decision_Tree', 'C4', 'LycoS18→LycoS17', 'tree')

print(f"\n{'='*60}")
print("  Stage 4.1 for LDA and DT Complete!")
print(f"{'='*60}")
print(f"  All outputs in: {RESULTS_DIR}")