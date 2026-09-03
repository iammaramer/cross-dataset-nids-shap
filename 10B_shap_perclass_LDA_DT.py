
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

# ── Recreate splits and scalers ───────────────────────────────────────────────
print("Preparing train/test splits and scalers...")

X17 = df17.drop(columns=['label']).values
y17 = df17['label'].values
X17_train_raw, X17_test_raw, y17_train, y17_test = train_test_split(
    X17, y17, test_size=0.2, random_state=42, stratify=y17
)

X18 = df18.drop(columns=['label']).values
y18 = df18['label'].values
X18_train_raw, X18_test_raw, y18_train, y18_test = train_test_split(
    X18, y18, test_size=0.2, random_state=42, stratify=y18
)

scaler17 = MinMaxScaler()
X17_train = scaler17.fit_transform(X17_train_raw)
X17_test  = scaler17.transform(X17_test_raw)

scaler18 = MinMaxScaler()
X18_train = scaler18.fit_transform(X18_train_raw)
X18_test  = scaler18.transform(X18_test_raw)

# Cross-dataset test sets using the same held-out test partitions
X18_test_scaled17 = scaler17.transform(X18_test_raw)  # C3: train on 17, test on 18
X17_test_scaled18 = scaler18.transform(X17_test_raw)  # C4: train on 18, test on 17

print("  ✅ Splits and scalers ready.\n")

# ── Helper: load model ────────────────────────────────────────────────────────
def load_model(exp, model_name):
    path = os.path.join(MODELS_DIR, f"{exp}_{model_name}.pkl")
    with open(path, 'rb') as f:
        return pickle.load(f)

# ── Helper: per-class SHAP ────────────────────────────────────────────────────
def run_shap_per_class(model, X_test, y_test, model_name, exp_name,
                       ds_name, explainer_type, X_background=None):
    print(f"\n{'-'*60}")
    print(f"Per-class SHAP — {model_name} | {exp_name} | {ds_name}")
    print(f"{'-'*60}")

    X_df = pd.DataFrame(X_test, columns=feature_names)

    # Build explainer once
    if explainer_type == 'linear':
        bg_idx = np.random.RandomState(42).choice(
            len(X_background), min(SHAP_SAMPLE, len(X_background)), replace=False
        )
        X_bg = X_background[bg_idx]
        explainer = shap.LinearExplainer(model, X_bg)
    elif explainer_type == 'tree':
        explainer = shap.TreeExplainer(model)
    else:
        raise ValueError(f"Unknown explainer_type: {explainer_type}")

    classes = np.unique(y_test)
    for cls in classes:
        cls_name = "Benign" if cls == 0 else "Attack"
        mask = (y_test == cls)
        n_cls = mask.sum()
        if n_cls == 0:
            continue

        print(f"\n  Class {cls} ({cls_name}) — {n_cls} test samples")

        idx_cls = np.where(mask)[0]
        if len(idx_cls) > SHAP_SAMPLE:
            idx_cls = np.random.RandomState(42).choice(idx_cls, SHAP_SAMPLE, replace=False)

        X_cls = X_df.iloc[idx_cls]

        shap_exp_cls = explainer(X_cls)
        vals_cls = np.array(shap_exp_cls.values)
        print(f"    SHAP values shape: {vals_cls.shape}")

        # Handle class dimension correctly
        if vals_cls.ndim == 3:
            # Tree models: (n_samples, n_features, n_classes)
            class_index = list(model.classes_).index(cls)
            vals_cls = vals_cls[:, :, class_index]
        elif vals_cls.ndim != 2:
            raise ValueError(f"Unexpected SHAP shape for {model_name}: {vals_cls.shape}")

        vals_abs = np.abs(vals_cls)
        mean_shap = vals_abs.mean(axis=0)

        global_importance = (
            pd.DataFrame({
                'Feature': feature_names,
                'Mean_SHAP': mean_shap
            })
            .sort_values('Mean_SHAP', ascending=False)
            .reset_index(drop=True)
        )
        global_importance['Rank'] = global_importance.index + 1

        csv_name = f"shap_perclass_{exp_name}_{model_name}_class{cls}.csv"
        csv_path = os.path.join(RESULTS_DIR, csv_name)
        global_importance.to_csv(csv_path, index=False)
        print(f"    ✅ Importance saved → {csv_name}")

        top20 = global_importance.head(20)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top20['Feature'][::-1], top20['Mean_SHAP'][::-1], color='seagreen')
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(f'Top 20 Features — {model_name} | {exp_name} | Class {cls} ({cls_name})')
        plt.tight_layout()
        bar_name = f"shap_perclass_bar_{exp_name}_{model_name}_class{cls}.png"
        plt.savefig(os.path.join(RESULTS_DIR, bar_name), dpi=150)
        plt.close()
        print(f"    ✅ Bar plot saved   → {bar_name}")

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            vals_cls,
            X_cls,
            feature_names=feature_names,
            max_display=20,
            show=False
        )
        plt.title(f'SHAP Summary — {model_name} | {exp_name} | Class {cls} ({cls_name})')
        plt.tight_layout()
        bee_name = f"shap_perclass_beeswarm_{exp_name}_{model_name}_class{cls}.png"
        plt.savefig(os.path.join(RESULTS_DIR, bee_name), dpi=150)
        plt.close()
        print(f"    ✅ Beeswarm saved   → {bee_name}")

        print("    Top 10 features for this class:")
        for _, row in global_importance.head(10).iterrows():
            print(f"      #{int(row['Rank']):2d}  {row['Feature']:35s}  SHAP={row['Mean_SHAP']:.5f}")

# ── C1: LycoS17 → LycoS17 ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Per-class SHAP — C1 (LycoS17→LycoS17)")
print(f"{'='*60}")

run_shap_per_class(
    load_model('C1', 'LDA'),
    X17_test,
    y17_test,
    'LDA',
    'C1',
    'LycoS17',
    'linear',
    X17_train
)

run_shap_per_class(
    load_model('C1', 'Decision_Tree'),
    X17_test,
    y17_test,
    'Decision_Tree',
    'C1',
    'LycoS17',
    'tree'
)

# ── C2: LycoS18 → LycoS18 ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Per-class SHAP — C2 (LycoS18→LycoS18)")
print(f"{'='*60}")

run_shap_per_class(
    load_model('C2', 'LDA'),
    X18_test,
    y18_test,
    'LDA',
    'C2',
    'LycoS18',
    'linear',
    X18_train
)

run_shap_per_class(
    load_model('C2', 'Decision_Tree'),
    X18_test,
    y18_test,
    'Decision_Tree',
    'C2',
    'LycoS18',
    'tree'
)

# ── C3: LycoS17 → LycoS18 ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Per-class SHAP — C3 (LycoS17→LycoS18)")
print(f"{'='*60}")

run_shap_per_class(
    load_model('C3', 'LDA'),
    X18_test_scaled17,
    y18_test,
    'LDA',
    'C3',
    'LycoS17→LycoS18',
    'linear',
    X17_train
)

run_shap_per_class(
    load_model('C3', 'Decision_Tree'),
    X18_test_scaled17,
    y18_test,
    'Decision_Tree',
    'C3',
    'LycoS17→LycoS18',
    'tree'
)

# ── C4: LycoS18 → LycoS17 ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Per-class SHAP — C4 (LycoS18→LycoS17)")
print(f"{'='*60}")

run_shap_per_class(
    load_model('C4', 'LDA'),
    X17_test_scaled18,
    y17_test,
    'LDA',
    'C4',
    'LycoS18→LycoS17',
    'linear',
    X18_train
)

run_shap_per_class(
    load_model('C4', 'Decision_Tree'),
    X17_test_scaled18,
    y17_test,
    'Decision_Tree',
    'C4',
    'LycoS18→LycoS17',
    'tree'
)

print(f"\n{'='*60}")
print("Stage 4.3 — Per-class SHAP (LDA & DT, C1–C4) complete!")
print(f"All outputs saved in: {RESULTS_DIR}")
print(f"{'='*60}")