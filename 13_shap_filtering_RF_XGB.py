import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (matthews_corrcoef, accuracy_score,
                              precision_score, recall_score, f1_score)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
LYCOS17      = r"D:\thesis_implementation\LycoS-IDS2017_FINAL.csv"
LYCOS18      = r"D:\thesis_implementation\LycoS-Unicas-IDS2018_CLEANED.csv"
SHAP_DIR     = r"D:\thesis_implementation\results\shap"
RESULTS_DIR  = r"D:\thesis_implementation\results"
PARAMS_FILE  = r"D:\thesis_implementation\results\best_params.json"
RANKING_CSV  = os.path.join(SHAP_DIR, "shap_feature_ranking_RF_C1C4.csv")
TUNED_CSV    = os.path.join(RESULTS_DIR, "tuned_results.csv")
OUTPUT_CSV   = os.path.join(RESULTS_DIR, "shap_filtered_results.csv")
COMPARE_CSV  = os.path.join(RESULTS_DIR, "shap_filtering_comparison.csv")
SAMPLE_18    = 500_000

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Step 1: identify stable features ─────────────────────────────────────────
print("Step 1 — Loading SHAP feature ranking and selecting stable features...")
rank_df = pd.read_csv(RANKING_CSV)

stable_feats = rank_df[
    (rank_df['Top20_Count_C1C4'] >= 3) &
    (rank_df['Stable'] == True)
]['Feature'].tolist()

print(f"  Stable features selected (Top20_Count≥3 AND Stable=True): {len(stable_feats)}")
print(f"  Features: {stable_feats}")

# ── Step 2: load data ─────────────────────────────────────────────────────────
print("\nStep 2 — Loading datasets...")
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
print(f"  LycoS18 : {df18.shape}")

all_features = [c for c in df17.columns if c != 'label']

missing = [f for f in stable_feats if f not in all_features]
if missing:
    print(f"  ⚠️ Missing features in dataset: {missing}")
    stable_feats = [f for f in stable_feats if f in all_features]
    print(f"  Adjusted stable features: {len(stable_feats)}")

# ── Step 3: recreate splits and scalers ──────────────────────────────────────
print("\nStep 3 — Recreating splits and scalers...")

X17_all = df17.drop(columns=['label']).values
y17     = df17['label'].values
X18_all = df18.drop(columns=['label']).values
y18     = df18['label'].values

X17_train_idx, X17_test_idx = train_test_split(
    range(len(df17)), test_size=0.2, random_state=42, stratify=y17)[0], \
    train_test_split(range(len(df17)), test_size=0.2, random_state=42, stratify=y17)[1]

X18_train_idx, X18_test_idx = train_test_split(
    range(len(df18)), test_size=0.2, random_state=42, stratify=y18)[0], \
    train_test_split(range(len(df18)), test_size=0.2, random_state=42, stratify=y18)[1]

scaler17 = MinMaxScaler()
scaler18 = MinMaxScaler()

X17_train_full = scaler17.fit_transform(X17_all[X17_train_idx])
X17_test_full  = scaler17.transform(X17_all[X17_test_idx])
y17_train      = y17[X17_train_idx]
y17_test       = y17[X17_test_idx]

X18_train_full = scaler18.fit_transform(X18_all[X18_train_idx])
X18_test_full  = scaler18.transform(X18_all[X18_test_idx])
y18_train      = y18[X18_train_idx]
y18_test       = y18[X18_test_idx]

X18_test_scaled17 = scaler17.transform(X18_all[X18_test_idx])
X17_test_scaled18 = scaler18.transform(X17_all[X17_test_idx])

stable_idx = [all_features.index(f) for f in stable_feats]

X17_train = X17_train_full[:, stable_idx]
X18_train = X18_train_full[:, stable_idx]
X17_test  = X17_test_full[:, stable_idx]
X18_test  = X18_test_full[:, stable_idx]
X18_test_s17 = X18_test_scaled17[:, stable_idx]
X17_test_s18 = X17_test_scaled18[:, stable_idx]

print(f"  ✅ Feature subset shape: {X17_train.shape[1]} features")

# ── Step 4: load best hyperparameters ─────────────────────────────────────────
print("\nStep 4 — Loading best hyperparameters...")
with open(PARAMS_FILE, 'r') as f:
    best_params = json.load(f)

rf_params_17  = best_params['LycoS17']['Random Forest']['params']
xgb_params_17 = best_params['LycoS17']['XGBoost']['params']
rf_params_18  = best_params['LycoS18']['Random Forest']['params']
xgb_params_18 = best_params['LycoS18']['XGBoost']['params']

print(f"  RF  params (LycoS17 / C3) : {rf_params_17}")
print(f"  XGB params (LycoS17 / C3) : {xgb_params_17}")
print(f"  RF  params (LycoS18 / C4) : {rf_params_18}")
print(f"  XGB params (LycoS18 / C4) : {xgb_params_18}")

# ── Step 5: retrain on stable features and evaluate C3, C4 ───────────────────
def eval_metrics(y_true, y_pred):
    return {
        'Accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1':        round(f1_score(y_true, y_pred, zero_division=0), 4),
        'MCC':       round(matthews_corrcoef(y_true, y_pred), 4),
    }

combos = {
    'C3': (X17_train, y17_train, X18_test_s17, y18_test),
    'C4': (X18_train, y18_train, X17_test_s18, y17_test),
}

param_map = {
    'C3': {'Random Forest': rf_params_17,  'XGBoost': xgb_params_17},
    'C4': {'Random Forest': rf_params_18,  'XGBoost': xgb_params_18},
}

print("\nStep 5 — Retraining and evaluating C3 and C4 on stable feature subset...")
results = []

for exp_name, (X_tr, y_tr, X_te, y_te) in combos.items():
    for model_name in ['Random Forest', 'XGBoost']:
        params = param_map[exp_name][model_name]

        if model_name == 'Random Forest':
            clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        else:
            clf = XGBClassifier(**params, random_state=42,
                                eval_metric='logloss', verbosity=0)

        print(f"  Training {model_name} on {exp_name} ({X_tr.shape[1]} features)...", end=' ')
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        metrics = eval_metrics(y_te, y_pred)
        print(f"MCC = {metrics['MCC']:.4f}")

        results.append({
            'Experiment': exp_name,
            'Model':      model_name,
            **metrics,
        })

# ── Step 6: save results and build comparison table ───────────────────────────
filtered_df = pd.DataFrame(results)
filtered_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Filtered results saved → {OUTPUT_CSV}")

tuned_df   = pd.read_csv(TUNED_CSV)

# use exact model name strings as they appear in tuned_results.csv
tuned_c3c4 = tuned_df[
    (tuned_df['Experiment'].isin(['C3', 'C4'])) &
    (tuned_df['Model'].isin(['Random Forest', 'XGBoost']))
][['Experiment', 'Model', 'MCC']].rename(columns={'MCC': 'MCC_original'})

compare = tuned_c3c4.merge(
    filtered_df[['Experiment', 'Model', 'MCC']].rename(columns={'MCC': 'MCC_filtered'}),
    on=['Experiment', 'Model']
)
compare['MCC_delta'] = (compare['MCC_filtered'] - compare['MCC_original']).round(4)

print("\nComparison: Original tuned MCC vs SHAP-filtered MCC (C3 and C4):")
print(compare.to_string(index=False))

compare.to_csv(COMPARE_CSV, index=False)
print(f"\n✅ Comparison table saved → {COMPARE_CSV}")

# ── Step 7: bar chart comparison ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

for ax, exp in zip(axes, ['C3', 'C4']):
    sub = compare[compare['Experiment'] == exp]
    x = np.arange(len(sub))
    width = 0.35
    ax.bar(x - width/2, sub['MCC_original'], width, label='Original (tuned)', color='steelblue')
    ax.bar(x + width/2, sub['MCC_filtered'], width, label='SHAP-filtered', color='seagreen')
    ax.set_xticks(x)
    ax.set_xticklabels(sub['Model'], rotation=15)
    ax.set_title(f'{exp}: Original vs SHAP-filtered MCC')
    ax.set_ylabel('MCC')
    ax.legend()
    ax.set_ylim(bottom=min(0, sub[['MCC_original', 'MCC_filtered']].min().min() - 0.05))

plt.tight_layout()
bar_path = os.path.join(RESULTS_DIR, "shap_filtering_mcc_comparison.png")
plt.savefig(bar_path, dpi=150)
plt.close()
print(f"✅ Bar chart saved → {bar_path}")

print("\nStage 6 — SHAP-guided feature filtering complete!")