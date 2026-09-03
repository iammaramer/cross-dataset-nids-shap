import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
LYCOS17      = r"data\LycoS-IDS2017_FINAL.csv"
LYCOS18      = r"data\LycoS-Unicas-IDS2018_CLEANED.csv"
SHAP_DIR     = r"results\shap"
RESULTS_DIR  = r"results"
PARAMS_FILE  = os.path.join(RESULTS_DIR, "best_params.json")
RANKING_CSV  = os.path.join(SHAP_DIR, "shap_feature_ranking_RF_C1C4.csv")
TUNED_CSV    = os.path.join(RESULTS_DIR, "tuned_results.csv")
OUTPUT_CSV   = os.path.join(RESULTS_DIR, "shap_filtered_LDA_DT_results.csv")
COMPARE_CSV  = os.path.join(RESULTS_DIR, "shap_filtering_LDA_DT_comparison.csv")
SAMPLE_18    = 500_000

os.makedirs(RESULTS_DIR, exist_ok=True)
RANDOM_STATE = 42

# ── Step 1: identify stable features (same as RF/XGB script) ─────────────────
print("Step 1 — Loading SHAP feature ranking and selecting stable features...")
rank_df = pd.read_csv(RANKING_CSV)

stable_feats = rank_df[
    (rank_df['Top20_Count_C1C4'] >= 3) &
    (rank_df['Stable'] == True)
]['Feature'].tolist()

print(f"  Stable features selected (Top20_Count≥3 AND Stable=True): {len(stable_feats)}")
print(f"  Features: {stable_feats}")

# ── Step 2: load data ────────────────────────────────────────────────────────
print("\nStep 2 — Loading datasets...")
df17 = pd.read_csv(LYCOS17)
print(f"  LycoS17 : {df17.shape}")

df18_full = pd.read_csv(LYCOS18)
parts = []
total = len(df18_full)
for label_val, group in df18_full.groupby('label'):
    frac = len(group) / total
    n = min(int(SAMPLE_18 * frac), len(group))
    parts.append(group.sample(n=n, random_state=RANDOM_STATE))
df18 = pd.concat(parts, ignore_index=True)
del df18_full
print(f"  LycoS18 : {df18.shape}")

all_features = [c for c in df17.columns if c != 'label']

missing = [f for f in stable_feats if f not in all_features]
if missing:
    print(f"  ⚠️ Missing features in dataset: {missing}")
    stable_feats = [f for f in stable_feats if f in all_features]
    print(f"  Adjusted stable features: {len(stable_feats)}")

# ── Step 3: recreate splits and scalers (copying structure) ──────────────────
print("\nStep 3 — Recreating splits and scalers...")

X17_all = df17.drop(columns=['label']).values
y17     = df17['label'].values
X18_all = df18.drop(columns=['label']).values
y18     = df18['label'].values

X17_train_idx, X17_test_idx = train_test_split(
    np.arange(len(df17)),
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y17
)

X18_train_idx, X18_test_idx = train_test_split(
    np.arange(len(df18)),
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y18
)

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

# Cross-dataset scaling
X18_test_scaled17 = scaler17.transform(X18_all[X18_test_idx])
X17_test_scaled18 = scaler18.transform(X17_all[X17_test_idx])

# SHAP-stable subset indices
stable_idx = [all_features.index(f) for f in stable_feats]

X17_train = X17_train_full[:, stable_idx]
X18_train = X18_train_full[:, stable_idx]
X17_test  = X17_test_full[:, stable_idx]
X18_test  = X18_test_full[:, stable_idx]
X18_test_s17 = X18_test_scaled17[:, stable_idx]
X17_test_s18 = X17_test_scaled18[:, stable_idx]

print(f"  ✅ Feature subset shape: {X17_train.shape[1]} features")

# ── Step 4: load best hyperparameters for LDA and DT ─────────────────────────
print("\nStep 4 — Loading best hyperparameters for LDA and Decision Tree...")
with open(PARAMS_FILE, 'r') as f:
    best_params = json.load(f)

lda_params_17 = best_params['LycoS17']['LDA']['params']
dt_params_17  = best_params['LycoS17']['Decision Tree']['params']
lda_params_18 = best_params['LycoS18']['LDA']['params']
dt_params_18  = best_params['LycoS18']['Decision Tree']['params']

print(f"  LDA params (LycoS17 / C3): {lda_params_17}")
print(f"  DT  params (LycoS17 / C3): {dt_params_17}")
print(f"  LDA params (LycoS18 / C4): {lda_params_18}")
print(f"  DT  params (LycoS18 / C4): {dt_params_18}")

# ── Step 5: helper for metrics ───────────────────────────────────────────────
def eval_metrics(y_true, y_pred):
    return {
        'Accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1':        round(f1_score(y_true, y_pred, zero_division=0), 4),
        'MCC':       round(matthews_corrcoef(y_true, y_pred), 4),
    }

# ── Step 6: retrain LDA/DT on full vs stable features and evaluate C3, C4 ────
print("\nStep 6 — Retraining and evaluating LDA & DT on full vs stable features...")

# Full-feature matrices for C3 and C4 (using original scalers)
X17_train_full_c3 = X17_train_full
X18_test_full_s17 = X18_test_scaled17
X18_train_full_c4 = X18_train_full
X17_test_full_s18 = X17_test_scaled18

combos_full = {
    'C3': (X17_train_full_c3, y17_train, X18_test_full_s17, y18_test,
           lda_params_17, dt_params_17),
    'C4': (X18_train_full_c4, y18_train, X17_test_full_s18, y17_test,
           lda_params_18, dt_params_18),
}

results = []

for exp_name, (X_tr_full, y_tr, X_te_full, y_te,
               lda_params, dt_params) in combos_full.items():

    # LDA full
    lda_full = LinearDiscriminantAnalysis(**lda_params)
    lda_full.fit(X_tr_full, y_tr)
    y_pred = lda_full.predict(X_te_full)
    metrics_full_lda = eval_metrics(y_te, y_pred)

    # LDA stable
    X_tr_stable = X_tr_full[:, stable_idx]
    X_te_stable = X_te_full[:, stable_idx]
    lda_stable = LinearDiscriminantAnalysis(**lda_params)
    lda_stable.fit(X_tr_stable, y_tr)
    y_pred_s = lda_stable.predict(X_te_stable)
    metrics_stable_lda = eval_metrics(y_te, y_pred_s)

    print(f"{exp_name} | LDA | full vs stable → "
          f"MCC_full={metrics_full_lda['MCC']:.4f}, "
          f"MCC_stable={metrics_stable_lda['MCC']:.4f}")

    results.append({
        'Experiment': exp_name,
        'Model': 'LDA',
        'Feature_Set': 'full',
        **metrics_full_lda
    })
    results.append({
        'Experiment': exp_name,
        'Model': 'LDA',
        'Feature_Set': 'SHAP_stable',
        **metrics_stable_lda
    })

    # Decision Tree full
    dt_full = DecisionTreeClassifier(random_state=RANDOM_STATE, **dt_params)
    dt_full.fit(X_tr_full, y_tr)
    y_pred = dt_full.predict(X_te_full)
    metrics_full_dt = eval_metrics(y_te, y_pred)

    # Decision Tree stable
    dt_stable = DecisionTreeClassifier(random_state=RANDOM_STATE, **dt_params)
    dt_stable.fit(X_tr_stable, y_tr)
    y_pred_s = dt_stable.predict(X_te_stable)
    metrics_stable_dt = eval_metrics(y_te, y_pred_s)

    print(f"{exp_name} | Decision Tree | full vs stable → "
          f"MCC_full={metrics_full_dt['MCC']:.4f}, "
          f"MCC_stable={metrics_stable_dt['MCC']:.4f}")

    results.append({
        'Experiment': exp_name,
        'Model': 'Decision Tree',
        'Feature_Set': 'full',
        **metrics_full_dt
    })
    results.append({
        'Experiment': exp_name,
        'Model': 'Decision Tree',
        'Feature_Set': 'SHAP_stable',
        **metrics_stable_dt
    })

filtered_df = pd.DataFrame(results)
filtered_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Filtered LDA/DT results saved → {OUTPUT_CSV}")

# ── Step 7: build comparison with tuned_results.csv ──────────────────────────
tuned_df = pd.read_csv(TUNED_CSV)

tuned_c3c4 = tuned_df[
    (tuned_df['Experiment'].isin(['C3', 'C4'])) &
    (tuned_df['Model'].isin(['LDA', 'Decision Tree']))
][['Experiment', 'Model', 'MCC']].rename(columns={'MCC': 'MCC_original'})

filtered_c3c4 = filtered_df[
    filtered_df['Feature_Set'] == 'SHAP_stable'
][['Experiment', 'Model', 'MCC']].rename(columns={'MCC': 'MCC_filtered'})

compare = tuned_c3c4.merge(filtered_c3c4, on=['Experiment', 'Model'])
compare['MCC_delta'] = (compare['MCC_filtered'] - compare['MCC_original']).round(4)

print("\nComparison: Original tuned MCC vs SHAP-filtered MCC (LDA/DT, C3 & C4):")
print(compare.to_string(index=False))

compare.to_csv(COMPARE_CSV, index=False)
print(f"\n✅ Comparison table saved → {COMPARE_CSV}")

# ── Step 8: bar chart comparison ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

for ax, exp in zip(axes, ['C3', 'C4']):
    sub = compare[compare['Experiment'] == exp]
    x = np.arange(len(sub))
    width = 0.35
    ax.bar(x - width/2, sub['MCC_original'], width,
           label='Original (tuned)', color='steelblue')
    ax.bar(x + width/2, sub['MCC_filtered'], width,
           label='SHAP-filtered', color='seagreen')
    ax.set_xticks(x)
    ax.set_xticklabels(sub['Model'], rotation=15)
    ax.set_title(f'{exp}: LDA/DT Original vs SHAP-filtered MCC')
    ax.set_ylabel('MCC')
    ax.legend()
    ax.set_ylim(bottom=min(0, sub[['MCC_original', 'MCC_filtered']].min().min() - 0.05))

plt.tight_layout()
bar_path = os.path.join(RESULTS_DIR, "shap_filtering_LDA_DT_mcc_comparison.png")
plt.savefig(bar_path, dpi=150)
plt.close()
print(f"✅ Bar chart saved → {bar_path}")

print("\nStage 6 — SHAP-guided feature filtering for LDA & DT complete!")