import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix,
                             classification_report)
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
LYCOS17      = r"data\LycoS-IDS2017_FINAL.csv"
LYCOS18      = r"data\LycoS-Unicas-IDS2018_CLEANED.csv"
RESULTS_DIR  = r"results"
MODELS_DIR   = r"models"
PARAMS_PATH  = r"results\best_params.json"
SAMPLE_18    = 500_000

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# ── Load best params ──────────────────────────────────────────────────────────
print("Loading best hyperparameters...")
with open(PARAMS_PATH, 'r') as f:
    best_params = json.load(f)
print("  ✅ best_params.json loaded.\n")

# ── Load datasets ─────────────────────────────────────────────────────────────
print("Loading datasets...")
df17 = pd.read_csv(LYCOS17)
print(f"  LycoS17 loaded : {df17.shape}")

df18_full = pd.read_csv(LYCOS18)
print(f"  LycoS18 full   : {df18_full.shape}")

total_18 = len(df18_full)
parts = []
for label_val, group in df18_full.groupby('label'):
    frac = len(group) / total_18
    n = min(int(SAMPLE_18 * frac), len(group))
    parts.append(group.sample(n=n, random_state=42))
df18 = pd.concat(parts, ignore_index=True)
del df18_full
print(f"  LycoS18 sampled: {df18.shape}\n")

# ── Prepare splits ────────────────────────────────────────────────────────────
print("Preparing train/test splits...")

X17 = df17.drop(columns=['label']).values
y17 = df17['label'].values
X17_train, X17_test, y17_train, y17_test = train_test_split(
    X17, y17, test_size=0.2, random_state=42, stratify=y17)

X18 = df18.drop(columns=['label']).values
y18 = df18['label'].values
X18_train, X18_test, y18_train, y18_test = train_test_split(
    X18, y18, test_size=0.2, random_state=42, stratify=y18)

# Scale — fit on train, apply to test (no leakage)
scaler17 = MinMaxScaler()
X17_train = scaler17.fit_transform(X17_train)
X17_test  = scaler17.transform(X17_test)

scaler18 = MinMaxScaler()
X18_train = scaler18.fit_transform(X18_train)
X18_test  = scaler18.transform(X18_test)

# For cross-dataset: apply source scaler to target test set
X18_test_scaled17 = scaler17.transform(
    df18.drop(columns=['label']).values[
        train_test_split(range(len(df18)), test_size=0.2, random_state=42, 
                        stratify=y18)[1]
    ]
)
X17_test_scaled18 = scaler18.transform(
    df17.drop(columns=['label']).values[
        train_test_split(range(len(df17)), test_size=0.2, random_state=42,
                        stratify=y17)[1]
    ]
)

print("  ✅ Splits ready.\n")

# ── Build model instances with tuned params ───────────────────────────────────
def get_models(ds_name):
    p = best_params[ds_name]
    return {
        'LDA': LinearDiscriminantAnalysis(
            **p['LDA']['params']),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, **p['Decision Tree']['params']),
        'Random Forest': RandomForestClassifier(
            random_state=42, n_jobs=2, **p['Random Forest']['params']),
        'XGBoost': XGBClassifier(
            random_state=42, eval_metric='logloss',
            verbosity=0, tree_method='hist', n_jobs=1,
            **p['XGBoost']['params']),
    }

# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, exp_name, model_name):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc  = matthews_corrcoef(y_true, y_pred)
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1        : {f1:.4f}")
    print(f"    MCC       : {mcc:.4f}")
    return {
        'Experiment' : exp_name,
        'Model'      : model_name,
        'Accuracy'   : round(acc,  4),
        'Precision'  : round(prec, 4),
        'Recall'     : round(rec,  4),
        'F1'         : round(f1,   4),
        'MCC'        : round(mcc,  4),
    }

# ── Run all experiments ───────────────────────────────────────────────────────
all_results = []

# ── C1: Train LycoS17 → Test LycoS17 ─────────────────────────────────────────
print(f"\n{'='*60}")
print("  C1: Train=LycoS17  |  Test=LycoS17")
print(f"{'='*60}")
models_17 = get_models('LycoS17')
for name, clf in models_17.items():
    print(f"\n  [{name}] Training...")
    clf.fit(X17_train, y17_train)
    y_pred = clf.predict(X17_test)
    result = evaluate(y17_test, y_pred, 'C1', name)
    all_results.append(result)
    # Save model
    with open(os.path.join(MODELS_DIR, f'C1_{name.replace(" ","_")}.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    print(f"    ✅ Model saved.")

# ── C2: Train LycoS18 → Test LycoS18 ─────────────────────────────────────────
print(f"\n{'='*60}")
print("  C2: Train=LycoS18  |  Test=LycoS18")
print(f"{'='*60}")
models_18 = get_models('LycoS18')
for name, clf in models_18.items():
    print(f"\n  [{name}] Training...")
    clf.fit(X18_train, y18_train)
    y_pred = clf.predict(X18_test)
    result = evaluate(y18_test, y_pred, 'C2', name)
    all_results.append(result)
    with open(os.path.join(MODELS_DIR, f'C2_{name.replace(" ","_")}.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    print(f"    ✅ Model saved.")

# ── C3: Train LycoS17 → Test LycoS18 ─────────────────────────────────────────
print(f"\n{'='*60}")
print("  C3: Train=LycoS17  |  Test=LycoS18  (cross-dataset)")
print(f"{'='*60}")
models_17_c3 = get_models('LycoS17')
for name, clf in models_17_c3.items():
    print(f"\n  [{name}] Training on LycoS17, testing on LycoS18...")
    clf.fit(X17_train, y17_train)
    y_pred = clf.predict(X18_test_scaled17)
    result = evaluate(y18_test, y_pred, 'C3', name)
    all_results.append(result)
    with open(os.path.join(MODELS_DIR, f'C3_{name.replace(" ","_")}.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    print(f"    ✅ Model saved.")

# ── C4: Train LycoS18 → Test LycoS17 ─────────────────────────────────────────
print(f"\n{'='*60}")
print("  C4: Train=LycoS18  |  Test=LycoS17  (cross-dataset)")
print(f"{'='*60}")
models_18_c4 = get_models('LycoS18')
for name, clf in models_18_c4.items():
    print(f"\n  [{name}] Training on LycoS18, testing on LycoS17...")
    clf.fit(X18_train, y18_train)
    y_pred = clf.predict(X17_test_scaled18)
    result = evaluate(y17_test, y_pred, 'C4', name)
    all_results.append(result)
    with open(os.path.join(MODELS_DIR, f'C4_{name.replace(" ","_")}.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    print(f"    ✅ Model saved.")

# ── Save results to CSV ───────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
csv_path = os.path.join(RESULTS_DIR, 'tuned_results.csv')
results_df.to_csv(csv_path, index=False)
print(f"\n✅ Tuned results saved to: {csv_path}")

# ── Final summary table ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  FINAL SUMMARY — TUNED EXPERIMENTS")
print(f"{'='*60}")
pivot = results_df.pivot_table(
    index='Model', columns='Experiment', values='MCC')
print(pivot.to_string())
print(f"\n{'='*60}")
print("  MCC per Experiment per Model (higher = better)")
print(f"{'='*60}")
for exp in ['C1', 'C2', 'C3', 'C4']:
    exp_df = results_df[results_df['Experiment'] == exp].sort_values('MCC', ascending=False)
    print(f"\n  {exp}:")
    for _, row in exp_df.iterrows():
        print(f"    {row['Model']:15s} | MCC={row['MCC']:.4f} | "
              f"F1={row['F1']:.4f} | Acc={row['Accuracy']:.4f}")