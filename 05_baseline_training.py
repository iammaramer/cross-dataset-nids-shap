import pandas as pd
import numpy as np
import os
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
LYCOS17     = r"D:\thesis_implementation\LycoS-IDS2017_FINAL.csv"
LYCOS18     = r"D:\thesis_implementation\LycoS-Unicas-IDS2018_CLEANED.csv"
RESULTS_DIR = r"D:\thesis_implementation\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Sampling config for LycoS18 (Option A) ───────────────────────────────────
SAMPLE_SIZE_18 = 500_000  # stratified sample size

# ── Classifiers ───────────────────────────────────────────────────────────────
CLASSIFIERS = {
    "LDA"          : LinearDiscriminantAnalysis(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost"      : XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
}

# ── Load datasets ─────────────────────────────────────────────────────────────
print("Loading datasets...")
df17 = pd.read_csv(LYCOS17)
print(f"  LycoS17 loaded      : {df17.shape}")

df18_full = pd.read_csv(LYCOS18)
print(f"  LycoS18 full loaded : {df18_full.shape}")

# Stratified sampling of LycoS18 to SAMPLE_SIZE_18
print(f"\nSampling LycoS18 down to {SAMPLE_SIZE_18:,} rows (stratified)...")
total_18 = len(df18_full)
df18_parts = []
for label_val, group in df18_full.groupby('label'):
    frac = len(group) / total_18
    n_samples = int(SAMPLE_SIZE_18 * frac)
    n_samples = min(n_samples, len(group))  # safety
    df18_parts.append(group.sample(n=n_samples, random_state=42))

df18 = pd.concat(df18_parts, ignore_index=True)
del df18_full  # free RAM

print(f"  LycoS18 sampled     : {df18.shape}")
print("  LycoS18 sample label distribution:")
print(df18['label'].value_counts(normalize=True))
print()

# ── Helper: split features and label ─────────────────────────────────────────
def get_Xy(df):
    X = df.drop(columns=['label']).values
    y = df['label'].values
    return X, y

# ── Helper: run one experiment ────────────────────────────────────────────────
def run_experiment(combo_name, train_df, test_df, clf_name, clf, within=False):
    X_train, y_train = get_Xy(train_df)
    X_test,  y_test  = get_Xy(test_df)

    if within:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )

    # MinMax scaling — fit on TRAIN only
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    start = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    elapsed = time.time() - start

    mcc  = matthews_corrcoef(y_test, y_pred)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)

    print(f"    [{combo_name}] {clf_name:15s} | "
          f"MCC={mcc:.4f} | Acc={acc:.4f} | F1={f1:.4f} | Time={elapsed:.1f}s")

    return {
        "Combination": combo_name,
        "Type"       : "Within" if within else "Cross",
        "Classifier" : clf_name,
        "MCC"        : round(mcc,  4),
        "Accuracy"   : round(acc,  4),
        "F1"         : round(f1,   4),
        "Precision"  : round(prec, 4),
        "Recall"     : round(rec,  4),
        "Time_sec"   : round(elapsed, 1)
    }

# ── Define 4 combinations ─────────────────────────────────────────────────────
combinations = [
    ("C1: LycoS17→LycoS17", df17, df17, True),   # within
    ("C2: LycoS18→LycoS18", df18, df18, True),   # within (sampled)
    ("C3: LycoS17→LycoS18", df17, df18, False),  # cross
    ("C4: LycoS18→LycoS17", df18, df17, False),  # cross
]

# ── Run all 16 experiments ────────────────────────────────────────────────────
results  = []
csv_path = os.path.join(RESULTS_DIR, "baseline_results.csv")

print("\nRunning 16 experiments...\n")

for combo_name, train_df, test_df, within in combinations:
    print(f"  {combo_name}")
    for clf_name, clf in CLASSIFIERS.items():
        row = run_experiment(
            combo_name=combo_name,
            train_df=train_df,
            test_df=test_df,
            clf_name=clf_name,
            clf=clf,
            within=within
        )
        results.append(row)
        # Save progress after each single run
        pd.DataFrame(results).to_csv(csv_path, index=False)
    print()

# ── Final summary table ───────────────────────────────────────────────────────
results_df = pd.DataFrame(results)

print("\n" + "="*75)
print("  BASELINE RESULTS SUMMARY — MCC SCORES")
print("="*75)
pivot = results_df.pivot(index="Combination", columns="Classifier", values="MCC")
pivot = pivot[["LDA", "Decision Tree", "Random Forest", "XGBoost"]]
print(pivot.to_string())
print("="*75)
print(f"\n✅ Full results saved to: {csv_path}")