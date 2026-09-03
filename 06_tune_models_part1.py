import pandas as pd
import numpy as np
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, make_scorer
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
LYCOS17     = r"data\LycoS-IDS2017_FINAL.csv"
LYCOS18     = r"data\LycoS-Unicas-IDS2018_CLEANED.csv"
RESULTS_DIR = r"results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SAMPLE_SIZE_18 = 500_000
MCC_SCORER     = make_scorer(matthews_corrcoef)
CV_FOLDS       = 3
N_ITER         = 20

# ── Load & prepare datasets ───────────────────────────────────────────────────
print("Loading datasets...")
df17 = pd.read_csv(LYCOS17)
print(f"  LycoS17 loaded      : {df17.shape}")

df18_full = pd.read_csv(LYCOS18)
print(f"  LycoS18 full loaded : {df18_full.shape}")

total_18 = len(df18_full)
df18_parts = []
for label_val, group in df18_full.groupby('label'):
    frac = len(group) / total_18
    n_samples = min(int(SAMPLE_SIZE_18 * frac), len(group))
    df18_parts.append(group.sample(n=n_samples, random_state=42))
df18 = pd.concat(df18_parts, ignore_index=True)
del df18_full
print(f"  LycoS18 sampled     : {df18.shape}\n")

# ── Helper: prepare train split ───────────────────────────────────────────────
def get_train_split(df):
    X = df.drop(columns=['label']).values
    y = df['label'].values
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    return X_train, y_train

# ── Search spaces ─────────────────────────────────────────────────────────────
lda_params = [
    {'solver': 'svd'},
    {'solver': 'lsqr', 'shrinkage': None},
    {'solver': 'lsqr', 'shrinkage': 'auto'},
    {'solver': 'eigen', 'shrinkage': 'auto'},
]

dt_space = {
    'max_depth'        : [5, 10, 15, 20, 30, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf' : [1, 2, 4, 8],
    'max_features'     : ['sqrt', 'log2', None],
}

rf_space = {
    'n_estimators'     : [100, 200, 300, 500],
    'max_depth'        : [10, 20, 30, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'max_features'     : ['sqrt', 'log2'],
}

xgb_space = {
    'n_estimators'     : [100, 200, 300],
    'max_depth'        : [3, 5, 6, 8, 10],
    'learning_rate'    : [0.01, 0.05, 0.1, 0.2],
    'subsample'        : [0.6, 0.8, 1.0],
    'colsample_bytree' : [0.6, 0.8, 1.0],
}

# ── Tune LDA (manual grid) ────────────────────────────────────────────────────
def tune_lda(X_train, y_train, dataset_name):
    print(f"  Tuning LDA on {dataset_name}...")
    best_mcc    = -99
    best_params = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    for params in lda_params:
        try:
            clf    = LinearDiscriminantAnalysis(**params)
            scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                clf.fit(X_train[train_idx], y_train[train_idx])
                pred = clf.predict(X_train[val_idx])
                scores.append(matthews_corrcoef(y_train[val_idx], pred))
            mcc = np.mean(scores)
            if mcc > best_mcc:
                best_mcc    = mcc
                best_params = params
        except Exception:
            continue

    print(f"    Best LDA params : {best_params}  |  CV MCC = {best_mcc:.4f}")
    return best_params, best_mcc

# ── Tune with RandomizedSearchCV ──────────────────────────────────────────────
def tune_model(clf, param_space, X_train, y_train, clf_name, dataset_name):
    print(f"  Tuning {clf_name} on {dataset_name}...")
    start = time.time()
    cv    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator           = clf,
        param_distributions = param_space,
        n_iter              = N_ITER,
        scoring             = MCC_SCORER,
        cv                  = cv,
        random_state        = 42,
        n_jobs              = -1,
        verbose             = 0
    )
    search.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"    Best {clf_name} params : {search.best_params_}")
    print(f"    CV MCC = {search.best_score_:.4f}  |  Time = {elapsed:.1f}s")
    return search.best_params_, search.best_score_

# ── Run tuning for both datasets ──────────────────────────────────────────────
datasets = {
    'LycoS17': df17,
    'LycoS18': df18,
}

all_best_params = {}

for ds_name, df in datasets.items():
    print(f"\n{'='*60}")
    print(f"  Tuning on {ds_name}")
    print(f"{'='*60}")

    X_train, y_train = get_train_split(df)
    all_best_params[ds_name] = {}

    bp, score = tune_lda(X_train, y_train, ds_name)
    all_best_params[ds_name]['LDA'] = {'params': bp, 'cv_mcc': round(score, 4)}

    bp, score = tune_model(
        DecisionTreeClassifier(random_state=42),
        dt_space, X_train, y_train, 'Decision Tree', ds_name)
    all_best_params[ds_name]['Decision Tree'] = {'params': bp, 'cv_mcc': round(score, 4)}

    bp, score = tune_model(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        rf_space, X_train, y_train, 'Random Forest', ds_name)
    all_best_params[ds_name]['Random Forest'] = {'params': bp, 'cv_mcc': round(score, 4)}

    bp, score = tune_model(
        XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
        xgb_space, X_train, y_train, 'XGBoost', ds_name)
    all_best_params[ds_name]['XGBoost'] = {'params': bp, 'cv_mcc': round(score, 4)}

# ── Save results ──────────────────────────────────────────────────────────────
output_path = os.path.join(RESULTS_DIR, 'best_params.json')
with open(output_path, 'w') as f:
    json.dump(all_best_params, f, indent=4)

# ── Print final summary ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  TUNING SUMMARY — BEST CV MCC SCORES")
print(f"{'='*60}")
for ds_name, models in all_best_params.items():
    print(f"\n  {ds_name}:")
    for model_name, info in models.items():
        print(f"    {model_name:15s} | CV MCC = {info['cv_mcc']:.4f} | {info['params']}")

print(f"\n✅ Best parameters saved to: {output_path}")