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
LYCOS17     = r"D:\thesis_implementation\LycoS-IDS2017_FINAL.csv"
LYCOS18     = r"D:\thesis_implementation\LycoS-Unicas-IDS2018_CLEANED.csv"
RESULTS_DIR = r"D:\thesis_implementation\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TUNE_SAMPLE_18 = 200_000   # LycoS18 sampled to 200k
MCC_SCORER     = make_scorer(matthews_corrcoef)
CV_FOLDS       = 2
N_ITER         = 10
N_JOBS         = 2

# ── Pre-fill known LycoS17 results (from previous run) ───────────────────────
output_path = os.path.join(RESULTS_DIR, 'best_params.json')

known_results = {
    "LycoS17": {
        "LDA": {
            "params": {"solver": "svd"},
            "cv_mcc": 0.9416
        },
        "Decision Tree": {
            "params": {
                "max_depth": 30,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": None
            },
            "cv_mcc": 0.9984
        },
        "Random Forest": {
            "params": {
                "n_estimators": 500,
                "min_samples_split": 2,
                "min_samples_leaf": 2,
                "max_features": "log2",
                "max_depth": None
            },
            "cv_mcc": 0.9988
        }
    }
}

if not os.path.exists(output_path):
    with open(output_path, 'w') as f:
        json.dump(known_results, f, indent=4)
    print("✅ Pre-filled known LycoS17 results into best_params.json")
else:
    print("✅ best_params.json already exists — loading it.")

with open(output_path, 'r') as f:
    all_best_params = json.load(f)

# ── Load datasets ─────────────────────────────────────────────────────────────
def load_and_sample(path, sample_size, name, skip_sample=False):
    print(f"\nLoading {name}...")
    df = pd.read_csv(path)
    print(f"  Full shape : {df.shape}")
    if skip_sample:
        print(f"  Using full dataset (no sampling).")
        return df
    parts = []
    total = len(df)
    for label_val, group in df.groupby('label'):
        frac = len(group) / total
        n = min(int(sample_size * frac), len(group))
        parts.append(group.sample(n=n, random_state=42))
    sampled = pd.concat(parts, ignore_index=True)
    del df
    print(f"  Sampled    : {sampled.shape}")
    return sampled

df17 = load_and_sample(LYCOS17, TUNE_SAMPLE_18, 'LycoS17', skip_sample=True)
df18 = load_and_sample(LYCOS18, TUNE_SAMPLE_18, 'LycoS18', skip_sample=False)

# ── Helper: prepare train split ───────────────────────────────────────────────
def get_train_split(df):
    X = df.drop(columns=['label']).values
    y = df['label'].values
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    print(f"  Train split shape : {X_train.shape}")
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
    'n_estimators'     : [100, 200, 300],
    'max_depth'        : [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'max_features'     : ['sqrt', 'log2'],
}

xgb_space = {
    'n_estimators'     : [100, 200, 300],
    'max_depth'        : [3, 5, 6, 8],
    'learning_rate'    : [0.05, 0.1, 0.2],
    'subsample'        : [0.7, 0.9, 1.0],
    'colsample_bytree' : [0.7, 0.9, 1.0],
}

# ── Tune LDA ──────────────────────────────────────────────────────────────────
def tune_lda(X_train, y_train, dataset_name):
    print(f"\n  [LDA] Tuning on {dataset_name}...")
    best_mcc, best_params = -99, {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    for i, params in enumerate(lda_params):
        try:
            clf, scores = LinearDiscriminantAnalysis(**params), []
            for tr, val in cv.split(X_train, y_train):
                clf.fit(X_train[tr], y_train[tr])
                scores.append(matthews_corrcoef(y_train[val], clf.predict(X_train[val])))
            mcc = np.mean(scores)
            print(f"    Combo {i+1}/{len(lda_params)} | params={params} | MCC={mcc:.4f}")
            if mcc > best_mcc:
                best_mcc, best_params = mcc, params
        except Exception as e:
            print(f"    Combo {i+1} failed: {e}")
            continue
    print(f"  ✅ Best LDA : {best_params}  |  CV MCC = {best_mcc:.4f}")
    return best_params, best_mcc

# ── Tune with RandomizedSearchCV ──────────────────────────────────────────────
def tune_model(clf, param_space, X_train, y_train, clf_name, dataset_name):
    print(f"\n  [{clf_name}] Tuning on {dataset_name}...")
    print(f"    N_ITER={N_ITER}, CV={CV_FOLDS}-fold, n_jobs={N_JOBS}")
    start = time.time()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator           = clf,
        param_distributions = param_space,
        n_iter              = N_ITER,
        scoring             = MCC_SCORER,
        cv                  = cv,
        random_state        = 42,
        n_jobs              = N_JOBS,
        verbose             = 2        # shows each fit progress
    )
    search.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  ✅ Best {clf_name} : {search.best_params_}")
    print(f"     CV MCC = {search.best_score_:.4f}  |  Time = {elapsed:.1f}s")
    return search.best_params_, search.best_score_

# ── Main loop ─────────────────────────────────────────────────────────────────
datasets = {'LycoS17': df17, 'LycoS18': df18}

for ds_name, df in datasets.items():
    if ds_name not in all_best_params:
        all_best_params[ds_name] = {}

    print(f"\n{'='*60}")
    print(f"  Tuning on {ds_name}")
    print(f"{'='*60}")

    X_train, y_train = get_train_split(df)

    # LDA
    if 'LDA' not in all_best_params[ds_name]:
        bp, score = tune_lda(X_train, y_train, ds_name)
        all_best_params[ds_name]['LDA'] = {'params': bp, 'cv_mcc': round(score, 4)}
        with open(output_path, 'w') as f:
            json.dump(all_best_params, f, indent=4)
    else:
        info = all_best_params[ds_name]['LDA']
        print(f"\n  [LDA] Already done — skipping.")
        print(f"    ✅ Best LDA : {info['params']}  |  CV MCC = {info['cv_mcc']}")

    # Decision Tree
    if 'Decision Tree' not in all_best_params[ds_name]:
        bp, score = tune_model(
            DecisionTreeClassifier(random_state=42),
            dt_space, X_train, y_train, 'Decision Tree', ds_name)
        all_best_params[ds_name]['Decision Tree'] = {'params': bp, 'cv_mcc': round(score, 4)}
        with open(output_path, 'w') as f:
            json.dump(all_best_params, f, indent=4)
    else:
        info = all_best_params[ds_name]['Decision Tree']
        print(f"\n  [Decision Tree] Already done — skipping.")
        print(f"    ✅ Best DT  : {info['params']}  |  CV MCC = {info['cv_mcc']}")

    # Random Forest
    if 'Random Forest' not in all_best_params[ds_name]:
        bp, score = tune_model(
            RandomForestClassifier(random_state=42, n_jobs=N_JOBS, max_samples=0.5),
            rf_space, X_train, y_train, 'Random Forest', ds_name)
        all_best_params[ds_name]['Random Forest'] = {'params': bp, 'cv_mcc': round(score, 4)}
        with open(output_path, 'w') as f:
            json.dump(all_best_params, f, indent=4)
    else:
        info = all_best_params[ds_name]['Random Forest']
        print(f"\n  [Random Forest] Already done — skipping.")
        print(f"    ✅ Best RF  : {info['params']}  |  CV MCC = {info['cv_mcc']}")

    # XGBoost
    if 'XGBoost' not in all_best_params[ds_name]:
        bp, score = tune_model(
            XGBClassifier(
                random_state = 42,
                eval_metric  = 'logloss',
                verbosity    = 0,
                tree_method  = 'hist',
                n_jobs       = 1
            ),
            xgb_space, X_train, y_train, 'XGBoost', ds_name)
        all_best_params[ds_name]['XGBoost'] = {'params': bp, 'cv_mcc': round(score, 4)}
        with open(output_path, 'w') as f:
            json.dump(all_best_params, f, indent=4)
    else:
        info = all_best_params[ds_name]['XGBoost']
        print(f"\n  [XGBoost] Already done — skipping.")
        print(f"    ✅ Best XGB : {info['params']}  |  CV MCC = {info['cv_mcc']}")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  TUNING SUMMARY — BEST CV MCC SCORES")
print(f"{'='*60}")
for ds_name, models in all_best_params.items():
    print(f"\n  {ds_name}:")
    for model_name, info in models.items():
        print(f"    {model_name:15s} | CV MCC = {info['cv_mcc']:.4f} | {info['params']}")

print(f"\n✅ All done! Best parameters saved to: {output_path}")