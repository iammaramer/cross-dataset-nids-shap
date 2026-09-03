import pandas as pd
import numpy as np
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
LYCOS17     = r"data\LycoS-IDS2017_FINAL.csv"
LYCOS18     = r"data\LycoS-Unicas-IDS2018_CLEANED.csv"
PARAMS_PATH = r"results\best_params.json"
RESULTS_DIR = r"results\10runs_results"   # <-- new folder

STABLE_FEATURES = [
    "bwd_pkt_len_std", "bwd_pkt_len_mean", "bwd_pkt_len_tot",
    "fwd_pkt_hdr_len_min", "pkt_len_var", "pkt_len_std",
    "bwd_pkt_len_max", "pkt_len_max",
    "bwd_tcp_init_win_bytes", "fwd_tcp_init_win_bytes"
]

N_RUNS      = 10
SAMPLE_SIZE = 500_000
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load best_params.json ─────────────────────────────────────────────────────
print("Loading best_params.json ...")
with open(PARAMS_PATH, "r") as f:
    best_params = json.load(f)
print("  Loaded.")

# ── Build model from best_params ──────────────────────────────────────────────
def build_model(model_name, dataset_key, seed):
    p = best_params[dataset_key][model_name]["params"]
    if model_name == "LDA":
        kwargs = {k: v for k, v in p.items() if v is not None}
        return LinearDiscriminantAnalysis(**kwargs)
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(**p, random_state=seed)
    elif model_name == "Random Forest":
        return RandomForestClassifier(**p, random_state=seed, n_jobs=2)
    elif model_name == "XGBoost":
        return XGBClassifier(**p, random_state=seed,
                             eval_metric="logloss", verbosity=0,
                             tree_method="hist", n_jobs=1)
    raise ValueError(f"Unknown model: {model_name}")

# ── Stratified sample ─────────────────────────────────────────────────────────
def stratified_sample(df, n):
    total = len(df)
    parts = []
    for _, grp in df.groupby("label"):
        frac = len(grp) / total
        k = min(int(n * frac), len(grp))
        parts.append(grp.sample(n=k, random_state=42))
    return pd.concat(parts, ignore_index=True)

# ── Load datasets ─────────────────────────────────────────────────────────────
print("Loading datasets ...")
df17 = pd.read_csv(LYCOS17)
print(f"  LycoS17 loaded: {df17.shape}")

df18_full = pd.read_csv(LYCOS18)
df18 = stratified_sample(df18_full, SAMPLE_SIZE)
del df18_full
print(f"  LycoS18 sampled: {df18.shape}")

# Keep only stable features + label
COLS = STABLE_FEATURES + ["label"]
df17 = df17[[c for c in COLS if c in df17.columns]]
df18 = df18[[c for c in COLS if c in df18.columns]]
print(f"  Columns kept: {df17.shape[1]} (stable features + label)")

# ── Four combinations ─────────────────────────────────────────────────────────
combinations = [
    ("C1", df17, df17, True,  "LycoS17"),
    ("C2", df18, df18, True,  "LycoS18"),
    ("C3", df17, df18, False, "LycoS17"),
    ("C4", df18, df17, False, "LycoS18"),
]
MODEL_NAMES = ["LDA", "Decision Tree", "Random Forest", "XGBoost"]

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_CSV     = os.path.join(RESULTS_DIR, "stability_results.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "stability_summary.csv")

# ── Resume support ────────────────────────────────────────────────────────────
if os.path.exists(OUT_CSV):
    done_df   = pd.read_csv(OUT_CSV)
    done_keys = set(zip(done_df["Combo"], done_df["Model"], done_df["Run"]))
    all_rows  = done_df.to_dict("records")
    print(f"  Resuming — {len(done_df)} rows already saved.")
else:
    done_keys = set()
    all_rows  = []

# ── Main loop ─────────────────────────────────────────────────────────────────
total = len(combinations) * len(MODEL_NAMES) * N_RUNS
print(f"\nStarting: {total} total experiments  |  Already done: {len(all_rows)}\n")

for combo_name, train_df, test_df, within, param_key in combinations:
    for model_name in MODEL_NAMES:
        for run in range(1, N_RUNS + 1):

            key = (combo_name, model_name, run)
            if key in done_keys:
                print(f"  SKIP  {combo_name} | {model_name:15s} | run {run}")
                continue

            seed = run * 7  # different seed each run, fully reproducible

            # Prepare features and labels
            X_tr = train_df.drop(columns=["label"]).values
            y_tr = train_df["label"].values
            X_te = test_df.drop(columns=["label"]).values
            y_te = test_df["label"].values

            if within:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_tr, y_tr, test_size=0.2,
                    random_state=seed, stratify=y_tr
                )

            scaler = MinMaxScaler()
            X_tr   = scaler.fit_transform(X_tr)
            X_te   = scaler.transform(X_te)

            # Train and evaluate
            clf = build_model(model_name, param_key, seed)
            t0  = time.time()
            clf.fit(X_tr, y_tr)
            y_pred  = clf.predict(X_te)
            elapsed = time.time() - t0

            mcc = matthews_corrcoef(y_te, y_pred)
            acc = accuracy_score(y_te, y_pred)
            f1  = f1_score(y_te, y_pred, zero_division=0)

            row = {
                "Combo": combo_name, "Model": model_name, "Run": run,
                "MCC": round(mcc, 6), "Accuracy": round(acc, 6),
                "F1": round(f1, 6), "TimeSec": round(elapsed, 2)
            }
            all_rows.append(row)
            done_keys.add(key)

            # Save after every single run
            pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)

            print(f"  {combo_name} | {model_name:15s} | run {run:2d} "
                  f"| MCC={mcc:.4f}  Acc={acc:.4f}  F1={f1:.4f}  t={elapsed:.1f}s")

# ── Final summary ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_rows)
summary = (
    results_df.groupby(["Combo", "Model"])["MCC"]
    .agg(Mean_MCC="mean", Std_MCC="std", Min_MCC="min", Max_MCC="max")
    .round(4).reset_index()
)

print("\n" + "="*70)
print("STABILITY SUMMARY — MCC across 10 runs")
print("="*70)
print(summary.to_string(index=False))

summary.to_csv(SUMMARY_CSV, index=False)

print(f"\nAll 160 results → {OUT_CSV}")
print(f"Summary         → {SUMMARY_CSV}")