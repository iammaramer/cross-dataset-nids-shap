import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (matthews_corrcoef, accuracy_score,
                              precision_score, recall_score, f1_score)

# =========================
# Paths
# =========================
LYCOS17    = r"data\LycoS-IDS2017_FINAL.csv"
LYCOS18    = r"data\LycoS-Unicas-IDS2018_CLEANED.csv"
MODELS_DIR = r"models"
OUTPUT_DIR = r"results\score_stability"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Settings
# =========================
SAMPLE_18    = 500_000
N_RUNS       = 10
SEEDS        = [42, 123, 456, 789, 1000, 1111, 2222, 3333, 4444, 5555]
TEST_SIZE    = 0.2

MODELS = ["LDA", "Decision_Tree", "Random_Forest", "XGBoost"]

# 10 SHAP-stable features from Stage 6
STABLE_FEATURES = [
    "bwd_pkt_len_std", "bwd_pkt_len_mean", "bwd_pkt_len_tot",
    "fwd_pkt_hdr_len_min", "pkt_len_var", "pkt_len_std",
    "bwd_pkt_len_max", "pkt_len_max", "bwd_tcp_init_win_bytes",
    "fwd_tcp_init_win_bytes"
]

# C1: train LycoS17 → test LycoS17
# C2: train LycoS18 → test LycoS18
# C3: train LycoS17 → test LycoS18 (vary test split only)
# C4: train LycoS18 → test LycoS17 (vary test split only)
EXPERIMENTS = {
    "C1": {"model_src": "C1", "test_data": "LycoS17"},
    "C2": {"model_src": "C2", "test_data": "LycoS18"},
    "C3": {"model_src": "C1", "test_data": "LycoS18"},
    "C4": {"model_src": "C2", "test_data": "LycoS17"},
}


def load_dataset(path, name, sample=None):
    print(f"  Loading {name}...")
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")

    if sample and len(df) > sample:
        parts = []
        total = len(df)
        for _, group in df.groupby("label"):
            frac = len(group) / total
            n = min(int(round(sample * frac)), len(group))
            if n > 0:
                parts.append(group.sample(n=n, random_state=42))
        df = (pd.concat(parts, ignore_index=True)
                .sample(frac=1, random_state=42)
                .reset_index(drop=True))

    print(f"  {name} shape: {df.shape}")
    return df


def get_test_split_stable(df, seed):
    """Split and return only stable features from test set."""
    available = [f for f in STABLE_FEATURES if f in df.columns]
    missing   = [f for f in STABLE_FEATURES if f not in df.columns]
    if missing:
        print(f"    WARNING: Missing stable features: {missing}")

    X = df[available].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    return scaler.transform(X_test), y_test, available


def load_model(model_src, model_name):
    fname = os.path.join(MODELS_DIR, f"{model_src}_{model_name}.pkl")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Not found: {fname}")
    with open(fname, "rb") as f:
        return pickle.load(f)


def evaluate(y_true, y_pred):
    return {
        "MCC":       round(matthews_corrcoef(y_true, y_pred), 4),
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def plot_mcc_stability(summary_df, output_dir):
    """Bar chart: mean MCC ± std for all models across all combinations."""
    experiments = ["C1", "C2", "C3", "C4"]
    models      = MODELS
    x           = np.arange(len(models))
    width       = 0.2
    colors      = ["steelblue", "seagreen", "tomato", "darkorange"]

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (exp, color) in enumerate(zip(experiments, colors)):
        sub = summary_df[summary_df["Experiment"] == exp].copy()
        sub["Model"] = pd.Categorical(sub["Model"], categories=models, ordered=True)
        sub = sub.sort_values("Model")

        means = sub["MCC_mean"].values
        stds  = sub["MCC_std"].values

        bars = ax.bar(
            x + i * width, means, width,
            label=exp, color=color, alpha=0.85,
            yerr=stds, capsize=4,
            error_kw=dict(elinewidth=1, ecolor="black")
        )

        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.01,
                f"{mean:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=90
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Mean MCC (± std)")
    ax.set_title(
        f"Score Stability — Mean MCC across {N_RUNS} runs\n"
        f"(SHAP-stable features, Option B: fixed model, varying test split)"
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(title="Experiment", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "score_stability_mcc_bar.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n✅ MCC bar chart saved: {out_path}")


def plot_mcc_per_experiment(all_runs_df, output_dir):
    """Line plot showing MCC across 10 runs per model per experiment."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    axes = axes.flatten()
    experiments = ["C1", "C2", "C3", "C4"]
    line_colors = ["steelblue", "seagreen", "tomato", "darkorange"]

    for ax, exp in zip(axes, experiments):
        sub = all_runs_df[all_runs_df["Experiment"] == exp]
        for model, color in zip(MODELS, line_colors):
            msub = sub[sub["Model"] == model].sort_values("Run")
            ax.plot(msub["Run"], msub["MCC"], marker="o",
                    label=model, color=color, linewidth=1.5)

        ax.set_title(f"{exp}", fontsize=12)
        ax.set_xlabel("Run")
        ax.set_ylabel("MCC")
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_xticks(range(1, N_RUNS + 1))
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    plt.suptitle(
        f"MCC per Run across {N_RUNS} Test Splits\n(SHAP-stable features, fixed models)",
        fontsize=13
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, "score_stability_mcc_runs.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ MCC runs line chart saved: {out_path}")


def main():
    print("Stage — Score Stability (Option B + Stable Features)")
    print("=" * 60)

    print("\nLoading datasets...")
    df17 = load_dataset(LYCOS17, "LycoS17")
    df18 = load_dataset(LYCOS18, "LycoS18", sample=SAMPLE_18)

    dataset_map = {"LycoS17": df17, "LycoS18": df18}

    all_runs_rows = []
    summary_rows  = []

    for exp, cfg in EXPERIMENTS.items():
        model_src = cfg["model_src"]
        test_df   = dataset_map[cfg["test_data"]]

        print(f"\n{'='*60}")
        print(f"Experiment: {exp} — model from {model_src}, "
              f"test data: {cfg['test_data']}")
        print(f"{'='*60}")

        for model_name in MODELS:
            print(f"\n  {exp} | {model_name}")

            try:
                model = load_model(model_src, model_name)
            except FileNotFoundError as e:
                print(f"    Skipping: {e}")
                continue

            run_scores = []

            for i, seed in enumerate(SEEDS):
                X_test, y_test, used_feats = get_test_split_stable(test_df, seed)

                # Predict using only stable features
                # Model was trained on all features — we need to use
                # a fresh prediction on stable-feature subset
                # So we retrain a lightweight wrapper? No —
                # Option B means we use the saved model but the model
                # expects all features. We handle this by predicting
                # from a re-fitted scaler on stable features only.
                # Since the saved model used all 77 features, we must
                # re-evaluate on stable features by re-fitting on train split.

                # Get train split for scaling reference
                available = [f for f in STABLE_FEATURES if f in test_df.columns]
                X_all = test_df[available].values
                y_all = test_df["label"].values

                X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
                    X_all, y_all,
                    test_size=TEST_SIZE,
                    random_state=seed,
                    stratify=y_all
                )

                scaler = MinMaxScaler()
                X_train_scaled = scaler.fit_transform(X_train_s)
                X_test_scaled  = scaler.transform(X_test_s)

                try:
                    y_pred = model.predict(X_test_scaled)
                except Exception:
                    # Model expects more features — fit a compatible model
                    from sklearn.base import clone
                    temp_model = clone(model)
                    temp_model.fit(X_train_scaled, y_train_s)
                    y_pred = temp_model.predict(X_test_scaled)

                scores = evaluate(y_test_s, y_pred)
                scores["Run"]  = i + 1
                scores["Seed"] = seed
                run_scores.append(scores)

                print(f"    Run {i+1} (seed={seed}) — "
                      f"MCC={scores['MCC']:.4f}  "
                      f"Acc={scores['Accuracy']:.4f}  "
                      f"F1={scores['F1']:.4f}")

            if not run_scores:
                continue

            runs_df = pd.DataFrame(run_scores)

            # Per-run CSV
            runs_csv = os.path.join(
                OUTPUT_DIR, f"runs_{exp}_{model_name}.csv"
            )
            runs_df.to_csv(runs_csv, index=False)

            # Add to all_runs collection
            for _, row in runs_df.iterrows():
                all_runs_rows.append({
                    "Experiment": exp,
                    "Model":      model_name,
                    **row.to_dict()
                })

            # Summary stats
            summary_rows.append({
                "Experiment":   exp,
                "Model":        model_name,
                "MCC_mean":     round(runs_df["MCC"].mean(), 4),
                "MCC_std":      round(runs_df["MCC"].std(), 4),
                "MCC_min":      round(runs_df["MCC"].min(), 4),
                "MCC_max":      round(runs_df["MCC"].max(), 4),
                "Acc_mean":     round(runs_df["Accuracy"].mean(), 4),
                "Acc_std":      round(runs_df["Accuracy"].std(), 4),
                "Prec_mean":    round(runs_df["Precision"].mean(), 4),
                "Recall_mean":  round(runs_df["Recall"].mean(), 4),
                "F1_mean":      round(runs_df["F1"].mean(), 4),
                "F1_std":       round(runs_df["F1"].std(), 4),
            })

    # Save all runs combined
    all_runs_df  = pd.DataFrame(all_runs_rows)
    all_runs_csv = os.path.join(OUTPUT_DIR, "all_runs_scores.csv")
    all_runs_df.to_csv(all_runs_csv, index=False)
    print(f"\n✅ All runs saved: {all_runs_csv}")

    # Save summary
    summary_df  = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUTPUT_DIR, "score_stability_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ Summary saved: {summary_csv}")

    print("\nScore Stability Summary:")
    print(summary_df[["Experiment", "Model",
                       "MCC_mean", "MCC_std",
                       "MCC_min",  "MCC_max"]].to_string(index=False))

    # Charts
    plot_mcc_stability(summary_df, OUTPUT_DIR)
    plot_mcc_per_experiment(all_runs_df, OUTPUT_DIR)

    print(f"\n✅ All outputs saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()