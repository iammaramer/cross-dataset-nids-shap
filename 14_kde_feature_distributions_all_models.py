import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Paths
# =========================
LYCOS17 = r"D:\thesis_implementation\LycoS-IDS2017_FINAL.csv"
LYCOS18 = r"D:\thesis_implementation\LycoS-Unicas-IDS2018_CLEANED.csv"

RESULTS_DIR = r"D:\thesis_implementation\results"
SHAP_DIR = os.path.join(RESULTS_DIR, "shap")
RANKING_CSV = os.path.join(SHAP_DIR, "shap_feature_ranking_RF_C1C4.csv")

KDE_DIR = os.path.join(RESULTS_DIR, "kde_plots")
os.makedirs(KDE_DIR, exist_ok=True)

# =========================
# Settings
# =========================
RANDOM_STATE = 42
SAMPLE_17 = 300000
SAMPLE_18 = 300000
TOP_N_STABLE = 10
TOP_N_DRIFTING = 10

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

# These are already confirmed from your Stage 6 script output
KNOWN_STABLE = [
    'bwd_pkt_len_std', 'bwd_pkt_len_mean', 'bwd_pkt_len_tot',
    'fwd_pkt_hdr_len_min', 'pkt_len_var', 'pkt_len_std',
    'bwd_pkt_len_max', 'pkt_len_max',
    'bwd_tcp_init_win_bytes', 'fwd_tcp_init_win_bytes'
]


def stratified_sample(df, label_col="label", n_samples=300000, random_state=42):
    if len(df) <= n_samples:
        return df.copy()

    parts = []
    total = len(df)

    for label_val, group in df.groupby(label_col):
        frac = len(group) / total
        n = min(int(round(n_samples * frac)), len(group))
        if n > 0:
            parts.append(group.sample(n=n, random_state=random_state))

    sampled = pd.concat(parts, ignore_index=True)
    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)


def get_feature_lists(ranking_csv):
    rank_df = pd.read_csv(ranking_csv)
    rank_df.columns = [c.strip() for c in rank_df.columns]

    print("  Ranking CSV columns:", rank_df.columns.tolist())

    top20_col = None
    mean_rank_col = None

    for c in rank_df.columns:
        if "Top20" in c:
            top20_col = c
        if c == "Mean_Rank" or "Mean_Rank" in c:
            mean_rank_col = c

    if top20_col is None:
        raise ValueError("Could not find Top20 count column in ranking CSV.")
    if mean_rank_col is None:
        raise ValueError("Could not find Mean_Rank column in ranking CSV.")

    # Use hardcoded stable features confirmed from Stage 6
    stable_features = KNOWN_STABLE[:TOP_N_STABLE]

    # Drifting: not in stable list, highest Top20 count, lowest Mean_Rank
    drifting_df = rank_df[
        ~rank_df["Feature"].isin(stable_features)
    ].copy()
    drifting_df = drifting_df.sort_values(
        [top20_col, mean_rank_col],
        ascending=[False, True]
    )
    drifting_features = drifting_df["Feature"].head(TOP_N_DRIFTING).tolist()

    return stable_features, drifting_features


def clip_outliers_for_plot(series, lower_q=0.01, upper_q=0.99):
    s = pd.to_numeric(series, errors="coerce").dropna()

    if s.empty:
        return s

    low = s.quantile(lower_q)
    high = s.quantile(upper_q)

    return s[(s >= low) & (s <= high)]


def plot_single_kde(feature, df17, df18, group_name):
    s17 = clip_outliers_for_plot(df17[feature])
    s18 = clip_outliers_for_plot(df18[feature])

    if len(s17) < 10 or len(s18) < 10:
        print(f"  Skipping {feature}: not enough valid values after clipping.")
        return

    plt.figure(figsize=(8, 5))

    sns.kdeplot(s17, label="LycoS17", fill=True, alpha=0.30, linewidth=2, color="steelblue")
    sns.kdeplot(s18, label="LycoS18", fill=True, alpha=0.30, linewidth=2, color="darkorange")

    plt.title(f"KDE Distribution: {feature}")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(KDE_DIR, f"kde_{group_name}_{feature}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {out_path}")


def plot_grid(features, df17, df18, title, out_name):
    n = len(features)

    if n == 0:
        print(f"  No features available for {out_name}. Skipping grid.")
        return

    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.8 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax in axes[n:]:
        ax.axis("off")

    for i, feature in enumerate(features):
        ax = axes[i]

        s17 = clip_outliers_for_plot(df17[feature])
        s18 = clip_outliers_for_plot(df18[feature])

        if len(s17) < 10 or len(s18) < 10:
            ax.text(0.5, 0.5, f"{feature}\nNot enough data", ha="center", va="center")
            ax.set_axis_off()
            continue

        sns.kdeplot(s17, ax=ax, label="LycoS17", fill=True, alpha=0.30, linewidth=2, color="steelblue")
        sns.kdeplot(s18, ax=ax, label="LycoS18", fill=True, alpha=0.30, linewidth=2, color="darkorange")

        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")

        if i == 0:
            ax.legend()
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    fig.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = os.path.join(KDE_DIR, out_name)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"  Saved grid: {out_path}")


def save_feature_lists(stable_features, drifting_features):
    max_len = max(len(stable_features), len(drifting_features), 1)

    stable_pad = stable_features + [None] * (max_len - len(stable_features))
    drifting_pad = drifting_features + [None] * (max_len - len(drifting_features))

    out_df = pd.DataFrame({
        "Stable_Features": stable_pad,
        "Drifting_Features": drifting_pad
    })

    out_csv = os.path.join(KDE_DIR, "kde_feature_lists.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"  Saved feature list CSV: {out_csv}")


def load_dataset(path, name):
    print(f"  Loading {name}...")
    try:
        df = pd.read_csv(path)
    except Exception:
        print(f"  Default parser failed for {name}, retrying with engine='python'...")
        df = pd.read_csv(path, engine='python', on_bad_lines='skip')
    print(f"  {name} shape: {df.shape}")
    return df


def main():
    print("Stage 7 — KDE Distribution Plots (All Models)")
    print("=" * 60)

    print("\n1. Loading feature ranking...")
    stable_features, drifting_features = get_feature_lists(RANKING_CSV)

    print(f"  Stable features ({len(stable_features)}):")
    print(f"  {stable_features}")

    print(f"\n  Drifting features ({len(drifting_features)}):")
    print(f"  {drifting_features}")

    save_feature_lists(stable_features, drifting_features)

    print("\n2. Loading datasets...")
    df17 = load_dataset(LYCOS17, "LycoS17")
    df18 = load_dataset(LYCOS18, "LycoS18")

    print("\n3. Stratified sampling for plotting speed...")
    df17_plot = stratified_sample(df17, label_col="label", n_samples=SAMPLE_17, random_state=RANDOM_STATE)
    df18_plot = stratified_sample(df18, label_col="label", n_samples=SAMPLE_18, random_state=RANDOM_STATE)

    print(f"  Sampled LycoS17 shape: {df17_plot.shape}")
    print(f"  Sampled LycoS18 shape: {df18_plot.shape}")

    needed_features = stable_features + drifting_features
    missing_17 = [f for f in needed_features if f not in df17_plot.columns]
    missing_18 = [f for f in needed_features if f not in df18_plot.columns]

    if missing_17:
        raise ValueError(f"Missing features in LycoS17: {missing_17}")
    if missing_18:
        raise ValueError(f"Missing features in LycoS18: {missing_18}")

    print("\n4. Plotting individual KDEs — stable features...")
    for feature in stable_features:
        plot_single_kde(feature, df17_plot, df18_plot, "stable")

    print("\n5. Plotting individual KDEs — drifting features...")
    for feature in drifting_features:
        plot_single_kde(feature, df17_plot, df18_plot, "drifting")

    print("\n6. Plotting grid figures...")
    plot_grid(
        stable_features,
        df17_plot,
        df18_plot,
        "KDE Distributions of SHAP-Stable Features: LycoS17 vs LycoS18",
        "kde_stable_features_grid.png"
    )

    plot_grid(
        drifting_features,
        df17_plot,
        df18_plot,
        "KDE Distributions of SHAP-Drifting Features: LycoS17 vs LycoS18",
        "kde_drifting_features_grid.png"
    )

    print("\n✅ Done. All KDE plots saved in:")
    print(KDE_DIR)


if __name__ == "__main__":
    main()