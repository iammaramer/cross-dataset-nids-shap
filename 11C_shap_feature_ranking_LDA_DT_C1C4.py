
import os
import pandas as pd

BASE_DIR    = r"D:\thesis_implementation"
RESULTS_DIR = os.path.join(BASE_DIR, "results", "shap")

models = ["LDA", "Decision_Tree"]
combos = ["C1", "C2", "C3", "C4"]

def build_ranking_table(model_name):
    dfs = []
    for combo in combos:
        fname = f"shap_importance_{combo}_{model_name}.csv"
        fpath = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing file: {fpath}")

        df = pd.read_csv(fpath)

        # Normalize column names just in case
        df = df.rename(columns={
            "Feature": "Feature",
            "Mean_SHAP": "Mean_SHAP",
            "Rank": "Rank"
        })

        df_combo = df[["Feature", "Rank", "Mean_SHAP"]].copy()
        df_combo = df_combo.rename(columns={
            "Rank":      f"Rank_{combo}",
            "Mean_SHAP": f"Mean_SHAP_{combo}"
        })

        dfs.append(df_combo)

    # Merge on feature
    out = dfs[0]
    for df_next in dfs[1:]:
        out = out.merge(df_next, on="Feature", how="outer")

    # Optional: fill missing ranks with large number & SHAP with 0
    for combo in combos:
        out[f"Rank_{combo}"] = out[f"Rank_{combo}"].fillna(9999).astype(int)
        out[f"Mean_SHAP_{combo}"] = out[f"Mean_SHAP_{combo}"].fillna(0.0)

    # Sort by average rank over C1–C4
    out["Rank_mean"] = out[[f"Rank_{c}" for c in combos]].mean(axis=1)
    out = out.sort_values("Rank_mean").reset_index(drop=True)

    # Recompute overall rank for convenience
    out["Global_Rank"] = out.index + 1

    # Put columns in a nice order
    cols = ["Global_Rank", "Feature"]
    for combo in combos:
        cols.append(f"Rank_{combo}")
        cols.append(f"Mean_SHAP_{combo}")
    cols.append("Rank_mean")
    out = out[cols]

    return out

if __name__ == "__main__":
    for model in models:
        print(f"Building feature ranking table for {model}...")
        ranking_df = build_ranking_table(model)
        out_name = f"shap_feature_ranking_{model}_C1C4.csv"
        out_path = os.path.join(RESULTS_DIR, out_name)
        ranking_df.to_csv(out_path, index=False)
        print(f"  ✅ Saved → {out_path}")

    print("\nAll ranking tables generated.")