import pandas as pd
import numpy as np
import os

# Paths
RESULTS_DIR = r"D:\thesis_implementation\results\shap"
OUTPUT_CSV  = os.path.join(RESULTS_DIR, "shap_feature_ranking_RF_C1C4.csv")

# Helper to load one SHAP importance file and rename columns
def load_shap(experiment):
    path = os.path.join(RESULTS_DIR, f"shap_importance_{experiment}_Random_Forest.csv")
    df = pd.read_csv(path)
    # Expect columns: Feature, Mean_SHAP, Rank
    df = df[['Feature', 'Mean_SHAP', 'Rank']].copy()
    df.rename(columns={
        'Mean_SHAP': f'Mean_SHAP_{experiment}',
        'Rank':      f'Rank_{experiment}'
    }, inplace=True)
    return df

print("Loading global SHAP importance for Random Forest (C1–C4)...")

c1 = load_shap('C1')
c2 = load_shap('C2')
c3 = load_shap('C3')
c4 = load_shap('C4')

# Merge all on Feature
df = c1.merge(c2, on='Feature', how='outer') \
       .merge(c3, on='Feature', how='outer') \
       .merge(c4, on='Feature', how='outer')

# Ensure ranks are numeric (in case of NaN for missing features)
for exp in ['C1', 'C2', 'C3', 'C4']:
    df[f'Rank_{exp}'] = pd.to_numeric(df[f'Rank_{exp}'], errors='coerce')

# Compute helper metrics
rank_cols = [f'Rank_{e}' for e in ['C1', 'C2', 'C3', 'C4']]
shap_within_cols = ['Mean_SHAP_C1', 'Mean_SHAP_C2']
shap_cross_cols  = ['Mean_SHAP_C3', 'Mean_SHAP_C4']

# Mean rank across all experiments (ignoring NaN)
df['Mean_Rank_C1C4'] = df[rank_cols].mean(axis=1, skipna=True)

# Count how many times feature is in top-20 for each experiment
def count_top20(row):
    count = 0
    for col in rank_cols:
        r = row[col]
        if pd.notna(r) and r <= 20:
            count += 1
    return count

df['Top20_Count_C1C4'] = df.apply(count_top20, axis=1)

# Separate within vs cross SHAP averages
df['Mean_SHAP_within'] = df[shap_within_cols].mean(axis=1, skipna=True)
df['Mean_SHAP_cross']  = df[shap_cross_cols].mean(axis=1, skipna=True)

# Stability flag from shap_cross_dataset_comparison_RF.csv (C1 vs C2 stability)
stable_path = os.path.join(RESULTS_DIR, "shap_cross_dataset_comparison_RF.csv")
if os.path.exists(stable_path):
    stable = pd.read_csv(stable_path)
    # columns: Feature, In_LycoS17_Top20, In_LycoS18_Top20, Stable
    stable.rename(columns={
        'In_LycoS17_Top20': 'In_C1_Top20',
        'In_LycoS18_Top20': 'In_C2_Top20'
    }, inplace=True)
    df = df.merge(stable[['Feature', 'In_C1_Top20', 'In_C2_Top20', 'Stable']],
                  on='Feature', how='left')
else:
    df['In_C1_Top20'] = np.nan
    df['In_C2_Top20'] = np.nan
    df['Stable']      = np.nan

# Sort by:
# 1) Top20_Count_C1C4 (desc), then 2) Mean_Rank_C1C4 (asc), then 3) Mean_SHAP_within (desc)
df_sorted = df.sort_values(
    by=['Top20_Count_C1C4', 'Mean_Rank_C1C4', 'Mean_SHAP_within'],
    ascending=[False, True, False]
).reset_index(drop=True)

# Add overall rank index
df_sorted['Global_Rank'] = df_sorted.index + 1

# Reorder columns for readability
cols_order = [
    'Global_Rank',
    'Feature',
    'Rank_C1', 'Rank_C2', 'Rank_C3', 'Rank_C4',
    'Top20_Count_C1C4',
    'Mean_Rank_C1C4',
    'Mean_SHAP_C1', 'Mean_SHAP_C2', 'Mean_SHAP_C3', 'Mean_SHAP_C4',
    'Mean_SHAP_within', 'Mean_SHAP_cross',
    'In_C1_Top20', 'In_C2_Top20', 'Stable'
]
df_sorted = df_sorted[cols_order]

# Save
df_sorted.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Global SHAP feature ranking table saved to:\n  {OUTPUT_CSV}")

print("\nTop 15 features by Global_Rank:")
print(df_sorted.head(15).to_string(index=False))