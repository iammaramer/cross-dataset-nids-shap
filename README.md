# Investigating Cross-Dataset Generalization in Machine Learning-Based Network Intrusion Detection Using SHAP-Based Explainability

This repository contains the full implementation code for my master's thesis
in Cybersecurity, evaluating the cross-dataset generalization of machine
learning models for network intrusion detection, with a SHAP-based
explainability analysis conducted on the LycoS-IDS2017 and
LycoS-Unicas-IDS2018 datasets.

## Author

Ammar Amer — Master's degree, Bahçeşehir University

## Repository structure

    .
    ├── 01_env_setup_and_preprocessing.ipynb   # Environment setup, dataset merging & cleaning
    ├── 02_dataset_verify.ipynb                # Verification of preprocessed data
    ├── 05_baseline_training.py                # Baseline model training
    ├── 06_tune_models_part1.py                # Hyperparameter tuning (part 1)
    ├── 06_tune_models_part2.py                # Hyperparameter tuning (part 2)
    ├── 07_tuned_experiments.py                # Experiments with tuned models
    ├── 08A_shap_analysis_RF_XGB.py            # Global SHAP analysis — RF & XGBoost
    ├── 08B_shap_analysis_DT_LDA.py            # Global SHAP analysis — DT & LDA
    ├── 09_shap_cross_C3C4.py                  # SHAP analysis, cross-dataset (C3/C4)
    ├── 10_shap_perclass_RF_XGB.py             # Per-class SHAP — RF & XGBoost
    ├── 10B_shap_perclass_LDA_DT.py            # Per-class SHAP — LDA & DT
    ├── 10C_shap_perclass_cross_C3C4.py        # Per-class SHAP, cross-dataset (C3/C4)
    ├── 11_shap_feature_rank_RF.py             # SHAP feature ranking — RF
    ├── 11b_shap_feature_rank_xgb.py           # SHAP feature ranking — XGBoost
    ├── 11C_shap_feature_ranking_LDA_DT_C1C4.py# SHAP feature ranking — LDA & DT (C1-C4)
    ├── 12_shap_stability_RF.py                # SHAP stability analysis — RF
    ├── 12b_shap_stability_xgb.py              # SHAP stability analysis — XGBoost
    ├── 12C_shap_stability_LDA_DT.py           # SHAP stability analysis — LDA & DT
    ├── 12D_combined_stability_heatmap.py      # Combined stability heatmap, all models
    ├── 13_shap_filtering_RF_XGB.py            # SHAP-based feature filtering — RF & XGBoost
    ├── 13B_shap_filtering_LDA_DT.py           # SHAP-based feature filtering — LDA & DT
    ├── 13C_shap_filtering_all_models.py       # SHAP-based feature filtering — all models
    ├── 14_kde_feature_distributions_all_models.py # KDE feature distribution plots
    ├── 15_shap_stability_C1C2_rerun_all_models_5seeds.py   # Stability rerun, C1/C2, 5 seeds
    ├── 15B_shap_stability_C1C2_rerun_all_models_10seeds.py # Stability rerun, C1/C2, 10 seeds
    ├── 15C_shap_stability_C1C2_merge_all_15runs.py         # Merge of all 15 stability runs
    ├── 16_shap_stability_Cross_C3C4_15seeds.py             # Stability, cross-dataset (C3/C4), 15 seeds
    ├── 17_scores_stability_all_models_all_runs.py          # Score stability across all runs
    ├── 18_stability_retraining_10runs.py                   # Stability under retraining, 10 runs
    ├── models/            # Trained model files (.pkl), 4 models × C1-C4 = 16 files
    ├── data/
    │   └── README.md      # Dataset sources, citations, and download links
    ├── requirements.txt   # Python package dependencies
    ├── LICENSE             # MIT License
    └── README.md           # This file

## Experimental setup notation

- **C1** — train and test on LycoS-IDS2017 (within-dataset)
- **C2** — train and test on LycoS-Unicas-IDS2018 (within-dataset)
- **C3** — train on LycoS-IDS2017, test on LycoS-Unicas-IDS2018 (cross-dataset)
- **C4** — train on LycoS-Unicas-IDS2018, test on LycoS-IDS2017 (cross-dataset)

Models evaluated: Random Forest (RF), XGBoost (XGB), Decision Tree (DT), and
Linear Discriminant Analysis (LDA).

## Datasets

See [`data/README.md`](data/README.md) for dataset sources, citations, and
download links for the exact processed data used in this study.

## Setup

1. Clone this repository.
2. Install dependencies:

pip install -r requirements.txt

3. Download the datasets as described in [`data/README.md`](data/README.md).

   > [!NOTE]
   > Make sure the downloaded files keep their exact original names and go in the
   > exact folders below — the scripts reference these specific relative paths:
   >
   > - `data/LycoS-IDS2017_FINAL.csv`
   > - `data/LycoS-Unicas-IDS2018_CLEANED.csv`
   > - `models/` — keep all 16 `.pkl` files here with their original filenames
   >
   > Placing files anywhere else, or renaming them, will cause a "file not found" error.

4. Run `01_env_setup_and_preprocessing.ipynb` to reproduce preprocessing, or
   place the pre-processed CSVs directly if using the provided download link.
5. Run the numbered scripts in order to reproduce training, tuning, and the
   SHAP-based explainability analysis.

## Trained models

Pretrained model files for all four models across all four experimental
configurations (C1-C4) are provided in [`models/`](models/), so SHAP analysis
and downstream scripts can be reproduced without retraining from scratch.

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.