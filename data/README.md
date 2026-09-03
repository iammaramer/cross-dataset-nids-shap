# Datasets

This repository contains the code used to preprocess, train, and analyze
models on two publicly available network intrusion detection datasets.

## Original dataset sources

### LycoS-IDS2017
- **Source:** https://lycos-ids.univ-lemans.fr/
- **Citation:** Rosay, A., Carlier, F., et al. (2021). *From CIC-IDS2017 to
  LYCOS-IDS2017: A corrected dataset for better performance.* IEEE/WIC/ACM
  International Conference on Web Intelligence and Intelligent Agent
  Technology (WI-IAT 2021). https://doi.org/10.1145/3486622.3493973

### LycoS-Unicas-IDS2018
- **Source:** https://github.com/MarcoCantone/LycoS-Unicas-IDS2018
- **Citation:** Cantone, M., Marrocco, C., & Bria, A. (2024). *On the
  Cross-Dataset Generalization of Machine Learning for Network Intrusion
  Detection.* arXiv:2402.10974. https://arxiv.org/abs/2402.10974

## Processed data used in this study

The exact cleaned/merged CSV files produced by
`01_env_setup_and_preprocessing.ipynb` and used throughout the scripts in
this repository are provided here for convenience and reproducibility:

- `LycoS-IDS2017_FINAL.csv` (647 MB)
- `LycoS-Unicas-IDS2018_CLEANED.csv` (4.81 GB)

Download both here: https://drive.google.com/drive/folders/12tfOQeNV9gfPf-MYQe5zTPkrLgaBfmPc?usp=sharing