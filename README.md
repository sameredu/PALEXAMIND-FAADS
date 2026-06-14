# PALEXAMIND-FAADS
## AI-Driven Automated Fairness Intervention for Infrastructure-Constrained E-Examinations

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset DOI](https://img.shields.io/badge/Dataset-10.5281%2Fzenodo.20366538-orange)](https://doi.org/10.5281/zenodo.20366538)
[![Software DOI](https://img.shields.io/badge/Software-10.5281%2Fzenodo.20538921-blue)](https://doi.org/10.5281/zenodo.20538921)
[![Version](https://img.shields.io/badge/Version-1.2.0-brightgreen)](https://github.com/sameredu/PALEXAMIND-FAADS)

**Part of the PALEXAMIND PhD Project** — Islamic University of Gaza (IUG) / University College of Applied Sciences (UCAS), Gaza, Palestine

---

## Overview

PALEXAMIND-FAADS is a **Fairness-Aware Automated Decision System** (FAADS) that transitions from retrospective behavioral analytics to a proactive, **real-time-ready** fairness intervention architecture for Moodle-based electronic examinations.

Building on **Phase I** findings (Cohen's d = 2.641, U = 27.50, p < 0.0001), the system classifies five disruption patterns using **Random Forest**, **XGBoost**, and **LightGBM** classifiers, augmented by a novel **Behavioral Fairness Index (BFI)** and a three-tier **Smart Validation Engine (SVE)**.

This repository contains the complete, reproducible Python implementation, the anonymized dataset reference, and all scripts that regenerate the tables and figures.

---

## What's New in v1.2

- **LightGBM baseline** added alongside Random Forest and XGBoost (three independent classifier families).
- **SHAP interpretability analysis** — model-agnostic, instance-level feature attribution with a summary plot.
- **BFI weight sensitivity analysis** — robustness check across three weight configurations.
- Updated `requirements.txt` (`shap`, `lightgbm`) and new `analysis/` scripts.

---

## Key Results (Real Dataset — DWEB 1318, 226 students, 675 exam records)

### Model Comparison (identical pipeline: SMOTE, 80/20 stratified split, 5-fold CV, seed = 42)

| Model | Accuracy | Weighted F1 | CV F1 (5-fold) |
|-------|----------|-------------|----------------|
| Random Forest | 96.05% | 0.960 | 0.950 ± 0.010 |
| XGBoost | 96.23% | 0.962 | 0.963 ± 0.010 |
| **LightGBM (added in v1.2)** | **96.57%** | **0.966** | **0.962 ± 0.007** |

The three classifiers agree within a 0.52-percentage-point accuracy band, indicating the discriminative signal resides in the engineered behavioral features rather than any single model family.

### Per-Pattern F1-Scores

| Pattern | Description | RF F1 | XGB F1 |
|---------|-------------|-------|--------|
| P1 | Complete Disconnection | 0.925 | 0.932 |
| P2 | Delayed Entry | 0.987 | 0.979 |
| P3 | Partial Interruption | 0.912 | 0.922 |
| P4 | Repeated Vulnerability | 0.991 | 0.991 |
| Normal | No disruption | 0.987 | 0.987 |

### BFI Weight Sensitivity (v1.2)

| Weight config. (α/β/γ) | Automatic (%) | Review (%) | No action (%) | Mean BFI |
|------------------------|---------------|------------|---------------|----------|
| Original (0.33/0.33/0.34) | 81.5 | 15.6 | 3.0 | 0.856 |
| Performance-weighted (0.50/0.25/0.25) | 84.6 | 11.7 | 3.7 | 0.877 |
| Balanced-moderate (0.40/0.30/0.30) | 83.6 | 13.5 | 3.0 | 0.865 |

The intervention-tier distribution remains stable under reasonable weight perturbation, indicating the framework is not brittle with respect to the empirical weight choice.

### SHAP Feature Attribution (v1.2)

Mean absolute SHAP value (Random Forest, test set): **RIF (0.283) > IL (0.183) > SER (0.159) > BFD (0.063) > IGV (0.058)**, confirming Repeated Infrastructure Failures (RIF) as the dominant, directionally consistent predictive signal.

---

## System Architecture

The FAADS pipeline consists of four layers:

```
Moodle Event Stream
     |
[1] Secure Data Acquisition  -> anonymized event logs (SHA-256 hashing)
     |
[2] Feature Engineering      -> V_i = {IGV, IL, SER, BFD, RIF}
     |
[3] Disruption Classification -> RF / XGBoost / LightGBM -> P_i
     |
[4] SVE Decision Engine + BFI -> BFI >= 0.80 Automatic | >= 0.60 Review | < 0.60 None
     |
    Golden Profile Update & Logging
```

**BFI Formula:** `BFI_i = α·P_i + β·R_i + γ·(1 − BFD_i)`  (default: α = β = 0.33, γ = 0.34)

---

## Repository Structure

```
PALEXAMIND-FAADS/
├── feature_engineering.py          # Feature extraction from DWEB 1318 dataset
├── train_and_analyze.py            # RF + XGBoost training, BFI, SVE pipeline
├── analysis/
│   ├── bfi_sensitivity.py          # BFI weight sensitivity analysis (v1.2)
│   ├── shap_analysis.py            # SHAP interpretability + summary plot (v1.2)
│   └── lightgbm_baseline.py        # LightGBM baseline comparison (v1.2)
├── requirements.txt                # Python dependencies
├── results/
│   ├── Figure3_FeatureImportance_RealData.png
│   ├── Figure4_SHAP_Summary_RF.png            # (v1.2)
│   ├── Figure5_ConfusionMatrix_RealData.png
│   ├── Figure6_ModelComparison.png
│   ├── Table2_FAADS_RF_XGB.csv
│   ├── Table_BFI_Sensitivity.csv              # (v1.2)
│   ├── Table_LightGBM_Baseline.csv            # (v1.2)
│   ├── Table_SHAP_ranking.csv                 # (v1.2)
│   ├── FAADS_SVE_results.csv
│   ├── classification_report_RF.csv
│   ├── classification_report_XGB.csv
│   └── processed_features_real.csv
├── CITATION.cff
├── .zenodo.json
└── README.md
```

---

## Installation

```bash
git clone https://github.com/sameredu/PALEXAMIND-FAADS.git
cd PALEXAMIND-FAADS
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Feature Engineering
```bash
python feature_engineering.py
# Output: results/processed_features_real.csv
```

### Step 2 — Train Models + Run FAADS Pipeline
```bash
python train_and_analyze.py
# Outputs: figures, classification reports, SVE results
```

### Step 3 — Additional Analyses (v1.2)
```bash
cd analysis
python lightgbm_baseline.py     # LightGBM baseline -> ../results/Table_LightGBM_Baseline.csv
python shap_analysis.py         # SHAP summary plot -> ../results/Figure4_SHAP_Summary_RF.png
python bfi_sensitivity.py       # BFI sensitivity  -> ../results/Table_BFI_Sensitivity.csv
```

### Quick Example
```python
import pandas as pd

df = pd.read_csv('results/processed_features_real.csv')
X = df[['IGV', 'IL', 'SER', 'BFD', 'RIF']]
y = df['Label']

# Behavioral Fairness Index (BFI)
alpha, beta, gamma = 0.33, 0.33, 0.34
# BFI_i = alpha*P_i + beta*R_i + gamma*(1 - BFD_i)
```

---

## Dataset

The DWEB 1318 anonymized Moodle examination dataset is available on Zenodo:

**DOI:** [10.5281/zenodo.20366538](https://doi.org/10.5281/zenodo.20366538)

```bibtex
@dataset{yaghi_2026_dweb1318,
  author    = {Yaghi, Samer and AbuSamra, Aiman Ahmed},
  title     = {{DWEB 1318 Web Databases — Anonymized Moodle Examination Dataset}},
  year      = 2026,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20366538},
  url       = {https://doi.org/10.5281/zenodo.20366538}
}
```

---

## Citation

If you use this software, please cite:

> Yaghi, S., & AbuSamra, A. A. (2026). *PALEXAMIND-FAADS: A Longitudinal Fairness-Aware Framework for Infrastructure-Constrained E-Examinations* (Version 1.2.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.20538921

```bibtex
@software{yaghi2026faads,
  author    = {Yaghi, Samer and AbuSamra, Aiman Ahmed},
  title     = {{PALEXAMIND-FAADS: A Longitudinal Fairness-Aware Framework
               for Infrastructure-Constrained E-Examinations}},
  version   = {1.2.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20538921},
  url       = {https://doi.org/10.5281/zenodo.20538921}
}
```

### Phase I (Foundational Study)

The empirical foundation for this work is established in Phase I, available as a preprint:

> Yaghi, S., & AbuSamra, A. A. (2026). *Fairness-Aware Learning Analytics for Detecting Internet-Induced Inequality in Moodle-Based E-Examinations.* Research Square (preprint). https://doi.org/10.21203/rs.3.rs-9843743/v1

```bibtex
@article{yaghi_2026_phase1,
  author  = {Yaghi, Samer and AbuSamra, Aiman Ahmed},
  title   = {{Fairness-Aware Learning Analytics for Detecting Internet-Induced
             Inequality in Moodle-Based E-Examinations}},
  journal = {Research Square (preprint)},
  year    = {2026},
  doi     = {10.21203/rs.3.rs-9843743/v1},
  url     = {https://doi.org/10.21203/rs.3.rs-9843743/v1}
}
```

---

## Authors

**Samer Yaghi** — PhD Candidate
University College of Applied Sciences (UCAS), Gaza, Palestine
Islamic University of Gaza (IUG), Computer Engineering Dept.
✉ syaghi@ucas.edu.ps
🔗 [ORCID: 0009-0001-0268-7163](https://orcid.org/0009-0001-0268-7163)

**Prof. Aiman Ahmed AbuSamra** — PhD Supervisor
Islamic University of Gaza (IUG), Computer Engineering Dept., Palestine
✉ aasamra@iugaza.edu.ps
🔗 [ORCID: 0000-0003-4652-3993](https://orcid.org/0000-0003-4652-3993)

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

This work is part of the **PALEXAMIND PhD Project** at the Islamic University of Gaza (IUG) and University College of Applied Sciences (UCAS), Gaza, Palestine.
Dataset DOI: [10.5281/zenodo.20366538](https://doi.org/10.5281/zenodo.20366538)
