# PALEXAMIND-FAADS
## A Longitudinal Machine Learning Framework for Fairness in Online Examinations under Infrastructure Constraints

[![Version](https://img.shields.io/badge/Version-2.0.0-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset DOI](https://img.shields.io/badge/Dataset-10.5281%2Fzenodo.20366538-orange)](https://doi.org/10.5281/zenodo.20366538)
[![Software DOI](https://img.shields.io/badge/Software-10.5281%2Fzenodo.20538921-blue)](https://doi.org/10.5281/zenodo.20538921)

---

> ## 🚀 v2.0.0 — Methodological Upgrade (July 2026)
> This release upgrades the evaluation protocol to a fully **leakage-controlled design**. All results on this page are produced by [`train_and_evaluate_v2.py`](train_and_evaluate_v2.py) under the v2.0 protocol and **supersede the v1.x figures**, which are retained in the [Legacy v1.x Results](#legacy-v1x-results-superseded) section for auditability. Details: [`CHANGELOG_v2.md`](CHANGELOG_v2.md).

---

## Overview

PALEXAMIND-FAADS is a fairness-aware decision-support framework that detects **infrastructure-induced disruption** in Moodle-based online examinations — distinguishing a student whose connection failed from a student who disengaged — using only the behavioral artifacts available in any standard Moodle deployment. Detected cases are routed through a three-tier **Smart Validation Engine (SVE)** governed by a composite **Behavioral Fairness Index (BFI)**, operationalizing *conservative autonomy*: automated action only at high confidence, human review for everything uncertain.

The framework is validated on 675 real examination records from 226 students at UCAS, Gaza — one of the most infrastructure-hostile assessment environments on record (connectivity routinely below 50 kbps) — with ground-truth labels cross-referenced against ISP connectivity interruption reports.

**Phase II of the PALEXAMIND PhD research program.** Phase I (empirical foundation): [Research Square preprint](https://doi.org/10.21203/rs.3.rs-9843743/v1).

---

## What's New in v2.0

- **Leakage-controlled evaluation protocol** — all metrics computed on genuine, natural-distribution records
- **Student-grouped cross-validation** (StratifiedGroupKFold): no student appears in both training and evaluation
- **In-fold SMOTE**: oversampling fitted inside training folds only
- **Strictly-prior longitudinal features** (`RIF_prior`, `BFD_prior`): computed exclusively from sessions preceding the one being classified
- **Binary disruption detection** as the primary benchmark task
- **Transparent baseline models** (logistic regression, depth-2 decision tree)
- **Updated reproducibility pipeline** — single script regenerates all corrected results
- **Corrected experimental results**, SHAP, ablation, and SVE/BFI sensitivity analyses

---

## Methodological Improvements in v2

| Previous protocol (v1.x) | Updated protocol (v2.0) |
|---|---|
| Random record-level split | **Student-grouped split** (StratifiedGroupKFold) |
| Global SMOTE (before split) | **Fold-only SMOTE** (training partition only) |
| Mixed session history in RIF/BFD | **Strictly-prior history** (chronological exam order) |
| Accuracy on balanced (partly synthetic) test set | **ROC-AUC / F1 on natural distribution** |
| Multi-class only | **Binary primary + multi-class exploratory** |
| Tree ensembles only | Ensembles **+ transparent baselines** |

---

## Key Results (v2.0 — leakage-controlled, natural distribution)

Dataset: DWEB 1318 — 226 students, 675 exam records (583 Normal, 92 disrupted: P1=33, P3=29, P4=24, P2=6).

### Primary task — Binary disruption detection (student-grouped 5-fold CV)

| Model | F1 (disrupted) | Precision | Recall | ROC-AUC | Accuracy |
|---|---|---|---|---|---|
| **XGBoost** | **0.565 ± 0.056** | 0.581 | 0.553 | **0.828** | 0.884 |
| Random Forest | 0.531 ± 0.051 | 0.550 | 0.531 | 0.834 | 0.871 |
| LightGBM | 0.525 ± 0.028 | 0.546 | 0.510 | 0.826 | 0.874 |
| Logistic Regression (baseline) | 0.401 ± 0.041 | 0.272 | 0.762 | 0.810 | 0.690 |

### Exploratory task — Five-class pattern discrimination (student-grouped 5-fold CV)

| Model | Accuracy | Macro-F1 | Weighted F1 |
|---|---|---|---|
| **LightGBM** | **82.8% ± 1.7** | **0.439** | 0.839 |
| XGBoost | 80.4% ± 3.0 | 0.421 | 0.826 |
| Random Forest | 76.4% ± 2.2 | 0.399 | 0.798 |

Minority disruption patterns are data-limited in the natural distribution (six real P2 sessions); pattern-level claims await multi-course data collection.

### Feature attribution (v2.0 — leakage-free features)

| Feature | Gini (RF) | Mean \|SHAP\| | Meaning |
|---|---|---|---|
| **IL — Initial Latency** | **0.329** (Rank 1) | 0.156 | Entry delay — primary disruption indicator |
| **SER — Submission Efficiency** | 0.241 | **0.203** (Rank 1) | Response completeness ratio |
| BFD_prior | 0.172 | 0.084 | Divergence from strictly-prior profile |
| RIF_prior | 0.154 | 0.156 | Prior disrupted-exam count |
| IGV | 0.105 | 0.071 | Unanswered-index-gap proxy |

**Initial latency and submission efficiency emerge as the dominant behavioral signals**; the strictly-prior repeated-failure history contributes the largest single ablation gain (+0.129 ROC-AUC).

### SVE / BFI simulation (v2.0 — n=110 scored-history records)

| Weight config. (α/β/γ) | Automatic (%) | Review (%) | No action (%) | Mean BFI |
|---|---|---|---|---|
| Original (0.33/0.33/0.34) | **7.3** | **51.8** | 40.9 | 0.587 |
| Performance-weighted (0.50/0.25/0.25) | 8.2 | 15.5 | 76.4 | 0.517 |
| Balanced-moderate (0.40/0.30/0.30) | 8.2 | 25.5 | 66.4 | 0.558 |

The automatic tier is stable across weight configurations (7.3–8.2%); the engine escalates roughly half of scored disruption cases to human review — the intended *conservative autonomy* profile.

---

## System Architecture (v2.0)

```
        Historical Moodle Sessions
                    │
     Strictly-Prior Feature Construction
        (IGV, IL, SER, BFD_prior, RIF_prior)
                    │
      Student-Grouped Cross-Validation
                    │
          Training-Fold SMOTE
                    │
       Binary Disruption Detection
        (RF / XGBoost / LightGBM
         + transparent baselines)
                    │
      Behavioral Fairness Index (BFI)
     BFI_i = a*P_i + b*R_i + g*(1 - BFD_i)
                    │
        Smart Validation Engine
   +----------------+----------------+
   |                |                |
   v                v                v
Automatic      Instructor        No action
(BFI >= 0.80)  review flag      (BFI < 0.60)
               (0.60-0.79)
```

---

## Repository Structure

```
PALEXAMIND-FAADS/
├── train_and_evaluate_v2.py        # * v2.0 leakage-controlled pipeline (authoritative)
├── CHANGELOG_v2.md                 # * v2.0 protocol upgrades, documented in full
├── results_v2/                     # * v2.0 corrected results (authoritative)
│   ├── Table_corrected_binary_groupedCV.csv
│   ├── Table_corrected_5class_groupedCV.csv
│   ├── Table_corrected_5class_holdout.csv
│   ├── Table_corrected_feature_importance.csv
│   ├── classification_report_RF_natural.csv
│   ├── corrected_features.csv      # dataset + strictly-prior features
│   ├── Figure_corrected_confusion_matrix.png
│   ├── Figure_corrected_feature_importance.png
│   └── Figure_corrected_SHAP.png
├── feature_engineering.py          # v1.x feature extraction (legacy, retained for audit)
├── train_and_analyze.py            # v1.x pipeline (legacy, retained for audit)
├── analysis/                       # v1.x additional analyses (legacy)
├── results/                        # v1.x results (legacy — superseded)
├── requirements.txt
├── CITATION.cff                    # v2.0.0
├── .zenodo.json                    # v2.0.0
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

### Reproduce all v2.0 results (single command)

```bash
python train_and_evaluate_v2.py
# Outputs -> results_v2/ : binary + 5-class tables, per-class report,
#                          corrected feature importance, confusion matrix, SHAP figure
```

### Quick example

```python
import pandas as pd

df = pd.read_csv('results_v2/corrected_features.csv')
X = df[['IGV', 'IL', 'SER', 'BFD_prior', 'RIF_prior']]   # strictly-prior features
y = (df['Label'] != 'Normal').astype(int)                  # binary primary task

# Behavioral Fairness Index (BFI)
alpha, beta, gamma = 0.33, 0.33, 0.34
# BFI_i = alpha*P_i + beta*R_i + gamma*(1 - BFD_i)
```

### Legacy v1.x scripts

`feature_engineering.py`, `train_and_analyze.py`, and `analysis/` reproduce the superseded v1.x results and are retained unmodified for auditability only.

---

## Legacy v1.x Results (superseded)

<details>
<summary>Click to expand — v1.x figures, retained for auditability. <b>Do not cite.</b></summary>

The v1.x protocol applied SMOTE before the train/test split and computed RIF/BFD over each student's full session history (including the classified session). Under that protocol the models reported ~96% five-class accuracy (RF 96.05%, XGBoost 96.23%, LightGBM 96.57%; per-pattern F1 0.91-0.99), RIF-dominant feature importance (34.9-40.4%), and an SVE distribution of 65.2% / 10.3% / 24.5%. These figures are artifacts of the evaluation protocol and are **superseded by the v2.0 results above** (see `CHANGELOG_v2.md`, UPG-1 and UPG-2). The v1.x scripts and outputs remain in `results/` and `analysis/` for verification.

</details>

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

All student and section identifiers are irreversibly pseudonymized (SHA-256 with institutional salt). Collection and publication are covered by IUG Ethical Research Committee approval no. 2026/29.

---

## Citation

If you use this software, please cite:

> Yaghi, S., & AbuSamra, A. A. (2026). *PALEXAMIND-FAADS: A Longitudinal Machine Learning Framework for Fairness in Online Examinations under Infrastructure Constraints* (Version 2.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.20538921

```bibtex
@software{yaghi2026faads,
  author    = {Yaghi, Samer and AbuSamra, Aiman Ahmed},
  title     = {{PALEXAMIND-FAADS: A Longitudinal Machine Learning Framework
               for Fairness in Online Examinations under Infrastructure Constraints}},
  version   = {2.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20538921},
  url       = {https://doi.org/10.5281/zenodo.20538921}
}
```

The concept DOI `10.5281/zenodo.20538921` always resolves to the latest release; the version-specific DOI for v2.0.0 is minted automatically by Zenodo upon release.

### Phase I (Foundational Study)

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
Ethics approval: IUG Ethical Research Committee, approval no. 2026/29 (30 June 2026).
Dataset DOI: [10.5281/zenodo.20366538](https://doi.org/10.5281/zenodo.20366538)
