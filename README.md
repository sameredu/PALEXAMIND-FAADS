# PALEXAMIND-FAADS
## AI-Driven Automated Fairness Intervention for Infrastructure-Constrained E-Examinations

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20366538.svg)](https://doi.org/10.5281/zenodo.20366538)
[![Dataset](https://img.shields.io/badge/Dataset-Zenodo-orange)](https://doi.org/10.5281/zenodo.20366538)

**Part of the EXAMIND PhD Project** — Islamic University of Gaza (IUG) / UCAS, Palestine

---

## Overview

PALEXAMIND-FAADS is a **Fairness-Aware Automated Decision System** (FAADS) that transitions from retrospective behavioral analytics to proactive, real-time fairness intervention in Moodle-based electronic examinations.

This repository contains the complete Python implementation of **Paper 2** of the PALEXAMIND/EXAMIND PhD project:

> **Yaghi, S. & AbuSamra, A. A. (2026).** *Beyond Manual Analytics: The PALEXAMIND-FAADS Framework for AI-Driven Automated Fairness Intervention in Infrastructure-Constrained E-Examinations.* Educational Assessment, Evaluation and Accountability (EAIT). [Under Review]

Building on **Phase I** findings (Cohen's d=2.641, U=27.50, p<0.0001), this system classifies five disruption patterns using Random Forest and XGBoost classifiers augmented by a novel **Behavioral Fairness Index (BFI)**.

---

## Key Results (Real Dataset — DWEB 1318, n=206 students, 675 exam records)

| Model | Accuracy | Weighted F1 | CV F1 (5-fold) |
|-------|----------|-------------|----------------|
| Random Forest | **96.1%** | 0.960 | 0.950 ± 0.010 |
| XGBoost | **96.2%** | 0.962 | 0.963 ± 0.010 |

### Per-Pattern F1-Scores

| Pattern | Description | RF F1 | XGB F1 |
|---------|-------------|-------|--------|
| P1 | Complete Disconnection | 0.925 | 0.932 |
| P2 | Delayed Entry | 0.987 | 0.979 |
| P3 | Partial Interruption | 0.912 | 0.922 |
| P4 | Repeated Vulnerability | 0.991 | 0.991 |
| Normal | No disruption | 0.987 | 0.987 |

### SVE Decision Distribution (BFI-based)

| Decision | Count | Percentage |
|----------|-------|-----------|
| Automatic Intervention (BFI ≥ 0.80) | 380 | 65.2% |
| Instructor Review (0.60 ≤ BFI < 0.80) | 60 | 10.3% |
| No Intervention (BFI < 0.60) | 143 | 24.5% |

---

## System Architecture

The FAADS pipeline consists of 5 layers:

```
Moodle Event Stream
     ↓
[1] Feature Engineering  → V_i = {IGV, IL, SER, BFD, RIF}
     ↓
[2] Disruption Classification  → RF / XGBoost → P_i
     ↓
[3] BFI Scoring  → BFI_i = α·P_i + β·R_i + γ·(1 − BFD_i)
     ↓
[4] SVE Decision Engine  → BFI ≥ 0.80 → Automatic | ≥ 0.60 → Review | < 0.60 → None
     ↓
[5] Golden Profile Update & Logging
```

**BFI Formula:** `BFI_i = 0.33·P_i + 0.33·R_i + 0.34·(1 − BFD_i)`

---

## Repository Structure

```
PALEXAMIND-FAADS/
├── feature_engineering.py      # Feature extraction from DWEB 1318 dataset
├── train_and_analyze.py        # RF + XGBoost training, BFI, SVE pipeline
├── requirements.txt            # Python dependencies
├── results/
│   ├── Figure3_FeatureImportance_RealData.png
│   ├── Figure5_ConfusionMatrix_RealData.png
│   ├── Figure6_ModelComparison.png
│   ├── Table2_FAADS_RF_XGB.csv
│   ├── FAADS_SVE_results.csv
│   ├── classification_report_RF.csv
│   ├── classification_report_XGB.csv
│   └── processed_features_real.csv
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

### Step 1: Feature Engineering
```bash
python feature_engineering.py
# Output: processed_features_real.csv
```

### Step 2: Train Models + Run FAADS Pipeline
```bash
python train_and_analyze.py
# Outputs: figures, classification reports, SVE results
```

### Quick Example
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load features
df = pd.read_csv('results/processed_features_real.csv')

# Features
X = df[['IGV', 'IL', 'SER', 'BFD', 'RIF']]
y = df['Label']

# BFI calculation
alpha, beta, gamma = 0.33, 0.33, 0.34
# BFI_i = alpha*P_i + beta*R_i + gamma*(1 - BFD_i)
```

---

## Dataset

The DWEB 1318 dataset is available on Zenodo:

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

If you use this code, please cite:

```bibtex
@software{yaghi_2026_palexamind_faads,
  author    = {Yaghi, Samer and AbuSamra, Aiman Ahmed},
  title     = {{PALEXAMIND-FAADS: Python Implementation for AI-Driven
                Automated Fairness Intervention in E-Examinations}},
  year      = 2026,
  publisher = {GitHub},
  url       = {https://github.com/sameredu/PALEXAMIND-FAADS},
  note      = {Part of the EXAMIND PhD Project, IUG/UCAS, Palestine}
}
```

**Companion Paper:**
```bibtex
@article{yaghi_2026_faads_paper,
  author  = {Yaghi, Samer and AbuSamra, Aiman Ahmed},
  title   = {Beyond Manual Analytics: The PALEXAMIND-FAADS Framework for
             AI-Driven Automated Fairness Intervention in
             Infrastructure-Constrained E-Examinations},
  journal = {Educational Assessment, Evaluation and Accountability (EAIT)},
  year    = {2026},
  note    = {Under Review}
}
```

**Phase I Paper (Dataset):**
```bibtex
@article{yaghi_2026_paper1,
  author  = {Yaghi, Samer and AbuSamra, Aiman Ahmed},
  title   = {Fairness-Aware Learning Analytics for Detecting Internet-Induced
             Inequality in Moodle-Based E-Examinations},
  journal = {Computers \& Education},
  year    = {2026},
  doi     = {10.5281/zenodo.20366538}
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

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

This work is part of the **EXAMIND PhD Project** at the Islamic University of Gaza (IUG).  
Dataset DOI: [10.5281/zenodo.20366538](https://doi.org/10.5281/zenodo.20366538)
