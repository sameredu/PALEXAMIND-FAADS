# PALEXAMIND-FAADS
## A Longitudinal Machine Learning Framework for Fairness in Online Examinations under Infrastructure Constraints

[![Version](https://img.shields.io/badge/Version-2.1.0-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset DOI](https://img.shields.io/badge/Dataset-10.5281%2Fzenodo.23643942-orange)](https://doi.org/10.5281/zenodo.23643942)

---

> ## v2.1.0 — Audit release (November 2026)
> This release audits the pipeline's own earlier results. It adds a script that rebuilds the features from the archived raw data, and a script that regenerates every number, table and figure of the accompanying manuscript. The v2.1.0 findings below **supersede the v2.0 binary benchmark and the v2.0 SVE/BFI simulation**, for the reasons given in [`CHANGELOG_v2.1.md`](CHANGELOG_v2.1.md). All earlier code and outputs are kept unchanged for auditability.

---

## Overview

PALEXAMIND-FAADS began as a fairness-aware decision-support pipeline for Moodle-based online examinations. Its aim was to tell a student whose connection failed from a student who disengaged, using only the behavioral records that any Moodle deployment exports.

Release v2.1.0 asks how much of that pipeline's reported performance is real. It uses 675 examination sessions from 226 students in a Web Databases course at the University College of Applied Sciences (UCAS), Gaza, Palestine. The data contain **no network telemetry**. Disruption is therefore a **weak label**, inferred from behavior: a session counts as substantially disrupted when at least half of its questions are unanswered (blank ratio r ≥ 0.50), following the published severity definition.

The analysis accompanies the manuscript *Leakage controlled evaluation of machine learning for connectivity disruption in online examinations*.

---

## Key findings (v2.1.0)

### 1. Protocol audit: what survives leakage control

Five-class task, released features and hyperparameters.

| Model | Naive protocol: accuracy | Naive: macro-F1 | Leakage-controlled: accuracy | Leakage-controlled: macro-F1 |
|---|---|---|---|---|
| Random Forest | 95.5% | 0.956 | 76.4% ± 2.8 | 0.407 ± 0.053 |
| XGBoost | 96.6% | 0.966 | 80.4% ± 3.0 | 0.421 ± 0.049 |
| LightGBM | 96.2% | 0.962 | 82.8% ± 1.7 | 0.439 ± 0.046 |

Always predicting the majority class gives **86.4%** accuracy. The naive protocol oversamples with SMOTE before splitting, splits records at random so the same student appears on both sides, and computes RIF and BFD over each student's whole history.

### 2. Concurrent detection is circular

| Features | Target | AUC |
|---|---|---|
| v2.0 feature set | P1–P4 vs Normal (as released in v2.0) | 0.830 |
| Examination type only | Substantial disruption | 0.757 |
| Answered share only (SER as defined in the documentation) | Substantial disruption | 1.000 |

The label is a rule on the session's own response pattern, so any model that sees that pattern reproduces the rule. The v2.0 figure of 0.83 does not measure detection, for two reasons. First, the v2.0 code counts answers awaiting manual grading as unanswered, so SER is 0 for every practical session (correlation with examination type −0.97). Second, the P4 class consists of Normal sessions of students who were disrupted elsewhere in the term; 20 of its 24 sessions have no unanswered question.

### 3. Prospective prediction: the sitting matters, not the student

Only information available before the sitting. Student-grouped CV, 10 × 5 folds; prevalence 9.5%.

| Model | AUC | AP |
|---|---|---|
| M0 Examination type only | 0.757 | 0.198 |
| M1 Individual history only | 0.641 | 0.178 |
| M2 Type + individual history | 0.794 | 0.293 |
| M4 Type + sitting position + history (logistic) | 0.789 | 0.301 |
| M4 (RF / XGBoost / LightGBM) | 0.780–0.787 | 0.274–0.282 |

- **AUC gain of M4 over examination type:** +0.008, 95% CI −0.026 to +0.041 (paired student-level bootstrap).
- **Practical sittings:** students who had been disrupted before were disrupted again at **20.6%** (7/34), against **21.1%** (43/204) for students who had not.
- **Repeat disruption:** 8 students had two or more disrupted sessions, against 6.4 expected if outcomes are shuffled within each sitting (p = 0.27).

### 4. Selective triage

The three-tier triage of out-of-fold risk never isolates a group precise enough for automatic individual action. The top tier is 30–37.5% precise across nine threshold pairs. The evidence supports **safeguards for a whole sitting before it** and **session-level review after it**. It does not support profiling individual students as repeatedly vulnerable.

---

## Reproducing v2.1.0

```bash
git clone https://github.com/sameredu/PALEXAMIND-FAADS.git
cd PALEXAMIND-FAADS
pip install -r requirements.txt

# (optional) rebuild the feature file from the seven archived data files and check it
#   download the files from https://doi.org/10.5281/zenodo.23643942 into ./data
FAADS_RAW=data python analysis/build_features_from_raw.py     # prints MATCH

# regenerate every number, table and figure of the manuscript -> results_v21/
python analysis/faads_v21.py
python analysis/fig1_pipeline.py
```

On Windows (PowerShell), set the variable first: `$env:FAADS_RAW="data"`.

| Output in `results_v21/` | Content |
|---|---|
| `Table_data_by_exam.csv` | Sessions and substantial disruption by sitting |
| `Table_protocol_audit_5class.csv` | Naive vs leakage-controlled five-class results |
| `Table_concurrent_diagnostics.csv` | Circularity diagnostics |
| `Table_prospective_groupedCV.csv`, `Table_prospective_temporal.csv` | Prospective models, grouped CV and forward validation |
| `Table_attribution.csv` | Logistic coefficients and mean absolute SHAP |
| `Table_triage.csv`, `Table_triage_sensitivity.csv`, `Table_calibration.csv`, `Table_section_audit.csv` | Selective triage, sensitivity, calibration, section audit |
| `summary.json` | Every other number quoted in the manuscript |
| `Fig1_pipeline.png` … `Fig5_triage.png` | Manuscript figures (600 dpi) |

---

## Repository structure

```
PALEXAMIND-FAADS/
├── analysis/
│   ├── build_features_from_raw.py   # v2.1: rebuild features from the Zenodo files and verify
│   ├── faads_v21.py                 # v2.1: audit, diagnostics, prospective models, triage (authoritative)
│   ├── fig1_pipeline.py             # v2.1: Figure 1
│   ├── bfi_sensitivity.py           # v1.2 (legacy)
│   ├── lightgbm_baseline.py         # v1.2 (legacy)
│   └── shap_analysis.py             # v1.2 (legacy)
├── results_v21/                     # v2.1 results (authoritative)
├── train_and_evaluate_v2.py         # v2.0 pipeline (superseded, retained)
├── results_v2/                      # v2.0 results (superseded, retained)
├── feature_engineering.py           # v1.x feature extraction (its features are used by v2.1)
├── train_and_analyze.py             # v1.x pipeline (superseded, retained)
├── results/                         # v1.x results; processed_features_real.csv is the v2.1 input
├── CHANGELOG_v2.1.md
├── CHANGELOG_v2.md
├── CITATION.cff
├── .zenodo.json
└── requirements.txt
```

---

## Earlier releases (superseded)

<details>
<summary><b>v2.0.0</b> (July 2026): leakage-controlled protocol. Do not cite the binary benchmark or the SVE/BFI simulation.</summary>

v2.0.0 introduced student-grouped folds, in-fold SMOTE and strictly-prior RIF/BFD. Its five-class results (76.4–82.8% accuracy) are confirmed by v2.1.0. Two of its results are withdrawn:

- **Binary benchmark (ROC-AUC ≈ 0.83):** see finding 2 above.
- **SVE/BFI simulation:** the linear composite BFI = α·P + β·R + γ·(1 − BFD) has no reproducing script in v2.0, and its terms work against each other: a disrupted session diverges from the student's baseline, so the (1 − BFD) term takes credit away from exactly the students the disruption term flags. A post-hoc index for judging whether a recorded score can be trusted is developed separately at [sameredu/bfi-behavioral-fairness-index](https://github.com/sameredu/bfi-behavioral-fairness-index).

The v2.0 README also stated that labels were cross-referenced against ISP connectivity reports. That statement was incorrect: the labels are rules on the behavioral record, and no external connectivity data were used.
</details>

<details>
<summary><b>v1.x</b> (to June 2026): about 96% five-class accuracy. Do not cite.</summary>

v1.x applied SMOTE before the train/test split and computed RIF and BFD over each student's whole history, including the session being classified. The reported 96.05–96.57% accuracy, the RIF-dominated feature importance and the 65.2% / 10.3% / 24.5% SVE distribution are artifacts of that protocol. The v1.x scripts and outputs are kept in `results/` and `analysis/` for verification.
</details>

---

## Dataset

Anonymized Moodle examination logs, version 1.1.0 (documentation update; data files identical to v1.0.0):

**DOI:** [10.5281/zenodo.23643942](https://doi.org/10.5281/zenodo.23643942) (CC BY 4.0)

```bibtex
@dataset{yaghi_2026_webdb_dataset,
  author    = {Yaghi, Samer and AbuSamra, Aiman Ahmed},
  title     = {{Web Databases: Anonymized Moodle Examination Dataset
               for Fairness-Aware Learning Analytics}},
  version   = {1.1.0},
  year      = 2026,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23643942},
  url       = {https://doi.org/10.5281/zenodo.23643942}
}
```

Identifiers are pseudonymized with salted SHA-256. Collection and release were approved by the IUG Ethical Research Committee (approval no. 2026/29, 30 June 2026).

The dataset was first analyzed in:

> Yaghi S, AbuSamra AA (2026) Fairness aware learning analytics for identifying internet induced inequality in online examinations. *Discover Education* 5:1009. https://doi.org/10.1007/s44217-026-02182-6

---

## Citation

Please cite the software through the version DOI shown on its Zenodo record. The repository's **Cite this repository** button uses [`CITATION.cff`](CITATION.cff).

> Yaghi S, AbuSamra AA (2026) PALEXAMIND-FAADS: A Longitudinal Machine Learning Framework for Fairness in Online Examinations under Infrastructure Constraints (Version 2.1.0) [Software]. Zenodo.

---

## Authors

**Samer Yaghi**, University College of Applied Sciences (UCAS), Gaza, Palestine; Department of Computer Engineering, Islamic University of Gaza (IUG).
syaghi@ucas.edu.ps · [ORCID 0009-0001-0268-7163](https://orcid.org/0009-0001-0268-7163)

**Prof. Aiman Ahmed AbuSamra**, Department of Computer Engineering, Islamic University of Gaza (IUG), Palestine.
aasamra@iugaza.edu.ps · [ORCID 0000-0003-4652-3993](https://orcid.org/0000-0003-4652-3993)

---

## License

MIT. See [LICENSE](LICENSE).
