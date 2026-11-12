# Changelog: v2.1.0 (November 2026)

v2.1.0 is an audit release. It adds code and results; no existing file is modified except the documentation and metadata (`README.md`, `CITATION.cff`, `.zenodo.json`, `requirements.txt`).

## Added

- `analysis/build_features_from_raw.py`: rebuilds `results/processed_features_real.csv` from the seven files of the archived dataset (https://doi.org/10.5281/zenodo.23643942) using the released v1.x feature code, and compares it with the committed copy. All features and labels match exactly.
- `analysis/faads_v21.py`: regenerates every number, table and figure of the accompanying manuscript (protocol audit, circularity diagnostics, prospective prediction, attribution, selective triage).
- `analysis/fig1_pipeline.py`: Figure 1 (study design).
- `results_v21/`: all v2.1.0 outputs.
- `scipy` added to `requirements.txt` (used for Fisher's exact and chi-square tests).

## Findings that change earlier claims

1. **Five-class results.** Under the leakage-controlled protocol (student-grouped folds, in-fold SMOTE, strictly-prior features), accuracy is 76.4–82.8%, below the 86.4% of a majority-class rule. This confirms the v2.0 five-class figures and adds the baseline that v2.0 did not report.

2. **SER coding.** `feature_engineering.py` computes SER from questions that already have a score. Answers awaiting manual grading (`Needs_Grading`) are therefore counted as unanswered. In the practical examinations every answered question awaits manual grading, so SER = 0 for all 245 practical sessions (correlation with examination type −0.97). SER as documented (answered / total) predicts the substantial-disruption label perfectly (AUC 1.000), because the label is a threshold on the same quantity.

3. **P4 class.** P4 is defined as a Normal session of a student with at least two disrupted sessions anywhere in the term, later sessions included. Of its 24 sessions, 20 have no unanswered question. Including P4 in the positive class of the v2.0 binary benchmark makes that benchmark partly a test of future information. **The v2.0 binary ROC-AUC (≈ 0.83) is withdrawn.**

4. **Initial latency (IL).** IL is implemented as the minute of the hour at which the attempt started, not as the delay after the examination opened. It correlates 0.73 with examination type because practical sittings opened at twenty past the hour.

5. **Individual history.** In a prospective task (information available before the sitting), the strictly-prior disruption count contributes essentially nothing once examination type is known. Mean absolute SHAP is 0.004 log-odds, against 1.28 for examination type. Students with and without a previous disruption are disrupted at the same rate within each examination type.

6. **SVE/BFI simulation.** The v2.0 SVE/BFI table has no reproducing script. In the linear composite, the (1 − BFD) term works against the disruption term, because a disrupted session diverges from the student's baseline and so loses credit. **It is withdrawn.** A post-hoc index is developed separately at https://github.com/sameredu/bfi-behavioral-fairness-index.

7. **Ground truth.** The v2.0 README stated that labels were cross-referenced against ISP connectivity reports. No such data were used; the labels are rules on the behavioral record. The statement is corrected.

## Unchanged

All v1.x and v2.0 code and outputs are kept as they were, for auditability.
