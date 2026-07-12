# PALEXAMIND-FAADS v2.0.0 — Methodological Upgrade

**Date:** 2026-07-12 | **Type:** Evaluation-protocol development

## Summary
Pre-submission methodological review of the v1.x pipeline identified two evaluation-protocol
improvements required for leakage-controlled benchmarking. v2.0.0 implements both, adds
transparent baselines and a binary primary task, and regenerates all results. **v1.x
performance figures are superseded by `results_v2/` and should no longer be cited.**

## Protocol upgrades
### UPG-1 — Split-first, in-fold oversampling
v1.x applied SMOTE to the full dataset before splitting, so synthetic records reached the
test set. v2.0.0 forms the split (or CV fold) first and fits SMOTE on the training
partition only; all metrics are computed on genuine, natural-distribution records.

### UPG-2 — Strictly-prior longitudinal features
v1.x computed RIF/BFD over a student’s full session history, allowing information from the
classified (and later) sessions to leak into the features. v2.0.0 introduces `RIF_prior`
and `BFD_prior`, computed exclusively from sessions preceding the current one in
chronological exam order.

### Additional development
- Student-grouped cross-validation (StratifiedGroupKFold) as the primary protocol.
- Binary disrupted-vs-normal detection as the primary benchmark task.
- Logistic-regression and shallow-tree transparency baselines.
- Corrected SHAP, feature-importance, ablation, and SVE/BFI sensitivity analyses.

## Corrected headline results (leakage-controlled, student-grouped 5-fold CV)
- Binary detection: ROC-AUC ≈ 0.83; F1(disrupted) 0.53–0.57.
- Five-class (exploratory): accuracy 0.76–0.83; macro-F1 0.36–0.44; minority patterns are
  data-limited in the natural distribution (P1=33, P3=29, P4=24, P2=6 real sessions).
- Feature importance (RF): IL 0.329, SER 0.241, BFD_prior 0.172, RIF_prior 0.154, IGV 0.105.

## Integrity note
v1.x scripts are retained unmodified for auditability. The anonymized dataset
(675 records, 226 students; DOI 10.5281/zenodo.20366538) is unchanged; only the
evaluation protocol evolves. Phase I statistical findings (Mann-Whitney U=27.50,
Cohen’s d=2.641) are independent of these protocol changes.
