"""
PALEXAMIND-FAADS v2.0.0 — Corrected, leakage-free evaluation pipeline.

Fixes vs v1.2.0:
  FIX-1  SMOTE was applied to the full dataset BEFORE the train/test split,
         placing synthetic records in the test set. Now: split/CV-fold first,
         SMOTE fitted on the training partition only; all metrics reported on
         genuine (natural-distribution) records.
  FIX-2  RIF counted a student's non-Normal labels across ALL sessions,
         including the current and future ones (label leakage). Now: RIF_prior
         counts strictly earlier sessions in chronological exam order.
         BFD likewise recomputed from strictly-prior sessions (BFD_prior).

Outputs (written to ./results_v2/):
  Table_corrected_5class_holdout.csv      Design A  (split-first, natural test set)
  Table_corrected_5class_groupedCV.csv    Design B  (StratifiedGroupKFold by student)
  Table_corrected_binary_groupedCV.csv    Binary disrupted-vs-normal task
  classification_report_RF_natural.csv    Per-class report, natural test set
  Table_corrected_feature_importance.csv  RF Gini importance, leakage-free features
  Figure_corrected_confusion_matrix.png
  Figure_corrected_feature_importance.png
"""
import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42
IN_CSV = os.environ.get('FAADS_FEATURES', 'results/processed_features_real.csv')
OUT = 'results_v2'
os.makedirs(OUT, exist_ok=True)

# ────────────────────────────────────────────────────────────────────
# 1. Load and rebuild longitudinal features WITHOUT leakage (FIX-2)
# ────────────────────────────────────────────────────────────────────
df = pd.read_csv(IN_CSV)
EXAM_ORDER = {'Midterm1': 1, 'Mid2_Theory': 2, 'Mid2_Prac': 3,
              'Final_Theory': 4, 'Final_Prac': 5, 'Supp_Theory': 6, 'Supp_Prac': 7}
df['exam_order'] = df['Exam'].map(EXAM_ORDER)
df = df.sort_values(['Student_Label', 'exam_order']).reset_index(drop=True)

df['is_disr_int'] = (df['Label'] != 'Normal').astype(int)
# RIF_prior: number of disrupted sessions strictly BEFORE the current one
df['RIF_prior'] = (df.groupby('Student_Label')['is_disr_int']
                     .transform(lambda s: s.shift(1).fillna(0).cumsum()).astype(int))

# BFD_prior: L2 divergence of (IGV, IL, SER) from the expanding mean of PRIOR sessions
def _bfd_prior(g):
    prior_mean = g[['IGV', 'IL', 'SER']].expanding().mean().shift(1)
    d = np.sqrt(((g[['IGV', 'IL', 'SER']] - prior_mean) ** 2).sum(axis=1))
    return d.fillna(0.0)
df['BFD_prior'] = df.groupby('Student_Label', group_keys=False).apply(_bfd_prior)


FEATS = ['IGV', 'IL', 'SER', 'BFD_prior', 'RIF_prior']
X = df[FEATS].values
le = LabelEncoder()
y = le.fit_transform(df['Label'].values)
yb = df['is_disr_int'].values
groups = df['Student_Label'].values
print('Natural class distribution:', df['Label'].value_counts().to_dict())
print('Unique students:', df['Student_Label'].nunique())

def smote_train(Xtr, ytr):
    """SMOTE fitted on the TRAINING partition only (FIX-1)."""
    mc = pd.Series(ytr).value_counts().min()
    if mc < 2:
        return RandomOverSampler(random_state=SEED).fit_resample(Xtr, ytr)
    return SMOTE(random_state=SEED, k_neighbors=max(1, min(3, mc - 1))).fit_resample(Xtr, ytr)

def make_models():
    return {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10,
                                                class_weight='balanced', random_state=SEED),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                 eval_metric='mlogloss', random_state=SEED, verbosity=0),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=SEED, verbosity=-1),
        'Logistic Regression': LogisticRegression(max_iter=2000, class_weight='balanced',
                                                  random_state=SEED),
        'Decision Stump (d=2)': DecisionTreeClassifier(max_depth=2, class_weight='balanced',
                                                       random_state=SEED),
    }

# ────────────────────────────────────────────────────────────────────
# 2. DESIGN A — split first, SMOTE on train only, NATURAL test set
# ────────────────────────────────────────────────────────────────────
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
Xtr_s, ytr_s = smote_train(Xtr, ytr)
rowsA = []
for name, m in make_models().items():
    if 'Logistic' in name:
        sc = StandardScaler().fit(Xtr_s)
        m.fit(sc.transform(Xtr_s), ytr_s)
        pred = m.predict(sc.transform(Xte))
    else:
        m.fit(Xtr_s, ytr_s)
        pred = m.predict(Xte)
    rowsA.append({'Model': name,
                  'Accuracy': round(accuracy_score(yte, pred), 4),
                  'Macro_F1': round(f1_score(yte, pred, average='macro'), 4),
                  'Weighted_F1': round(f1_score(yte, pred, average='weighted'), 4)})
pd.DataFrame(rowsA).to_csv(f'{OUT}/Table_corrected_5class_holdout.csv', index=False)
print('\nDesign A (natural holdout, n=%d):' % len(yte))
print(pd.DataFrame(rowsA).to_string(index=False))

# Per-class report + confusion matrix (RF)
rf = make_models()['Random Forest'].fit(Xtr_s, ytr_s)
pred_rf = rf.predict(Xte)
rep = classification_report(yte, pred_rf, target_names=le.classes_, digits=3,
                            zero_division=0, output_dict=True)
pd.DataFrame(rep).T.round(3).to_csv(f'{OUT}/classification_report_RF_natural.csv')

cm = confusion_matrix(yte, pred_rf, labels=range(len(le.classes_)))
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(le.classes_))); ax.set_xticklabels(le.classes_)
ax.set_yticks(range(len(le.classes_))); ax.set_yticklabels(le.classes_)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=9)
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('RF confusion matrix — corrected pipeline, natural test set (n=%d)' % len(yte), fontsize=9)
fig.colorbar(im, fraction=0.046)
fig.tight_layout(); fig.savefig(f'{OUT}/Figure_corrected_confusion_matrix.png', dpi=300); plt.close(fig)

# Corrected feature importance
imp = pd.DataFrame({'Feature': FEATS, 'Importance': rf.feature_importances_.round(4)}) \
        .sort_values('Importance', ascending=False)
imp.to_csv(f'{OUT}/Table_corrected_feature_importance.csv', index=False)
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.barh(imp['Feature'][::-1], imp['Importance'][::-1], color='#2c6fad')
ax.set_xlabel('Gini importance')
ax.set_title('RF feature importance — leakage-free features', fontsize=10)
fig.tight_layout(); fig.savefig(f'{OUT}/Figure_corrected_feature_importance.png', dpi=300); plt.close(fig)
print('\nCorrected feature importance:\n', imp.to_string(index=False))

# ────────────────────────────────────────────────────────────────────
# 3. DESIGN B — StratifiedGroupKFold(5) by student, natural test folds
# ────────────────────────────────────────────────────────────────────
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
rowsB = []
for name in make_models():
    accs, mf1s, wf1s = [], [], []
    pc = np.zeros((5, len(le.classes_))); k = 0
    for tr, te in sgkf.split(X, y, groups):
        Xs, ys = smote_train(X[tr], y[tr])
        m = make_models()[name]
        if 'Logistic' in name:
            sc = StandardScaler().fit(Xs)
            m.fit(sc.transform(Xs), ys); pred = m.predict(sc.transform(X[te]))
        else:
            m.fit(Xs, ys); pred = m.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        mf1s.append(f1_score(y[te], pred, average='macro'))
        wf1s.append(f1_score(y[te], pred, average='weighted'))
        pc[k] = f1_score(y[te], pred, average=None, labels=range(len(le.classes_)), zero_division=0)
        k += 1
    row = {'Model': name,
           'Accuracy_mean': round(np.mean(accs), 4), 'Accuracy_sd': round(np.std(accs), 4),
           'Macro_F1_mean': round(np.mean(mf1s), 4), 'Macro_F1_sd': round(np.std(mf1s), 4),
           'Weighted_F1_mean': round(np.mean(wf1s), 4)}
    row.update({f'F1_{c}': round(v, 3) for c, v in zip(le.classes_, pc.mean(0))})
    rowsB.append(row)
pd.DataFrame(rowsB).to_csv(f'{OUT}/Table_corrected_5class_groupedCV.csv', index=False)
print('\nDesign B (student-grouped 5-fold CV):')
print(pd.DataFrame(rowsB).to_string(index=False))

# ────────────────────────────────────────────────────────────────────
# 4. BINARY TASK — disrupted vs normal (primary corrected headline)
# ────────────────────────────────────────────────────────────────────
rowsC = []
for name in ['Random Forest', 'XGBoost', 'LightGBM', 'Logistic Regression']:
    f1s, precs, recs, aucs, accs = [], [], [], [], []
    for tr, te in sgkf.split(X, yb, groups):
        Xs, ys = SMOTE(random_state=SEED, k_neighbors=3).fit_resample(X[tr], yb[tr])
        m = make_models()[name]
        if 'Logistic' in name:
            sc = StandardScaler().fit(Xs)
            m.fit(sc.transform(Xs), ys)
            pred = m.predict(sc.transform(X[te])); proba = m.predict_proba(sc.transform(X[te]))[:, 1]
        else:
            m.fit(Xs, ys)
            pred = m.predict(X[te]); proba = m.predict_proba(X[te])[:, 1]
        f1s.append(f1_score(yb[te], pred)); precs.append(precision_score(yb[te], pred, zero_division=0))
        recs.append(recall_score(yb[te], pred)); aucs.append(roc_auc_score(yb[te], proba))
        accs.append(accuracy_score(yb[te], pred))
    rowsC.append({'Model': name,
                  'F1_disrupted': round(np.mean(f1s), 3), 'F1_sd': round(np.std(f1s), 3),
                  'Precision': round(np.mean(precs), 3), 'Recall': round(np.mean(recs), 3),
                  'ROC_AUC': round(np.mean(aucs), 3), 'Accuracy': round(np.mean(accs), 3)})
pd.DataFrame(rowsC).to_csv(f'{OUT}/Table_corrected_binary_groupedCV.csv', index=False)
print('\nBinary disrupted-vs-normal (student-grouped 5-fold CV):')
print(pd.DataFrame(rowsC).to_string(index=False))
print('\nAll v2 outputs written to', OUT)
