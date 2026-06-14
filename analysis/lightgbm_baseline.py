"""
Additional Baseline: LightGBM Comparison
Trains a LightGBM classifier on the same 5-feature pipeline (identical SMOTE,
train/test split, SEED=42) used for RF and XGBoost, and reports accuracy,
weighted F1, and 5-fold CV F1 for direct comparison with Table 2.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42
np.random.seed(SEED)

feat = pd.read_csv('../results/processed_features_real.csv')
FEATURES = ['IGV', 'IL', 'SER', 'BFD', 'RIF']
X = feat[FEATURES].fillna(0).values
y = feat['Label'].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

min_class = min(feat['Label'].value_counts())
k_neighbors = max(1, min(3, min_class - 1))
smote = SMOTE(random_state=SEED, k_neighbors=k_neighbors)
X_res, y_res = smote.fit_resample(X, y_enc)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_res, y_res, test_size=0.2, random_state=SEED, stratify=y_res)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

results = []

# ---- RF (reference, recompute for consistency) ----
rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                             class_weight='balanced', random_state=SEED)
rf.fit(X_tr, y_tr)
y_pred = rf.predict(X_te)
acc = accuracy_score(y_te, y_pred)
f1w = f1_score(y_te, y_pred, average='weighted')
cv_s = cross_val_score(rf, X_res, y_res, cv=cv, scoring='f1_weighted')
results.append(('Random Forest', acc, f1w, cv_s.mean(), cv_s.std()))

# ---- XGBoost (reference) ----
xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                     eval_metric='mlogloss', random_state=SEED, verbosity=0)
xgb.fit(X_tr, y_tr)
y_pred = xgb.predict(X_te)
acc = accuracy_score(y_te, y_pred)
f1w = f1_score(y_te, y_pred, average='weighted')
cv_s = cross_val_score(xgb, X_res, y_res, cv=cv, scoring='f1_weighted')
results.append(('XGBoost', acc, f1w, cv_s.mean(), cv_s.std()))

# ---- LightGBM (new baseline) ----
lgbm = LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                       random_state=SEED, verbose=-1)
lgbm.fit(X_tr, y_tr)
y_pred_lgbm = lgbm.predict(X_te)
acc = accuracy_score(y_te, y_pred_lgbm)
f1w = f1_score(y_te, y_pred_lgbm, average='weighted')
cv_s = cross_val_score(lgbm, X_res, y_res, cv=cv, scoring='f1_weighted')
results.append(('LightGBM', acc, f1w, cv_s.mean(), cv_s.std()))

print(f"{'Model':<18}{'Accuracy':>10}{'Weighted F1':>14}{'CV F1 (mean)':>16}{'CV F1 (std)':>14}")
for name, acc, f1w, cvm, cvs in results:
    print(f"{name:<18}{acc*100:>9.2f}%{f1w:>14.3f}{cvm:>16.3f}{cvs:>14.3f}")

# Classification report for LightGBM
print("\nLightGBM Classification Report:")
print(classification_report(y_te, y_pred_lgbm, target_names=le.classes_))

# Save comparison table
df_out = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Weighted_F1', 'CV_F1_mean', 'CV_F1_std'])
df_out['Accuracy'] = (df_out['Accuracy'] * 100).round(2)
df_out[['Weighted_F1','CV_F1_mean','CV_F1_std']] = df_out[['Weighted_F1','CV_F1_mean','CV_F1_std']].round(3)
df_out.to_csv('../results/Table_LightGBM_Baseline.csv', index=False)

# Feature importance for LightGBM
lgbm_imp = lgbm.feature_importances_
ranked = sorted(zip(FEATURES, lgbm_imp), key=lambda x: -x[1])
total = sum(lgbm_imp)
print("\nLightGBM Feature Importance:")
for f, v in ranked:
    print(f"  {f:<6} {v/total*100:.1f}%")
