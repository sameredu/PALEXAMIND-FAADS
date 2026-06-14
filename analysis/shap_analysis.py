"""
SHAP Analysis for PALEXAMIND-FAADS
Trains RF and XGBoost on the 5 behavioral features (IGV, IL, SER, BFD, RIF)
following the exact pipeline in train_and_analyze.py (SMOTE + train/test split,
SEED=42), then computes SHAP values for the held-out test set and produces
a SHAP summary (beeswarm) plot for the RF model (primary model).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

SEED = 42
np.random.seed(SEED)

feat = pd.read_csv('../results/processed_features_real.csv')
FEATURES = ['IGV', 'IL', 'SER', 'BFD', 'RIF']
X = feat[FEATURES].fillna(0).values
y = feat['Label'].values

le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Classes:", list(le.classes_))

min_class = min(feat['Label'].value_counts())
k_neighbors = max(1, min(3, min_class - 1))
smote = SMOTE(random_state=SEED, k_neighbors=k_neighbors)
X_res, y_res = smote.fit_resample(X, y_enc)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_res, y_res, test_size=0.2, random_state=SEED, stratify=y_res)

# ---- Random Forest ----
rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                             class_weight='balanced', random_state=SEED)
rf.fit(X_tr, y_tr)

# ---- XGBoost ----
xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                     eval_metric='mlogloss', random_state=SEED, verbosity=0)
xgb.fit(X_tr, y_tr)

X_te_df = pd.DataFrame(X_te, columns=FEATURES)

# ── SHAP for RF (multi-class -> use TreeExplainer, average abs SHAP across classes) ──
print("Computing SHAP values for Random Forest...")
explainer_rf = shap.TreeExplainer(rf)
shap_values_rf = explainer_rf.shap_values(X_te_df)

# shap_values_rf shape for multi-class: (n_samples, n_features, n_classes) in recent SHAP versions
shap_values_rf = np.array(shap_values_rf)
print("SHAP values shape:", shap_values_rf.shape)

if shap_values_rf.ndim == 3:
    # (n_samples, n_features, n_classes) -> mean abs over classes
    mean_abs_shap = np.abs(shap_values_rf).mean(axis=2)  # (n_samples, n_features)
else:
    mean_abs_shap = np.abs(shap_values_rf)

# Build a "summary" matrix using signed SHAP for the predicted class per sample
y_pred_rf = rf.predict(X_te)
signed_shap = np.zeros((X_te_df.shape[0], len(FEATURES)))
for i in range(X_te_df.shape[0]):
    cls = y_pred_rf[i]
    if shap_values_rf.ndim == 3:
        signed_shap[i, :] = shap_values_rf[i, :, cls]
    else:
        signed_shap[i, :] = shap_values_rf[i, :]

# ── Beeswarm-style summary plot (custom, matching PALEXAMIND figure style) ──
mean_abs_importance = np.abs(signed_shap).mean(axis=0)
order = np.argsort(mean_abs_importance)

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

last_sc = None
for rank, fidx in enumerate(order):
    feat_name = FEATURES[fidx]
    shap_vals_f = signed_shap[:, fidx]
    feat_vals_f = X_te_df.values[:, fidx]
    # Per-feature normalization (percentile-based, robust to outliers)
    lo, hi = np.percentile(feat_vals_f, 5), np.percentile(feat_vals_f, 95)
    if hi <= lo:
        hi = lo + 1e-9
    norm_f = Normalize(vmin=lo, vmax=hi)
    jitter = (np.random.rand(len(shap_vals_f)) - 0.5) * 0.35
    last_sc = ax.scatter(shap_vals_f, np.full_like(shap_vals_f, rank) + jitter,
                          c=feat_vals_f, cmap='coolwarm', norm=norm_f,
                          s=22, alpha=0.75, edgecolors='none')

ax.set_yticks(range(len(order)))
ax.set_yticklabels([FEATURES[i] for i in order], fontsize=12)
ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
ax.set_xlabel('SHAP value (impact on predicted-class probability)', fontsize=11)
ax.set_title('Figure 4. SHAP Summary Plot — Random Forest\n(Test Set, Predicted-Class Attribution)',
              fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Generic low-high colorbar (per-feature scaling applied above; colorbar shows relative position)
sm = ScalarMappable(cmap='coolwarm', norm=Normalize(vmin=0, vmax=1))
cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Low', 'High'])
cbar.set_label('Feature value (per-feature scaled)', fontsize=10)

plt.tight_layout()
plt.savefig('../results/Figure4_SHAP_Summary_RF.png', dpi=180,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved Figure4_SHAP_Summary_RF.png")

# ── Print mean |SHAP| ranking for text reporting ──
ranking = sorted(zip(FEATURES, mean_abs_importance), key=lambda x: -x[1])
print("\nMean |SHAP value| ranking (RF, predicted class):")
for f, v in ranking:
    print(f"  {f:<6} {v:.4f}")

# Save ranking to CSV
pd.DataFrame(ranking, columns=['Feature', 'Mean_abs_SHAP']).to_csv(
    '../results/Table_SHAP_ranking.csv', index=False)
