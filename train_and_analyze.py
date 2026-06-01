"""
PALEXAMIND-FAADS — Real Dataset Feature Engineering + RF + XGBoost
Dataset: DWEB 1318 Web Databases — 7 exam files
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings, os
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
os.makedirs('/home/claude/results', exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING FROM REAL DATASET
# ══════════════════════════════════════════════════════════════════

def parse_duration_sec(s):
    """'1 hour 39 minutes' → seconds"""
    s = str(s).strip()
    if s in ['nan','None','']: return 0
    total = 0
    import re
    h = re.search(r'(\d+)\s*hour', s);   total += int(h.group(1))*3600 if h else 0
    m = re.search(r'(\d+)\s*minute', s); total += int(m.group(1))*60   if m else 0
    sc= re.search(r'(\d+)\s*second', s); total += int(sc.group(1))     if sc else 0
    return total

def parse_start_minute(s):
    """'24 March 2026, 12:20 PM' → minutes past exam open hour"""
    try:
        dt = pd.to_datetime(str(s), dayfirst=True)
        return dt.minute
    except: return 0

EXAM_MAX = {
    'Midterm1': 25, 'Mid2_Theory': 15, 'Mid2_Prac': 10,
    'Final_Theory': 20, 'Final_Prac': 10,
    'Supp_Theory': 10, 'Supp_Prac': 10
}

files = {
    'Midterm1':    '/home/claude/dataset/DataBase_WEB_Midterm1_Anonymized.xlsx',
    'Mid2_Theory': '/home/claude/dataset/DataBase_WEB_Midterm2_Theory_Anonymized.xlsx',
    'Mid2_Prac':   '/home/claude/dataset/DataBase_WEB_Midterm2_Practical_Anonymized.xlsx',
    'Final_Theory':'/home/claude/dataset/DataBase_WEB_Final_Theory_Anonymized.xlsx',
    'Final_Prac':  '/home/claude/dataset/DataBase_WEB_Final_Practical_Anonymized.xlsx',
    'Supp_Theory': '/home/claude/dataset/DataBase_WEB_Supp_Theory_Anonymized.xlsx',
    'Supp_Prac':   '/home/claude/dataset/DataBase_WEB_Supp_Practical_Anonymized.xlsx',
}

all_records = []

for exam_name, path in files.items():
    df = pd.read_excel(path)
    q_cols = [c for c in df.columns if c.startswith('Q')]
    score_col = [c for c in df.columns if 'Total_Score' in c][0]
    exam_max = EXAM_MAX[exam_name]

    for _, row in df.iterrows():
        dur_sec  = parse_duration_sec(row['Duration'])
        start_min= parse_start_minute(row['Start_Time'])

        # Q-scores and blanks
        q_vals   = [str(row[c]).strip() for c in q_cols]
        n_q      = len(q_cols)
        n_unans  = sum(1 for v in q_vals if v == 'Unanswered')
        n_needs  = sum(1 for v in q_vals if v == 'Needs_Grading')
        n_scored = n_q - n_unans - n_needs

        # Numeric score
        score_raw = str(row[score_col]).strip()
        score_num = pd.to_numeric(score_raw, errors='coerce')
        score_pct = (score_num / exam_max * 100) if pd.notna(score_num) else np.nan

        # F1: IGV — variance of positions of unanswered questions
        unans_idx = [i for i,v in enumerate(q_vals) if v=='Unanswered']
        if len(unans_idx) >= 2:
            igv = float(np.var(np.diff(unans_idx)))
        elif len(unans_idx)==1:
            igv = float(unans_idx[0]) / max(n_q,1)
        else:
            igv = 0.0

        # F2: IL — Initial Latency (minutes past exam start)
        il = float(start_min)

        # F3: SER — Submission Efficiency Ratio
        ser = n_scored / max(n_q, 1)

        # Disruption label
        blank_ratio = n_unans / max(n_q, 1)

        # P1: Unanswered ALL + duration > 60min (sat full session)
        is_p1 = (n_unans == n_q and n_q > 0 and dur_sec >= 3600)
        # P2: Delayed entry — started >30min late (start_min > 30)
        is_p2 = (start_min > 30 and dur_sec < 600)
        # P3: Partial — >50% blank
        is_p3 = (blank_ratio > 0.5 and not is_p1 and n_q > 1)
        # P4: Repeated (will fill after cross-exam aggregation)

        label = 'P1' if is_p1 else 'P2' if is_p2 else 'P3' if is_p3 else 'Normal'

        all_records.append({
            'Student_Label': row['Student_Label'],
            'Section': row['Section'],
            'Exam': exam_name,
            'IGV': igv,
            'IL': il,
            'SER': ser,
            'n_unanswered': n_unans,
            'n_questions': n_q,
            'blank_ratio': blank_ratio,
            'duration_sec': dur_sec,
            'start_min': start_min,
            'score_pct': score_pct,
            'Label': label,
        })

feat = pd.DataFrame(all_records)

# BFD — Behavioral Fairness Divergence: L2 norm vs student historical centroid
student_means = feat.groupby('Student_Label')[['IGV','IL','SER']].transform('mean')
feat['BFD'] = np.sqrt(((feat[['IGV','IL','SER']] - student_means)**2).sum(axis=1))
feat['BFD'] = (feat['BFD'] - feat['BFD'].min()) / (feat['BFD'].max() - feat['BFD'].min() + 1e-9)

# RIF — Infrastructure Vulnerability: # disrupted exams per student
feat['is_disrupted'] = feat['Label'] != 'Normal'
rif_map = feat.groupby('Student_Label')['is_disrupted'].sum().to_dict()
feat['RIF'] = feat['Student_Label'].map(rif_map)

# P4: repeated vulnerability (RIF >= 2 and currently Normal)
p4_mask = (feat['Label'] == 'Normal') & (feat['RIF'] >= 2)
feat.loc[p4_mask, 'Label'] = 'P4'

feat.to_csv('/home/claude/results/processed_features_real.csv', index=False)

print("=== DATASET SUMMARY ===")
print(f"Total records: {len(feat)}")
print(f"Unique students: {feat['Student_Label'].nunique()}")
print(f"\nLabel distribution:")
vc = feat['Label'].value_counts()
for label, count in vc.items():
    print(f"  {label:<10} {count:3d} ({count/len(feat)*100:.1f}%)")

print(f"\nFeature statistics:")
print(feat[['IGV','IL','SER','BFD','RIF']].describe().round(3).to_string())

# ══════════════════════════════════════════════════════════════════
# 2. MODEL TRAINING — RF + XGBoost
# ══════════════════════════════════════════════════════════════════
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — using GradientBoosting as fallback")

FEATURES = ['IGV','IL','SER','BFD','RIF']
X = feat[FEATURES].fillna(0).values
y = feat['Label'].values

le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"\nClasses: {list(le.classes_)}")

# SMOTE
min_class = min(feat['Label'].value_counts())
k_neighbors = max(1, min(3, min_class - 1))
smote = SMOTE(random_state=SEED, k_neighbors=k_neighbors)
X_res, y_res = smote.fit_resample(X, y_enc)
res_dist = pd.Series(le.inverse_transform(y_res)).value_counts()
print(f"\nAfter SMOTE: {res_dist.to_dict()}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_res, y_res, test_size=0.2, random_state=SEED, stratify=y_res)

# ── Random Forest ─────────────────────────────────────────────────
print("\n" + "="*55)
print("MODEL 1: Random Forest")
print("="*55)
rf = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    class_weight='balanced', random_state=SEED)
rf.fit(X_tr, y_tr)
y_pred_rf = rf.predict(X_te)

rf_acc  = accuracy_score(y_te, y_pred_rf)
rf_f1   = f1_score(y_te, y_pred_rf, average='weighted')
cv_rf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_rf_s = cross_val_score(rf, X_res, y_res, cv=cv_rf, scoring='f1_weighted')

print(f"Accuracy    : {rf_acc*100:.2f}%")
print(f"Weighted F1 : {rf_f1:.3f}")
print(f"CV F1       : {cv_rf_s.mean():.3f} ± {cv_rf_s.std():.3f}")
print(f"\nClassification Report:\n")
print(classification_report(y_te, y_pred_rf, target_names=le.classes_))

rf_imp = rf.feature_importances_

# ── XGBoost / GradientBoosting ────────────────────────────────────
print("="*55)
print(f"MODEL 2: {'XGBoost' if HAS_XGB else 'GradientBoosting (fallback)'}")
print("="*55)

if HAS_XGB:
    xgb = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=SEED, verbosity=0)
else:
    xgb = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=SEED)

xgb.fit(X_tr, y_tr)
y_pred_xgb = xgb.predict(X_te)

xgb_acc = accuracy_score(y_te, y_pred_xgb)
xgb_f1  = f1_score(y_te, y_pred_xgb, average='weighted')
cv_xgb  = cross_val_score(xgb, X_res, y_res, cv=cv_rf, scoring='f1_weighted')

print(f"Accuracy    : {xgb_acc*100:.2f}%")
print(f"Weighted F1 : {xgb_f1:.3f}")
print(f"CV F1       : {cv_xgb.mean():.3f} ± {cv_xgb.std():.3f}")
print(f"\nClassification Report:\n")
print(classification_report(y_te, y_pred_xgb, target_names=le.classes_))

xgb_imp = xgb.feature_importances_

# ══════════════════════════════════════════════════════════════════
# 3. FIGURES
# ══════════════════════════════════════════════════════════════════

# ── Figure 3: Feature Importance — RF vs XGBoost side by side ────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')
fig.suptitle('Figure 3: Feature Importance Ranking (Gini Impurity) — Real Dataset',
             fontsize=13, fontweight='bold', y=1.02)

for ax, imp, title, color in [
    (axes[0], rf_imp,  'Random Forest', '#2E5FA3'),
    (axes[1], xgb_imp, 'XGBoost / GradientBoosting', '#E65100'),
]:
    ax.set_facecolor('white')
    sorted_idx = np.argsort(imp)
    bars = ax.barh(range(len(sorted_idx)), imp[sorted_idx]*100,
                   color=color, edgecolor='white', height=0.6)
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([FEATURES[i] for i in sorted_idx], fontsize=12)
    ax.set_xlabel('Importance Weight (%)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    for i, (bar, idx) in enumerate(zip(bars, sorted_idx)):
        ax.text(bar.get_width()+0.3, i,
                f"{imp[idx]*100:.1f}%", va='center', fontsize=10.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max(imp)*130)

plt.tight_layout()
plt.savefig('/home/claude/results/Figure3_FeatureImportance_RealData.png',
            dpi=180, bbox_inches='tight', facecolor='white')
plt.close()

# ── Figure 5: Confusion Matrices — RF vs XGBoost ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('white')
fig.suptitle('Figure 5: Confusion Matrices — RF vs XGBoost (Real Dataset)',
             fontsize=13, fontweight='bold', y=1.02)

for ax, y_pred, title in [
    (axes[0], y_pred_rf,  f'Random Forest (Acc={rf_acc*100:.1f}%)'),
    (axes[1], y_pred_xgb, f'XGBoost (Acc={xgb_acc*100:.1f}%)'),
]:
    cm = confusion_matrix(y_te, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=0.5, ax=ax, annot_kws={'size':11})
    ax.set_xlabel('Predicted Pattern', fontsize=11, labelpad=8)
    ax.set_ylabel('Actual Pattern',    fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/results/Figure5_ConfusionMatrix_RealData.png',
            dpi=180, bbox_inches='tight', facecolor='white')
plt.close()

# ── Figure: Model Comparison Bar ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')

models  = ['Random Forest', 'XGBoost\n(Gradient)']
accs    = [rf_acc*100, xgb_acc*100]
f1s     = [rf_f1*100,  xgb_f1*100]
cv_m    = [cv_rf_s.mean()*100, cv_xgb.mean()*100]
cv_s    = [cv_rf_s.std()*100,  cv_xgb.std()*100]

x = np.arange(len(models)); w = 0.25
b1 = ax.bar(x-w, accs, w, label='Test Accuracy', color='#2E5FA3', zorder=3)
b2 = ax.bar(x,   f1s,  w, label='Weighted F1',   color='#3B6D11', zorder=3)
b3 = ax.bar(x+w, cv_m, w, label='CV F1 (mean)',  color='#E65100',
            yerr=cv_s, capsize=4, zorder=3)

ax.set_ylim(0, 110)
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel('Score (%)', fontsize=11)
ax.set_title('Model Performance Comparison — Real Dataset (DWEB 1318)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{bar.get_height():.1f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/results/Figure_ModelComparison.png',
            dpi=180, bbox_inches='tight', facecolor='white')
plt.close()

# ── Save classification reports ───────────────────────────────────
rf_rep  = classification_report(y_te, y_pred_rf,  target_names=le.classes_, output_dict=True)
xgb_rep = classification_report(y_te, y_pred_xgb, target_names=le.classes_, output_dict=True)
pd.DataFrame(rf_rep).T.to_csv('/home/claude/results/classification_report_RF.csv')
pd.DataFrame(xgb_rep).T.to_csv('/home/claude/results/classification_report_XGB.csv')

# ══════════════════════════════════════════════════════════════════
# 4. FINAL COMPARISON TABLE (vs Paper Claims)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("COMPARISON: PAPER CLAIMS vs REAL DATASET RESULTS")
print("="*65)
print(f"{'Metric':<35} {'Paper':<12} {'RF Real':<12} {'XGB Real'}")
print("-"*70)
metrics = [
    ('Overall Accuracy',     '92.3%',  f'{rf_acc*100:.1f}%',  f'{xgb_acc*100:.1f}%'),
    ('Weighted F1',          '0.94',   f'{rf_f1:.3f}',        f'{xgb_f1:.3f}'),
    ('CV F1 (mean)',         '0.921',  f'{cv_rf_s.mean():.3f}',f'{cv_xgb.mean():.3f}'),
    ('CV F1 std',            '±0.018', f'±{cv_rf_s.std():.3f}',f'±{cv_xgb.std():.3f}'),
]
for k,p,r,x in metrics:
    print(f"  {k:<33} {p:<12} {r:<12} {x}")

print(f"\n  Feature Importance Ranking (RF Real):")
ranked = sorted(zip(FEATURES, rf_imp), key=lambda x: -x[1])
for i,(f,imp_val) in enumerate(ranked, 1):
    print(f"    {i}. {f:<6} {imp_val*100:.1f}%")

print(f"\n  Feature Importance Ranking (XGB Real):")
ranked_xgb = sorted(zip(FEATURES, xgb_imp), key=lambda x: -x[1])
for i,(f,imp_val) in enumerate(ranked_xgb, 1):
    print(f"    {i}. {f:<6} {imp_val*100:.1f}%")

print("\n=== FILES SAVED ===")
import os
for f in os.listdir('/home/claude/results'):
    sz = os.path.getsize(f'/home/claude/results/{f}')
    print(f"  {f} ({sz//1024}KB)")
EOF
