"""
PALEXAMIND-FAADS v2.1 analysis: protocol audit, construct-validity diagnostics,
prospective (pre-examination) risk prediction, attribution and selective triage.
Run from the repository root:  python analysis/faads_v21.py
Input : results/processed_features_real.csv   (675 sessions, 226 students)
Output: results_v21/*.csv, results_v21/*.png, results_v21/summary.json
"""
import os, json, warnings, logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             accuracy_score, f1_score, roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42
IN = os.environ.get('FAADS_FEATURES', 'results/processed_features_real.csv')
OUT = os.environ.get('FAADS_OUT', 'results_v21'); os.makedirs(OUT, exist_ok=True)
S = {}                                   # summary of every number used in the manuscript

# ------------------------------------------------------------------ data
df = pd.read_csv(IN)
ORDER = {'Midterm1': 1, 'Mid2_Theory': 2, 'Mid2_Prac': 3, 'Final_Theory': 4,
         'Final_Prac': 5, 'Supp_Theory': 6, 'Supp_Prac': 7}
df['order'] = df.Exam.map(ORDER)
df = df.sort_values(['Student_Label', 'order']).reset_index(drop=True)
df['practical'] = df.Exam.str.contains('Prac').astype(int)
# Session-level target: substantial disruption, blank ratio >= 0.50 (published severity definition)
df['y'] = (df.blank_ratio >= 0.5).astype(int)
df['complete'] = (df.blank_ratio == 1).astype(int)
g = df.groupby('Student_Label')
# strictly-prior history
df['n_prior'] = g.cumcount()
df['dis_prior'] = g['y'].transform(lambda s: s.shift(1).fillna(0).cumsum())
df['blank_prior_mean'] = g['blank_ratio'].transform(lambda s: s.expanding().mean().shift(1)).fillna(0)
df['blank_prior_max'] = g['blank_ratio'].transform(lambda s: s.expanding().max().shift(1)).fillna(0)
df['answered_share'] = 1 - df.blank_ratio          # SER as defined in the text
# v2.0 feature set as coded in the released pipeline
df['is_disr_v2'] = (df.Label != 'Normal').astype(int)
df['RIF_prior_v2'] = g['is_disr_v2'].transform(lambda s: s.shift(1).fillna(0).cumsum())
def _bfd(x):
    pm = x[['IGV', 'IL', 'SER']].expanding().mean().shift(1)
    return np.sqrt(((x[['IGV', 'IL', 'SER']] - pm) ** 2).sum(axis=1)).fillna(0)
df['BFD_prior_v2'] = df.groupby('Student_Label', group_keys=False).apply(_bfd)
grp = df.Student_Label.values

S['data'] = dict(sessions=len(df), students=int(df.Student_Label.nunique()),
                 substantial=int(df.y.sum()), prevalence=round(df.y.mean(), 4),
                 complete=int(df.complete.sum()), students_complete=int(df[df.complete == 1].Student_Label.nunique()),
                 students_any=int(df[df.y == 1].Student_Label.nunique()),
                 students_repeat=int((g['y'].sum() >= 2).sum()),
                 rate_practical=round(df[df.practical == 1].y.mean(), 4), n_practical=int(df.practical.sum()),
                 rate_theory=round(df[df.practical == 0].y.mean(), 4), n_theory=int((1 - df.practical).sum()),
                 sessions_with_history=int((df.n_prior > 0).sum()),
                 label_counts=df.Label.value_counts().to_dict(),
                 p4_no_blank=int(((df.Label == 'P4') & (df.n_unanswered == 0)).sum()),
                 ser_zero_practical=round((df[df.practical == 1].SER == 0).mean(), 3),
                 corr_SER_practical=round(df.SER.corr(df.practical), 3),
                 corr_IL_practical=round(df.IL.corr(df.practical), 3))
per_exam = df.groupby('Exam').agg(order=('order', 'first'), sessions=('y', 'size'), substantial=('y', 'sum'),
                                  complete=('complete', 'sum')).sort_values('order')
per_exam['rate_pct'] = (100 * per_exam.substantial / per_exam.sessions).round(1)
per_exam.to_csv(f'{OUT}/Table_data_by_exam.csv')

def models5():
    return {'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=SEED),
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='mlogloss', random_state=SEED, verbosity=0),
            'LightGBM': LGBMClassifier(n_estimators=100, random_state=SEED, verbosity=-1)}

# ------------------------------------------------------------------ A. protocol audit (five-class, original labels)
le = LabelEncoder(); y5 = le.fit_transform(df.Label)
rows = []
# A1 naive protocol (v1.x): SMOTE on all records, then random 80/20 split, whole-history features
Xn = df[['IGV', 'IL', 'SER', 'BFD', 'RIF']].fillna(0).values
Xr, yr = SMOTE(random_state=SEED, k_neighbors=3).fit_resample(Xn, y5)
Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=SEED, stratify=yr)
n_synth_test = None
for name, m in models5().items():
    m.fit(Xtr, ytr); p = m.predict(Xte)
    if name == 'Random Forest': S['naive_rf_importance'] = dict(zip(['IGV', 'IL', 'SER', 'BFD', 'RIF'], m.feature_importances_.round(3).tolist()))
    rows.append(dict(Protocol='Naive (oversample, then split; whole-history features)', Model=name,
                     Accuracy=accuracy_score(yte, p), Macro_F1=f1_score(yte, p, average='macro'), n_test=len(yte)))
S['naive_test_n'] = int(len(yte)); S['naive_resampled_n'] = int(len(yr)); S['naive_real_n'] = int(len(y5))
# A2 leakage-controlled (v2.0): student-grouped folds, in-fold SMOTE, strictly-prior features
Xc = df[['IGV', 'IL', 'SER', 'BFD_prior_v2', 'RIF_prior_v2']].values
sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
def smote_tr(X, y):
    mc = pd.Series(y).value_counts().min()
    if mc < 2: return RandomOverSampler(random_state=SEED).fit_resample(X, y)
    return SMOTE(random_state=SEED, k_neighbors=max(1, min(3, mc - 1))).fit_resample(X, y)
for name in models5():
    acc, mf = [], []
    for tr, te in sg.split(Xc, y5, grp):
        Xs, ys = smote_tr(Xc[tr], y5[tr]); m = models5()[name].fit(Xs, ys); p = m.predict(Xc[te])
        acc.append(accuracy_score(y5[te], p)); mf.append(f1_score(y5[te], p, average='macro'))
    rows.append(dict(Protocol='Leakage-controlled (student-grouped folds, in-fold oversampling, prior-only features)',
                     Model=name, Accuracy=np.mean(acc), Macro_F1=np.mean(mf), Accuracy_sd=np.std(acc), Macro_F1_sd=np.std(mf), n_test=len(y5)))
A = pd.DataFrame(rows).round(4); A.to_csv(f'{OUT}/Table_protocol_audit_5class.csv', index=False)
S['majority_class_accuracy'] = round((df.Label == 'Normal').mean(), 4)

# ------------------------------------------------------------------ helpers for binary tasks
def cv_oof(feats, ycol, make, seeds=range(10), oversample=False):
    """Repeated student-grouped 5-fold CV. Metrics are computed inside each test fold and averaged
    (50 folds), so fold-to-fold calibration shifts do not distort the ranking metrics.
    Also returns the out-of-fold probabilities averaged over repeats."""
    X = df[feats].values.astype(float); y = df[ycol].values
    oof = np.zeros((len(list(seeds)), len(y))); rep = []
    for i, sd in enumerate(seeds):
        k = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=sd); f = []
        for tr, te in k.split(X, y, grp):
            Xa, ya = (SMOTE(random_state=SEED, k_neighbors=3).fit_resample(X[tr], y[tr]) if oversample else (X[tr], y[tr]))
            oof[i, te] = make().fit(Xa, ya).predict_proba(X[te])[:, 1]
            f.append((roc_auc_score(y[te], oof[i, te]), average_precision_score(y[te], oof[i, te]), brier_score_loss(y[te], oof[i, te])))
        rep.append(np.mean(f, axis=0))
    rep = np.array(rep)
    return dict(AUC=rep[:, 0].mean(), AUC_sd=rep[:, 0].std(), AP=rep[:, 1].mean(), AP_sd=rep[:, 1].std(),
                Brier=rep[:, 2].mean()), oof.mean(0), oof
LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
RF = lambda: RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=10, random_state=SEED)
XGB = lambda: XGBClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, eval_metric='logloss', random_state=SEED, verbosity=0)
LGB = lambda: LGBMClassifier(n_estimators=150, max_depth=2, num_leaves=4, learning_rate=0.05, min_child_samples=15, subsample=0.8, subsample_freq=1, random_state=SEED, verbosity=-1)
RFv2 = lambda: RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=SEED)

# ------------------------------------------------------------------ B. construct-validity diagnostics (concurrent detection)
B = []
for lab, feats, ycol, mk, ov in [
    ('Released feature set, released target (P1-P4 vs Normal)', ['IGV', 'IL', 'SER', 'BFD_prior_v2', 'RIF_prior_v2'], 'is_disr_v2', RFv2, True),
    ('Released feature set, substantial disruption', ['IGV', 'IL', 'SER', 'BFD_prior_v2', 'RIF_prior_v2'], 'y', RFv2, True),
    ('Examination type only', ['practical'], 'y', LR, False),
    ('Answered share only (SER as defined)', ['answered_share'], 'y', RFv2, False),
    ('IGV + IL + answered share', ['IGV', 'IL', 'answered_share'], 'y', RFv2, False)]:
    r, _, _ = cv_oof(feats, ycol, mk, oversample=ov); r = dict(Setting=lab, **r); B.append(r)
Bt = pd.DataFrame(B).round(4); Bt.to_csv(f'{OUT}/Table_concurrent_diagnostics.csv', index=False)

# ------------------------------------------------------------------ C. prospective task (information available before the session)
HIST = ['dis_prior', 'blank_prior_mean', 'blank_prior_max']      # the student's own record
FULL = ['practical', 'n_prior'] + HIST                            # + position of the sitting in the term
C = []; OOF = {}
for lab, feats, mk in [('M0 Examination type only (logistic)', ['practical'], LR),
                       ('M1 Individual history only (logistic)', HIST, LR),
                       ('M2 Type + individual history (logistic)', ['practical'] + HIST, LR),
                       ('M3 Type + sitting position (logistic)', ['practical', 'n_prior'], LR),
                       ('M4 Type + position + history: logistic regression', FULL, LR),
                       ('M4 Type + position + history: random forest', FULL, RF),
                       ('M4 Type + position + history: XGBoost', FULL, XGB),
                       ('M4 Type + position + history: LightGBM', FULL, LGB)]:
    r, o, oo = cv_oof(feats, 'y', mk); C.append(dict(Model=lab, **r)); OOF[lab] = (o, oo)
Ct = pd.DataFrame(C).round(4); Ct.to_csv(f'{OUT}/Table_prospective_groupedCV.csv', index=False)
S['prevalence_AP_baseline'] = round(df.y.mean(), 4)
S['brier_prevalence'] = round(brier_score_loss(df.y, np.full(len(df), df.y.mean())), 4)

# paired student-level bootstrap of the AUC / AP gain of history over examination type
def boot_delta(a, b, y, n=2000, seed=SEED):
    rng = np.random.default_rng(seed); st = df.Student_Label.values; u = np.unique(st)
    idx = {s: np.where(st == s)[0] for s in u}; da, dp = [], []
    for _ in range(n):
        ii = np.concatenate([idx[s] for s in rng.choice(u, len(u))])
        if y[ii].sum() == 0: continue
        da.append(roc_auc_score(y[ii], a[ii]) - roc_auc_score(y[ii], b[ii]))
        dp.append(average_precision_score(y[ii], a[ii]) - average_precision_score(y[ii], b[ii]))
    return [round(float(x), 4) for x in (np.mean(da), *np.percentile(da, [2.5, 97.5]), np.mean(dp), *np.percentile(dp, [2.5, 97.5]))]
yv = df.y.values
IND = df.practical.values.astype(float)      # examination-type indicator, no fitting required
S['auc_indicator'] = round(roc_auc_score(yv, IND), 4); S['ap_indicator'] = round(average_precision_score(yv, IND), 4)
S['delta_M2_vs_M0'] = boot_delta(OOF['M2 Type + individual history (logistic)'][0], IND, yv)
S['delta_M3_vs_M0'] = boot_delta(OOF['M3 Type + sitting position (logistic)'][0], IND, yv)
S['delta_M4_vs_M3'] = boot_delta(OOF['M4 Type + position + history: logistic regression'][0], OOF['M3 Type + sitting position (logistic)'][0], yv)
S['delta_M4_vs_M0'] = boot_delta(OOF['M4 Type + position + history: logistic regression'][0], IND, yv)
def boot_ci(p, y, n=2000, seed=SEED):
    rng = np.random.default_rng(seed); st = df.Student_Label.values; u = np.unique(st)
    idx = {s: np.where(st == s)[0] for s in u}; a = []
    for _ in range(n):
        ii = np.concatenate([idx[s] for s in rng.choice(u, len(u))])
        if 0 < y[ii].sum() < len(ii): a.append(roc_auc_score(y[ii], p[ii]))
    return [round(float(x), 3) for x in np.percentile(a, [2.5, 97.5])]
S['auc_ci'] = {k: boot_ci(v[0], yv) for k, v in OOF.items()}; S['auc_ci']['indicator'] = boot_ci(IND, yv)

# within examination type: does the student's own record rank sessions?
from scipy.stats import fisher_exact
W = {}
pM2 = OOF['M2 Type + individual history (logistic)'][0]
for nm, mask in [('practical', df.practical == 1), ('theory', df.practical == 0)]:
    m = mask.values & (df.n_prior.values > 0)
    W[nm] = dict(n=int(m.sum()), events=int(yv[m].sum()), AUC_M2=round(roc_auc_score(yv[m], pM2[m]), 3))
    sub = df[m]; a_ = sub[sub.dis_prior > 0].y; b_ = sub[sub.dis_prior == 0].y
    W[nm].update(with_prior=[int(a_.sum()), int(len(a_)), round(a_.mean(), 3)], without_prior=[int(b_.sum()), int(len(b_)), round(b_.mean(), 3)],
                 fisher_p=round(float(fisher_exact([[a_.sum(), len(a_) - a_.sum()], [b_.sum(), len(b_) - b_.sum()]])[1]), 3))
S['within_type'] = W
# is disruption repeated within students more than the sitting-level rates imply? (permute outcomes within sitting)
rng = np.random.default_rng(SEED); obs = int((df.groupby('Student_Label').y.sum() >= 2).sum()); sim = []
for _ in range(5000):
    yp = df.groupby('Exam').y.transform(lambda s: rng.permutation(s.values)); sim.append(int((yp.groupby(df.Student_Label).sum() >= 2).sum()))
sim = np.array(sim)
S['repeat_permutation'] = dict(observed=obs, expected=round(float(sim.mean()), 2), p_ge=round(float((sim >= obs).mean()), 4),
                               q025=float(np.percentile(sim, 2.5)), q975=float(np.percentile(sim, 97.5)))

# temporal (forward) validation: fit on midterm sittings, test on final and supplementary sittings
tr = (df.order <= 3).values; te = ~tr; T = []
for lab, feats, mk in [('M0 Examination type only (logistic)', ['practical'], LR), ('M2 Type + individual history (logistic)', ['practical'] + HIST, LR),
                       ('M4 Type + position + history: logistic regression', FULL, LR), ('M4 Type + position + history: random forest', FULL, RF),
                       ('M4 Type + position + history: XGBoost', FULL, XGB), ('M4 Type + position + history: LightGBM', FULL, LGB)]:
    X = df[feats].values.astype(float); p = mk().fit(X[tr], yv[tr]).predict_proba(X[te])[:, 1]
    T.append(dict(Model=lab, n_train=int(tr.sum()), n_test=int(te.sum()), events_test=int(yv[te].sum()),
                  AUC=roc_auc_score(yv[te], p), AP=average_precision_score(yv[te], p), Brier=brier_score_loss(yv[te], p)))
Tt = pd.DataFrame(T).round(4); Tt.to_csv(f'{OUT}/Table_prospective_temporal.csv', index=False)
S['temporal_test_prevalence'] = round(yv[te].mean(), 4)

# ------------------------------------------------------------------ D. attribution (logistic coefficients + SHAP for the boosted model)
sc = StandardScaler().fit(df[FULL].values); lr = LogisticRegression(max_iter=2000).fit(sc.transform(df[FULL].values), yv)
coef = pd.DataFrame({'Feature': FULL, 'Std_coefficient': lr.coef_[0].round(3), 'Odds_ratio_per_SD': np.exp(lr.coef_[0]).round(2)})
try:
    import shap
    sv = np.zeros((len(df), len(FULL)))
    for a, b in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(df[FULL].values, yv, grp):
        m = XGB().fit(df[FULL].values[a], yv[a]); sv[b] = shap.TreeExplainer(m).shap_values(df[FULL].values[b])
    coef['Mean_abs_SHAP_XGB'] = np.abs(sv).mean(0).round(3); np.save(f'{OUT}/shap_values.npy', sv)
except Exception as e:
    print('SHAP unavailable:', e); sv = None
coef.to_csv(f'{OUT}/Table_attribution.csv', index=False)

# ------------------------------------------------------------------ E. selective triage on out-of-fold risk
p = OOF['M4 Type + position + history: logistic regression'][0]
def tiers(p, lo, hi): return np.where(p >= hi, 2, np.where(p >= lo, 1, 0))
def tier_table(lo, hi):
    t = tiers(p, lo, hi); r = []
    for k, nm in [(2, 'Tier 1: high risk (p >= 3 x base rate)'), (1, 'Tier 2: elevated risk (base rate <= p < 3 x base rate)'), (0, 'Tier 3: low risk (p < base rate)')]:
        m = t == k
        r.append(dict(Tier=nm, Sessions=int(m.sum()), Share_pct=round(100 * m.mean(), 1), Disrupted=int(yv[m].sum()),
                      Observed_rate_pct=round(100 * yv[m].mean(), 1) if m.sum() else np.nan,
                      Share_of_disruptions_pct=round(100 * yv[m].sum() / yv.sum(), 1)))
    return pd.DataFrame(r)
base = float(df.y.mean())
TT = tier_table(base, 3 * base); TT.to_csv(f'{OUT}/Table_triage.csv', index=False)
grid = []
for lo in [0.05, base, 0.15]:
    for hi in [0.25, 3 * base, 0.35]:
        t = tier_table(lo, hi)
        grid.append(dict(lower=round(lo, 3), upper=round(hi, 3), tier1_share=t.Share_pct[0], tier1_rate=t.Observed_rate_pct[0],
                         tier2_share=t.Share_pct[1], tier2_rate=t.Observed_rate_pct[1], tier3_share=t.Share_pct[2], tier3_rate=t.Observed_rate_pct[2],
                         captured_t1_t2=round(t.Share_of_disruptions_pct[0] + t.Share_of_disruptions_pct[1], 1)))
pd.DataFrame(grid).to_csv(f'{OUT}/Table_triage_sensitivity.csv', index=False)
S['triage_thresholds'] = [round(base, 3), round(3 * base, 3)]
# reliability of the pooled out-of-fold probabilities
df['_p'] = p; df['_bin'] = pd.qcut(df._p.rank(method='first'), 5, labels=False)
cal = df.groupby('_bin').agg(mean_pred=('_p', 'mean'), observed=('y', 'mean'), n=('y', 'size')).round(3); cal.to_csv(f'{OUT}/Table_calibration.csv')
# section-level audit of the triage (delivery sections are the only group attribute in the data)
t = tiers(p, base, 3 * base); df['_flag'] = (t >= 1).astype(int)
sec = df.groupby('Section').agg(sessions=('y', 'size'), disruption_rate=('y', 'mean'), flagged_share=('_flag', 'mean')).round(3)
sec['recall_flagged'] = df[df.y == 1].groupby('Section')['_flag'].mean().round(3)
sec['disrupted'] = df.groupby('Section')['y'].sum()
sec.to_csv(f'{OUT}/Table_section_audit.csv')
from scipy.stats import chi2_contingency
S['section_flag_chi2'] = [round(float(x), 3) for x in chi2_contingency(pd.crosstab(df.Section, df._flag))[:2]]
S['section_recall_chi2'] = [round(float(x), 3) for x in chi2_contingency(pd.crosstab(df[df.y == 1].Section, df[df.y == 1]._flag))[:2]] if df[df.y==1]._flag.nunique()>1 else None

# ------------------------------------------------------------------ figures
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5})   # journal: Arial/Helvetica, 8-12 pt
BLUE, GREY, ORANGE = '#2c6fad', '#8a8f98', '#d1701f'
# Fig 2: what survives each protocol
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.0))
nv = A[A.Protocol.str.startswith('Naive')]; ct = A[A.Protocol.str.startswith('Leak')]
x = np.arange(3); wd = 0.36
for a_, col, ttl in [(ax[0], 'Accuracy', 'Accuracy'), (ax[1], 'Macro_F1', 'Macro-F1')]:
    b1 = a_.bar(x - wd / 2, nv[col], wd, color=GREY, label='Naive protocol'); b2 = a_.bar(x + wd / 2, ct[col], wd, color=BLUE, hatch='///', edgecolor='white', linewidth=0, label='Leakage-controlled')
    for b in b1: a_.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f'{b.get_height():.2f}', ha='center', fontsize=8)
    for b in b2: a_.text(b.get_x() + b.get_width() / 2, b.get_height() - 0.07, f'{b.get_height():.2f}', ha='center', fontsize=8, color='white', fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.15', fc=BLUE, ec='none'))
    a_.set_xticks(x); a_.set_xticklabels(nv.Model, fontsize=8); a_.set_ylim(0, 1.12); a_.set_xlim(-0.6, 2.6); a_.set_ylabel(ttl)
ax[0].axhline(S['majority_class_accuracy'], ls='--', lw=0.8, color='k'); ax[0].text(2.58, S['majority_class_accuracy'] + 0.012, 'majority class', ha='right', va='bottom', fontsize=8)
hd, lb = ax[1].get_legend_handles_labels(); fig.legend(hd, lb, frameon=False, fontsize=8, loc='upper center', ncol=2)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f'{OUT}/Fig2_protocol_audit.png', dpi=600); plt.close(fig)
# Fig 3: ROC and PR, prospective task
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.1))
for lab, c, ls, short in [('M0 Examination type only (logistic)', GREY, '--', 'M0 examination type'), ('M1 Individual history only (logistic)', ORANGE, ':', 'M1 individual history'),
                          ('M2 Type + individual history (logistic)', '#1b3a5c', '-.', 'M2 type + history'), ('M4 Type + position + history: logistic regression', BLUE, '-', 'M4 type + position + history')]:
    o = IND if lab.startswith('M0') else OOF[lab][0]; f, tp, _ = roc_curve(yv, o); pr, rc, _ = precision_recall_curve(yv, o)
    ax[0].plot(f, tp, color=c, ls=ls, lw=1.4, label=short)
    if lab.startswith('M0'): ax[1].plot([yv[IND == 1].sum() / yv.sum()], [yv[IND == 1].mean()], 'o', color=c, ms=6)
    else: ax[1].step(rc[:-1], pr[:-1], where='post', color=c, ls=ls, lw=1.2)
ax[0].plot([0, 1], [0, 1], color='k', lw=0.5); ax[0].set_xlabel('False-positive rate'); ax[0].set_ylabel('True-positive rate')
ax[1].axhline(base, color='k', lw=0.5); ax[1].text(0.99, base - 0.012, 'prevalence', ha='right', va='top', fontsize=8)
ax[1].set_xlabel('Recall'); ax[1].set_ylabel('Precision'); ax[1].set_ylim(0, 0.8); ax[0].legend(frameon=False, fontsize=8, loc='lower right', handlelength=2.6)
fig.tight_layout(); fig.savefig(f'{OUT}/Fig3_prospective_roc_pr.png', dpi=600); plt.close(fig)
# Fig 4: attribution
NAMES = {'practical': 'Practical examination', 'n_prior': 'Sitting position (prior sessions)', 'dis_prior': 'Prior disrupted sessions',
         'blank_prior_mean': 'Prior mean blank ratio', 'blank_prior_max': 'Prior maximum blank ratio'}
if sv is not None:
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.0), gridspec_kw={'width_ratios': [1, 1.15]})
    o = np.argsort(np.abs(sv).mean(0)); ax[0].barh([NAMES[FULL[i]] for i in o], np.abs(sv).mean(0)[o], color=BLUE)
    ax[0].set_xlabel('Mean |SHAP| (log-odds)'); ax[0].tick_params(axis='y', labelsize=8.5)
    rng = np.random.default_rng(SEED)
    for r_, i in enumerate(o):
        v = df[FULL[i]].values.astype(float); cn = (v - v.min()) / (v.max() - v.min() + 1e-9)
        scat = ax[1].scatter(sv[:, i], r_ + rng.uniform(-0.28, 0.28, len(v)), c=cn, cmap='coolwarm', s=5, alpha=0.7, linewidths=0)
    ax[1].axvline(0, color='k', lw=0.5); ax[1].set_yticks(range(len(o))); ax[1].set_yticklabels([]); ax[1].set_xlabel('SHAP value (log-odds of disruption)')
    cb = fig.colorbar(scat, ax=ax[1], fraction=0.04, pad=0.02); cb.set_ticks([0, 1]); cb.set_ticklabels(['Low', 'High']); cb.set_label('Feature value', fontsize=8)
    fig.tight_layout(); fig.savefig(f'{OUT}/Fig4_attribution_shap.png', dpi=600); plt.close(fig)
# Fig 5: triage tiers
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.0))
lab_ = ['Tier 1\nhigh risk', 'Tier 2\nelevated risk', 'Tier 3\nlow risk']; cols = [ORANGE, BLUE, GREY]
b = ax[0].bar(lab_, TT.Share_pct, color=cols); ax[0].set_ylabel('Share of sessions (%)')
for i, bb in enumerate(b): ax[0].text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 1, f'{TT.Share_pct[i]:.1f}%\n(n={TT.Sessions[i]})', ha='center', fontsize=8)
ax[0].set_ylim(0, max(TT.Share_pct) + 16)
b = ax[1].bar(lab_, TT.Observed_rate_pct, color=cols); ax[1].set_ylabel('Observed disruption rate (%)')
for i, bb in enumerate(b): ax[1].text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.8, f'{TT.Observed_rate_pct[i]:.1f}%\n({TT.Disrupted[i]}/{TT.Sessions[i]})', ha='center', fontsize=8)
ax[1].axhline(100 * base, color='k', lw=0.6, ls='--'); ax[1].text(2.45, 100 * base + 0.8, 'overall rate', ha='right', fontsize=8); ax[1].set_ylim(0, max(TT.Observed_rate_pct) + 12)
for a_ in ax: a_.tick_params(axis='x', labelsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/Fig5_triage.png', dpi=600); plt.close(fig)

json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1, default=str)
for nm, t_ in [('A protocol audit', A), ('B concurrent diagnostics', Bt), ('C prospective grouped CV', Ct), ('C temporal', Tt), ('D attribution', coef), ('E triage', TT), ('calibration', cal), ('sections', sec), ('by exam', per_exam)]:
    print('\n==', nm); print(t_.to_string())
print('\n', json.dumps(S, indent=1, default=str))
