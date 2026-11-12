"""
Rebuild results/processed_features_real.csv from the seven Zenodo files
(https://doi.org/10.5281/zenodo.23643942) and compare it with the committed copy.
Usage (from the repository root):
    FAADS_RAW=/path/to/zenodo/files python analysis/build_features_from_raw.py
The feature and label code is the released v1.x/v2.0 code, unchanged, so that the audit in
analysis/faads_v21.py runs on exactly the features the pipeline used.
"""
import os, re, sys
import numpy as np, pandas as pd

RAW = os.environ.get('FAADS_RAW', 'data')
REF = 'results/processed_features_real.csv'
FILES = {'Midterm1': 'Midterm1', 'Mid2_Theory': 'Midterm2_Theory', 'Mid2_Prac': 'Midterm2_Practical',
         'Final_Theory': 'Final_Theory', 'Final_Prac': 'Final_Practical',
         'Supp_Theory': 'Supp_Theory', 'Supp_Prac': 'Supp_Practical'}
EXAM_MAX = {'Midterm1': 25, 'Mid2_Theory': 15, 'Mid2_Prac': 10, 'Final_Theory': 20, 'Final_Prac': 10,
            'Supp_Theory': 10, 'Supp_Prac': 10}

def dur_sec(s):
    s = str(s); t = 0
    for pat, k in ((r'(\d+)\s*hour', 3600), (r'(\d+)\s*min', 60), (r'(\d+)\s*sec', 1)):
        m = re.search(pat, s); t += int(m.group(1)) * k if m else 0
    return t

rows = []
for exam, stem in FILES.items():
    df = pd.read_excel(os.path.join(RAW, f'DataBase_WEB_{stem}_Anonymized.xlsx'))
    q = [c for c in df.columns if c.startswith('Q')]
    sc = [c for c in df.columns if 'Total_Score' in c][0]
    for _, r in df.iterrows():
        v = [str(r[c]).strip() for c in q]; nq = len(q)
        un = sum(x == 'Unanswered' for x in v); ng = sum(x == 'Needs_Grading' for x in v)
        d = dur_sec(r['Duration'])
        try: sm = pd.to_datetime(str(r['Start_Time']), dayfirst=True).minute
        except Exception: sm = 0
        idx = [i for i, x in enumerate(v) if x == 'Unanswered']
        igv = float(np.var(np.diff(idx))) if len(idx) >= 2 else (idx[0] / max(nq, 1) if len(idx) == 1 else 0.0)
        s = pd.to_numeric(str(r[sc]).strip(), errors='coerce')
        br = un / max(nq, 1)
        p1 = un == nq and nq > 0 and d >= 3600
        p2 = sm > 30 and d < 600
        p3 = br > 0.5 and not p1 and nq > 1
        rows.append(dict(Student_Label=r['Student_Label'], Section=r['Section'], Exam=exam, IGV=igv, IL=float(sm),
                         SER=(nq - un - ng) / max(nq, 1), n_unanswered=un, n_questions=nq, blank_ratio=br,
                         duration_sec=d, start_min=sm, score_pct=(s / EXAM_MAX[exam] * 100) if pd.notna(s) else np.nan,
                         Label='P1' if p1 else 'P2' if p2 else 'P3' if p3 else 'Normal'))
f = pd.DataFrame(rows)
mu = f.groupby('Student_Label')[['IGV', 'IL', 'SER']].transform('mean')
f['BFD'] = np.sqrt(((f[['IGV', 'IL', 'SER']] - mu) ** 2).sum(axis=1))
f['BFD'] = (f.BFD - f.BFD.min()) / (f.BFD.max() - f.BFD.min() + 1e-9)
f['is_disrupted'] = f.Label != 'Normal'
f['RIF'] = f.Student_Label.map(f.groupby('Student_Label').is_disrupted.sum())
f.loc[(f.Label == 'Normal') & (f.RIF >= 2), 'Label'] = 'P4'
print('sessions', len(f), '| students', f.Student_Label.nunique(), '| labels', f.Label.value_counts().to_dict())
print('substantial disruption (blank ratio >= 0.50):', int((f.blank_ratio >= 0.5).sum()))
if os.path.exists(REF):
    ref = pd.read_csv(REF); a = f.sort_values(['Student_Label', 'Exam']).reset_index(drop=True)
    b = ref.sort_values(['Student_Label', 'Exam']).reset_index(drop=True); ok = True
    for c in ['IGV', 'IL', 'SER', 'BFD', 'RIF', 'blank_ratio', 'duration_sec']:
        dmax = float(np.abs(a[c].astype(float) - b[c].astype(float)).max()); ok &= dmax < 1e-6
        print(f'  {c:12s} max abs difference vs committed file: {dmax:.2e}')
    same = bool((a.Label.values == b.Label.values).all()); ok &= same
    print('  labels identical:', same); print('MATCH' if ok else 'MISMATCH'); sys.exit(0 if ok else 1)
f.to_csv(REF, index=False); print('written', REF)
