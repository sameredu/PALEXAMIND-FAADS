"""Figure 1 of the v2.1.0 analysis (study design). Run from the repository root: python analysis/fig1_pipeline.py"""
import os, logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family': ['Arial', 'Liberation Sans', 'DejaVu Sans']})
OUT = os.environ.get('FAADS_OUT', 'results_v21'); os.makedirs(OUT, exist_ok=True)
W, H = 6.5, 5.9
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
INK, BLUE, LBLUE, GREY, LGREY, ORANGE, LORANGE = '#12263a', '#2c6fad', '#eaf2fa', '#5f6670', '#f3f4f6', '#c8641a', '#fdf2e7'

def box(x, y, w, h, fc, ec, ls='-', lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2', fc=fc, ec=ec, lw=lw, linestyle=ls))
def title(x, y, s, size=9, color=INK, ha='center'): ax.text(x, y, s, ha=ha, va='center', fontsize=size, fontweight='bold', color=color)
def body(x, y, s, size=8, color='#222222', style='normal', ha='center'):
    ax.text(x, y, s, ha=ha, va='top', fontsize=size, color=color, linespacing=1.35, fontstyle=style)
def arrow(x1, y1, x2, y2, c=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=11, lw=1.2, color=c, shrinkA=0, shrinkB=0))
def band(y, s, x=1): ax.text(x, y, s, ha='left', va='center', fontsize=8.5, fontweight='bold', color=GREY)

# ---- band 1: data and label
band(97.2, 'DATA AND LABEL')
for x, t, b, fc, ec, ls in [
    (1, 'Attempt reports', '675 sessions\n226 students\n7 sittings', LBLUE, BLUE, '-'),
    (35, 'Session record', 'blank ratio r\nstart time, duration\nsitting type, prior history', LBLUE, BLUE, '-'),
    (69, 'Weak label', 'substantial disruption\nr ≥ 0.50\n64 sessions (9.5%)', LGREY, GREY, '--')]:
    box(x, 76, 30, 18.5, fc, ec, ls); title(x + 15, 91, t); body(x + 15, 87.6, b)
arrow(31.4, 85.2, 34.6, 85.2); arrow(65.4, 85.2, 68.6, 85.2)

# ---- protocol bar
box(1, 63, 98, 9, LGREY, GREY, '--')
title(50, 69.4, 'Leakage-controlled protocol')
body(50, 67.2, 'folds grouped by student   ·   resampling inside training folds   ·   features from prior sessions only')
arrow(50, 75.6, 50, 72.4)

# ---- band 2: three analyses
band(59.6, 'THREE ANALYSES', x=19)
for x, t, b, q in [
    (1, 'A  Protocol audit', 'released five-class pipeline\nnaive vs controlled protocol', 'How much accuracy\nsurvives?'),
    (35, 'B  Concurrent detection', 'features and label from\nthe same session', 'Is the task circular?'),
    (69, 'C  Prospective prediction', 'only information available\nbefore the sitting (M0–M4)', 'Does individual history\nadd to sitting type?')]:
    box(x, 31, 30, 26, LBLUE, BLUE); title(x + 15, 53.6, t); body(x + 15, 50.4, b)
    ax.plot([x + 3, x + 27], [42.4, 42.4], color=BLUE, lw=0.6)
    body(x + 15, 41.0, q, style='italic', color=INK)
for x in (16, 50, 84): arrow(x, 62.6, x, 57.4)

# ---- band 3: decision layer
band(26.6, 'DECISION LAYER')
box(1, 1, 98, 23, LORANGE, ORANGE)
title(50, 20.6, 'D  Selective triage of out-of-fold risk (three tiers)')
for x, t, b in [(3, 'Before the sitting', 'safeguards for the\nwhole sitting'),
                (35.5, 'After the sitting', 'session-level review\n(companion index)'),
                (68, 'Automatic individual action', 'only if a tier is\nprecise enough')]:
    box(x, 3, 29, 14, 'white', ORANGE, lw=0.8); title(x + 14.5, 14.0, t, size=8.5); body(x + 14.5, 11.6, b)
arrow(84, 30.6, 84, 24.4, c=ORANGE)
fig.savefig(f'{OUT}/Fig1_pipeline.png', dpi=600)
