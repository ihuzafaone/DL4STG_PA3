"""Build report/report.pdf with reportlab. Every number is read from results/*.json|csv (no hand-typed numbers).

Usage:  python src/build_report.py
"""
import os, re, sys, json, datetime
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, CondPageBreak, KeepTogether, Preformatted
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from common import ROOT, RES, FIG, PROC, load_json

REPO = 'https://github.com/ihuzafaone/DL4STG_PA3.git'
FD = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
for nm, fn in (('DV', 'DejaVuSans.ttf'), ('DV-B', 'DejaVuSans-Bold.ttf'), ('DV-I', 'DejaVuSans-Oblique.ttf'), ('DV-BI', 'DejaVuSans-BoldOblique.ttf'), ('DVMono', 'DejaVuSansMono.ttf')):
    pdfmetrics.registerFont(TTFont(nm, os.path.join(FD, fn)))
pdfmetrics.registerFontFamily('DV', normal='DV', bold='DV-B', italic='DV-I', boldItalic='DV-BI')

# ------------------------------------------------------------------ styles
BODY = ParagraphStyle('body', fontName='DV', fontSize=8.4, leading=11.1, alignment=TA_JUSTIFY, spaceAfter=3.5)
SMALL = ParagraphStyle('small', parent=BODY, fontSize=7.6, leading=10, spaceAfter=2)
CAP = ParagraphStyle('cap', parent=BODY, fontSize=7.8, leading=10, alignment=TA_CENTER, textColor=colors.HexColor('#333333'), spaceAfter=7)
H1 = ParagraphStyle('h1', fontName='DV-B', fontSize=14, leading=17, spaceBefore=6, spaceAfter=6, textColor=colors.HexColor('#12355b'))
H2 = ParagraphStyle('h2', fontName='DV-B', fontSize=10.6, leading=13, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor('#12355b'))
H3 = ParagraphStyle('h3', fontName='DV-B', fontSize=9.2, leading=12, spaceBefore=5, spaceAfter=2, textColor=colors.HexColor('#2b2b2b'))
TITLE = ParagraphStyle('title', fontName='DV-B', fontSize=20, leading=25, alignment=TA_CENTER, textColor=colors.HexColor('#12355b'), spaceAfter=6)
SUB = ParagraphStyle('sub', fontName='DV', fontSize=11, leading=15, alignment=TA_CENTER, spaceAfter=3)
CELL = ParagraphStyle('cell', fontName='DV', fontSize=7.4, leading=9.2)
CELLB = ParagraphStyle('cellb', parent=CELL, fontName='DV-B')
CODE = ParagraphStyle('code', fontName='DVMono', fontSize=7, leading=8.6, backColor=colors.HexColor('#f3f3f3'), borderPadding=3, spaceAfter=5)

STORY, FIGN = [], [0]


def esc(t): return t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def md(t):
    """tiny markdown -> reportlab markup (bold, italic, code); <sub>, <super>/<sup>, <b>, <i> pass through."""
    t = t.replace('<sup>', '<super>').replace('</sup>', '</super>')
    t = esc(t)
    t = re.sub(r'&lt;(/?(?:sub|super|b|i))&gt;', r'<\1>', t)
    t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t, flags=re.S)
    t = re.sub(r'(?<![\*\w])\*(?!\*)(.+?)(?<!\*)\*(?![\*\w])', r'<i>\1</i>', t, flags=re.S)
    t = re.sub(r'`([^`]+)`', r'<font name="DVMono" size="7.4">\1</font>', t)
    return t


def P(t, st=BODY, raw=False): STORY.append(Paragraph(t if raw else md(t), st))
def paras(t, st=BODY):
    for b in [x.strip() for x in t.split('\n\n') if x.strip()]: P(b, st)
def H(level, t): STORY.append(Paragraph(esc(t), {1: H1, 2: H2, 3: H3}[level]))
def SP(h=4): STORY.append(Spacer(1, h))
def code(t): STORY.append(Preformatted(t, CODE))


def fig(name, width_cm, caption):
    path = os.path.join(FIG, name); assert os.path.exists(path), path
    w, h = PILImage.open(path).size; FIGN[0] += 1
    W = min(width_cm * 0.9, 16.4) * cm; img = Image(path, width=W, height=W * h / w)
    STORY.append(KeepTogether([img, Paragraph(f'<b>Figure {FIGN[0]}.</b> {md(caption)}', CAP)]))


def table(rows, widths=None, header=True, size=None, align_right_from=None, zebra=True):
    st = CELL if size is None else ParagraphStyle('c', parent=CELL, fontSize=size, leading=size * 1.25)
    stb = ParagraphStyle('cb', parent=st, fontName='DV-B')
    data = [[Paragraph(md(str(c)), stb if (header and i == 0) else st) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths] if widths else None, repeatRows=1 if header else 0)
    ts = [('GRID', (0, 0), (-1, -1), .3, colors.HexColor('#999999')), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2)]
    if header: ts.append(('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dfe8f3')))
    if zebra:
        for i in range(2, len(rows), 2): ts.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f6f8fb')))
    t.setStyle(TableStyle(ts)); STORY.append(t); SP(5)


def nice(n):
    base = {'temporal_only': 'Temporal-only', 'stgnn_fixed': 'Fixed', 'stgnn_adaptive': 'Adaptive', 'stgnn_hybrid': 'Hybrid'}
    return base[n.replace('_skip', '')] + (' + self path' if n.endswith('_skip') else '')
def f1(x): return f'{x:.1f}'
def f2(x): return f'{x:.2f}'
def f3(x): return f'{x:.3f}'
def pct(x, d=1): return f'{100*x:.{d}f} %'
def tfmt(s): return f'{s:.0f} s ({s/60:.1f} min)'


# ------------------------------------------------------------------ load results
q1, q2, q3 = load_json(f'{RES}/q1_results.json'), load_json(f'{RES}/q2_results.json'), load_json(f'{RES}/q3_results.json')
S45 = load_json(f'{RES}/q4_q5_summary.json'); q6 = load_json(f'{RES}/q6_results.json'); NARR = load_json(f'{RES}/q6_narrative.json'); HW = load_json(f'{RES}/hardware.json')
NAMES = ['temporal_only', 'stgnn_fixed', 'stgnn_adaptive', 'stgnn_hybrid', 'stgnn_fixed_skip', 'stgnn_adaptive_skip', 'stgnn_hybrid_skip']
R42 = {n: load_json(f'{RES}/train_{n}_seed42.json') for n in NAMES}
IT1 = {n: load_json(f'{RES}/iter1_global_norm/train_{n}_seed42.json') for n in ('temporal_only', 'stgnn_fixed')}
CFG = R42['temporal_only']['cfg']
NARR['6b'] = re.sub(r'\*\*\((\d)\) ', lambda m: f'**Q6(b)-{m.group(1)} — ', NARR['6b'])
NARR['6c'] = re.sub(r'\*\*\((\d)\) ', lambda m: f'**Q6(c)-{m.group(1)} — ', NARR['6c'])
ms_df = pd.read_csv(f'{RES}/q6a_multiseed.csv')
from models import TemporalEncoder
KS = TemporalEncoder(CFG['hidden_d'], 8).c1.kernel_size[0]
miss = pd.read_csv(f'{RES}/q3_missing_per_sensor.csv')
now = datetime.datetime.now().strftime('%d %b %Y')


def mean_std_row(n):
    r = ms_df[ms_df['name'] == n].iloc[0]
    return int(r['n_seeds']), r['test_MAE_mean'], r['test_MAE_std'], r['test_RMSE_mean'], r['test_RMSE_std'], r['time_mean'], r['time_std']


# =================================================================== COVER
STORY.append(Spacer(1, 1.2 * cm))
STORY.append(Paragraph('Spatio-Temporal Traffic Forecasting on LargeST San Diego', TITLE))
STORY.append(Paragraph('Fixed, learnable and hybrid graph adjacency', SUB)); SP(14)
for ln in ('<b>Name:</b> Huzaifa', '<b>Roll No:</b> 25280009', '<b>Course:</b> AI651 – Deep Learning for Space, Time and Graphs (Fall 2026)', '<b>Assignment:</b> Assignment 3 (due Sun 22 Nov 2026, 11:55 pm)',
           f'<b>Source code (GitHub):</b> <link href="{REPO}" color="blue">{REPO}</link>', f'<b>Report generated:</b> {now} · global seed 42 · hardware: {HW.get("gpu") or HW["cpu_model"]}'):
    STORY.append(Paragraph(ln, SUB))
SP(16)
H(2, 'Summary of results (test period Apr–Dec 2021, vehicles per 15 min; mean ± std over seeds 42, 43, 44)')
rows = [['Model', 'Adjacency', 'Test MAE — exact assignment formula H = ReLU(Â Z W<sub>g</sub>) (main table)', 'Test MAE — with self path (labelled extension)', 'Test RMSE (exact / self path)', 'Training time, seed 42 (exact formula)']]
for n, m, a in (('temporal_only', 'Temporal-only', 'None'), ('stgnn_fixed', 'STGNN', 'Fixed'), ('stgnn_adaptive', 'STGNN', 'Adaptive'), ('stgnn_hybrid', 'STGNN', 'Hybrid')):
    k0, mm0, ss0, rm0, rs0, _, _ = mean_std_row(n)
    if n == 'temporal_only':
        rows.append([m, a, f'**{mm0:.1f} ± {ss0:.1f}**', '— (no graph layer; same model)', f'{rm0:.1f} ± {rs0:.1f}', tfmt(R42[n]['train_time_sec'])])
    else:
        k1, mm1, ss1, rm1, rs1, _, _ = mean_std_row(n + '_skip')
        rows.append([m, a, f'{mm0:.1f} ± {ss0:.1f}', f'**{mm1:.1f} ± {ss1:.1f}**', f'{rm0:.1f} ± {rs0:.1f} / {rm1:.1f} ± {rs1:.1f}', tfmt(R42[n]['train_time_sec'])])
table(rows, [2.6, 2.0, 3.9, 3.4, 3.1, 2.4], size=7.2)
_skip_better = all(R42[n]['test']['mae'] < R42['temporal_only']['test']['mae'] for n in ('stgnn_fixed_skip', 'stgnn_adaptive_skip', 'stgnn_hybrid_skip'))
_lit_best = R42['temporal_only']['test']['mae'] < min(R42[n]['test']['mae'] for n in ('stgnn_fixed', 'stgnn_adaptive', 'stgnn_hybrid'))
P(f"With the graph layer exactly as specified, H = ReLU(Â Z W<sub>g</sub>), {'the temporal-only model is the best of the four (no graph variant beats it)' if _lit_best else 'see the table'}. Adding a self path, H = ReLU(Â Z W<sub>g</sub> + Z W<sub>s</sub>) (a documented extension), {'reverses this' if _skip_better else 'gives'}: test MAE {f2(R42['stgnn_fixed_skip']['test']['mae'])} (fixed), {f2(R42['stgnn_adaptive_skip']['test']['mae'])} (adaptive), {f2(R42['stgnn_hybrid_skip']['test']['mae'])} (hybrid) versus {f2(R42['temporal_only']['test']['mae'])} for temporal-only. Details, multi-seed statistics and the learned-graph analysis are in Q4–Q6. "
  f"All code, notebooks and results are in the repository above (to be public at submission).", BODY, raw=True)
SP(6)
H(3, 'How to read this report')
P('Sections follow the assignment: Q1 → Q2 → Q3 → Q4 → Q5 → Q6 → Q7 (reproducibility, hardware, configuration) → Generative AI usage. Every sub-part is labelled as in the assignment (e.g. “Q3(b)-2”). All numbers are produced by the notebooks and read from `results/*.json|csv` by `src/build_report.py`; all figures are saved in `figures/` and embedded here.')
STORY.append(PageBreak())

# =================================================================== Q1
H(1, 'Question 1 — What does a sum remember?')
H(2, 'Q1(a) Affine message')
a1 = q1['a']
paras(f"""**Aggregated message.** For ϕ(x) = w·x + c and a neighbourhood N(v) with n<sub>v</sub> = |N(v)| and S<sub>v</sub> = Σ<sub>u∈N(v)</sub> x<sub>u</sub>:
Σ<sub>u</sub> ϕ(x<sub>u</sub>) = Σ<sub>u</sub> (w x<sub>u</sub> + c) = w Σ<sub>u</sub> x<sub>u</sub> + c Σ<sub>u</sub> 1 = **w·S<sub>v</sub> + c·n<sub>v</sub>**.

**Why equal (n<sub>v</sub>, S<sub>v</sub>) force equal outputs.** h<sub>v</sub> = ρ(Σ<sub>u</sub> ϕ(x<sub>u</sub>)) = ρ(w S<sub>v</sub> + c n<sub>v</sub>). The nonlinearity ρ acts *after* aggregation and there is no degree normalisation, so the only thing ρ ever sees is the single scalar w S<sub>v</sub> + c n<sub>v</sub>. If two neighbourhoods have the same n<sub>v</sub> and S<sub>v</sub>, this scalar is identical for **every** choice of (w, c), hence ρ receives the same input and returns the same output for **every** function ρ (ReLU, tanh, any MLP). The layer cannot tell how many neighbours are steady (0): a 0 only contributes through c (i.e. through n<sub>v</sub>), which is the same as a +1 and a −1 cancelling.

**Construction.** N₁ = {{+1, −1, 0, 0}} and N₂ = {{0, 0, 0, 0}}: both have n<sub>v</sub> = {a1['n']} and S<sub>v</sub> = {a1['S']}, but N₁ has {a1['steady_N1']} steady neighbours and N₂ has {a1['steady_N2']}.
**Numerical verification:** for {a1['n_random_trials']:,} random draws of (w, c) and of a randomly initialised MLP ρ, the outputs for N₁ and N₂ are identical: the maximum difference is {a1['max_output_gap_closed_form']:.1f} when ρ is fed w·S+c·n and {a1['max_output_gap_sum_of_messages']:.1e} (floating-point round-off of summing the messages in a different order) when the four messages are summed explicitly.""")

H(2, 'Q1(b) A single number that remembers all three counts')
b1 = q1['b']
paras(f"""Let n<sub>−</sub>, n<sub>0</sub>, n<sub>+</sub> be the numbers of slowing (−1), steady (0) and speeding-up (+1) neighbours, with 1 ≤ n<sub>−</sub> + n<sub>0</sub> + n<sub>+</sub> ≤ K. Take the base **B = K + 1** and **ϕ(−1) = 1, ϕ(0) = B, ϕ(+1) = B²**. The summed message is s = n<sub>−</sub> + n<sub>0</sub>·B + n<sub>+</sub>·B². Each count satisfies 0 ≤ n ≤ K < B, so each is a legal base-B digit and **no carry** can occur; by uniqueness of base-B representation (n<sub>+</sub>, n<sub>0</sub>, n<sub>−</sub>) are exactly the digits of s. A base of only K would fail because a count of K would carry into the next digit.

**Decoding:** n<sub>−</sub> = s mod B, n<sub>0</sub> = ⌊s/B⌋ mod B, n<sub>+</sub> = ⌊s/B²⌋.

**Verification for K = 8** (B = {b1['B']}, ϕ = (1, {b1['B']}, {b1['B']**2})): over **every** triple of counts with 1 ≤ n<sub>−</sub>+n<sub>0</sub>+n<sub>+</sub> ≤ 8 — **{b1['n_triples']} triples** (= C(11,3) − 1) — all {b1['n_triples']} sums are distinct (assert passed; sums range from {b1['min_sum']} to {b1['max_sum']}) and decoding recovers the three counts exactly in every case.

**Counting non-steady neighbours with one message:** ϕ(x) = x² (or |x|) gives ϕ(±1) = 1 and ϕ(0) = 0, so Σ ϕ(x<sub>u</sub>) = n<sub>−</sub> + n<sub>+</sub> = number of non-steady neighbours for any neighbourhood size (verified on all {b1['n_triples']} triples).""")

H(2, 'Q1(c) Sum, mean and max with one-hot messages')
c1 = q1['c']; ce = c1['counterexamples']
def setstr(x): return '{' + ', '.join(f'{int(v):+d}' if v else '0' for v in x) + '}'
def vec(v): return '(' + ', '.join(f'{x:.3g}' if abs(x - round(x)) > 1e-9 else f'{int(round(x))}' for x in v) + ')'
def no_reason(key):
    e = ce[key]; return f"**No.** N<sub>A</sub> = {setstr(e['nbhd1'])} and N<sub>B</sub> = {setstr(e['nbhd2'])} give the same vector {vec(e['vector'])}, but the correct answers are {e['ans1']} and {e['ans2']}, so ρ cannot separate them."
T = c1['table']
rows = [['Task', 'Method', 'Possible?', 'Reason']]
rows += [['(i) count steady neighbours', 'Sum', 'Yes' if T['(i) #steady|Sum'] else 'No', 'The sum vector is (n<sub>−</sub>, n<sub>0</sub>, n<sub>+</sub>); ρ returns **coordinate 2**, which equals the number of steady neighbours.'],
         ['', 'Mean', 'No' if not T['(i) #steady|Mean'] else 'Yes', no_reason('(i) #steady|Mean')],
         ['', 'Max', 'No' if not T['(i) #steady|Max'] else 'Yes', no_reason('(i) #steady|Max')],
         ['(ii) more than half slowing', 'Sum', 'Yes' if T['(ii) >half slowing|Sum'] else 'No', 'Uses **all three coordinates**: since n = v<sub>1</sub>+v<sub>2</sub>+v<sub>3</sub>, “more than half slowing” ⇔ v<sub>1</sub> > n/2 ⇔ **v<sub>1</sub> > v<sub>2</sub> + v<sub>3</sub>** (compare coordinate 1 with n − coordinate 1).'],
         ['', 'Mean', 'Yes' if T['(ii) >half slowing|Mean'] else 'No', 'Coordinate 1 equals n<sub>−</sub>/n; the test is **m<sub>1</sub> > 0.5** (only coordinate 1 is used).'],
         ['', 'Max', 'No' if not T['(ii) >half slowing|Max'] else 'Yes', no_reason('(ii) >half slowing|Max')],
         ['(iii) any neighbour slowing', 'Sum', 'Yes' if T['(iii) any slowing|Sum'] else 'No', 'Test **v<sub>1</sub> > 0** (coordinate 1).'],
         ['', 'Mean', 'Yes' if T['(iii) any slowing|Mean'] else 'No', 'Test **m<sub>1</sub> > 0** (coordinate 1): n<sub>−</sub>/n > 0 ⇔ n<sub>−</sub> > 0.'],
         ['', 'Max', 'Yes' if T['(iii) any slowing|Max'] else 'No', 'Coordinate 1 of the max vector is 1 iff some neighbour is slowing; test **v<sub>1</sub> = 1**.']]
table(rows, [3.0, 1.6, 1.9, 10.5], size=7.4)
P(f"**Brute-force proof.** Aggregators are permutation-invariant, so a neighbourhood is fully described by its count triple; there are {c1['n_neighbourhoods_checked']} such neighbourhoods of size 1–8. For each (task, aggregator) pair the neighbourhoods were grouped by their aggregated vector; a task is solvable by some ρ **iff the correct answer is constant inside every group**. The code confirms exactly the table above (all 9 entries; assertion passed), and also checks every “Yes” rule (the explicit ρ in the table reproduces the correct answer for all {c1['n_neighbourhoods_checked']} neighbourhoods). The “No” examples are the smallest conflicting pairs found by the search (notebook `Q1_Q2_analysis`).")

# =================================================================== Q2
STORY.append(CondPageBreak(7 * cm))
H(1, 'Question 2 — Which way is the slowdown moving?')
H(2, 'Q2(a)(i) Best symmetric layer for one impulse')
ai = q2['a_i']
P("Impulse x<sub>20</sub>(0) = 1. The true next snapshot is 1 at sensor 19 and 0 everywhere else (including sensor 20). The layer predicts x̂<sub>i</sub> = a·x<sub>i</sub> + b(x<sub>i−1</sub> + x<sub>i+1</sub>). Predicted versus correct readings:")
table([['sensor', 'prediction', 'correct'], ['19', 'a·0 + b(x<sub>18</sub> + x<sub>20</sub>) = **b**', '**1**'], ['20', 'a·1 + b(x<sub>19</sub> + x<sub>21</sub>) = **a**', '**0**'], ['21', 'a·0 + b(x<sub>20</sub> + x<sub>22</sub>) = **b**', '**0**']], [2, 7, 2.5])
paras(f"""All other sensors have prediction 0 and truth 0. Hence the sum of squared errors is L(a, b) = (b − 1)² + a² + b². Setting ∂L/∂a = 2a = 0 gives **a = 0**; setting ∂L/∂b = 2(b − 1) + 2b = 4b − 2 = 0 gives **b = 1/2**. The Hessian diag(2, 4) is positive definite, so this is the global minimum, with L<sub>min</sub> = (1/2 − 1)² + 0 + (1/2)² = **1/2**.

Numerical check (least squares over all 40 sensors): a = {ai['a']:.4f}, b = {ai['b']:.4f}, **SSE = {ai['sse']:.4f}**; predictions at sensors 19, 20, 21 are {[round(x, 4) + 0.0 for x in ai['pred_19_20_21']]} versus truth {[round(x, 4) + 0.0 for x in ai['target_19_20_21']]}. The residual error cannot be removed: sensors 19 and 21 are mirror images and always receive the same prediction, but the truth needs 1 at sensor 19 and 0 at sensor 21.""")

H(2, 'Q2(a)(ii) Applying the fitted layer 20 times to a smooth bump')
aii = q2['a_ii']
fig('q2a_heatmaps.png', 15.5, 'Q2(a)(ii): left, the symmetric layer (a = 0, b = 1/2) applied iteratively to the Gaussian bump; right, the true motion (np.roll(x, −1) at each step). Both heatmaps use the same colour scale (vmin = 0, vmax = 1); x = sensor index, y = time step t = 0…20.')
P(f"**Comparison.** The predicted bump never leaves sensor 20 (its peak is at sensor 20 for all t = 0…20): the layer is a symmetric diffusion filter, so the peak decays from {aii['pred_peak_t0']:.2f} to {aii['pred_peak_t20']:.3f} and the width grows from σ = {aii['pred_std_t0']:.2f} to {aii['pred_std_t20']:.2f} sensors while the total mass ({aii['mass_t0']:.3f}) is conserved. The true bump translates one sensor per step against traffic without changing shape or height (peak {aii['truth_peak_t20']:.2f}, width σ = {aii['truth_std_t20']:.2f} at t = 20, now centred on sensor 0).")

H(2, 'Q2(b) Direction-aware weights')
qb = q2['b']
paras(f"""x̂<sub>i</sub>(t+1) = a x<sub>i</sub> + w<sub>−</sub> x<sub>i−1</sub> + w<sub>+</sub> x<sub>i+1</sub>. Since the truth is x<sub>i</sub>(t+1) = x<sub>i+1</sub>(t), choose **a = 0, w<sub>−</sub> = 0, w<sub>+</sub> = 1**. **Verification by substitution:** x̂<sub>i</sub> = 0·x<sub>i</sub> + 0·x<sub>i−1</sub> + 1·x<sub>i+1</sub> = x<sub>i+1</sub>(t) = x<sub>i</sub>(t+1) for every i and every input (indices mod 40). Numerically this was checked on {qb['n_random_inputs']:,} snapshots (random and one-hot) with exact equality, and a least-squares fit of (a, w<sub>−</sub>, w<sub>+</sub>) on random data returns {[round(x, 6) + 0.0 for x in qb['lstsq']]}, i.e. the solution is unique.

**Why this removes the difficulty of (a).** In (a) both neighbours share the weight b, so without sensor IDs the layer is invariant under reflection i → 40 − i; sensors 19 and 21, mirror images around an impulse at 20, must get identical predictions although the truth needs 19 → 1 and 21 → 0. Weighting x<sub>i−1</sub> and x<sub>i+1</sub> differently breaks the left/right symmetry: the layer is told which neighbour is ahead (i+1) and which is behind (i−1), and can simply copy the one ahead.

**As a 1D convolution.** The kernel entries, in the order that multiplies [x<sub>i−1</sub>, x<sub>i</sub>, x<sub>i+1</sub>], are **[w<sub>−</sub>, a, w<sub>+</sub>] = [0, 0, 1]**. PyTorch `Conv1d` computes cross-correlation (it does not flip the kernel), so the kernel is used exactly as written; a flipped kernel [1, 0, 0] would shift the pattern the other way (verified: [0,0,1] moves an impulse at 20 to {qb['impulse20_moves_to']}). The ring boundary is handled with **circular padding** (`padding=1, padding_mode='circular'`), so sensor 39's right neighbour is sensor 0 and sensor 0's left neighbour is 39. Verified with `torch.nn.Conv1d`: maximum absolute difference to `np.roll(x, −1)` = {qb['conv_max_abs_err']:.1e} (float32 round-off).""")

H(2, 'Q2(c) Using history with equal neighbour weights')
cc = q2['c']
paras(f"""x̂<sub>i</sub>(t+1) = a x<sub>i</sub>(t) + b(x<sub>i−1</sub>(t) + x<sub>i+1</sub>(t)) + c x<sub>i</sub>(t−1).

**Q2(c)(i) Backward motion, x<sub>i</sub>(t) = x<sub>i+1</sub>(t−1).** Applying the rule at sensor i−1 gives x<sub>i−1</sub>(t) = x<sub>i</sub>(t−1): the reading that was at sensor i at t−1 has moved to sensor i−1. Hence x<sub>i</sub>(t−1) = x<sub>i−1</sub>(t) can be rewritten as a current reading, and the target is x<sub>i+1</sub>(t) = [x<sub>i−1</sub>(t) + x<sub>i+1</sub>(t)] − x<sub>i−1</sub>(t) = [x<sub>i−1</sub>(t) + x<sub>i+1</sub>(t)] − x<sub>i</sub>(t−1). Therefore **a = 0, b = 1, c = −1** predicts the next snapshot exactly for every pair (x(t−1), x(t)) generated by the backward rule. Numerically: maximum error {cc['max_err_backward']:.1e} over {cc['n_pairs']:,} random pairs; least squares on backward pairs returns (a, b, c) = {[round(x, 6) + 0.0 for x in cc['lstsq_backward']]} (unique).

**Q2(c)(ii) Forward motion, x<sub>i</sub>(t+1) = x<sub>i−1</sub>(t), with the *same* (0, 1, −1).** Under forward motion the previous snapshot satisfies x<sub>i+1</sub>(t) = x<sub>i</sub>(t−1), i.e. x<sub>i</sub>(t−1) = x<sub>i+1</sub>(t). Substitution: x̂<sub>i</sub>(t+1) = x<sub>i−1</sub>(t) + x<sub>i+1</sub>(t) − x<sub>i</sub>(t−1) = x<sub>i−1</sub>(t) + x<sub>i+1</sub>(t) − x<sub>i+1</sub>(t) = x<sub>i−1</sub>(t), which is exactly the correct forward step. **So the same coefficients still work** (maximum error {cc['max_err_forward']:.1e} over {cc['n_pairs']:,} random pairs).

**Q2(c)(iii) Why the history disambiguates the direction.** The spatial term x<sub>i−1</sub>(t) + x<sub>i+1</sub>(t) is symmetric: it contains the right answer and the wrong one mixed together and cannot tell which is which. The previous snapshot carries the direction: the pair (x(t−1), x(t)) shows how the pattern just moved, because x<sub>i</sub>(t−1) equals the current reading of the neighbour the pattern came *from* (x<sub>i−1</sub>(t) under backward motion, x<sub>i+1</sub>(t) under forward motion). Subtracting x<sub>i</sub>(t−1) therefore cancels exactly the neighbour the pattern moved away from and leaves the other one automatically, with no direction-dependent spatial weights — the direction is encoded in *time* (the temporal difference), not in asymmetric spatial weights. Demonstration: for the same x(t) (impulse at 20) with a backward-motion history the layer predicts the impulse at sensor {q2['c_iii']['backward_pred_at']}; with a forward-motion history it predicts sensor {q2['c_iii']['forward_pred_at']}; the purely spatial symmetric layer gives {[round(x, 3) + 0.0 for x in q2['c_iii']['symmetric_layer_19_20_21']]} at sensors 19, 20, 21 and cannot choose.""")

# =================================================================== Q3
STORY.append(CondPageBreak(7 * cm))
H(1, 'Question 3 — Dataset preparation and graph construction (LargeST, San Diego)')
H(2, 'Q3(a) Data extraction and resampling')
P('**Q3(a)-1 Extract all 716 San Diego sensors.** The Kaggle release (`liuxu77/largest`) contains only the California files (`ca_his_raw_YYYY.h5`, `ca_meta.csv`, `ca_rn_adj.npy`; 8,600 sensors). As in the official LargeST notebook `data/sd/generate_sd_dataset.ipynb`, San Diego is the set of sensors with `District == 11`: **716 sensors, all in San Diego county**; their flow columns are the string sensor IDs and their rows in the adjacency are given by the meta column `ID2`. Only the 716 columns are read from the large HDF5 files.')
P(f"**Q3(a)-2 Keep 2020 and 2021.** Only `ca_his_raw_2020.h5` and `ca_his_raw_2021.h5` are used: 366×288 + 365×288 = {q3['resample']['T15']*3:,} five-minute rows (1 Jan 2020 00:00 → 31 Dec 2021 23:55, no gaps in the 5-min grid).")
fu = q3['flow_units']; rs = q3['resample']; mo = q3['missing_overall_5min']
P(f"**Sum vs mean (Q3(a)-3 decision).** PeMS 5-minute flow is the *number of vehicles that crossed the station in the 5-minute interval* (vehicles/5 min), not an hourly rate. This is confirmed by the data: all readings are non-negative integers (fraction integer-valued = {fu['frac_integer']:.2f}), and the fraction divisible by 12 is {fu['frac_div12']:.4f} ≈ 1/12 = {fu['expected_if_counts']:.4f}; hourly rates built from 5-minute counts would all be multiples of 12. **Decision: SUM** — X¹⁵<sub>t</sub> = Σ<sub>k=0..2</sub> X⁵<sub>3t+k</sub> = **vehicles per 15 minutes**; averaging would silently change the unit to “average vehicles per 5 min” and shrink all values by 3×. (The official helper notebook uses `resample('15T').mean()`; we sum because the assignment requires the unit to be preserved. Our values are therefore 3× the official LargeST scale.)")
P(f"**Q3(a)-3 Resampling.** Bins are left-labelled; the grid starts at 00:00 with no gaps so bin t is rows 3t…3t+2, giving T = {rs['T15']:,} 15-minute steps. NaN-aware rule: 3/3 valid readings → plain sum ({rs['bins_full']:,} bins); 2/3 valid → sum × 3/2 ({rs['bins_two']:,} bins); ≤ 1/3 valid → missing ({rs['bins_missing']:,} bins). The result was verified against `pandas.resample('15min').sum(min_count=3)`.")
H(3, 'Q3(a)-4 Missing values: detection and handling')
P(f"Three problems were measured per sensor (CSV `results/q3_missing_per_sensor.csv`) and overall on the raw 5-minute data ({mo['total_readings']:,} readings = 716 sensors × 210,528 steps): **NaN** (explicit gaps): {mo['nan']:,} ({f3(mo['nan_pct'])} %); **zeros**: {mo['zeros_total']:,} ({f3(mo['zeros_total_pct'])} %; a single 0 at night can be a genuine empty interval, runs of zeros indicate a dead sensor); **flat-lines**: runs of ≥ 36 consecutive *identical* readings (≥ 3 h), which are treated as sensor outages — {mo['stuck_zero']:,} readings in zero runs ({f3(mo['stuck_zero_pct'])} %) and {mo['stuck_positive']:,} in constant non-zero runs ({f3(mo['stuck_positive_pct'])} %). Runs are detected *within each split segment* (2020 | Jan–Mar 2021 | Apr–Dec 2021), so no run crosses a split boundary. Total treated as missing at 5-minute level: {mo['missing_5min']:,} = **{f3(mo['missing_5min_pct'])} %**; after 15-minute aggregation **{f3(rs['missing_15min_pct'])} %** of the entries are missing.")
rows = [['Statistic (per sensor, % of 5-min readings)', 'mean', 'median', 'max'],
        ['NaN', f3(miss.nan_pct.mean()), f3(miss.nan_pct.median()), f1(miss.nan_pct.max())], ['zero (any)', f3(miss.zero_pct.mean()), f3(miss.zero_pct.median()), f1(miss.zero_pct.max())],
        ['zero inside ≥3 h run', f3(miss.stuck_zero_pct.mean()), f3(miss.stuck_zero_pct.median()), f1(miss.stuck_zero_pct.max())], ['constant non-zero ≥3 h run', f3(miss.stuck_positive_pct.mean()), f3(miss.stuck_positive_pct.median()), f1(miss.stuck_positive_pct.max())],
        ['missing (NaN or ≥3 h run), all years', f3(miss.missing_5min_pct.mean()), f3(miss.missing_5min_pct.median()), f1(miss.missing_5min_pct.max())],
        ['missing, 2020', f3(miss.missing_5min_pct_2020.mean()), f3(miss.missing_5min_pct_2020.median()), f1(miss.missing_5min_pct_2020.max())], ['missing, 2021', f3(miss.missing_5min_pct_2021.mean()), f3(miss.missing_5min_pct_2021.median()), f1(miss.missing_5min_pct_2021.max())]]
table(rows, [8, 2.4, 2.4, 2.4])
P(f"Sensors that are entirely missing in both years: **{int((miss.missing_5min_pct == 100).sum())}**; sensors with > 50 % missing: **{int((miss.missing_5min_pct > 50).sum())}**; sensors with no missing reading at all: {int((miss.missing_5min_pct == 0).sum())}. Valid (observed) 15-min entries per split: " + ', '.join(f"{k}: {v['valid_pct']:.2f} %" for k, v in q3['valid_pct_per_split'].items()) + '.')
fig('q3a_missingness.png', 15.5, 'Q3(a)-4: left, histogram of the percentage of missing 5-min readings per sensor (log count); right, fraction of each day that is missing for every sensor (rows) over 2020–2021. The sensors with no data at all (19) appear as horizontal black lines across both years.')

_d = q3['dst']; _sp = [v for v in _d['days'].values() if v['kind'] == 'spring-forward']; _fb = [v for v in _d['days'].values() if v['kind'] == 'fall-back']
_sp_in = ', '.join('%.2f' % v['nan_frac_in_hour'] for v in _sp); _sp_out = ', '.join('%.3f' % v['nan_frac_other_hours_mean'] for v in _sp); _fb_in = ', '.join('%.3f' % v['nan_frac_in_hour'] for v in _fb)
P(f"**Daylight-saving time.** The 5-minute grid is in *local wall-clock time*: the morning rush-hour flow on weekdays starts rising (half of its morning peak) at {_d['morning_rise_hour_winter']:.2f} h in January–February 2021 and at {_d['morning_rise_hour_summer']:.2f} h in July–August 2021 (peak at {_d['morning_peak_hour_winter']:.2f} h and {_d['morning_peak_hour_summer']:.2f} h), i.e. no one-hour seasonal shift, so the data are DST-adjusted. Consequences: on the two spring-forward days (the clock skips 02:00–03:00) the non-existent hour is missing for every well-observed sensor (NaN fraction in that hour: {_sp_in} against {_sp_out} in the other hours of the same days) — four consecutive missing 15-minute steps, which the missing-value policy below handles like any other gap (targets masked; the causal 4-step forward fill bridges exactly one hour for inputs); on the two fall-back days the repeated hour is not duplicated in the grid (all {_fb[0]['steps_in_day']} five-minute rows of the day are present, NaN fraction in 01:00–02:00 = {_fb_in}), so those days contain 24 one-hour slots for a 25-hour day.")
im = q3['imputation']
paras(f"""**Handling policy (no leakage).** (1) NaN and ≥3 h stuck runs → missing; 15-min bins with ≤ 1 valid reading → missing. (2) **Targets, loss and metrics** use only genuinely observed entries (valid mask) — imputed values are never used as targets. (3) **Model inputs** need a number everywhere: (i) *causal forward-fill* from the last observed value, at most 4 steps (1 hour), applied separately inside each split so no information crosses split boundaries and no future value is used ({im['forward_filled']:,} entries = {f3(im['forward_filled_pct'])} %); (ii) everything still missing is set to the **per-sensor mean of the training split (2020) observed values** (global training mean {f1(im['global_train_mean'])} veh/15 min for the {im['sensors_without_2020_obs']} sensors with no 2020 observation): {im['train_mean_filled']:,} entries = {f2(im['train_mean_filled_pct'])} %. Observed entries: {im['observed']:,} = {f2(im['observed_pct'])} % of all {im['total']:,}. No validation or test statistic is used for any imputation. The processed tensor X ∈ ℝ<super>T×716×1</super> (T = {rs['T15']:,}) with timestamps and the validity mask are saved in `data/processed/` (`sd_flow_15min_filled.npy`, `sd_flow_15min_raw.npy` (NaN = missing), `sd_valid_mask.npy`, `sd_timestamps.npy`).""")
wk = q3['week_plot']
fig('q3a_three_sensors_week.png', 15.5, f"Q3(a)-5: flow of three sensors ({', '.join(wk['sensors'])}; the 10th/50th/90th percentile of mean flow among the {wk['n_candidates']} sensors fully observed that week) over Mon 3 – Sun 9 Feb 2020, 15-min resolution. Weekday double peaks and the quieter weekend are clearly visible.")

H(2, 'Q3(b) Constructing the traffic graph')
ad = q3['adjacency']
P(f"**Ordering check (asserted in code).** A<sub>SD</sub> = A<sub>CA</sub>[ID2][:, ID2]; the notebook asserts that the list of sensor IDs of the traffic tensor columns equals the `ID` column of the San Diego meta rows and the IDs of the CA meta rows indexed by `ID2`, and that all 716 IDs exist in the HDF5 file. As an independent sanity check, connected sensors are geographically close (median geographic distance of connected pairs {q3['ordering_check']['median_km_connected']:.2f} km vs {q3['ordering_check']['median_km_all_pairs']:.2f} km for all pairs), while a shuffled ordering would give ≈ {q3['ordering_check']['median_km_if_rows_shuffled']:.1f} km.")
rows = [['Q3(b)-1 / -2', 'Value'], ['Number of nodes', ad['nodes']], ['Nonzero entries including the stored diagonal (all 1.0)', f"{ad['nonzero_including_diagonal']:,}"], ['**Nonzero edges (self-loops excluded)**', f"**{ad['nonzero_edges_excl_self_loops']:,}** (density {f2(ad['density_pct'])} %)"],
        ['Edge weights (min / mean / max)', f"{ad['weight_min']:.4f} / {ad['weight_mean']:.3f} / {ad['weight_max']:.4f}"], ['`np.allclose(A, A.T)`', str(ad['allclose_symmetric'])], ['Unequal entries A<sub>ij</sub> ≠ A<sub>ji</sub> → asymmetric pairs', f"{ad['unequal_entries']:,} → **{ad['asymmetric_pairs']:,} pairs**"],
        ['Edges present in one direction only / both directions with different weights / both equal', f"{ad['edges_present_in_one_direction_only']:,} / {ad['edges_both_directions_unequal_weight']:,} / {ad['edges_both_directions_equal']}"],
        ['Out-neighbour count (min / mean / median / max; nodes with 0)', f"{ad['out_degree']['min']} / {ad['out_degree']['mean']:.2f} / {ad['out_degree']['median']:.0f} / {ad['out_degree']['max']}; {ad['out_degree']['zero']}"], ['In-neighbour count (min / mean / median / max; nodes with 0)', f"{ad['in_degree']['min']} / {ad['in_degree']['mean']:.2f} / {ad['in_degree']['median']:.0f} / {ad['in_degree']['max']}; {ad['in_degree']['zero']}"],
        ['Isolated nodes / weakly / strongly connected components', f"{ad['isolated_nodes']} / {q3['q3c_oversmoothing']['weak_components']} / {q3['q3c_oversmoothing']['strong_components']}"]]
table(rows, [10.5, 6.5])
P('**Q3(b)-2 Symmetry.** The matrix is **not symmetric** (`np.allclose(A, A.T)` is False): the provided road-distance graph is **directed**. Most connections exist in both directions but with different weights (' + pct(ad['edges_both_directions_unequal_weight'] / ad['nonzero_edges_excl_self_loops'], 0) + ' of the edges), ' + pct(ad['edges_present_in_one_direction_only'] / ad['nonzero_edges_excl_self_loops'], 0) + ' exist in one direction only and ' + pct(ad['edges_both_directions_equal'] / ad['nonzero_edges_excl_self_loops'], 2) + ' have identical weights both ways.')
fig('q3b_adjacency_heatmap.png', 15.5, 'Q3(b)-3: the 716×716 adjacency A_SD (left: weights; right: log10 of the non-zero weights). Sensors are in meta order, which follows the freeways, giving the banded structure along the diagonal; off-diagonal blocks link intersecting freeways and opposite carriageways.')
fig('q3b_degree_distribution.png', 14.5, 'Q3(b)-4: distribution of the number of neighbours of each sensor (out-neighbours |{j: A_ij ≠ 0}| in blue, in-neighbours in orange; self-loops excluded) and of the non-zero edge weights.')
P("**Q3(b)-5 Meaning of a non-zero A<sub>ij</sub>.** A<sub>ij</sub> ≠ 0 means sensor j is *close to sensor i along the road network*: with d<sub>ij</sub> the road-network distance from i to j, LargeST sets A<sub>ij</sub> = exp(−d<sub>ij</sub>²/σ²) (σ = standard deviation of all distances) and removes entries below the threshold 0.01. The value in (0.01, 1] is the **strength of the road-distance-based relationship**: near 1 for sensors that are very close along the road, near 0.01 for barely connected ones. Road distance depends on direction (one-way carriageways), so A is a **directed** weighted graph; the stored diagonal 1 is the self-loop (distance 0).")

H(2, 'Q3(c) Understanding graph neighbourhoods')
qc = q3['q3c']; corr = q3['q3c_correlation']; osm = q3['q3c_oversmoothing']
one = pd.read_csv(f'{RES}/q3c_one_hop.csv'); two = pd.read_csv(f'{RES}/q3c_two_hop.csv')
meta = pd.read_csv(f'{PROC}/sd_meta.csv')
P(f"Chosen sensor (median out-degree, best coverage in 2020): index {qc['sensor_index']}, ID **{qc['sensor_id']}** ({meta.Fwy[qc['sensor_index']]}, {meta.Lanes[qc['sensor_index']]} lanes, lat {meta.Lat[qc['sensor_index']]:.4f}, lon {meta.Lng[qc['sensor_index']]:.4f}). It has **{qc['one_hop']} one-hop neighbours** {{j : A<sub>ij</sub> ≠ 0}} and **{qc['within_two_hop']} neighbours within ≤ 2 edges**, of which {qc['strict_two_hop']} are reachable only in exactly two edges (two-hop ring).")
rows = [['idx', 'sensor ID', 'freeway', 'dir', 'A<sub>ij</sub>', 'A<sub>ji</sub>', 'distance (km)']] + [[int(r.idx), int(r.sensor_id), r.fwy, r.dir, f'{r.A_ij:.4f}', f'{r.A_ji:.4f}', f'{r.geo_km:.2f}'] for r in one.itertuples()]
P('**One-hop neighbours (all %d):**' % len(one)); table(rows, [1.2, 2.4, 2.2, 1.0, 2, 2, 2.4], size=7)
rows = [['idx', 'sensor ID', 'freeway', 'dir', 'distance (km)', '# one-hop neighbours leading there']] + [[int(r.idx), int(r.sensor_id), r.fwy, r.dir, f'{r.geo_km:.2f}', int(r.via_nbr_count)] for r in two.head(12).itertuples()]
P('**Two-hop ring: %d sensors (nearest 12 shown, sorted by distance; the complete list is `results/q3c_two_hop.csv`):**' % len(two)); table(rows, [1.2, 2.4, 2.4, 1.0, 2.6, 5], size=7)
fig('q3c_neighbourhood_map.png', 9.5, 'Q3(c): geographic map of the chosen sensor (red star), its one-hop neighbours (blue) and its two-hop ring (orange) among all 716 sensors.')
raw_, res_ = corr['raw'], corr['residual']
paras(f"""**Q3(c)-1 Why do neighbouring sensors help?** Congestion is a spatial process: a queue that forms downstream grows back upstream along the road, and neighbouring sensors share demand (the same commuters), capacity, weather and incidents; averaging neighbours also reduces sensor noise and can stand in for a temporarily missing sensor. The data support this, but modestly: raw 15-min flows are highly correlated for every pair of sensors in the county because of the shared daily cycle (mean Pearson correlation: one-hop {f3(raw_['one_hop'])}, two-hop ring {f3(raw_['two_hop'])}, random non-neighbours {f3(raw_['random'])}; training year; {raw_['n']} of the {raw_['n_with_both_rings']} sensors that have both a one-hop and a two-hop ring have defined correlations — the {raw_['n_with_both_rings']-raw_['n']} others have no 2020 data), so the useful signal is in the residual after removing each sensor's own weekly profile: there one-hop neighbours ({f3(res_['one_hop'])}) and the two-hop ring ({f3(res_['two_hop'])}) are clearly more correlated than random sensors ({f3(res_['random'])}); for {pct(res_['frac_sensors_onehop_gt_random'], 0)} of the sensors the one-hop neighbours are more correlated than random ones.""")
fig('q3c_neighbour_correlation.png', 16, 'Q3(c)-1: correlation of a sensor with its one-hop neighbours, its two-hop ring and random non-neighbours (training year 2020; one point per sensor); left: raw flow, right: residual flow after removing each sensor’s weekly profile.')
paras(f"""**Q3(c)-2 One-hop vs two-hop aggregation.** One layer with Â gives each node a weighted average over itself and its one-hop neighbours (the receptive field is the row support of Â: {qc['one_hop']+1} nodes for the chosen sensor including itself). Two layers give Â²: row i is non-zero on every node reachable in ≤ 2 edges ({qc['within_two_hop']+1} nodes for the chosen sensor), i.e. each node now also sees its neighbours' neighbours, with weights that are products of one-hop weights and therefore decay with graph distance. Averaged over all sensors the receptive field grows from {osm['receptive_field_mean']['0']:.0f} (k = 0) to {osm['receptive_field_mean']['1']:.1f} (k = 1), {osm['receptive_field_mean']['2']:.1f} (k = 2), {osm['receptive_field_mean']['3']:.1f} (k = 3) and {osm['receptive_field_mean']['10']:.1f} (k = 10) nodes. Two hops capture slower, larger-scale propagation (a queue spreading over several sensors in 15–60 minutes) at the cost of mixing in less relevant sensors.""")
fig('q3c_oversmoothing.png', 16, 'Q3(c)-3: numerical demonstration on the real graph Â (row-normalised D_out⁻¹(A+I)). Left: standard deviation across nodes of Â^k x relative to k = 0, for a real rush-hour traffic snapshot and for random features; middle: Dirichlet energy (mean squared difference along edges, log scale); right: number of nodes inside the k-hop receptive field (mean and 10–90 % range).')
paras(f"""**Q3(c)-3 Could too many graph-convolution layers hurt? Yes.** (i) *Over-smoothing:* each propagation is a low-pass filter on the graph, so repeated application makes node signals converge. In the experiment the Dirichlet energy of a real traffic snapshot (squared difference between connected sensors) falls to {pct(osm['dirichlet_real']['8'], 1)} of its initial value after k = 8 propagations and to {pct(osm['dirichlet_real']['40'], 2)} after k = 40, so connected sensors become indistinguishable; the across-node standard deviation falls to {pct(osm['std_ratio_real']['8'], 0)} (k = 8) and {pct(osm['std_ratio_real']['40'], 0)} (k = 40) of its initial value for the snapshot and {pct(osm['std_ratio_random']['40'], 0)} for random features, and then flattens rather than reaching 0 because the directed road graph has {osm['weak_components']} weakly ({osm['strong_components']} strongly) connected components that each converge to their own level. (ii) *Over-squashing / bottleneck:* the receptive field grows quickly (mean {osm['receptive_field_mean']['10']:.0f} of 716 nodes after 10 hops), so information from hundreds of sensors must be squeezed into one d-dimensional vector (d ∈ [16, 32] here). (iii) *Noise:* distant sensors are mostly irrelevant for a one-hour forecast and add variance. (iv) *Cost and optimisation:* each layer adds a graph multiplication ({osm['dense_cost_per_layer_flops']/1e6:.1f} MFLOP dense or {osm['sparse_cost_per_layer_flops']/1e6:.2f} MFLOP sparse per sample for d = 32) and parameters; deeper stacks are harder to train and over-fit. Hence shallow graph convolutions combined with temporal modelling (here one layer) are the usual choice.""")

# =================================================================== Q4
STORY.append(CondPageBreak(7 * cm))
H(1, 'Question 4 — Spatio-temporal traffic forecasting')
H(2, 'Q4(a) Forecasting windows, splits and normalisation')
q4 = S45['Q4a']; sp = q4['splits']
rows = [['Split', 'Period', 'Rows (15-min)', 'Windows', 'First input → last target']]
for k, lab in (('train', 'Train: Jan–Dec 2020'), ('val', 'Validation: Jan–Mar 2021'), ('test', 'Test: Apr–Dec 2021')):
    rows.append([k, lab, f"{sp[k]['rows']:,}", f"{sp[k]['windows']:,}", f"{sp[k]['first_input_time'][:16]} → {sp[k]['last_target_time'][:16]}"])
table(rows, [1.6, 4.3, 2.4, 2, 6.7])
paras(f"""Windows: X<sub>t</sub> = [x<sub>t−7</sub>, …, x<sub>t</sub>] ∈ ℝ<super>8×716×1</super> and Y<sub>t</sub> = [x<sub>t+1</sub>, …, x<sub>t+4</sub>] ∈ ℝ<super>4×716×1</super> (stored as (4, 716) targets). Windows are generated **inside each split segment only**, so the first input row and the last target row of every window lie in the same split; with L rows in a split there are L − 11 windows (printed above). The code asserts this for the first/last window of each split and an exhaustive check over *all* windows of all splits; the splits are contiguous, disjoint and in chronological order (train ends exactly where validation begins and validation where test begins).

**Normalisation (train only).** For every sensor the mean and standard deviation are computed over the *observed values of the training split only* and applied to all three splits (sensors with < 1000 training observations — the {q3['imputation']['sensors_without_2020_obs']} sensors without 2020 data — use the global training statistics: mean {f2(q4['global_mean'])}, std {f2(q4['global_std'])}); per-sensor means range from {f1(q4['sensor_mean_min'])} to {f1(q4['sensor_mean_max'])} and stds from {f1(q4['sensor_std_min'])} to {f1(q4['sensor_std_max'])}. Predictions are inverted (× std<sub>j</sub> + mean<sub>j</sub>) before every metric, which is therefore in original units (vehicles per 15 min). *Design note (a documented iteration):* my first attempt used one global scalar mean/std. With it the fixed-graph STGNN reached validation MAE {f1(IT1['stgnn_fixed']['val']['mae'])} versus {f1(IT1['temporal_only']['val']['mae'])} for the temporal-only model (files kept in `results/iter1_global_norm/`), because graph averaging mixes sensors whose mean flow levels differ by a factor of {q4['sensor_mean_max']/q4['sensor_mean_min']:.0f}. Per-sensor normalisation fixes the scale mismatch (and also improves the temporal-only model); this choice was made on validation results only and applies identically to all models. The MAE loss is weighted by σ<sub>j</sub>/σ<sub>global</sub> so that it equals the raw-unit MAE (divided by the global σ) that is reported.

**DataLoader / batch size.** `torch.utils.data.DataLoader` with batch size **{q4['batch_size']}** ({q4['steps_per_epoch']} optimisation steps per epoch). One step already processes {q4['batch_size']} × 716 = {q4['batch_size']*716:,} sensor series, so memory is small, while a larger batch would reduce the number of updates per epoch; 64 is also the LargeST benchmark default. Training batches are shuffled with a seeded generator; validation/test are not shuffled.""")

H(2, 'Q4(b) The forecasting architecture')
paras(f"""All shapes (B = batch, N = 716): input `x: (B, 8, N, 1)` → temporal encoder → **Z: (B, N, d)** with **d = {CFG['hidden_d']}** → graph convolution → **H: (B, N, d)** → head → `(B, N, 4)` → permuted to **Ŷ: (B, 4, N)** (the trailing feature axis F = 1 is squeezed; targets are stored as (B, 4, N)).


1. **Temporal encoder** (Component 1): the 8-step series of every sensor is processed independently: `(B,8,N,1)` is reshaped to `(B·N, 1, 8)` and passed through two `Conv1d` layers (kernel size {KS}, same padding, ReLU, {CFG['hidden_d']} channels), flattened over (channels × time) and projected by a `Linear(8·{CFG['hidden_d']} → {CFG['hidden_d']})` to Z.

2. **Spatial graph convolution** (Component 2): **H = ReLU(Â Z W<sub>g</sub>)** with W<sub>g</sub> ∈ ℝ<super>{CFG['hidden_d']}×{CFG['hidden_d']}</super> (+bias); the product Â Z is `einsum('nm,bmd->bnd', Â, Z)`: the dense 716×716 matrix multiplies the node axis independently for every batch element. This is exactly the formula of the assignment.

3. **Forecast head** (Component 3): `Linear({CFG['hidden_d']}→{CFG['head_hidden']}) → ReLU → Linear({CFG['head_hidden']}→4)` applied to each sensor; the four outputs are the horizons 15, 30, 45 and 60 min.

**Loss and optimisation.** Masked **MAE (L1)** loss: robust to outages and spikes in flow data and the same quantity as the reported MAE; only target entries that are *observed* (valid mask) contribute. Adam, learning rate {CFG['lr']}, no weight decay, batch size {CFG['batch_size']}, gradient-norm clipping {CFG['grad_clip']}, up to {CFG['max_epochs']} epochs, **early stopping on the validation MAE** (original units, patience {CFG['patience']}), best-validation checkpoint restored for the final evaluation (checkpoints saved in `checkpoints/`). Seed {CFG['seed']}.

**Metric convention for zero/missing targets (identical for all models).** MAE and RMSE are computed in original units over target entries that are *observed* (not NaN, not part of a ≥ 3 h outage run; see Q3(a)); imputed values are never targets. Genuine observed zeros are kept. Overall metrics average all valid (window, horizon, sensor) entries; per-horizon metrics average the valid entries of that horizon.""")
rows = [['Model', 'Params', 'Epochs run', 'Best epoch', 'Val MAE', 'Val RMSE', 'Test MAE', 'Test RMSE']]
for n, lab in (('temporal_only', 'Temporal-only (Q4(c))'), ('stgnn_fixed', 'STGNN, fixed graph'), ('stgnn_adaptive', 'STGNN, adaptive graph'), ('stgnn_hybrid', 'STGNN, hybrid graph')):
    r = R42[n]; rows.append([lab, f"{r['params']:,}", r['epochs_run'], r['best_epoch'], f2(r['val']['mae']), f2(r['val']['rmse']), f2(r['test']['mae']), f2(r['test']['rmse'])])
P('**Validation and test MAE / RMSE in original flow units (vehicles per 15 min), overall — seed 42, model formula exactly as in Q4(b):**'); table(rows, [4.2, 1.6, 1.7, 1.7, 1.9, 1.9, 1.9, 1.9])
rows = [['Model', 'split'] + [f'MAE {15*h}' for h in (1, 2, 3, 4)] + [f'RMSE {15*h}' for h in (1, 2, 3, 4)]]
for n, lab in (('temporal_only', 'Temporal-only'), ('stgnn_fixed', 'STGNN-Fixed'), ('stgnn_adaptive', 'STGNN-Adaptive'), ('stgnn_hybrid', 'STGNN-Hybrid')):
    for spn in ('val', 'test'):
        rows.append([lab, spn] + [f2(x) for x in R42[n][spn]['mae_h']] + [f2(x) for x in R42[n][spn]['rmse_h']])
P('**Per horizon (columns: horizon in minutes):**'); table(rows, [3.3, 1.2] + [1.55] * 8, size=7)
H2_ = S45['historical_last']
P(f"Reference (not a trained model): “historical last” persistence x̂<sub>t+h</sub> = x<sub>t</sub> has validation MAE {f2(H2_['val']['mae'])} / RMSE {f2(H2_['val']['rmse'])} and test MAE {f2(H2_['test']['mae'])} / RMSE {f2(H2_['test']['rmse'])}, evaluated with the same masked, de-normalised metric.")
fig('q4q5_training_curves.png', 15, 'Q4/Q5: validation MAE (left) and training loss in original units (right) per epoch for the four models of the main table (seed 42). Training stops when the validation MAE has not improved for 6 epochs.')

H(2, 'Q4(c) Temporal-only baseline')
P("The temporal-only baseline uses the same temporal encoder, the same forecast head, the same data, optimiser, learning rate, batch size, seed and early-stopping rule; the only difference is that the graph layer is removed (H = ReLU(Z)). Its parameter count is therefore smaller by exactly the W<sub>g</sub> block (see the table above). Whether spatial modelling helps is analysed in Q6(a) together with the graph variants, because the answer depends on how the graph layer is formulated (see the extension in Q5 and Q6(a)).")
P('**In the main table (the exact formula of the assignment) the temporal-only model wins.** **Does explicit spatial modelling improve forecasting accuracy?** ' + NARR['6a'].split('**Does explicit spatial modelling improve accuracy? (Q4(c))**')[1].split('**Training time.**')[0].strip())

# =================================================================== Q5
STORY.append(CondPageBreak(7 * cm))
H(1, 'Question 5 — Fixed versus learnable versus hybrid adjacency')
H(2, 'Q5(a) Fixed adjacency')
fx = R42['stgnn_fixed']
from models import fixed_adjacency
_Afix = fixed_adjacency(np.load(f'{PROC}/sd_adj.npy'), 'row').numpy(); _rs = _Afix.sum(1)
paras(f"""**Normalisation for a directed graph.** The road-distance matrix is directed and asymmetric (Q3(b)-2), so Kipf & Welling's D<sup>−1/2</sup>(A+I)D<sup>−1/2</sup> is not appropriate: for A<sub>ij</sub> ≠ A<sub>ji</sub> it is no longer a proper propagation operator. I use the **row-normalised random-walk operator Â = D<sub>out</sub><sup>−1</sup>(A + I)** (the diffusion operator of DCRNN / Graph WaveNet): row i is a convex combination of node i and its out-neighbours N(i) = {{j : A<sub>ij</sub> ≠ 0}}, rows sum to 1 (verified from the matrix: row sums in [{_rs.min():.6f}, {_rs.max():.6f}]) and edge direction is preserved. The stored diagonal (value 1) is zeroed before adding the identity so every node has exactly one unit self-loop before normalisation. Sensors with no out-neighbours keep all their weight on themselves. Â is stored as a `register_buffer` (not an `nn.Parameter`): it is **frozen** and never updated.

**Result (seed 42).** Test MAE **{f2(fx['test']['mae'])}**, RMSE **{f2(fx['test']['rmse'])}** (validation MAE {f2(fx['val']['mae'])}); training time **{tfmt(fx['train_time_sec'])}** for {fx['epochs_run']} epochs (best epoch {fx['best_epoch']}); {fx['params']:,} parameters. This is {'worse' if fx['test']['mae'] > R42['temporal_only']['test']['mae'] else 'better'} than the temporal-only baseline (test MAE {f2(R42['temporal_only']['test']['mae'])}); Q6(a) analyses why.""")
H(2, 'Q5(b) Learnable adjacency (own implementation)')
chk = S45['Q5b_check']; ad_ = R42['stgnn_adaptive']
paras(f"""**Implementation** (`src/models.py::AdaptiveAdjacency`, no PyG): trainable node embeddings E<sub>1</sub>, E<sub>2</sub> ∈ ℝ<super>716×{CFG['d_e']}</super>, randomly initialised from N(0, 0.5²) and updated by back-propagation of the forecasting loss on the **training** data only (nothing from validation or test ever enters the graph). The scores are S = E<sub>1</sub>E<sub>2</sub><super>⊤</super> and **A<sub>adaptive</sub> = Softmax<sub>row</sub>(ReLU(S))**.

**Top-k (k = {CFG['top_k']}) — before or after the normalisation?** The selection is made **before** the softmax: for every row the k = {CFG['top_k']} largest raw scores S<sub>ij</sub> are kept, ReLU is applied to the kept scores and the softmax is taken **only over those {CFG['top_k']} entries**; all other entries are exactly 0. Each row therefore sums to 1 (a distribution over its {CFG['top_k']} selected neighbours), the same normalisation as the row-normalised fixed graph. (The alternative “softmax over all 716 entries, then keep the top-k and renormalise” would give nearly the same ranking, but masking the logits first is cheaper and makes the kept weights independent of the 696 discarded entries.)

**How gradients propagate.** The index selection `topk` is piecewise constant in (E<sub>1</sub>, E<sub>2</sub>) and has no gradient, but the *values* gathered at the selected indices are ordinary differentiable functions of E<sub>1</sub>, E<sub>2</sub> and pass through ReLU → softmax → `scatter` → Â Z. Consequently ∂L/∂S<sub>ij</sub> ≠ 0 **only for the {chk['nonzero_per_row']} selected entries of each row**; the other entries receive exactly zero gradient at that step (they can still enter the top-k later, because updating E<sub>1</sub>, E<sub>2</sub> through the selected entries changes every score). Verified numerically: the gradient of a loss with respect to S has {chk['grad_nonzero']:,} non-zero entries = the {chk['selected_entries']:,} selected entries (716 rows × {chk['nonzero_per_row']}), with **{chk['grad_outside_selection']} non-zero gradients outside the selection**, and all rows of A sum to 1.

**Result (seed 42).** Test MAE **{f2(ad_['test']['mae'])}**, RMSE **{f2(ad_['test']['rmse'])}** (validation MAE {f2(ad_['val']['mae'])}), training time **{tfmt(ad_['train_time_sec'])}**, {ad_['epochs_run']} epochs, {ad_['params']:,} parameters (the {ad_['params']-fx['params']:,} extra parameters are the two embedding matrices). Compared with the fixed graph the test MAE is {f2(ad_['test']['mae'])} vs {f2(fx['test']['mae'])} ({100*(ad_['test']['mae']-fx['test']['mae'])/fx['test']['mae']:+.1f} %).""")
fig('q5b_adjacency_fixed_vs_learned.png', 16.5, 'Q5(b)-4: the fixed road-distance adjacency Â (left), the learned adaptive adjacency (middle: exactly 20 non-zero softmax weights per row) and the hybrid adjacency (right), as 716×716 heatmaps with sensors in meta order (seed 42). Logarithmic colour scale (10⁻³ … 1) with zeros left white, so that weak weights are visible. The road graph is concentrated in the diagonal band along the freeways; the learned graph is spread over the whole matrix.')
H(2, 'Q5(c) Hybrid adjacency')
hy = R42['stgnn_hybrid']
P(f"**A<sub>hybrid</sub> = α·Â<sub>fixed</sub> + (1 − α)·A<sub>adaptive</sub> with a *learnable* α = sigmoid(θ)**, θ initialised to 0 (α₀ = 0.5) and trained by back-propagation. Both operands are row-stochastic (row-normalised fixed graph; row-softmax adaptive graph), i.e. normalised consistently, so the mixture is row-stochastic too. **Learned α = {f3(hy['alpha'])}** (i.e. {pct(1-hy['alpha'], 0)} of the weight goes to the learned graph). Result (seed 42): test MAE **{f2(hy['test']['mae'])}**, RMSE **{f2(hy['test']['rmse'])}** (validation MAE {f2(hy['val']['mae'])}), training time **{tfmt(hy['train_time_sec'])}**, {hy['epochs_run']} epochs, {hy['params']:,} parameters. (With the self-path extension below the learned α is {f3(R42['stgnn_hybrid_skip']['alpha'])}.)")
H(3, 'Identical protocol for all four models')
rows = [['Setting', 'Value (all four models)'], ['Seed', CFG['seed']], ['Optimiser / learning rate', f"Adam / {CFG['lr']}"], ['Batch size', CFG['batch_size']], ['Maximum epochs / early stopping', f"{CFG['max_epochs']} / validation MAE, patience {CFG['patience']}"],
        ['Test period (same for all)', 'Apr–Dec 2021'], ['Epochs actually run (temporal-only / fixed / adaptive / hybrid)', ' / '.join(str(R42[n]['epochs_run']) for n in NAMES[:4])], ['Best epoch', ' / '.join(str(R42[n]['best_epoch']) for n in NAMES[:4])],
        ['Wall-clock training time (`time.perf_counter` around the epoch loop incl. per-epoch validation)', ' / '.join(tfmt(R42[n]['train_time_sec']) for n in NAMES[:4])]]
table(rows, [8, 9])
H(3, 'Extension (clearly separate from the assignment formula): graph layer with a self path')
P(f"In H = ReLU(Â Z W<sub>g</sub>) a sensor's output is built only from a weighted average over its neighbourhood. To test whether the graph helps once a sensor also keeps its own signal, all three graphs were also run with **H = ReLU(Â Z W<sub>g</sub> + Z W<sub>s</sub>)** (a separate self weight W<sub>s</sub>, the “root weight” of PyG's `GraphConv`); everything else is identical and the temporal-only model is the same run. These runs are an *extension*; the main table of Q6(a) uses the exact formula of the assignment.")
rows = [['Model', 'Params', 'Epochs', 'Test MAE', 'Test RMSE', 'Val MAE', 'Training time']]
for n, lab in (('stgnn_fixed_skip', 'STGNN-Fixed + self path'), ('stgnn_adaptive_skip', 'STGNN-Adaptive + self path'), ('stgnn_hybrid_skip', 'STGNN-Hybrid + self path (α = %s)' % f3(R42['stgnn_hybrid_skip']['alpha']))):
    r = R42[n]; rows.append([lab, f"{r['params']:,}", r['epochs_run'], f2(r['test']['mae']), f2(r['test']['rmse']), f2(r['val']['mae']), tfmt(r['train_time_sec'])])
table(rows, [5.4, 1.6, 1.7, 1.8, 1.9, 1.7, 3.0])

# =================================================================== Q6
STORY.append(CondPageBreak(7 * cm))
H(1, 'Question 6 — Experimental analysis and interpretation')
H(2, 'Q6(a) Forecasting comparison')
P('**Key finding.** With the graph layer exactly as specified (H = ReLU(Â Z W<sub>g</sub>), main table) the temporal-only model has the lowest error; with a self path added (a separate, clearly labelled extension table) all graph models beat it, and the learned graph is best. Both results are reported, with 3 seeds.')
P('**Main table — exact format requested (test period Apr–Dec 2021, original units: vehicles per 15 min; seed 42).**')
rows = [['Model', 'Adjacency', 'MAE', 'RMSE', 'Training time']]
for n, m, a in (('temporal_only', 'Temporal-only', 'None'), ('stgnn_fixed', 'STGNN', 'Fixed'), ('stgnn_adaptive', 'STGNN', 'Adaptive'), ('stgnn_hybrid', 'STGNN', 'Hybrid')):
    rows.append([m, a, f2(R42[n]['test']['mae']), f2(R42[n]['test']['rmse']), f"{tfmt(R42[n]['train_time_sec'])}; {R42[n]['epochs_run']} epochs"])
table(rows, [3.4, 3, 2.2, 2.2, 5])
P('**Extension table — same graphs with a self path (not the assignment formula):**')
rows = [['Model', 'Adjacency', 'MAE', 'RMSE', 'Training time']]
rows.append(['Temporal-only', 'None', f2(R42['temporal_only']['test']['mae']), f2(R42['temporal_only']['test']['rmse']), f"{tfmt(R42['temporal_only']['train_time_sec'])}; {R42['temporal_only']['epochs_run']} epochs"])
for n, a in (('stgnn_fixed_skip', 'Fixed'), ('stgnn_adaptive_skip', 'Adaptive'), ('stgnn_hybrid_skip', 'Hybrid')):
    rows.append(['STGNN + self path', a, f2(R42[n]['test']['mae']), f2(R42[n]['test']['rmse']), f"{tfmt(R42[n]['train_time_sec'])}; {R42[n]['epochs_run']} epochs"])
table(rows, [3.4, 3, 2.2, 2.2, 5])
P('**Per-horizon results (test period; MAE and RMSE in vehicles per 15 min):**')
rows = [['Model', 'Adjacency'] + [f'MAE {15*h} min' for h in (1, 2, 3, 4)] + [f'RMSE {15*h} min' for h in (1, 2, 3, 4)]]
for n, m, a in (('temporal_only', 'Temporal-only', 'None'), ('stgnn_fixed', 'STGNN', 'Fixed'), ('stgnn_adaptive', 'STGNN', 'Adaptive'), ('stgnn_hybrid', 'STGNN', 'Hybrid'), ('stgnn_fixed_skip', 'STGNN + self path', 'Fixed'), ('stgnn_adaptive_skip', 'STGNN + self path', 'Adaptive'), ('stgnn_hybrid_skip', 'STGNN + self path', 'Hybrid')):
    rows.append([m, a] + [f2(x) for x in R42[n]['test']['mae_h']] + [f2(x) for x in R42[n]['test']['rmse_h']])
table(rows, [3.0, 2.1] + [1.45] * 8, size=7)
ns_, *_ = mean_std_row('temporal_only')
P(f'**Mean ± sample std over {ns_} seeds (42, 43, 44; test period; training time is a single wall-clock measurement per run):**')
rows = [['Model', 'Adjacency', 'Test MAE', 'Test RMSE', 'Training time (s)']]
for n, m, a in (('temporal_only', 'Temporal-only', 'None'), ('stgnn_fixed', 'STGNN', 'Fixed'), ('stgnn_adaptive', 'STGNN', 'Adaptive'), ('stgnn_hybrid', 'STGNN', 'Hybrid'), ('stgnn_fixed_skip', 'STGNN + self path', 'Fixed'), ('stgnn_adaptive_skip', 'STGNN + self path', 'Adaptive'), ('stgnn_hybrid_skip', 'STGNN + self path', 'Hybrid')):
    k, mm, ss, rm, rsd, tm, ts = mean_std_row(n); rows.append([m, a, f'{mm:.2f} ± {ss:.2f}', f'{rm:.2f} ± {rsd:.2f}', f'{tm:.0f} ± {ts:.0f}'])
table(rows, [3.6, 3, 3.2, 3.2, 3.6])
fig('q6a_pred_vs_actual.png', 17, 'Q6(a): predicted vs actual traffic flow for three sensors (10th/50th/90th percentile of mean flow among fully observed sensors) on four test days (Mon 7 – Thu 10 June 2021). Left column: 15-min-ahead forecasts; right column: 60-min-ahead forecasts. Black = actual (observed); coloured = the four models of the main table.')
fig('q6a_pred_vs_actual_selfpath.png', 17, 'Q6(a), extension: the same sensors and days for the models with a self path (compared with the temporal-only baseline).')
fig('q6a_mae_by_hour.png', 12.5, 'Q6(a): test MAE by hour of day (all horizons) for all seven models (dashed: models with a self path).')
paras(NARR['6a'])

STORY.append(CondPageBreak(7 * cm))
H(2, 'Q6(b) Graph-structure analysis')
P('The analysed graph is the learned adjacency of the adaptive model (seed 42) — the model that follows the assignment formula; the same statistics for the other learned graphs are in `results/q6_results.json`. Row i lists the sensors whose features are aggregated into sensor i, the same convention as the road graph. All statistics use only the learned matrix, the sensor meta data (lat/long) and the training year (2020) for correlations.')
fig('q6b_learned_vs_road_stats.png', 17, 'Q6(b)-1 and -2: left, share of the connected pairs in each log-spaced distance bin (road edges, all learned edges, random pairs); middle, cumulative distribution of the same distances; right, overlap@k — the fraction of each sensor’s k strongest learned (non-self) edges that are road edges, against the chance level given by the road-graph density.')
fig('q6b_graph_visualisation.png', 17, 'Q6(b)-3: left, a random sample of road-graph edges on the map; middle, the 400 strongest learned edges that are not in the road graph (many span the whole county); right, distribution of the residual-flow correlation (training year) between the endpoints of learned non-road edges, learned road edges, road edges and random pairs.')
fig('q6b_nonroad_pair_series.png', 13, 'Q6(b)-3: traffic series (Sep 2020) of the strongest learned non-road pair between sensors that have data: the two series follow each other although there is no road edge.')
ex = pd.read_csv(f'{RES}/q6b_top_nonroad_edges.csv')
rows = [['i', 'j', 'sensor i', 'sensor j', 'fwy i', 'fwy j', 'learned weight', 'km apart', 'corr (raw)', 'corr (resid.)']]
rows += [[r.i, r.j, r.sensor_i, r.sensor_j, r.fwy_i, r.fwy_j, f'{r.weight:.3f}', f'{r.km:.1f}', f3(r.corr_raw), f3(r.corr_resid)] for r in ex.head(12).itertuples()]
P('**Strongest learned edges that are *not* in the road graph (either direction) between sensors with data — top 12 of 15 in `results/q6b_top_nonroad_edges.csv`; correlations use the training year:**'); table(rows, [0.9, 0.9, 1.9, 1.9, 1.7, 1.7, 1.8, 1.4, 1.5, 1.8], size=6.8)
lg = q6['learned_graph']['adaptive model']; cs = q6['corr_stats']
rows = [['Quantity (adaptive model, seed 42)', 'Value'], ['non-self learned edges', f"{lg['n_learned_edges_nonself']:,}"], ['fraction that are road edges (directed i→j / either direction)', f"{pct(lg['frac_learned_edges_in_road'])} / {pct(lg['frac_learned_edges_in_road_either'])}"],
        ['chance level (road-graph density, directed / either)', f"{pct(lg['road_edge_density'])} / {pct(lg['road_edge_density_either'])}"], ['fraction of road edges re-discovered', pct(lg['frac_road_edges_recovered'])],
        ['overlap@1 / @5 / @10 / @19 (directed)', ' / '.join(pct(lg[f'overlap_at_{k}_directed'], 0) for k in (1, 5, 10, 19))], ['global top-500 / top-2000 strongest edges that are road edges', f"{pct(lg['global_top500_in_road'], 0)} / {pct(lg['global_top2000_in_road'], 0)}"],
        ['median distance km: road edges / learned edges / top-2000 learned / random pairs', ' / '.join(f"{lg['dist_km'][k]['median']:.1f}" for k in ('road_edges', 'learned_all', 'learned_top2000', 'random_pairs'))],
        ['mean residual correlation: learned non-road / learned road / road edges / random pairs', ' / '.join(f3(cs['residual'][k]['mean']) for k in ('learned_nonroad', 'learned_road', 'road_edges', 'random_pairs'))],
        ['mean raw correlation: learned non-road / learned road / road edges / random pairs', ' / '.join(f3(cs['raw'][k]['mean']) for k in ('learned_nonroad', 'learned_road', 'road_edges', 'random_pairs'))]]
table(rows, [10.5, 6.5])
H(3, 'Answers to the four questions (Q6(b)-1 … Q6(b)-4)')
paras(NARR['6b'])

STORY.append(CondPageBreak(7 * cm))
H(2, 'Q6(c) Critical discussion')
H(3, 'Q6(c)-1, Q6(c)-2 and Q6(c)-3')
P('*Model references used below:* “main-table model” = the model with the layer exactly as specified in the assignment (temporal-only / fixed / adaptive / hybrid rows of the main table); “with self path” = the labelled extension H = ReLU(Â Z W<sub>g</sub> + Z W<sub>s</sub>). Each statement says which one it refers to.')
paras(NARR['6c'])

# =================================================================== Q7
STORY.append(CondPageBreak(7 * cm))
H(1, 'Question 7 — Reproducibility and reporting')
H(2, 'Implementation choices (summary)')
bul = [f"**Data:** LargeST San Diego only (716 sensors, `District == 11`), 2020–2021, flow variable, 5 → 15 min by NaN-aware **sum** (flow = vehicle counts), splits Train 2020 / Val Jan–Mar 2021 / Test Apr–Dec 2021, input 8 steps → output 4 steps.",
       f"**Missing data:** NaN and ≥ 3 h stuck runs → missing ({f3(q3['resample']['missing_15min_pct'])} % of 15-min entries); causal ≤ 1 h forward-fill inside each split, then train-split per-sensor mean for inputs; loss and metrics only on observed targets.",
       "**Normalisation:** per-sensor z-score from the training split only; MAE loss weighted by σ<sub>j</sub>/σ<sub>global</sub> (= raw-unit MAE).",
       "**Graphs:** fixed Â = D<sub>out</sub><sup>−1</sup>(A+I) (the road graph is directed); adaptive softmax(ReLU(E<sub>1</sub>E<sub>2</sub><super>⊤</super>)) with top-20 selected before the softmax, own implementation; hybrid α·Â + (1−α)·A<sub>adaptive</sub> with learnable α = σ(θ).",
       "**Model:** two-layer temporal Conv1d encoder (d = 32) → one graph convolution H = ReLU(Â Z W<sub>g</sub>) → MLP head; temporal-only baseline = same without the graph layer; extension with a self path reported separately."]
for b in bul: P('• ' + b)
H(2, 'Training configuration')
rows = [['Hyper-parameter', 'Value'], ['Random seed', f"{CFG['seed']} (python, numpy, torch, torch.cuda; deterministic cuDNN/cuBLAS flags); extra seeds 43, 44 for mean ± std"], ['Optimiser', 'Adam'], ['Learning rate / weight decay', f"{CFG['lr']} / {CFG['weight_decay']}"],
        ['Batch size', CFG['batch_size']], ['Maximum epochs / early stopping', f"{CFG['max_epochs']} / patience {CFG['patience']} on validation MAE (original units); best checkpoint restored"], ['Gradient clipping (norm)', CFG['grad_clip']],
        ['Loss', CFG['loss']], ['Normalisation', CFG['norm']], ['Hidden dimension d / head hidden / temporal kernel', f"{CFG['hidden_d']} / {CFG['head_hidden']} / {KS}"], ['Adaptive graph: d<sub>e</sub> / top-k / init std', f"{CFG['d_e']} / {CFG['top_k']} / 0.5"],
        ['Hybrid α', f"learnable, α = sigmoid(θ), θ₀ = 0; learned α = {f3(R42['stgnn_hybrid']['alpha'])} (spec formula), {f3(R42['stgnn_hybrid_skip']['alpha'])} (self-path extension)"],
        ['Epochs run (seed 42)', ', '.join(f"{nice(n)}: {R42[n]['epochs_run']}" for n in NAMES)]]
table(rows, [5.5, 11.5])
H(2, 'Hardware and software (collected programmatically, `results/hardware.json`)')
rows = [['Item', 'Value'], ['CPU', HW['cpu_model']], ['CPU cores (physical / logical)', f"{HW['cpu_physical_cores']} / {HW['cpu_logical_cores']}"], ['RAM', f"{HW['ram_gb']} GB"], ['GPU', f"{HW.get('gpu')} ({HW.get('gpu_memory_gb')} GB, CUDA {HW.get('cuda_version')})"], ['OS', HW['os']],
        ['Python / PyTorch / NumPy', f"{HW['python']} / {HW['torch']} / {HW['numpy']}"], ['pandas / h5py / matplotlib / scikit-learn', f"{HW.get('pandas')} / {HW.get('h5py')} / {HW.get('matplotlib')} / {HW.get('sklearn')}"],
        ['Training device used for all runs', R42['temporal_only']['device']], ['Estimated epoch time before training (GPU / CPU)', f"{S45['Q4a']['epoch_time_estimate_sec'].get('cuda', float('nan')):.0f} s / {S45['Q4a']['epoch_time_estimate_sec'].get('cpu', float('nan')):.0f} s (40 timed steps extrapolated to a full epoch + validation)"]]
table(rows, [5.5, 11.5])

import os as _os
if _os.path.exists(f'{RES}/cpu_benchmark.json'):
    _cb = load_json(f'{RES}/cpu_benchmark.json')
    P(f"**CPU feasibility (measured).** One complete training epoch ({_cb['steps_per_epoch']} steps, batch {_cb['batch_size']}) plus one validation pass was run on the CPU alone ({_cb['threads']} threads, {_cb['cpu']}) for each model of the main table (`notebooks/Q7_cpu_benchmark.ipynb`, `results/cpu_benchmark.json`):")
    rows = [['Model', 'Parameters', 'Train epoch (CPU)', 'Validation pass (CPU)', 'Epoch total (CPU)', 'Epoch total (GPU, run)']]
    for n, lab in (('temporal_only', 'Temporal-only'), ('stgnn_fixed', 'STGNN-Fixed'), ('stgnn_adaptive', 'STGNN-Adaptive'), ('stgnn_hybrid', 'STGNN-Hybrid')):
        m_ = _cb['models'][n]; rows.append([lab, f"{m_['params']:,}", f"{m_['train_epoch_sec']:.1f} s", f"{m_['val_pass_sec']:.1f} s", f"{m_['epoch_total_sec']:.1f} s", f"{R42[n]['train_time_sec']/R42[n]['epochs_run']:.1f} s"])
    table(rows, [3.4, 2.2, 3, 3.3, 2.9, 2.7])
    _worst = max(m_['epoch_total_sec'] for m_ in _cb['models'].values())
    P(f"At most {_worst:.0f} s per epoch on the CPU, so a full {CFG['max_epochs']}-epoch run of any model takes at most {CFG['max_epochs']*_worst/60:.0f} minutes even without a GPU (early stopping usually ends sooner).")
H(2, 'Measured training times (seed 42; wall-clock `time.perf_counter` around the training loop incl. validation)')
rows = [['Model', 'Adjacency', 'Epochs', 'Time', 'Seconds / epoch']]
for n, m, a in (('temporal_only', 'Temporal-only', 'None'), ('stgnn_fixed', 'STGNN', 'Fixed'), ('stgnn_adaptive', 'STGNN', 'Adaptive'), ('stgnn_hybrid', 'STGNN', 'Hybrid'), ('stgnn_fixed_skip', 'STGNN + self path', 'Fixed'), ('stgnn_adaptive_skip', 'STGNN + self path', 'Adaptive'), ('stgnn_hybrid_skip', 'STGNN + self path', 'Hybrid')):
    r = R42[n]; rows.append([m, a, r['epochs_run'], tfmt(r['train_time_sec']), f"{r['train_time_sec']/r['epochs_run']:.1f}"])
table(rows, [4, 3, 2, 4, 3])
H(2, 'Reproducibility instructions')
code("""conda activate pa1                                    # existing env (Python 3.11, torch 2.4.1+cu124)
python -m pip install -r requirements.txt             # only adds packages pa1 lacked
python -m ipykernel install --user --name pa1 --display-name pa1
# data -> data/raw/ : ca_meta.csv, ca_rn_adj.npy, ca_his_raw_2020.h5, ca_his_raw_2021.h5
#   (Kaggle liuxu77/largest:  kaggle datasets download liuxu77/largest -f <file> -p data/raw ; unzip)
python src/py2nb.py notebooks_src/Q1_Q2_analysis.py     --run
python src/py2nb.py notebooks_src/Q3_data_and_graph.py  --run
python src/py2nb.py notebooks_src/Q4_Q5_training.py     --run     # seed 42, 7 trainings
python src/py2nb.py notebooks_src/Q5_extra_seeds.py     --run     # seeds 43, 44 (optional)
python src/py2nb.py notebooks_src/Q6_analysis.py        --run
python src/py2nb.py notebooks_src/Q7_cpu_benchmark.py   --run     # CPU timing (optional; run on an idle machine)
python src/build_report.py                                       # report/report.pdf""")
P(f"Source code: <link href=\"{REPO}\" color=\"blue\">{REPO}</link>. The notebooks (notebooks/*.ipynb, outputs preserved) run top to bottom with kernel pa1; seed 42 is set in src/common.py (set_seed). Training used the GPU above; the code runs unchanged on CPU (estimated epoch time in the hardware table). Large artefacts (raw data, processed arrays, checkpoints) are git-ignored and regenerated by the notebooks.", BODY, raw=True)
import glob as _glob
_all_runs = [load_json(f) for f in sorted(_glob.glob(f'{RES}/train_*_seed*.json')) if '/iter1_' not in f]
_capped = [r for r in _all_runs if r['epochs_run'] >= r['cfg']['max_epochs']]
_capped_txt = ', '.join(nice(r['name']) + ' (seed ' + str(r['seed']) + ')' for r in _capped)
_cap_note = (f"**Epoch cap.** {len(_capped)} of the {len(_all_runs)} training runs (all seeds and models) ended at the {CFG['max_epochs']}-epoch cap instead of by early stopping ({_capped_txt}), so their validation MAE may still have been improving slowly; I did not measure how much. All models use the same cap, and the gaps between the models are much larger than the seed-to-seed standard deviations reported in Q6(a).")
H(2, 'Honest notes and limitations')
lim = [f"**Design iteration disclosed.** My first normalisation (one global mean/std) gave validation MAE {f1(IT1['stgnn_fixed']['val']['mae'])} for the fixed STGNN versus {f1(IT1['temporal_only']['val']['mae'])} for temporal-only; I switched to per-sensor normalisation (chosen on validation results only, applied to all models). The discarded runs are kept in `results/iter1_global_norm/`. The test set was never used for any choice.",
       "**The main table follows the assignment formula literally and, with it, no graph model beats the temporal-only model.** The self-path extension (not part of the specification) is reported separately and clearly labelled; I did not replace the main table with it.",
       f"**Training times are single measurements** that include early-stopping dynamics (the number of epochs differs between models); with {ns_} seeds the std of the times is given in Q6(a).",
       "**Distribution shift.** The test period (Apr–Dec 2021) has a different traffic level than the training year (COVID-affected 2020), see Q6(a); no re-calibration was applied.",
       f"**Data quality.** {q3['imputation']['sensors_without_2020_obs']} sensors have no 2020 observation (and are entirely missing in the files); they stay in the graph (716 nodes required) but are never evaluated; the strongest learned edges between sensors with data are the ones analysed in Q6(b).",
       _cap_note,
       "**Learned-graph interpretation** is correlational (see Q6(c)); the 6(b) statistics use the seed-42 model only."]
for b in lim: P('• ' + b)

# =================================================================== AI usage (written by the student)
# The text of this section is NOT generated here: the student writes it in report/ai_usage_student.md and this script only renders it.
# Supported markup: '## heading', blank-line separated paragraphs, '- ' bullets, **bold**, *italic*, `code`, ``` fenced blocks ``` (verbatim, e.g. prompts), <!-- comments --> (ignored).
import textwrap
STORY.append(CondPageBreak(7 * cm))
H(1, 'Generative AI Usage')
_ai_path = os.path.join(ROOT, 'report', 'ai_usage_student.md')
_ai_txt = open(_ai_path, encoding='utf-8').read() if os.path.exists(_ai_path) else ''
_ai_txt = re.sub(r'<!--.*?-->', '', _ai_txt, flags=re.S)


def _verbatim(text):
    wrapped = []
    for ln in text.split('\n'):
        wrapped += textwrap.wrap(ln, 69, break_long_words=True, break_on_hyphens=False, drop_whitespace=False, replace_whitespace=False) or ['']
    PPS = ParagraphStyle('pp', fontName='DVMono', fontSize=5.3, leading=6.3); L = 30; rows_ = []
    for i in range(0, len(wrapped), 2 * L):
        rows_.append([Preformatted('\n'.join(wrapped[i:i + L]), PPS), Preformatted('\n'.join(wrapped[i + L:i + 2 * L]), PPS)])
    t_ = Table(rows_, colWidths=[8.5 * cm, 8.5 * cm]); t_.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f6f6f6')), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3), ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2)]))
    STORY.append(t_); SP(4)


def _has_body(t):
    return any(ln.strip() and not ln.strip().startswith('#') for ln in t.split('\n'))


if not _has_body(_ai_txt):
    print('\n' + '!' * 78 + '\n WARNING: report/ai_usage_student.md has no text yet -> the PDF contains a "to be completed" notice.\n Write your Generative AI Usage section in that file, then run: python src/build_report.py\n' + '!' * 78 + '\n')
    P('**[TO BE COMPLETED BY THE STUDENT BEFORE SUBMISSION]** This section must state the prompt(s) used, the output obtained and the edits made (assignment instructions, page 1). Write it in `report/ai_usage_student.md` and rebuild the report with `python src/build_report.py`.')
else:
    parts = re.split(r'(```.*?```)', _ai_txt, flags=re.S)
    for part in parts:
        if part.startswith('```'):
            _verbatim(part.strip('`').strip('\n')); continue
        for blk in [x.strip() for x in part.split('\n\n') if x.strip()]:
            lines_ = blk.split('\n')
            if lines_[0].startswith('## '):
                H(2, lines_[0][3:].strip()); blk = '\n'.join(lines_[1:]).strip(); lines_ = blk.split('\n') if blk else []
            if lines_ and all(ln.lstrip().startswith(('- ', '• ')) for ln in lines_ if ln.strip()):
                for ln in lines_:
                    if ln.strip(): P('• ' + ln.lstrip()[2:].strip())
            elif blk:
                P(blk.replace('\n', ' '))


# ------------------------------------------------------------------ build
def _footer(c, d):
    c.saveState(); c.setFont('DV', 7); c.setFillColor(colors.grey)
    c.drawString(2 * cm, 1.1 * cm, 'AI651 Assignment 3 — Huzaifa (25280009) — ' + REPO)
    c.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f'page {d.page}'); c.restoreState()


out = os.path.join(ROOT, 'report', 'report.pdf')
doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm, bottomMargin=1.7 * cm, title='AI651 Assignment 3 — Huzaifa (25280009)', author='Huzaifa')
doc.build(STORY, onFirstPage=_footer, onLaterPages=_footer)
print('wrote', out, os.path.getsize(out) // 1024, 'KB,', FIGN[0], 'figures')
