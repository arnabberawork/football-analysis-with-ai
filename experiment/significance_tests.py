"""Statistical significance of the main comparison (no new runs).

Re-analyses the saved results only: experiment/results/main_comparison.csv
(per-query retrieval and generation metrics) and the three completed human
evaluation sheets. For each metric it reports the paired Wilcoxon
signed-rank test (Agentic vs Naive), a 95% bootstrap confidence interval
for the mean difference (10,000 resamples, seed 42), the rank-biserial
effect size, and Holm-adjusted p-values within each family of tests.
Retrieval and generation use the 25 queries both systems answer by
retrieval; human ratings use the 20 blind pairs, averaged over raters.

Usage:
    uv run python experiment/significance_tests.py
"""
import csv, glob, os, re, sys
import numpy as np
from scipy.stats import wilcoxon
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results') + os.sep
rng = np.random.default_rng(42)

def boot_ci(d, n=10000):
    d = np.asarray(d, float); idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1); return np.percentile(m, 2.5), np.percentile(m, 97.5)

def rank_biserial(d):
    d = np.asarray(d, float); d = d[d != 0]
    if len(d) == 0: return 0.0
    from scipy.stats import rankdata
    r = rankdata(np.abs(d)); return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()

def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = [0]*m; run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (m-k)*ps[i])); adj[i] = run
    return adj

def test(rows, label):
    out = []
    for name, a, b in rows:  # a=naive, b=agentic
        a = np.asarray(a, float); b = np.asarray(b, float); d = b - a
        nz = int((d != 0).sum())
        p = wilcoxon(b, a).pvalue if nz > 0 else 1.0
        lo, hi = boot_ci(d)
        out.append([name, len(d), a.mean(), b.mean(), d.mean(), lo, hi, nz, p, rank_biserial(d)])
    adj = holm([o[8] for o in out])
    print(f'\n== {label}')
    print(f'{"metric":22}{"n":>3}{"naive":>8}{"agentic":>9}{"diff":>8}{"95% CI":>18}{"nz":>4}{"p":>10}{"p_holm":>10}{"r_rb":>7}')
    res = []
    for o, pa in zip(out, adj):
        print(f'{o[0]:22}{o[1]:>3}{o[2]:>8.3f}{o[3]:>9.3f}{o[4]:>8.3f}   [{o[5]:6.3f},{o[6]:6.3f}]{o[7]:>4}{o[8]:>10.2g}{pa:>10.2g}{o[9]:>7.2f}')
        res.append(o + [pa])
    return res

# ---- main comparison
rows = list(csv.DictReader(open(R + 'main_comparison.csv', encoding='utf-8')))
by = {}
for r in rows: by.setdefault((r['match_id'], r['query']), {})[r['system']] = r
common = [v for v in by.values() if v['agentic']['precision_at_5'] not in ('N/A', '')]
print('common queries:', len(common))
def col(sys, k): return [float(v[sys][k]) for v in common]
retr = test([(m, col('naive', k), col('agentic', k)) for m, k in
             [('Precision@5', 'precision_at_5'), ('Recall@5', 'recall_at_5'), ('Precision@10', 'precision_at_10'), ('Recall@10', 'recall_at_10')]],
            'Retrieval, common 25 queries')
gen = test([(m, col('naive', k), col('agentic', k)) for m, k in
            [('ROUGE-1', 'rouge1'), ('ROUGE-2', 'rouge2'), ('ROUGE-L', 'rougeL'), ('BLEU', 'bleu'), ('BERTScore F1', 'bertscore_f1')]],
           'Generation, common 25 queries')

# ---- human evaluation
key = {r['pair_id']: r for r in csv.DictReader(open(R + 'human_eval_key.csv', encoding='utf-8'))}
files = {}
for f in sorted(glob.glob(R + 'human_eval_sheet_rater*_completed*.csv')):
    rid = re.search(r'rater(\d+)', f).group(1)
    if rid not in files or f.endswith('_fixed.csv'): files[rid] = f
print('\nrater files:', {k: os.path.basename(v) for k, v in files.items()})
dims = ['relevance', 'coherence', 'factual_consistency', 'fluency']
score = {}  # (pair, dim, system) -> list
for rid, f in files.items():
    for r in csv.DictReader(open(f, encoding='utf-8')):
        pid = r['pair_id']; k = key[pid]
        for dm in dims:
            for side in 'ab':
                v = r.get(f'{dm}_{side}', '').strip()
                if v:
                    score.setdefault((pid, dm, k[f'{side}_system']), []).append(float(v))
pairs = sorted(key, key=int)
hum = test([(dm, [np.mean(score[(p, dm, 'naive')]) for p in pairs], [np.mean(score[(p, dm, 'agentic')]) for p in pairs]) for dm in dims],
           'Human evaluation, 20 pairs (mean of 3 raters per pair)')

with open(R + 'significance_tests.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['family', 'metric', 'n', 'naive_mean', 'agentic_mean', 'mean_diff', 'ci95_low', 'ci95_high', 'nonzero_pairs', 'p_wilcoxon', 'r_rank_biserial', 'p_holm'])
    for fam, res in (('retrieval', retr), ('generation', gen), ('human', hum)):
        for o in res:
            w.writerow([fam, o[0], o[1]] + [round(float(x), 6) for x in o[2:7]] + [o[7], float(o[8]), round(float(o[9]), 4), float(o[10])])
print('saved', R + 'significance_tests.csv')
