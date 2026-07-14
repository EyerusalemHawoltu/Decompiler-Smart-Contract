"""Compare Gigahorse vs CFG name-recovery on the SAME contracts (runtime-only)."""
import json, csv, os
from collections import defaultdict
csv.field_size_limit(10**9)
from eval_cfg_csv import parse_sol_functions, classify_bytecode, SOLIDITY_DIR

ROOT = os.path.dirname(os.path.abspath(__file__))
CE   = os.path.join(ROOT, 'results', 'cfg_eval')
GHJ  = os.path.join(ROOT, 'results', 'gigahorse_names.jsonl')

# 1. gigahorse names {stem -> (version, names)}
gh = {}
with open(GHJ) as f:
    for line in f:
        r = json.loads(line)
        gh[r['stem']] = (r['version'], set(r.get('names') or []))

# 2. CFG V4 per-contract metrics {stem -> (p,r,f1,btype)}
cfg = {}
for fn in os.listdir(CE):
    if not fn.endswith('_results.csv'):
        continue
    with open(os.path.join(CE, fn)) as f:
        for row in csv.DictReader(f):
            stem = row['address'] + '_' + row['contract_name']
            cfg[stem] = (float(row['precision'] or 0), float(row['recall'] or 0),
                         float(row['f1'] or 0), row.get('bytecode_type', ''))

gh_sum = defaultdict(lambda: defaultdict(float)); gh_n = defaultdict(int)
cf_sum = defaultdict(lambda: defaultdict(float)); cf_n = defaultdict(int)
common = 0
for stem, (ver, names) in gh.items():
    if stem not in cfg:
        continue
    p_c, r_c, f_c, btype = cfg[stem]
    if btype != 'runtime':
        continue
    sol = parse_sol_functions(os.path.join(SOLIDITY_DIR, ver, stem + '.sol.cleaned'))
    if not sol and not names:
        continue
    common += 1
    m = names & sol
    p = len(m)/len(names) if names else 0.0
    r = len(m)/len(sol) if sol else 0.0
    f1 = 2*p*r/(p+r) if p+r else 0.0
    gh_sum[ver]['p'] += p; gh_sum[ver]['r'] += r; gh_sum[ver]['f1'] += f1; gh_n[ver]+=1
    cf_sum[ver]['p'] += p_c; cf_sum[ver]['r'] += r_c; cf_sum[ver]['f1'] += f_c; cf_n[ver]+=1

def wavg(sums, ns, k):
    tot = sum(ns.values())
    return sum(sums[v][k] for v in sums)/tot if tot else 0.0

print(f'common runtime contracts compared: {common}')
print()
print('%-12s %22s %22s' % ('', 'GIGAHORSE', 'CFG (V4)'))
print('%-12s %7s %7s %7s   %7s %7s %7s' % ('', 'P','R','F1','P','R','F1'))
for k in ['p','r','f1']:
    pass
print('%-12s %7.3f %7.3f %7.3f   %7.3f %7.3f %7.3f' % (
    'WEIGHTED', wavg(gh_sum,gh_n,'p'), wavg(gh_sum,gh_n,'r'), wavg(gh_sum,gh_n,'f1'),
    wavg(cf_sum,cf_n,'p'), wavg(cf_sum,cf_n,'r'), wavg(cf_sum,cf_n,'f1')))
print()
print('per-version  GH(P/R/F1)  vs  CFG(P/R/F1):')
def vk(v): p=v.split('.'); return (int(p[0]),int(p[1]),int(p[2]))
for v in sorted(gh_n, key=vk):
    n=gh_n[v]
    print('  %-7s n=%-5d  GH %.3f/%.3f/%.3f   CFG %.3f/%.3f/%.3f' % (
        v, n, gh_sum[v]['p']/n, gh_sum[v]['r']/n, gh_sum[v]['f1']/n,
        cf_sum[v]['p']/n, cf_sum[v]['r']/n, cf_sum[v]['f1']/n))
