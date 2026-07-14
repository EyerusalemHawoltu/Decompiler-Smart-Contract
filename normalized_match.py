"""
normalized_match.py — fair "correct modulo variable names" metric.

Variable/parameter names are erased at compile time and unrecoverable, so
exact-match/BLEU unfairly penalise correct decompilations that pick different
names (e.g. rescueToken: identical logic, renamed params -> BLEU 0).

This metric alpha-renames local identifiers to canonical tokens, then compares.
It CREDITS rename-only near-misses and still FAILS genuine hallucinations
(different #params / different statements).

Reads results/full_scores_clean_table.csv, writes results/full_scores_clean_normalized.csv.
"""
import csv, re, os
csv.field_size_limit(10**9)

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_IN  = os.path.join(ROOT, 'results', 'full_scores_clean_table.csv')
CSV_OUT = os.path.join(ROOT, 'results', 'full_scores_clean_normalized.csv')

# tokens kept verbatim (semantically meaningful): keywords, types, builtins
KEEP = {
 'function','external','public','internal','private','view','pure','payable',
 'returns','return','if','else','for','while','do','require','assert','revert',
 'emit','new','delete','memory','storage','calldata','mapping','struct','enum',
 'modifier','constructor','event','using','is','virtual','override','constant',
 'immutable','unchecked','try','catch','assembly','address','bool','string',
 'bytes','byte','true','false','this','super','wei','gwei','ether','seconds',
 'minutes','hours','days','weeks','msg','block','tx','abi','type','indexed',
 'anonymous','library','contract','interface','as','import','pragma','solidity',
}
KEEP |= {f'uint{8*i}' for i in range(1,33)} | {'uint'}
KEEP |= {f'int{8*i}'  for i in range(1,33)} | {'int'}
KEEP |= {f'bytes{i}'  for i in range(1,33)}

_TOK = re.compile(r'[A-Za-z_$][A-Za-z0-9_$]*|//[^\n]*|/\*.*?\*/|"[^"]*"|\'[^\']*\'|\S', re.DOTALL)
_IDENT = re.compile(r'[A-Za-z_$][A-Za-z0-9_$]*')

def normalize(code: str) -> str:
    code = re.sub(r'/\*.*?\*/', ' ', code, flags=re.DOTALL)
    code = re.sub(r'//[^\n]*', ' ', code)
    toks = _TOK.findall(code)
    out, mapping, cnt, prev = [], {}, 0, None
    for t in toks:
        if t.startswith('"') or t.startswith("'"):
            out.append('STR'); prev = t; continue
        if _IDENT.fullmatch(t):
            is_member = prev == '.'
            if t in KEEP or is_member or t[0].isupper():
                out.append(t)               # keyword / type / member / Type/Event name
            else:
                if t not in mapping:
                    cnt += 1; mapping[t] = f'V{cnt}'
                out.append(mapping[t])       # local/param/modifier name -> canonical
        elif not t.isspace():
            out.append(t)
        prev = t
    return ' '.join(out)

rows = list(csv.DictReader(open(CSV_IN)))
norm_match = 0
out_rows = []
for r in rows:
    nm = 1 if normalize(r['expected_solidity']) == normalize(r['model_output']) else 0
    norm_match += nm
    out_rows.append({'function': r['function'], 'version': r['version'],
                     'bleu4': r['bleu4'], 'exact_match': r['exact_match'],
                     'normalized_match': nm})

with open(CSV_OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['function','version','bleu4','exact_match','normalized_match'])
    w.writeheader(); w.writerows(out_rows)

n = len(rows)
exact = sum(1 for r in rows if r['exact_match'] == '1')
print('===== fair "correct modulo variable names" metric =====')
print(f'  functions          : {n}')
print(f'  exact match (raw)  : {exact} ({100*exact/n:.2f}%)')
print(f'  NORMALIZED match   : {norm_match} ({100*norm_match/n:.2f}%)')
# per version
from collections import defaultdict
vv = defaultdict(lambda: [0,0])
for r, o in zip(rows, out_rows):
    vv[r['version']][0] += o['normalized_match']; vv[r['version']][1] += 1
print('  per-version normalized-match:')
for v in sorted(vv, key=lambda x: tuple(int(p) for p in x.split('.'))):
    m, c = vv[v]
    print(f'    {v:<8} {m}/{c} = {100*m/c:.1f}%')
print('wrote', CSV_OUT)
