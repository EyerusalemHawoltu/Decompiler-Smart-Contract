"""
codebert_sim.py — semantic similarity of decompiler outputs vs ground truth
using CodeBERT (microsoft/codebert-base) cosine similarity.

Reads results/full_scores_clean_table.csv (expected_solidity, model_output),
writes results/full_scores_clean_codebert.csv with a codebert_sim column, and
prints mean similarity + fraction above thresholds (overall and per version).
"""
import csv, os
import torch
from transformers import AutoTokenizer, AutoModel
csv.field_size_limit(10**9)

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_IN  = os.environ.get('CB_IN',  os.path.join(ROOT, 'results', 'full_scores_clean_table.csv'))
CSV_OUT = os.environ.get('CB_OUT', os.path.join(ROOT, 'results', 'full_scores_clean_codebert.csv'))
MODEL   = 'microsoft/codebert-base'
BATCH   = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device = {device}', flush=True)
tok   = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device).eval()


@torch.no_grad()
def embed(texts):
    enc = tok([t or '' for t in texts], padding=True, truncation=True,
              max_length=512, return_tensors='pt').to(device)
    out  = model(**enc).last_hidden_state                  # B,T,H
    mask = enc.attention_mask.unsqueeze(-1).float()
    emb  = (out * mask).sum(1) / mask.sum(1).clamp(min=1)   # mean-pool
    return torch.nn.functional.normalize(emb, dim=-1)


rows = list(csv.DictReader(open(CSV_IN)))
print(f'{len(rows)} rows', flush=True)
sims = []
out_rows = []
for i in range(0, len(rows), BATCH):
    chunk = rows[i:i+BATCH]
    e1 = embed([r['expected_solidity'] for r in chunk])
    e2 = embed([r['model_output']      for r in chunk])
    cs = (e1 * e2).sum(-1).cpu().tolist()
    for r, c in zip(chunk, cs):
        out_rows.append((r, c)); sims.append(c)
    if i % (BATCH*20) == 0:
        print(f'  {i}/{len(rows)}', flush=True)

with open(CSV_OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['function', 'version', 'bleu4', 'exact_match', 'codebert_sim'])
    for r, c in out_rows:
        w.writerow([r['function'], r['version'], r['bleu4'], r['exact_match'], round(c, 4)])

n = len(sims)
print('\n===== CodeBERT semantic similarity =====')
print(f'  pairs        : {n}')
print(f'  mean cosine  : {sum(sims)/n:.4f}')
for th in (0.95, 0.9, 0.8, 0.7):
    c = sum(1 for s in sims if s >= th)
    print(f'  >= {th:>4}    : {c:>5} ({100*c/n:.1f}%)')
# per version mean
from collections import defaultdict
vs = defaultdict(list)
for r, c in out_rows:
    vs[r['version']].append(c)
print('\n  per-version mean cosine:')
for v in sorted(vs, key=lambda x: tuple(int(p) for p in x.split('.'))):
    print(f'    {v:<8} n={len(vs[v]):<5} {sum(vs[v])/len(vs[v]):.4f}')
print(f'\nwrote {CSV_OUT}', flush=True)
