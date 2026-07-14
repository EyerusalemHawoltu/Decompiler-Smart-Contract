"""
build_eval.py — merge inference_test.json + scores_test.csv into
                static/eval_results.json for the evaluation page.

Run on HPC:
    python build_eval.py
"""
import json, csv, os, statistics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
STATIC_DIR  = os.path.join(os.path.dirname(__file__), 'static')

full_path   = os.path.join(RESULTS_DIR, 'full_eval.json')     # 32K results from run_full_eval.py
inf_path    = os.path.join(RESULTS_DIR, 'inference_test.json') # 500 sample fallback
scores_path = os.path.join(RESULTS_DIR, 'scores_test.csv')
out_path    = os.path.join(STATIC_DIR,  'eval_results.json')

# ── Load inference data ────────────────────────────────────────────────────
rows = []

if os.path.exists(full_path):
    # ── Full 32K results from run_full_eval.py ─────────────────────────────
    print(f'Using full eval results: {full_path}')
    with open(full_path) as f:
        full_data = json.load(f)

    for key, item in full_data.items():
        rows.append({
            'key':          key,
            'name':         item.get('name', key.rsplit('_v', 1)[0] if '_v' in key else key),
            'bleu4':        float(item.get('bleu4', 0)),
            'codebert_sim': float(item.get('codebert_sim', item.get('bleu4', 0))),  # fallback
            'exact_match':  int(item.get('exact_match', 0)),
            'cfg':          item.get('cfg', ''),
            'ground_truth': item.get('ground_truth', ''),
            'predicted':    item.get('predicted', ''),
        })

else:
    # ── Fallback: 500 sample results ───────────────────────────────────────
    print(f'Full eval not found, using 500-sample fallback: {inf_path}')
    with open(inf_path) as f:
        inf_data = json.load(f)

    scores = {}
    with open(scores_path, newline='') as f:
        for row in csv.DictReader(f):
            scores[row['function']] = {
                'bleu4':        float(row['bleu4']),
                'codebert_sim': float(row.get('codebert_sim', row['bleu4'])),
                'exact_match':  int(row['exact_match']),
            }

    for key, item in inf_data.items():
        sc = scores.get(key, {})
        preds = item.get('predictions', [])
        predicted = preds[0] if isinstance(preds, list) and preds else str(preds)
        display_name = key.rsplit('_v', 1)[0] if '_v' in key else key
        rows.append({
            'key':          key,
            'name':         display_name,
            'bleu4':        sc.get('bleu4', 0),
            'codebert_sim': sc.get('codebert_sim', 0),
            'exact_match':  sc.get('exact_match', 0),
            'cfg':          item.get('cfg', ''),
            'ground_truth': item.get('ground_truth', ''),
            'predicted':    predicted,
        })

# Sort by bleu4 descending
rows.sort(key=lambda r: r['bleu4'], reverse=True)

# ── Summary stats ──────────────────────────────────────────────────────────
bleus   = [r['bleu4'] for r in rows]
cbs     = [r['codebert_sim'] for r in rows]
exacts  = [r['exact_match'] for r in rows]

summary = {
    'total':          len(rows),
    'exact_match_n':  sum(exacts),
    'exact_match_pct': round(100 * sum(exacts) / len(exacts), 1),
    'avg_bleu4':      round(statistics.mean(bleus), 4),
    'avg_codebert':   round(statistics.mean(cbs), 4),
    'bleu4_gte_09':   sum(1 for b in bleus if b >= 0.9),
    'bleu4_gte_07':   sum(1 for b in bleus if b >= 0.7),
    'bleu4_lt_05':    sum(1 for b in bleus if b < 0.5),
}

output = {'summary': summary, 'results': rows}

os.makedirs(STATIC_DIR, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(output, f)

print(f"Wrote {len(rows)} rows to {out_path}")
print(f"Summary: {summary}")
