"""
openai_decompile.py — test a strong (OpenAI) model on CFG->Solidity.

For a sample of the CLEAN test set, builds a prompt that EXPLICITLY hands the
model the recovered function signature (parsed from the CFG) plus the CFG, and
asks for the Solidity function. Lets us compare a large prompted model against
the fine-tuned 1.3B Nova on identical, leakage-free test functions.

Requires:  pip install openai   and   export OPENAI_API_KEY=...
Optional:  OPENAI_MODEL (default gpt-4o), N_SAMPLE (default 200)

Outputs results/openai_decompile.csv  with body-only + full BLEU/exact.
"""
import csv, json, math, os, re, sys, time

ROOT      = os.path.dirname(os.path.abspath(__file__))
TEST_JSON = os.path.join(ROOT, 'data-tokenized', 'test_set.json')
OUT_CSV   = os.path.join(ROOT, 'results', 'openai_decompile.csv')
MODEL     = os.environ.get('OPENAI_MODEL', 'gpt-4o')
N_SAMPLE  = int(os.environ.get('N_SAMPLE', '200'))

# ── metrics ──────────────────────────────────────────────────────────────────
def _ng(t, n):
    from collections import Counter
    return Counter(tuple(t[i:i+n]) for i in range(len(t)-n+1))

def bleu4(ref, hyp):
    ref, hyp = ref.split(), hyp.split()
    if not hyp: return 0.0
    sc = []
    for n in range(1, 5):
        h = _ng(hyp, n)
        if not h: return 0.0
        r = _ng(ref, n)
        m = sum(min(c, r[g]) for g, c in h.items()); tot = max(sum(h.values()), 1)
        p = m/tot
        if p == 0: return 0.0
        sc.append(p)
    bp = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref)/max(len(hyp), 1))
    return bp * math.exp(sum(math.log(s) for s in sc)/4)

def body(code):
    i = code.find('{')
    if i < 0: return code.strip()
    j = code.rfind('}')
    return code[i+1:j].strip() if j > i else code[i+1:].strip()

def norm(s): return ' '.join(s.split())

def sig_from_cfg(cfg):
    """First 'Function <sig>' line of the CFG -> the recovered signature."""
    m = re.search(r'Function\s+([^\n]+)', cfg)
    return m.group(1).strip() if m else ''

# ── prompt ───────────────────────────────────────────────────────────────────
SYS = ("You are an expert EVM decompiler. Given a control-flow graph (CFG) "
       "extracted from EVM bytecode and the recovered function signature, "
       "reconstruct the most likely original Solidity function. Output ONLY the "
       "Solidity function, no explanation, no markdown fences.")

def make_prompt(cfg, version, signature):
    return (f"Solidity compiler version: {version}\n"
            f"Recovered function signature: {signature}\n\n"
            f"EVM CFG:\n{cfg}\n\n"
            f"Reconstruct the full Solidity function (signature + body).")

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY

    data = json.load(open(TEST_JSON))
    keys = list(data.keys())[:N_SAMPLE]
    print(f'model={MODEL}  sampling {len(keys)} of {len(data)} test functions', flush=True)

    rows = []
    for i, k in enumerate(keys):
        e = data[k]
        cfg = e.get('cfg_representation', '').strip()
        gt  = e.get('solidity_definition', '').strip()
        ver = e.get('version', '0.8.x')
        sig = sig_from_cfg(cfg)
        if not cfg:
            continue
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=0,
                messages=[{'role': 'system', 'content': SYS},
                          {'role': 'user', 'content': make_prompt(cfg, ver, sig)}])
            pred = resp.choices[0].message.content.strip()
            pred = re.sub(r'^```[a-zA-Z]*\n?|```$', '', pred).strip()
        except Exception as ex:
            print(f'  [WARN] {k}: {ex}', flush=True); pred = ''
        rows.append({'function': k, 'version': ver, 'expected': gt, 'output': pred,
                     'bleu4_full': round(bleu4(norm(gt), norm(pred)), 4),
                     'bleu4_body': round(bleu4(norm(body(gt)), norm(body(pred))), 4),
                     'exact_body': 1 if norm(body(gt)) == norm(body(pred)) else 0})
        if (i+1) % 20 == 0:
            print(f'  {i+1}/{len(keys)}', flush=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['function','version','expected','output',
                                          'bleu4_full','bleu4_body','exact_body'])
        w.writeheader(); w.writerows(rows)

    n = len(rows)
    print('\n===== OpenAI (%s) on %d clean test functions =====' % (MODEL, n))
    print('  BLEU-4 full  : %.4f' % (sum(r['bleu4_full'] for r in rows)/n))
    print('  BLEU-4 body  : %.4f' % (sum(r['bleu4_body'] for r in rows)/n))
    print('  exact body   : %.2f%%' % (100*sum(r['exact_body'] for r in rows)/n))
    print('  (Nova on full set: BLEU-4 full 0.171 / body 0.072)')
    print('wrote', OUT_CSV, flush=True)

if __name__ == '__main__':
    main()
