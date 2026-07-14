"""
prepare_gigahorse_dataset.py — build the Nova training dataset from Gigahorse TAC.

Mirrors prepare_solidity_dataset.py exactly, but the model INPUT is the Gigahorse
per-function TAC (instead of the CFG). Pipeline:
  1. read datasets/gigahorse_dataset/gigahorse_tac.jsonl  (per-contract per-func TAC)
  2. align each named TAC function to its Solidity source body (.sol.cleaned), by
     function name (+arity tiebreak)
  3. dedup by NORMALIZED SOLIDITY BODY (pragma-invariant) -> no train/test leakage
  4. group-split 60/20/20 by body
  5. emit HF DatasetDict (columns: version, func_name, input, output, char_types)
     + test_set.json   -> identical format to the CFG dataset, so finetune_full.py
     and dataset.py are reused unchanged.
"""
import hashlib, json, os, random, re
from collections import defaultdict
from datasets import Dataset, DatasetDict

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAC_JSONL  = os.path.join(ROOT, 'datasets', 'gigahorse_dataset', 'gigahorse_tac.jsonl')
SOL_DIR    = os.path.join(ROOT, 'Contracts_By_Version_Cleaned')
OUTPUT_DIR = os.path.join(ROOT, 'datasets', 'gigahorse_dataset', 'nova-gigahorse')
TEST_JSON  = os.path.join(ROOT, 'datasets', 'gigahorse_dataset', 'test_set.json')

MAX_CHAR_LEN = 8192
TRAIN_RATIO, VALID_RATIO, TEST_RATIO = 0.60, 0.20, 0.20
RANDOM_SEED = 42

# ── TAC normalization: <label-N> after each statement line (hierarchical attn) ──
def normalize_tac(tac_text: str) -> str:
    out, idx = [], 1
    for line in tac_text.split('\n'):
        s = line.strip()
        if s and not s.startswith('function') and not s.startswith('block') and idx <= 256:
            out.append(line + f' <label-{idx}>'); idx += 1
        else:
            out.append(line)
    return '\n'.join(out)

# ── Solidity source -> {name: [(arity, full_function_text)]} ────────────────────
_FN = re.compile(r'\bfunction\s+(\w+)\s*\(')
def extract_sol_functions(src: str):
    src = re.sub(r'/\*.*?\*/', ' ', src, flags=re.DOTALL)
    src = re.sub(r'//[^\n]*', ' ', src)
    funcs = defaultdict(list)
    for m in _FN.finditer(src):
        name = m.group(1)
        # params
        i = m.end(); depth = 1; j = i
        while j < len(src) and depth:
            if src[j] == '(': depth += 1
            elif src[j] == ')': depth -= 1
            j += 1
        params = src[i:j-1].strip()
        arity = 0 if not params else params.count(',') + 1
        # body: find first '{' after params, brace-match
        k = src.find('{', j)
        if k == -1:
            continue
        d = 0; e = k
        while e < len(src):
            if src[e] == '{': d += 1
            elif src[e] == '}':
                d -= 1
                if d == 0: break
            e += 1
        full = ('function ' + src[m.end(1):e+1]).strip()
        funcs[name].append((arity, full))
    return funcs

def match_body(sol_funcs, gh_sig):
    name = gh_sig.split('(', 1)[0]
    inner = gh_sig[gh_sig.find('(')+1:gh_sig.rfind(')')] if '(' in gh_sig else ''
    arity = 0 if not inner.strip() else inner.count(',') + 1
    cands = sol_funcs.get(name, [])
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0][1]
    for a, full in cands:           # disambiguate by arity
        if a == arity:
            return full
    return None

def _body_key(sol): return hashlib.md5(' '.join(sol.split()).encode()).hexdigest()

def main():
    print('reading', TAC_JSONL, flush=True)
    groups = defaultdict(list)      # body_key -> [record]
    n_pairs = n_contracts = 0
    sol_cache = {}
    for line in open(TAC_JSONL):
        r = json.loads(line)
        tac_map = r.get('tac') or {}
        if not tac_map:
            continue
        ver, stem = r['version'], r['stem']
        sp = os.path.join(SOL_DIR, ver, stem + '.sol.cleaned')
        if sp not in sol_cache:
            try: sol_cache = {sp: extract_sol_functions(open(sp).read())}
            except Exception: sol_cache = {sp: {}}
        sol_funcs = sol_cache[sp]
        if not sol_funcs:
            continue
        n_contracts += 1
        for sig, tac in tac_map.items():
            body = match_body(sol_funcs, sig)
            if not body:
                continue
            tac_norm = normalize_tac(tac)
            pb = f'# This is the EVM TAC for a Solidity {ver} function:\n'
            pa = '\nWhat is the Solidity source code?\n'
            inp = pb + tac_norm + pa
            if len(inp + body) > MAX_CHAR_LEN:
                continue
            ct = '0'*len(pb) + '1'*len(tac_norm) + '0'*len(pa) + '0'*len(body)
            groups[_body_key(body)].append({
                'version': ver, 'func_name': sig, 'tac': tac,
                'input': inp, 'output': body, 'char_types': ct})
            n_pairs += 1
    print(f'aligned: {n_pairs} (tac,solidity) pairs from {n_contracts} contracts', flush=True)
    print(f'unique solidity bodies: {len(groups)}', flush=True)

    random.seed(RANDOM_SEED)
    keys = list(groups); random.shuffle(keys)
    te = int(len(keys)*TEST_RATIO); va = te + int(len(keys)*VALID_RATIO)
    parts = {'test': keys[:te], 'valid': keys[te:va], 'train': keys[va:]}

    def collect(ks):
        # ONE representative per unique body (richest TAC) -> removes pragma/contract
        # duplication; each function appears exactly once.
        o = {c: [] for c in ['version', 'func_name', 'input', 'output', 'char_types']}
        for k in ks:
            rec = max(groups[k], key=lambda r: len(r['tac']))
            for c in o: o[c].append(rec[c])
        return o
    dd = {s: collect(ks) for s, ks in parts.items()}
    for s in parts:
        print(f'  {s}: {len(dd[s]["input"])} samples ({len(parts[s])} unique bodies)', flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DatasetDict({s: Dataset.from_dict(dd[s]) for s in dd}).save_to_disk(OUTPUT_DIR)
    print('saved', OUTPUT_DIR, flush=True)

    tj = {}
    for i, k in enumerate(parts['test']):
        rec = max(groups[k], key=lambda r: len(r['tac']))
        key = f"{rec['func_name']}_v{rec['version']}___{i}"
        tj[key] = {'tac_representation': rec['tac'],
                   'solidity_definition': rec['output'], 'version': rec['version']}
    os.makedirs(os.path.dirname(TEST_JSON), exist_ok=True)
    json.dump(tj, open(TEST_JSON, 'w'))
    print(f'test_set.json: {len(tj)} functions', flush=True)

if __name__ == '__main__':
    main()
