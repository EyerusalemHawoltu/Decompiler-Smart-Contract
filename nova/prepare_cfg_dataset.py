"""
prepare_cfg_dataset.py — build the Nova training dataset from the IMPROVED CFG.

Identical pipeline to prepare_gigahorse_dataset.py (align -> dedup-by-body ->
60/20/20 leakage-free split -> Nova format), but the model INPUT is the improved
CFG text instead of Gigahorse TAC. Reuses the shared alignment/dedup helpers so
the two datasets are built by truly identical logic.
"""
import hashlib, json, os, random
from collections import defaultdict
from datasets import Dataset, DatasetDict
from prepare_gigahorse_dataset import extract_sol_functions, match_body
from prepare_solidity_dataset import normalize_cfg

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_JSONL  = os.path.join(ROOT, 'datasets', 'cfg_dataset', 'cfg.jsonl')
SOL_DIR    = os.path.join(ROOT, 'Contracts_By_Version_Cleaned')
OUTPUT_DIR = os.path.join(ROOT, 'datasets', 'cfg_dataset', 'nova-cfg')
TEST_JSON  = os.path.join(ROOT, 'datasets', 'cfg_dataset', 'test_set.json')

MAX_CHAR_LEN = 8192
TRAIN_RATIO, VALID_RATIO, TEST_RATIO = 0.60, 0.20, 0.20
RANDOM_SEED = 42

def _body_key(sol): return hashlib.md5(' '.join(sol.split()).encode()).hexdigest()

def main():
    print('reading', CFG_JSONL, flush=True)
    groups = defaultdict(list)
    n_pairs = n_contracts = 0
    for line in open(CFG_JSONL):
        r = json.loads(line)
        cfg_map = r.get('cfg') or {}
        if not cfg_map:
            continue
        ver, stem = r['version'], r['stem']
        sp = os.path.join(SOL_DIR, ver, stem + '.sol.cleaned')
        try:
            sol_funcs = extract_sol_functions(open(sp).read())
        except Exception:
            continue
        if not sol_funcs:
            continue
        n_contracts += 1
        for sig, cfg in cfg_map.items():
            body = match_body(sol_funcs, sig)
            if not body:
                continue
            cfg_norm = normalize_cfg(cfg)
            pb = f'# This is the EVM CFG for a Solidity {ver} function:\n'
            pa = '\nWhat is the Solidity source code?\n'
            inp = pb + cfg_norm + pa
            if len(inp + body) > MAX_CHAR_LEN:
                continue
            ct = '0'*len(pb) + '1'*len(cfg_norm) + '0'*len(pa) + '0'*len(body)
            groups[_body_key(body)].append({
                'version': ver, 'func_name': sig, 'cfg': cfg,
                'input': inp, 'output': body, 'char_types': ct})
            n_pairs += 1
    print(f'aligned: {n_pairs} (cfg,solidity) pairs from {n_contracts} contracts', flush=True)
    print(f'unique solidity bodies: {len(groups)}', flush=True)

    random.seed(RANDOM_SEED)
    keys = list(groups); random.shuffle(keys)
    te = int(len(keys)*TEST_RATIO); va = te + int(len(keys)*VALID_RATIO)
    parts = {'test': keys[:te], 'valid': keys[te:va], 'train': keys[va:]}

    def collect(ks):
        o = {c: [] for c in ['version', 'func_name', 'input', 'output', 'char_types']}
        for k in ks:
            rec = max(groups[k], key=lambda r: len(r['cfg']))
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
        rec = max(groups[k], key=lambda r: len(r['cfg']))
        tj[f"{rec['func_name']}_v{rec['version']}___{i}"] = {
            'cfg_representation': rec['cfg'],
            'solidity_definition': rec['output'], 'version': rec['version']}
    json.dump(tj, open(TEST_JSON, 'w'))
    print(f'test_set.json: {len(tj)} functions', flush=True)

if __name__ == '__main__':
    main()
