"""
run_cfg_dataset.py — harvest per-function CFG text for the whole corpus using the
IMPROVED evm_cfg_builder (BFS dispatcher reconstruction + PUSH0 + 453k selector
table). Mirrors run_gigahorse_dataset.py so the CFG and Gigahorse datasets are
built by parallel pipelines over the identical contracts.

Output (inode-safe, one JSONL): {stem, version, n_func, cfg: {func: cfg_text}}.
Resumable. Parallel via ProcessPoolExecutor.
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT     = os.path.dirname(os.path.abspath(__file__))
BYTECODE = os.path.join(ROOT, 'Contracts_Bytecode')
OUT      = os.environ.get('CFG_OUT', os.path.join(ROOT, 'datasets', 'cfg_dataset', 'cfg.jsonl'))
WORKERS  = int(os.environ.get('CFG_WORKERS', '16'))

def _setup():
    for p in (os.path.join(ROOT, 'evm_cfg_builder'), os.path.join(ROOT, 'nova')):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
_setup()

def process(job):
    stem, hexpath, ver = job
    _setup()
    try:
        from bytecode_to_cfg import extract_function_cfgs
        bc = open(hexpath).read().strip()
        cfgs = extract_function_cfgs(bc)
        named = {n: t for n, t in cfgs.items()
                 if not n.startswith(('_', '0x', 'func_'))}
    except Exception:
        named = {}
    return {'stem': stem, 'version': ver, 'n_func': len(named), 'cfg': named}

def done_stems():
    d = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try: d.add(json.loads(line)['stem'])
                except Exception: pass
    return d

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = done_stems()
    jobs = []
    for ver in sorted(os.listdir(BYTECODE)):
        d = os.path.join(BYTECODE, ver)
        if not os.path.isdir(d): continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.hex') and fn[:-4] not in done:
                jobs.append((fn[:-4], os.path.join(d, fn), ver))
    print(f'resume: {len(done)} done | remaining: {len(jobs)} | workers: {WORKERS}', flush=True)
    t0 = time.time(); n = 0
    with open(OUT, 'a') as out_f, ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(process, j): j for j in jobs}
        for fut in as_completed(futs):
            try: rec = fut.result(timeout=120)
            except Exception:
                j = futs[fut]; rec = {'stem': j[0], 'version': j[2], 'n_func': 0, 'cfg': {}}
            out_f.write(json.dumps(rec) + '\n'); n += 1
            if n % 5000 == 0:
                r = n / (time.time() - t0)
                print(f'  [{n}/{len(jobs)}] {r:.0f}/s ETA {(len(jobs)-n)/r/60:.1f}min', flush=True)
                out_f.flush()
    print('done', flush=True)

if __name__ == '__main__':
    main()
