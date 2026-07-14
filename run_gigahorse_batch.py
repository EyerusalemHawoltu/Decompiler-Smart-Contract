"""
run_gigahorse_batch.py — run Gigahorse over the whole corpus, inode-safely.

Gigahorse emits ~150 files per contract, so we never keep them around:
for each BATCH of contracts we (1) stage their .hex into a temp dir,
(2) run gigahorse via singularity over that dir, (3) harvest only the
recovered public-function names from HighLevelFunctionName.csv into a single
append-only JSONL, then (4) delete the batch output. Resumable: contracts
already in the JSONL are skipped.

Run inside the SLURM job (singularity module loaded).
"""
import json, os, re, shutil, subprocess, sys, time

ROOT       = os.path.dirname(os.path.abspath(__file__))
BYTECODE   = os.path.join(ROOT, 'Contracts_Bytecode')
SIF        = os.environ.get('GH_SBX', os.path.join(ROOT, 'gigahorse_sbx'))
CACHE_BIND = os.environ['GH_CACHE']                       # host dir -> container cache
WORK       = os.environ.get('GH_WORK', '/scratch/eh3115/gh_work')
OUT_JSONL  = os.environ.get('GH_NAMES', os.path.join(ROOT, 'results', 'gigahorse_names.jsonl'))
JOBS       = os.environ.get('GH_JOBS', '32')
BATCH      = int(os.environ.get('GH_BATCH', '300'))
TIMEOUT    = int(os.environ.get('GH_TIMEOUT', '120'))      # per-contract souffle timeout

_SKIP = {'__function_selector__', 'fallback', 'receive', ''}
_IDENT = re.compile(r'[A-Za-z_$][A-Za-z0-9_$]*')


def parse_names(out_dir):
    p = os.path.join(out_dir, 'HighLevelFunctionName.csv')
    if not os.path.isfile(p):
        return None
    names = set()
    with open(p) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            block, sig = parts[0], parts[1]
            if sig == block:
                continue
            bare = sig.split('(', 1)[0].strip()
            if bare in _SKIP or bare.startswith('0x'):
                continue
            if _IDENT.fullmatch(bare):
                names.add(bare)
    return sorted(names)


def done_set():
    done = set()
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['stem'])
                except Exception:
                    pass
    return done


def run_batch(stems_hex, version, out_f):
    """stems_hex: list of (stem, hexpath). Runs gigahorse on a staged dir."""
    bdir = os.path.join(WORK, 'hex')
    tdir = os.path.join(WORK, '.temp')
    shutil.rmtree(bdir, ignore_errors=True); shutil.rmtree(tdir, ignore_errors=True)
    os.makedirs(bdir, exist_ok=True)
    for stem, hp in stems_hex:
        shutil.copy(hp, os.path.join(bdir, stem + '.hex'))

    cmd = [
        'singularity', 'exec',
        '--bind', '/scratch/eh3115',
        '--bind', f'{CACHE_BIND}:/opt/gigahorse/gigahorse-toolchain/cache',
        '--pwd', WORK,
        SIF, 'python3', '/opt/gigahorse/gigahorse-toolchain/gigahorse.py',
        '--jobs', JOBS, '-T', str(TIMEOUT), bdir,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=TIMEOUT * len(stems_hex) + 600)
    except subprocess.TimeoutExpired:
        pass

    for stem, _ in stems_hex:
        out_dir = os.path.join(tdir, stem, 'out')
        names = parse_names(out_dir)
        rec = {'stem': stem, 'version': version,
               'names': names if names is not None else [],
               'status': 'ok' if names is not None else 'noout'}
        out_f.write(json.dumps(rec) + '\n')
    out_f.flush()
    shutil.rmtree(bdir, ignore_errors=True); shutil.rmtree(tdir, ignore_errors=True)


def main():
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    done = done_set()
    print(f'resume: {len(done)} already harvested', flush=True)

    jobs = []
    for version in sorted(os.listdir(BYTECODE)):
        d = os.path.join(BYTECODE, version)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.hex') and fn[:-4] not in done:
                jobs.append((fn[:-4], os.path.join(d, fn), version))
    print(f'remaining: {len(jobs)}', flush=True)

    t0 = time.time()
    with open(OUT_JSONL, 'a') as out_f:
        for i in range(0, len(jobs), BATCH):
            chunk = jobs[i:i + BATCH]
            ver = chunk[0][2]
            run_batch([(s, h) for s, h, _ in chunk], ver, out_f)
            n = i + len(chunk)
            rate = n / (time.time() - t0)
            eta = (len(jobs) - n) / rate / 3600 if rate else 0
            print(f'  [{n}/{len(jobs)}] {rate:.1f}/s ETA {eta:.1f}h', flush=True)
    print('done.', flush=True)


if __name__ == '__main__':
    main()
