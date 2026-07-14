"""
run_gigahorse_dataset.py — harvest per-function Gigahorse TAC for the whole corpus.

Like run_gigahorse_batch.py, but instead of just function names it serializes the
full per-function TAC (via gigahorse_tac_serialize) and appends, inode-safely, to
one JSONL: {stem, version, function, tac}. Resumable; deletes each batch's output.

Alignment to Solidity is a separate post-step (match stem+signature to .sol.cleaned).
"""
import json, os, shutil, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gigahorse_tac_serialize import serialize_tac

ROOT     = os.path.dirname(os.path.abspath(__file__))
BYTECODE = os.path.join(ROOT, 'Contracts_Bytecode')
SBX      = os.environ['GH_SBX']
CACHE    = os.environ['GH_CACHE']
WORK     = os.environ.get('GH_WORK', '/scratch/eh3115/gh_ds_work')
OUT      = os.environ.get('GH_TAC', os.path.join(ROOT, 'datasets', 'gigahorse_dataset', 'gigahorse_tac.jsonl'))
JOBS     = os.environ.get('GH_JOBS', '32')
BATCH    = int(os.environ.get('GH_BATCH', '300'))
TIMEOUT  = int(os.environ.get('GH_TIMEOUT', '60'))


def done_stems():
    d = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try: d.add(json.loads(line)['stem'])
                except Exception: pass
    return d


def run_batch(items, out_f):
    bdir = os.path.join(WORK, 'hex'); tdir = os.path.join(WORK, '.temp')
    shutil.rmtree(bdir, ignore_errors=True); shutil.rmtree(tdir, ignore_errors=True)
    os.makedirs(bdir, exist_ok=True)
    for stem, hp, _ in items:
        shutil.copy(hp, os.path.join(bdir, stem + '.hex'))
    cmd = ['singularity', 'exec', '--bind', '/scratch/eh3115',
           '--bind', f'{CACHE}:/opt/gigahorse/gigahorse-toolchain/cache',
           '--pwd', WORK, SBX, 'python3',
           '/opt/gigahorse/gigahorse-toolchain/gigahorse.py',
           '--jobs', JOBS, '-T', str(TIMEOUT), bdir]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=TIMEOUT * len(items) + 600)
    except subprocess.TimeoutExpired:
        pass
    for stem, _, ver in items:
        out_dir = os.path.join(tdir, stem, 'out')
        funcs = {}
        try:
            if os.path.isdir(out_dir):
                funcs = serialize_tac(out_dir)
        except Exception:
            funcs = {}
        # one record per stem, with all its functions' TAC
        out_f.write(json.dumps({'stem': stem, 'version': ver,
                                'n_func': len(funcs), 'tac': funcs}) + '\n')
    out_f.flush()
    shutil.rmtree(bdir, ignore_errors=True); shutil.rmtree(tdir, ignore_errors=True)


def main():
    os.makedirs(WORK, exist_ok=True); os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = done_stems()
    print(f'resume: {len(done)} stems already harvested', flush=True)
    jobs = []
    for ver in sorted(os.listdir(BYTECODE)):
        d = os.path.join(BYTECODE, ver)
        if not os.path.isdir(d): continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.hex') and fn[:-4] not in done:
                jobs.append((fn[:-4], os.path.join(d, fn), ver))
    print(f'remaining: {len(jobs)}', flush=True)
    t0 = time.time()
    with open(OUT, 'a') as out_f:
        for i in range(0, len(jobs), BATCH):
            run_batch(jobs[i:i+BATCH], out_f)
            n = i + min(BATCH, len(jobs) - i)
            rate = n / (time.time() - t0)
            print(f'  [{n}/{len(jobs)}] {rate:.1f}/s ETA {(len(jobs)-n)/rate/3600:.1f}h', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
