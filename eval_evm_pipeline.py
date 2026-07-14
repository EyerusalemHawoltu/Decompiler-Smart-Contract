"""
eval_evm_pipeline.py
====================
Batch-test the evm_cfg_builder pipeline (BFS + known_hashes) on all contracts
in Contracts_Bytecode/.

For each .hex file:
  - Run extract_function_cfgs() from nova/bytecode_to_cfg.py
  - Record: version, contract name, address, status, functions found,
    block counts, name resolution (real name vs hex ID), error message

Outputs:
  results/evm_pipeline_results.jsonl   — one JSON record per contract (streaming)
  results/evm_pipeline_summary.json    — aggregate stats

Supports resume: already-processed contracts (by file path) are skipped.

Usage:
    python eval_evm_pipeline.py                  # all contracts
    python eval_evm_pipeline.py --workers 8      # parallel workers
    python eval_evm_pipeline.py --version 0.8.4  # single version only
    python eval_evm_pipeline.py --limit 1000     # first N contracts (quick test)
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeout

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.abspath(__file__))
BYTECODE_DIR = os.path.join(ROOT, 'Contracts_Bytecode')
RESULTS_DIR  = os.path.join(ROOT, 'results')
JSONL_OUT    = os.path.join(RESULTS_DIR, 'evm_pipeline_results.jsonl')
SUMMARY_OUT  = os.path.join(RESULTS_DIR, 'evm_pipeline_summary.json')

TIMEOUT_PER_CONTRACT = 30   # seconds — skip if CFG extraction hangs


# ── Worker (runs in subprocess) ───────────────────────────────────────────────
def _process_one(args):
    """
    Called in a worker process.
    Returns a dict record for one contract.
    """
    hex_path, version, address, contract_name = args

    # Set up path in each worker process
    _local_ecb = os.path.join(ROOT, 'evm_cfg_builder')
    _nova_dir  = os.path.join(ROOT, 'nova')
    for p in (_local_ecb, _nova_dir):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    record = {
        'version':       version,
        'address':       address,
        'contract_name': contract_name,
        'file':          os.path.relpath(hex_path, ROOT),
        'status':        'ok',
        'n_functions':   0,
        'functions':     [],   # [{'name': str, 'blocks': int, 'named': bool}]
        'n_named':       0,    # functions with a real name (not hex ID)
        'n_hex_id':      0,    # functions whose name is still a hex ID
        'error':         None,
    }

    try:
        with open(hex_path) as f:
            bytecode = f.read().strip()

        if not bytecode:
            record['status'] = 'empty'
            return record

        from bytecode_to_cfg import extract_function_cfgs
        cfgs = extract_function_cfgs(bytecode)

        fn_list = []
        for name, cfg_text in cfgs.items():
            bb_count = cfg_text.count('\n\t- @')
            is_hex   = name.startswith('0x') or name.startswith('func_')
            fn_list.append({'name': name, 'blocks': bb_count, 'named': not is_hex})

        record['n_functions'] = len(fn_list)
        record['functions']   = fn_list
        record['n_named']     = sum(1 for f in fn_list if f['named'])
        record['n_hex_id']    = sum(1 for f in fn_list if not f['named'])

        if not fn_list:
            record['status'] = 'no_functions'

    except Exception as e:
        record['status'] = 'error'
        record['error']  = str(e)[:300]

    return record


# ── Main ──────────────────────────────────────────────────────────────────────
def collect_jobs(version_filter=None, limit=None):
    """Collect all (hex_path, version, address, contract_name) tuples."""
    jobs = []
    for version in sorted(os.listdir(BYTECODE_DIR)):
        ver_dir = os.path.join(BYTECODE_DIR, version)
        if not os.path.isdir(ver_dir):
            continue
        if version_filter and version != version_filter:
            continue
        for fname in sorted(os.listdir(ver_dir)):
            if not fname.endswith('.hex'):
                continue
            base = fname[:-4]                        # strip .hex
            parts = base.split('_', 1)
            address       = parts[0]
            contract_name = parts[1] if len(parts) > 1 else base
            jobs.append((os.path.join(ver_dir, fname), version, address, contract_name))
    if limit:
        jobs = jobs[:limit]
    return jobs


def load_done(jsonl_path):
    """Return set of file paths already recorded in the JSONL output."""
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(rec['file'])
            except Exception:
                pass
    return done


def run(workers=4, version_filter=None, limit=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    jobs   = collect_jobs(version_filter, limit)
    done   = load_done(JSONL_OUT)
    jobs   = [j for j in jobs if os.path.relpath(j[0], ROOT) not in done]

    total_all  = len(collect_jobs(version_filter, limit))
    n_skip     = total_all - len(jobs)
    n_todo     = len(jobs)

    print(f'Contracts total : {total_all}')
    print(f'Already done    : {n_skip}  (resuming)')
    print(f'To process      : {n_todo}')
    print(f'Workers         : {workers}')
    print(f'Output          : {JSONL_OUT}')
    print()

    if not n_todo:
        print('Nothing to do — all contracts already processed.')
        _write_summary()
        return

    t0        = time.time()
    n_done    = 0
    n_ok      = 0
    n_err     = 0
    n_empty   = 0
    n_nofunc  = 0

    with open(JSONL_OUT, 'a') as out_f:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, j): j for j in jobs}
            for fut in as_completed(futures):
                n_done += 1
                job = futures[fut]
                try:
                    rec = fut.result(timeout=TIMEOUT_PER_CONTRACT)
                except FuturesTimeout:
                    _, version, address, contract_name = job
                    rel = os.path.relpath(job[0], ROOT)
                    rec = {
                        'version': version, 'address': address,
                        'contract_name': contract_name, 'file': rel,
                        'status': 'timeout', 'n_functions': 0,
                        'functions': [], 'n_named': 0, 'n_hex_id': 0,
                        'error': f'Exceeded {TIMEOUT_PER_CONTRACT}s',
                    }
                except Exception as e:
                    _, version, address, contract_name = job
                    rel = os.path.relpath(job[0], ROOT)
                    rec = {
                        'version': version, 'address': address,
                        'contract_name': contract_name, 'file': rel,
                        'status': 'error', 'n_functions': 0,
                        'functions': [], 'n_named': 0, 'n_hex_id': 0,
                        'error': str(e)[:300],
                    }

                # Tally
                s = rec['status']
                if s == 'ok':       n_ok    += 1
                elif s == 'error':  n_err   += 1
                elif s == 'empty':  n_empty += 1
                else:               n_nofunc+= 1

                # Stream to file
                out_f.write(json.dumps(rec) + '\n')
                out_f.flush()

                # Progress
                if n_done % 500 == 0 or n_done == n_todo:
                    elapsed = time.time() - t0
                    rate    = n_done / elapsed if elapsed > 0 else 0
                    eta     = (n_todo - n_done) / rate if rate > 0 else 0
                    print(f'  [{n_done:>6}/{n_todo}]  '
                          f'ok={n_ok} err={n_err} empty={n_empty} nofunc={n_nofunc}  '
                          f'{rate:.1f}/s  ETA {eta/60:.1f}min')

    print(f'\nDone in {(time.time()-t0)/60:.1f} min')
    _write_summary()


def _write_summary():
    """Read JSONL and write aggregate summary JSON."""
    if not os.path.exists(JSONL_OUT):
        return

    from collections import defaultdict, Counter

    stats = {
        'total': 0, 'ok': 0, 'error': 0, 'empty': 0,
        'no_functions': 0, 'timeout': 0,
        'total_functions': 0, 'total_named': 0, 'total_hex_id': 0,
        'total_blocks': 0,
        'by_version': defaultdict(lambda: {
            'total': 0, 'ok': 0, 'error': 0,
            'functions': 0, 'named': 0,
        }),
        'top_errors': Counter(),
        'top_functions': Counter(),   # most common function names
    }

    with open(JSONL_OUT) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            stats['total'] += 1
            s = r.get('status', 'error')
            if s in stats:
                stats[s] += 1
            ver = r.get('version', 'unknown')
            stats['by_version'][ver]['total'] += 1
            if s == 'ok':
                stats['by_version'][ver]['ok'] += 1
            else:
                stats['by_version'][ver]['error'] += 1
                if r.get('error'):
                    msg = r['error'][:80]
                    stats['top_errors'][msg] += 1

            for fn in r.get('functions', []):
                stats['total_functions'] += 1
                stats['total_blocks']    += fn.get('blocks', 0)
                if fn.get('named'):
                    stats['total_named'] += 1
                    stats['top_functions'][fn['name']] += 1
                else:
                    stats['total_hex_id'] += 1
            stats['by_version'][ver]['functions'] += r.get('n_functions', 0)
            stats['by_version'][ver]['named']     += r.get('n_named', 0)

    # Convert Counters to lists
    stats['top_errors']    = stats['top_errors'].most_common(20)
    stats['top_functions'] = stats['top_functions'].most_common(30)
    stats['by_version']    = dict(stats['by_version'])

    n = stats['total']
    nf = stats['total_functions']
    stats['success_rate_pct']      = round(stats['ok'] / n * 100, 2) if n else 0
    stats['name_resolution_pct']   = round(stats['total_named'] / nf * 100, 2) if nf else 0
    stats['avg_functions_per_contract'] = round(nf / max(stats['ok'], 1), 2)
    stats['avg_blocks_per_function']    = round(stats['total_blocks'] / max(nf, 1), 2)

    with open(SUMMARY_OUT, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f'\n=== Summary ===')
    print(f'  Total contracts  : {n}')
    print(f'  Success          : {stats["ok"]}  ({stats["success_rate_pct"]}%)')
    print(f'  Errors           : {stats["error"]}')
    print(f'  Empty bytecode   : {stats["empty"]}')
    print(f'  No functions     : {stats["no_functions"]}')
    print(f'  Timeouts         : {stats["timeout"]}')
    print(f'  Total functions  : {nf}')
    print(f'  Named (resolved) : {stats["total_named"]}  ({stats["name_resolution_pct"]}%)')
    print(f'  Hex ID (unresolved): {stats["total_hex_id"]}')
    print(f'  Avg funcs/contract : {stats["avg_functions_per_contract"]}')
    print(f'  Avg blocks/func    : {stats["avg_blocks_per_function"]}')
    print(f'\nTop error messages:')
    for msg, cnt in stats['top_errors'][:5]:
        print(f'  [{cnt:>5}x]  {msg}')
    print(f'\nTop resolved function names:')
    for name, cnt in stats['top_functions'][:10]:
        print(f'  [{cnt:>5}x]  {name}')
    print(f'\nFull summary saved to {SUMMARY_OUT}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Batch EVM pipeline test')
    ap.add_argument('--workers',  type=int, default=4,    help='Parallel workers (default 4)')
    ap.add_argument('--version',  type=str, default=None, help='Test one version only (e.g. 0.8.4)')
    ap.add_argument('--limit',    type=int, default=None, help='Cap total contracts (quick test)')
    ap.add_argument('--summary',  action='store_true',    help='Just recompute summary from existing JSONL')
    args = ap.parse_args()

    if args.summary:
        _write_summary()
    else:
        run(workers=args.workers, version_filter=args.version, limit=args.limit)
