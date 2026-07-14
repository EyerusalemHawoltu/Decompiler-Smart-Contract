"""
validate_cfg_names.py
=====================
Validates evm_cfg_builder by comparing extracted CFG function names against
the public/external functions declared in the matching Solidity source.

Metrics computed per contract and aggregated:
  - Precision   : CFG names that appear in Solidity  (how accurate is the extractor)
  - Recall      : Solidity public functions found by CFG  (how complete)
  - F1          : harmonic mean of precision + recall
  - Jaccard     : |matched| / |cfg ∪ sol|  (set overlap)
  - Exact Match : 1.0 if CFG names == Solidity public names, else 0.0
  - BLEU-1..4   : treating sorted function name list as a token sequence

Usage:
    python validate_cfg_names.py                   # all contracts
    python validate_cfg_names.py --version 0.8.4   # one version
    python validate_cfg_names.py --limit 500        # quick test
    python validate_cfg_names.py --show-misses      # print missed names

Output:
    results/cfg_name_validation.json          per-contract records
    results/cfg_name_validation_summary.json  aggregate + per-version
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT         = os.path.dirname(os.path.abspath(__file__))
BYTECODE_DIR = os.path.join(ROOT, 'Contracts_Bytecode')
SOLIDITY_DIR = os.path.join(ROOT, 'Contracts_By_Version_Cleaned')
RESULTS_DIR  = os.path.join(ROOT, 'results')
OUT_JSON     = os.path.join(RESULTS_DIR, 'cfg_name_validation.json')
JSONL_OUT    = os.path.join(RESULTS_DIR, 'cfg_name_validation.jsonl')
SUMMARY_JSON = os.path.join(RESULTS_DIR, 'cfg_name_validation_summary.json')

def _setup_path():
    for p in (os.path.join(ROOT, 'evm_cfg_builder'), os.path.join(ROOT, 'nova')):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

_setup_path()


# ── BLEU (no external deps) ───────────────────────────────────────────────────
def _ngrams(tokens: list, n: int) -> dict:
    counts = {}
    for i in range(len(tokens) - n + 1):
        g = tuple(tokens[i:i+n])
        counts[g] = counts.get(g, 0) + 1
    return counts

def bleu_score(hypothesis: list, reference: list, max_n: int = 4) -> dict:
    """
    Sentence BLEU treating each function name as one token.
    Returns {'bleu1': ..., 'bleu2': ..., 'bleu3': ..., 'bleu4': ..., 'bleu': ...}
    """
    scores = {}
    log_sum = 0.0
    valid   = 0

    for n in range(1, max_n + 1):
        hyp_ng  = _ngrams(hypothesis, n)
        ref_ng  = _ngrams(reference,  n)
        clipped = sum(min(c, ref_ng.get(g, 0)) for g, c in hyp_ng.items())
        total   = max(sum(hyp_ng.values()), 1)
        prec    = clipped / total if total > 0 else 0.0
        scores[f'bleu{n}'] = round(prec, 4)
        if prec > 0:
            log_sum += math.log(prec)
            valid   += 1

    # Brevity penalty
    bp = 1.0 if len(hypothesis) >= len(reference) else \
         math.exp(1 - len(reference) / max(len(hypothesis), 1))

    geo_mean = math.exp(log_sum / max_n) if valid == max_n else 0.0
    scores['bleu'] = round(bp * geo_mean, 4)
    return scores


# ── Parse public/external function names from Solidity ───────────────────────
_FN_RE       = re.compile(
    r'\bfunction\s+(\w+)\s*\([^)]*\)[^{;]*?(public|external|private|internal)',
    re.DOTALL
)
_CONTRACT_RE = re.compile(r'\bcontract\s+(\w+)\b[^{]*\{')

def _extract_contract_body(src: str, contract_name: str) -> str:
    """
    Return just the body of `contract <contract_name> { ... }`.
    Falls back to full source if the contract is not found.
    """
    for m in _CONTRACT_RE.finditer(src):
        if m.group(1) == contract_name:
            start = m.end()          # position after the opening {
            depth = 1
            i     = start
            while i < len(src) and depth > 0:
                if src[i] == '{':  depth += 1
                elif src[i] == '}': depth -= 1
                i += 1
            return src[start:i-1]   # body without outer braces
    return src                       # fallback: full file

def parse_sol_functions(sol_path: str, contract_name: str = '') -> set:
    """Extract public/external function names from the main contract only."""
    try:
        with open(sol_path) as f:
            src = f.read()
        body = _extract_contract_body(src, contract_name) if contract_name else src
        return {m.group(1) for m in _FN_RE.finditer(body)
                if m.group(2) in ('public', 'external')}
    except Exception:
        return set()


# ── Process one contract (also works as subprocess worker target) ─────────────
def process(args_or_hex, sol_path=None, version=None, contract_name=None) -> dict:
    # Support both process(tuple) and process(hex, sol, ver, name) call styles
    if isinstance(args_or_hex, tuple):
        hex_path, sol_path, version, contract_name = args_or_hex
    else:
        hex_path = args_or_hex
    _setup_path()  # ensure imports work in subprocess workers
    record = {
        'version': version, 'contract_name': contract_name,
        'status': 'ok',
        # ALL extracted functions (named + hex IDs, excl. fallback/dispatcher)
        'cfg_all': [],
        # Named only (resolved via known_hashes)
        'cfg_named': [],
        # Public/external functions declared in Solidity (full file)
        'sol_functions': [],
        # Name-level comparison (named CFG vs sol)
        'name_matched': [], 'name_cfg_only': [], 'name_sol_only': [],
        # Count-level metrics (all CFG functions vs sol public count)
        'count_recall': 0.0,   # how many sol functions did CFG recover (by count)
        # Name-level metrics
        'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
        'jaccard': 0.0, 'exact_match': 0.0,
        'bleu1': 0.0, 'bleu2': 0.0, 'bleu3': 0.0, 'bleu4': 0.0, 'bleu': 0.0,
        'error': None,
    }

    # Parse sol — use full file (inherited contracts included, since deployed
    # bytecode includes all inherited public functions)
    sol_fns = parse_sol_functions(sol_path, '')
    record['sol_functions'] = sorted(sol_fns)

    try:
        with open(hex_path) as f:
            bytecode = f.read().strip()
        if not bytecode:
            record['status'] = 'empty'; return record

        from bytecode_to_cfg import extract_function_cfgs
        cfgs = extract_function_cfgs(bytecode)

        # ALL user functions (named + hex IDs), excluding fallback/dispatcher
        all_fns   = [n for n in cfgs if n not in ('_fallback', '_dispatcher')]
        named_fns = {n for n in all_fns
                     if not n.startswith('0x') and not n.startswith('func_')}

        record['cfg_all']   = sorted(all_fns)
        record['cfg_named'] = sorted(named_fns)
    except Exception as e:
        record['status'] = 'error'; record['error'] = str(e)[:200]; return record

    if not all_fns and not sol_fns:
        record['status'] = 'both_empty'; return record

    # ── Count recall: how many sol functions did we recover (any form)? ───────
    # Compare counts — each extracted function (even hex ID) counts as a recovery
    n_sol = len(sol_fns)
    n_cfg = len(all_fns)
    count_recall = min(n_cfg, n_sol) / n_sol if n_sol else 0.0
    record['count_recall'] = round(count_recall, 4)

    # ── Name-level metrics (named CFG only vs sol) ────────────────────────────
    matched  = named_fns & sol_fns
    cfg_only = named_fns - sol_fns
    sol_only = sol_fns  - named_fns
    record.update({'name_matched':   sorted(matched),
                   'name_cfg_only':  sorted(cfg_only),
                   'name_sol_only':  sorted(sol_only)})

    precision = len(matched) / len(named_fns) if named_fns else 0.0
    recall    = len(matched) / len(sol_fns)   if sol_fns   else 0.0
    f1        = (2*precision*recall/(precision+recall)
                 if precision+recall > 0 else 0.0)
    union     = named_fns | sol_fns
    jaccard   = len(matched) / len(union) if union else 0.0
    exact     = 1.0 if named_fns == sol_fns else 0.0
    bleu      = bleu_score(sorted(named_fns), sorted(sol_fns))

    record.update({
        'precision': round(precision, 4), 'recall': round(recall, 4),
        'f1': round(f1, 4), 'jaccard': round(jaccard, 4), 'exact_match': exact,
        **{k: round(v, 4) for k, v in bleu.items()}
    })
    return record


# ── Main ──────────────────────────────────────────────────────────────────────
def _collect_pairs(version_filter=None, limit=None):
    pairs = []
    for version in sorted(os.listdir(BYTECODE_DIR)):
        if version_filter and version != version_filter:
            continue
        hex_dir = os.path.join(BYTECODE_DIR, version)
        sol_dir = os.path.join(SOLIDITY_DIR, version)
        if not os.path.isdir(hex_dir) or not os.path.isdir(sol_dir):
            continue
        for fname in sorted(os.listdir(hex_dir)):
            if not fname.endswith('.hex'):
                continue
            base     = fname[:-4]
            sol_file = os.path.join(sol_dir, base + '.sol.cleaned')
            if not os.path.isfile(sol_file):
                continue
            parts = base.split('_', 1)
            pairs.append((os.path.join(hex_dir, fname), sol_file,
                          version, parts[1] if len(parts) > 1 else base))
    if limit:
        pairs = pairs[:limit]
    return pairs


def _load_done_jsonl():
    """Return set of (hex_path) already in JSONL output (for resume)."""
    done = set()
    if not os.path.exists(JSONL_OUT):
        return done
    with open(JSONL_OUT) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if 'hex_path' in rec:
                    done.add(rec['hex_path'])
            except Exception:
                pass
    return done


def run(version_filter=None, limit=None, show_misses=False, workers=1):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_pairs = _collect_pairs(version_filter, limit)
    done      = _load_done_jsonl()
    pairs     = [(hx, sl, ver, nm) for hx, sl, ver, nm in all_pairs
                 if hx not in done]

    n_skip = len(all_pairs) - len(pairs)
    print(f'Paired contracts : {len(all_pairs)}')
    print(f'Already done     : {n_skip}  (resuming from JSONL)')
    print(f'To process       : {len(pairs)}')
    print(f'Workers          : {workers}\n')

    METRICS = ['precision','recall','f1','jaccard','exact_match',
               'bleu1','bleu2','bleu3','bleu4','bleu','count_recall']

    sums   = defaultdict(float)
    by_ver = defaultdict(lambda: defaultdict(float))
    n_ok = n_err = n_empty = 0
    n_done = 0
    t0 = time.time()

    with open(JSONL_OUT, 'a') as out_f:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(process, job): job for job in pairs}
                for fut in as_completed(futures):
                    n_done += 1
                    try:
                        rec = fut.result(timeout=60)
                    except Exception as e:
                        job = futures[fut]
                        rec = {'version': job[2], 'contract_name': job[3],
                               'hex_path': job[0], 'status': 'error',
                               'error': str(e)[:200],
                               **{m: 0.0 for m in METRICS}}
                    rec['hex_path'] = futures[fut][0]

                    # tally
                    s = rec.get('status', 'error')
                    if s == 'ok':
                        n_ok += 1
                        ver = rec.get('version', '')
                        for m in METRICS:
                            sums[m]        += rec.get(m, 0)
                            by_ver[ver][m] += rec.get(m, 0)
                        by_ver[ver]['n'] += 1
                    elif s == 'error':
                        n_err += 1
                    else:
                        n_empty += 1

                    out_f.write(json.dumps(rec) + '\n')
                    if n_done % 500 == 0 or n_done == len(pairs):
                        p  = sums['precision'] / n_ok if n_ok else 0
                        r  = sums['recall']    / n_ok if n_ok else 0
                        f  = sums['f1']        / n_ok if n_ok else 0
                        b  = sums['bleu']      / n_ok if n_ok else 0
                        em = sums['exact_match']/ n_ok if n_ok else 0
                        rate = n_done / (time.time() - t0)
                        eta  = (len(pairs) - n_done) / rate if rate > 0 else 0
                        print(f'  [{n_done:>6}/{len(pairs)}]  '
                              f'P={p:.3f}  R={r:.3f}  F1={f:.3f}  '
                              f'BLEU={b:.3f}  ExactMatch={em:.1%}  '
                              f'ok={n_ok} err={n_err}  {rate:.0f}/s  ETA {eta/60:.1f}min')
                        out_f.flush()
        else:
            for i, job in enumerate(pairs):
                rec = process(job)
                rec['hex_path'] = job[0]
                n_done += 1

                s = rec.get('status', 'error')
                ver = rec.get('version', '')
                if s == 'ok':
                    n_ok += 1
                    for m in METRICS:
                        sums[m]        += rec.get(m, 0)
                        by_ver[ver][m] += rec.get(m, 0)
                    by_ver[ver]['n'] += 1
                    if show_misses and rec.get('name_sol_only'):
                        print(f"  [{rec['contract_name']} v{ver}]  missed: {rec['name_sol_only'][:5]}")
                elif s == 'error':
                    n_err += 1
                else:
                    n_empty += 1

                out_f.write(json.dumps(rec) + '\n')
                if (i+1) % 500 == 0 or (i+1) == len(pairs):
                    p  = sums['precision'] / n_ok if n_ok else 0
                    r  = sums['recall']    / n_ok if n_ok else 0
                    f  = sums['f1']        / n_ok if n_ok else 0
                    b  = sums['bleu']      / n_ok if n_ok else 0
                    em = sums['exact_match']/ n_ok if n_ok else 0
                    print(f'  [{i+1:>6}/{len(pairs)}]  '
                          f'P={p:.3f}  R={r:.3f}  F1={f:.3f}  '
                          f'BLEU={b:.3f}  ExactMatch={em:.1%}  '
                          f'ok={n_ok} err={n_err}')
                    out_f.flush()

    elapsed = time.time() - t0
    total_processed = n_ok + n_err + n_empty
    print(f'\nDone in {elapsed/60:.1f} min  ({total_processed} contracts processed)\n')

    # ── Recompute totals including previously-done JSONL records ─────────────
    # (If we resumed, sums only cover this run's batch. Recompute from full JSONL.)
    if n_skip > 0:
        sums   = defaultdict(float)
        by_ver = defaultdict(lambda: defaultdict(float))
        n_ok = n_err = n_empty = 0
        with open(JSONL_OUT) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                s = r.get('status', 'error')
                ver = r.get('version', '')
                if s == 'ok':
                    n_ok += 1
                    for m in METRICS:
                        sums[m]        += r.get(m, 0)
                        by_ver[ver][m] += r.get(m, 0)
                    by_ver[ver]['n'] += 1
                elif s == 'error':
                    n_err += 1
                else:
                    n_empty += 1

    # ── Averages ─────────────────────────────────────────────────────────────
    avgs = {m: round(sums[m]/n_ok, 4) if n_ok else 0.0 for m in METRICS}

    ver_summary = {}
    for ver, d in by_ver.items():
        n = d['n']
        ver_summary[ver] = {m: round(d[m]/n, 4) for m in METRICS}
        ver_summary[ver]['n_contracts'] = int(n)

    summary = {
        'total_pairs': len(pairs),
        'ok': n_ok, 'errors': n_err, 'empty_or_nofunc': n_empty,
        **{f'avg_{m}': avgs[m] for m in METRICS},
        'by_version': dict(sorted(ver_summary.items())),
    }

    with open(SUMMARY_JSON, 'w') as f: json.dump(summary, f, indent=2)

    print(f"""
╔══════════════════════════════════════════════════════╗
║         CFG Name Validation — Final Scores          ║
╠══════════════════════════════════════════════════════╣
  Contracts compared  : {n_ok} / {len(pairs)}
  Errors              : {n_err}
  Empty / no-func     : {n_empty}
╠══════════════════════════════════════════════════════╣
  Count Recall        : {avgs['count_recall']:.4f}   (fraction of sol functions recovered by count)
╠══════════════════════════════════════════════════════╣
  Precision           : {avgs['precision']:.4f}   (named CFG functions that exist in Solidity)
  Recall              : {avgs['recall']:.4f}   (public Solidity functions found by name)
  F1 Score            : {avgs['f1']:.4f}
  Jaccard Similarity  : {avgs['jaccard']:.4f}   (set overlap: matched / union)
  Exact Match         : {avgs['exact_match']:.2%}   (contracts where CFG names == Solidity names)
╠══════════════════════════════════════════════════════╣
  BLEU-1              : {avgs['bleu1']:.4f}
  BLEU-2              : {avgs['bleu2']:.4f}
  BLEU-3              : {avgs['bleu3']:.4f}
  BLEU-4              : {avgs['bleu4']:.4f}
  BLEU (combined)     : {avgs['bleu']:.4f}
╠══════════════════════════════════════════════════════╣
  Per-version summary :
""")
    for ver, d in sorted(ver_summary.items()):
        print(f"    {ver:>8}  P={d['precision']:.3f}  R={d['recall']:.3f}  "
              f"F1={d['f1']:.3f}  BLEU={d['bleu']:.3f}  "
              f"ExactMatch={d['exact_match']:.1%}  n={d['n_contracts']}")
    print(f"""
  Saved: {JSONL_OUT}
         {SUMMARY_JSON}
╚══════════════════════════════════════════════════════╝""")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Validate CFG function names against Solidity source')
    ap.add_argument('--version',    default=None,   help='Test one pragma version (e.g. 0.8.4)')
    ap.add_argument('--limit',      type=int, default=None, help='Cap total contracts (quick test)')
    ap.add_argument('--workers',    type=int, default=1,    help='Parallel workers (default 1)')
    ap.add_argument('--show-misses',action='store_true',    help='Print missed sol functions')
    args = ap.parse_args()
    run(version_filter=args.version, limit=args.limit,
        show_misses=args.show_misses, workers=args.workers)
