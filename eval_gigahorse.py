"""
eval_gigahorse.py — name-recovery evaluation for the Gigahorse front-end.

Assumes `gigahorse.py` has already been run over a directory of .hex files,
producing per-contract output under  <GH_OUT>/<stem>/out/HighLevelFunctionName.csv.

For each contract it:
  - reads the recovered PUBLIC function names from HighLevelFunctionName.csv
    (strips typed signature -> bare name; drops unresolved / fallback / selector)
  - parses the deployed public/external functions from the Solidity source
    (same interface-excluding parser as eval_cfg_csv.py)
  - computes precision / recall / F1 / BLEU / exact-match
and writes per-version + summary CSVs, mirroring eval_cfg_csv.py so the numbers
are directly comparable to the CFG front-end.
"""
import argparse
import csv
import os
import re
import sys

# reuse the exact Solidity parser + metrics from the CFG evaluator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_cfg_csv import (parse_sol_functions, bleu_score, classify_bytecode,
                          METRICS, BYTECODE_DIR, SOLIDITY_DIR)

GH_OUT = os.environ.get('GH_OUT', '')   # dir with <stem>/out/ subdirs
RESULTS_DIR = os.environ.get('CFG_EVAL_OUT',
                             os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'results', 'cfg_eval_gigahorse'))

_SKIP = {'__function_selector__', 'fallback', 'receive', ''}


def gigahorse_named(stem: str) -> set:
    """Recovered public function names for one contract, or set() if no output."""
    path = os.path.join(GH_OUT, stem, 'out', 'HighLevelFunctionName.csv')
    if not os.path.isfile(path):
        return None                      # gigahorse produced nothing (fail/timeout)
    names = set()
    with open(path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            block, sig = parts[0], parts[1]
            if sig == block:             # unresolved internal function (name == block id)
                continue
            bare = sig.split('(', 1)[0].strip()
            if bare in _SKIP or bare.startswith('0x'):
                continue
            if re.fullmatch(r'[A-Za-z_$][A-Za-z0-9_$]*', bare):
                names.add(bare)
    return names


def main(sample_per_version, version_filter):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    from collections import defaultdict
    ver_sums = defaultdict(lambda: defaultdict(float))
    ver_cnt = defaultdict(int)
    rows_by_ver = defaultdict(list)
    n_total = n_noout = 0

    for version in sorted(os.listdir(BYTECODE_DIR)):
        if version_filter and version != version_filter:
            continue
        hexd = os.path.join(BYTECODE_DIR, version)
        sold = os.path.join(SOLIDITY_DIR, version)
        if not (os.path.isdir(hexd) and os.path.isdir(sold)):
            continue
        cnt = 0
        for fn in sorted(os.listdir(hexd)):
            if not fn.endswith('.hex'):
                continue
            stem = fn[:-4]
            sol_file = os.path.join(sold, stem + '.sol.cleaned')
            if not os.path.isfile(sol_file):
                continue
            cnt += 1
            if sample_per_version and cnt > sample_per_version:
                break
            n_total += 1

            bc = open(os.path.join(hexd, fn)).read()
            btype = classify_bytecode(bc)
            named = gigahorse_named(stem)
            if named is None:
                n_noout += 1
                named = set()
            sol = parse_sol_functions(sol_file)
            if not sol and not named:
                continue

            matched = named & sol
            p = len(matched) / len(named) if named else 0.0
            r = len(matched) / len(sol) if sol else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            union = named | sol
            jac = len(matched) / len(union) if union else 0.0
            exact = 1 if named == sol else 0
            cr = (min(len(named), len(sol)) / len(sol)) if sol else 0.0
            bl = bleu_score(sorted(named), sorted(sol))
            row = {'precision': p, 'recall': r, 'f1': f1, 'jaccard': jac,
                   'exact_match': exact, 'count_recall': cr, **bl}
            rows_by_ver[version].append((stem, btype, row))
            if btype == 'runtime':
                for m in METRICS:
                    ver_sums[version][m] += row.get(m, 0.0)
                ver_cnt[version] += 1

    # write summary (runtime-only, like eval_cfg_csv no_stubs)
    out = os.path.join(RESULTS_DIR, 'version_summary_no_stubs.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['version', 'n_runtime'] + METRICS)
        for v in sorted(ver_sums):
            n = ver_cnt[v]
            w.writerow([v, n] + [round(ver_sums[v][m] / n, 4) if n else 0 for m in METRICS])
    print(f'contracts: {n_total} | no gigahorse output: {n_noout}')
    print(f'wrote {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=None)
    ap.add_argument('--version', default=None)
    main(ap.parse_args().sample, ap.parse_args().version)
