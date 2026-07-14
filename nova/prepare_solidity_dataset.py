"""
Prepare the Solidity CFG dataset for Nova-style fine-tuning.

Key design decisions vs the previous version:
  - ALL pragma versions pooled into a single dataset (no version-stratified splits).
    This prevents overfitting caused by the same function appearing in both train and
    test just because it was compiled with a slightly different pragma.
  - Exact-duplicate removal: pairs with the same (function_name, solidity_definition)
    are deduplicated before splitting.
  - Group-aware split: all samples that share the same function_name are assigned to
    ONE partition (train, valid, or test).  This ensures the model never sees
    transfer(address,uint256) at train time and is then tested on a pragma-variant of
    the exact same function.
  - Split ratio: 60 / 20 / 20  (train / valid / test)

Run from the nova/ directory:
    python prepare_solidity_dataset.py

Output:
    ../data-tokenized/nova-solidity-bcr/train   (60 %)
    ../data-tokenized/nova-solidity-bcr/valid   (20 %)
    ../data-tokenized/nova-solidity-bcr/test    (20 %)
    ../data-tokenized/test_set.json             (test set for inference / eval)
"""

import hashlib
import json
import os
import random
from collections import defaultdict
from datasets import Dataset, DatasetDict

DATA_DIR   = '../Cleaned_Aligned_JSON'
OUTPUT_DIR = '../data-tokenized/nova-solidity-bcr'
TEST_JSON  = '../data-tokenized/test_set.json'

MAX_CHAR_LEN = 8192
TRAIN_RATIO  = 0.60
VALID_RATIO  = 0.20
TEST_RATIO   = 0.20
RANDOM_SEED  = 42

# ── EVM opcode set ────────────────────────────────────────────────────────────
EVM_OPCODES = {
    'STOP','ADD','MUL','SUB','DIV','SDIV','MOD','SMOD','ADDMOD','MULMOD','EXP',
    'SIGNEXTEND','LT','GT','SLT','SGT','EQ','ISZERO','AND','OR','XOR','NOT',
    'BYTE','SHL','SHR','SAR','SHA3','KECCAK256','ADDRESS','BALANCE','ORIGIN',
    'CALLER','CALLVALUE','CALLDATALOAD','CALLDATASIZE','CALLDATACOPY','CODESIZE',
    'CODECOPY','GASPRICE','EXTCODESIZE','EXTCODECOPY','RETURNDATASIZE',
    'RETURNDATACOPY','EXTCODEHASH','BLOCKHASH','COINBASE','TIMESTAMP','NUMBER',
    'DIFFICULTY','GASLIMIT','CHAINID','SELFBALANCE','BASEFEE','POP','MLOAD',
    'MSTORE','MSTORE8','SLOAD','SSTORE','JUMP','JUMPI','PC','MSIZE','GAS',
    'JUMPDEST',
}
EVM_OPCODES |= {f'PUSH{i}'  for i in range(33)}
EVM_OPCODES |= {f'DUP{i}'   for i in range(1, 17)}
EVM_OPCODES |= {f'SWAP{i}'  for i in range(1, 17)}
EVM_OPCODES |= {f'LOG{i}'   for i in range(5)}
EVM_OPCODES |= {'CREATE','CALL','CALLCODE','RETURN','DELEGATECALL','CREATE2',
                'STATICCALL','REVERT','INVALID','SELFDESTRUCT'}


def normalize_cfg(cfg_text: str) -> str:
    """Add <label-N> markers to instruction lines (training format)."""
    lines     = cfg_text.split('\n')
    result    = []
    label_idx = 1
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- ') and label_idx <= 256:
            parts = stripped[2:].split()
            if parts and parts[0].upper() in EVM_OPCODES:
                result.append(line + f' <label-{label_idx}>')
                label_idx += 1
                continue
        result.append(line)
    return '\n'.join(result)


def _dedup_key(cfg: str, solidity: str) -> str:
    """Stable hash for (cfg, solidity) PAIR deduplication.

    Keyed on CONTENT — not the per-sample suffixed func_name — so genuinely
    identical (cfg, solidity) pairs collapse to one. (The old key used
    func_name, which carries a unique '_vN___i' suffix, so it never actually
    deduplicated and let identical functions leak across the split.)
    """
    raw = ' '.join(cfg.split()) + '\x00' + ' '.join(solidity.split())
    return hashlib.md5(raw.encode()).hexdigest()


def _body_key(solidity: str) -> str:
    """Split-group key = the normalized solidity body.

    All samples that share a body (across pragmas / contracts) are assigned to
    the SAME partition, guaranteeing no function body appears in both train and
    test — the fix for the train/test memorization leakage.
    """
    return hashlib.md5(' '.join(solidity.split()).encode()).hexdigest()


def load_all_data():
    """
    Load every combined_*.json file and deduplicate to UNIQUE FUNCTIONS.

    A function is identified by its normalized solidity body (``_body_key``),
    which is pragma-invariant — so the same function appearing under several
    pragma versions / contracts collapses to a SINGLE representative. We keep
    the first occurrence (richest non-empty CFG wins on ties).

    Returns a list of (version, func_name, cfg, solidity) tuples, one per
    unique function.
    """
    best: dict[str, tuple[str, str, str, str]] = {}   # body_key -> sample
    total_raw = 0
    total_dup = 0

    for fname in sorted(os.listdir(DATA_DIR)):
        if not (fname.startswith('combined_') and fname.endswith('.json')):
            continue
        if fname.startswith('combined_.'):        # .DS_Store artefact
            continue
        version = fname.replace('combined_', '').replace('.json', '')
        fpath   = os.path.join(DATA_DIR, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception as e:
            print(f'  Skip {fname}: {e}')
            continue
        if not isinstance(data, dict) or not data:
            continue

        file_raw = file_dup = 0
        for func_name, entry in data.items():
            if not isinstance(entry, dict):
                continue
            cfg = entry.get('cfg_representation', '').strip()
            sol = entry.get('solidity_definition', '').strip()
            if not cfg or not sol:
                continue
            file_raw += 1
            bk = _body_key(sol)                      # dedup ignores pragma
            if bk in best:
                file_dup += 1
                # keep the variant with the richer CFG (more information)
                if len(cfg) > len(best[bk][2]):
                    best[bk] = (version, func_name, cfg, sol)
                continue
            best[bk] = (version, func_name, cfg, sol)

        total_raw += file_raw
        total_dup += file_dup
        print(f'  {version:>8}: {file_raw} raw, {file_dup} duplicate-body samples removed')

    raw_samples = list(best.values())
    print(f'\nTotal raw: {total_raw} | Cross-pragma duplicate bodies removed: '
          f'{total_dup} | Unique functions: {len(raw_samples)}')
    return raw_samples


def main():
    print(f'Loading data from {DATA_DIR} ...')
    raw_samples = load_all_data()

    # ── Build one record per UNIQUE function, apply length filter ─────────────
    records: list[dict] = []
    skipped_len = 0

    for version, func_name, cfg, sol in raw_samples:
        cfg_norm      = normalize_cfg(cfg)
        prompt_before = f'# This is the EVM CFG for a Solidity {version} function:\n'
        prompt_after  = '\nWhat is the Solidity source code?\n'
        inp           = prompt_before + cfg_norm + prompt_after

        if len(inp + sol) > MAX_CHAR_LEN:
            skipped_len += 1
            continue

        ct = ('0' * len(prompt_before)
              + '1' * len(cfg_norm)
              + '0' * len(prompt_after)
              + '0' * len(sol))

        records.append({
            'version':    version,
            'func_name':  func_name,
            'cfg':        cfg,          # raw CFG kept for test_set.json
            'input':      inp,
            'output':     sol,
            'char_types': ct,
        })

    print(f'After length filter: {len(records)} unique functions kept, '
          f'{skipped_len} skipped (too long)')

    # ── 60 / 20 / 20 split over UNIQUE functions ──────────────────────────────
    # Every function body is already unique, so a plain shuffle-and-slice gives
    # train / valid / test partitions that share NO function body → zero leakage.
    random.seed(RANDOM_SEED)
    random.shuffle(records)

    n          = len(records)
    test_end   = int(n * TEST_RATIO)
    valid_end  = test_end + int(n * VALID_RATIO)
    test_recs  = records[:test_end]
    valid_recs = records[test_end:valid_end]
    train_recs = records[valid_end:]

    COLS = ['version', 'func_name', 'input', 'output', 'char_types']
    def to_cols(recs):
        return {c: [r[c] for r in recs] for c in COLS}

    train_raw = to_cols(train_recs)
    valid_raw = to_cols(valid_recs)
    test_raw  = to_cols(test_recs)

    n_train, n_valid, n_test = len(train_recs), len(valid_recs), len(test_recs)
    print(f'\nSplit summary (unique functions, leakage-free):')
    print(f'  Train : {n_train:>6}  ({n_train/n*100:.1f}%)')
    print(f'  Valid : {n_valid:>6}  ({n_valid/n*100:.1f}%)')
    print(f'  Test  : {n_test:>6}  ({n_test/n*100:.1f}%)')

    # ── Save HuggingFace datasets ─────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    DatasetDict({
        'train': Dataset.from_dict(train_raw),
        'valid': Dataset.from_dict(valid_raw),
        'test':  Dataset.from_dict(test_raw),
    }).save_to_disk(OUTPUT_DIR)
    print(f'\nHuggingFace datasets saved to {OUTPUT_DIR}')

    # ── Save test_set.json (unique functions, raw CFG) ────────────────────────
    os.makedirs(os.path.dirname(TEST_JSON), exist_ok=True)
    test_json = {}
    for i, rec in enumerate(test_recs):
        key = f"{rec['func_name']}_v{rec['version']}___{i}"
        test_json[key] = {
            'cfg_representation':  rec['cfg'],
            'solidity_definition': rec['output'],
            'version':             rec['version'],
        }
    with open(TEST_JSON, 'w') as f:
        json.dump(test_json, f, indent=2)
    print(f'Test JSON saved to {TEST_JSON}  ({len(test_json)} functions)')


if __name__ == '__main__':
    main()
