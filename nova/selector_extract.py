"""
selector_extract.py — the lightest front-end.

Recovers PUBLIC function names by scanning the dispatcher for 4-byte selectors
(no CFG construction, no data-flow). For each recovered selector it also grabs a
raw linear opcode slice of the function body (entry -> first terminator), so the
output can double as a minimal model input.

Returned dict matches bytecode_to_cfg.extract_function_cfgs:
    { function_name_or_hexselector : opcode_slice_text }

Resolved names are bare (e.g. 'transfer'); unresolved selectors appear as the
hex string (e.g. '0x12345678') so the evaluator counts them as un-named, exactly
like the CFG extractor's convention.
"""
from __future__ import annotations
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Use the same evm_cfg_builder checkout (for pyevmasm + the PUSH0 patch + table).
_ECB = os.environ.get('ECB_DIR', os.path.join(_ROOT, 'evm_cfg_builder'))
if os.path.isdir(_ECB) and _ECB not in sys.path:
    sys.path.insert(0, _ECB)

_BLOCK_END = {'STOP', 'RETURN', 'REVERT', 'INVALID', 'SELFDESTRUCT', 'SUICIDE', 'JUMP'}


def _patch_push0() -> None:
    """Register PUSH0 (0x5f) so Shanghai bytecode disassembles correctly."""
    try:
        from pyevmasm.evmasm import instruction_tables
        entry = ("PUSH0", 0, 0, 1, 2, "Place value 0 on stack.")
        for tbl in instruction_tables.values():
            if 0x5F not in tbl._instruction_list:
                tbl._instruction_list[0x5F] = entry
                setattr(tbl, "_InstructionTable__name_to_opcode", None)
    except Exception:
        pass


_KNOWN_HASHES: dict | None = None


def _load_known_hashes() -> dict:
    """Load the same merged selector->bare-name table the CFG path uses."""
    global _KNOWN_HASHES
    if _KNOWN_HASHES is not None:
        return _KNOWN_HASHES
    path = os.path.join(_ECB, 'evm_cfg_builder', 'known_hashes', 'known_hashes.json')
    try:
        with open(path) as f:
            raw = json.load(f)
        _KNOWN_HASHES = {int(k): v for k, v in raw.items()}
    except Exception:
        _KNOWN_HASHES = {}
    return _KNOWN_HASHES


def extract_functions_by_selector(bytecode_hex: str,
                                  skip_dispatcher: bool = True) -> dict[str, str]:
    _patch_push0()
    from pyevmasm import disassemble_all

    bc = bytecode_hex.strip()
    if bc.lower().startswith('0x'):
        bc = bc[2:]
    bc = ''.join(bc.split())
    try:
        raw = bytes.fromhex(bc)
    except ValueError as e:
        raise ValueError(f'invalid hex: {e}')

    instrs = list(disassemble_all(raw))
    if not instrs:
        return {}

    # pc -> instruction index, for slicing the body later
    pc_to_idx = {ins.pc: i for i, ins in enumerate(instrs)}
    kh = _load_known_hashes()

    # ── Dispatcher scan ───────────────────────────────────────────────────────
    # A public function compare looks like:  PUSH4 <selector> ... JUMPI
    # We record any PUSH4 whose value sits a few instructions before a JUMPI.
    found: dict[int, int | None] = {}   # selector -> dest_pc (or None)
    n = len(instrs)
    for i, ins in enumerate(instrs):
        if ins.name == 'PUSH4' and ins.operand is not None:
            window = instrs[i + 1:i + 7]
            if not any(w.name == 'JUMPI' for w in window):
                continue
            sel = ins.operand
            # dest = the PUSH operand that immediately precedes the JUMPI
            dest = None
            for j in range(i + 1, min(i + 7, n)):
                if (instrs[j].name.startswith('PUSH') and instrs[j].operand is not None
                        and j + 1 < n and instrs[j + 1].name == 'JUMPI'):
                    dest = instrs[j].operand
                    break
            # keep first dest seen for a selector
            found.setdefault(sel, dest)

    # ── Build result dict ───────────────────────────────────────────────────────
    results: dict[str, str] = {}
    for sel, dest in found.items():
        name = kh.get(sel, hex(sel))
        body = _slice_body(instrs, pc_to_idx, dest) if dest is not None else ''
        key = name
        if key in results:                      # rare selector/name clash
            key = f'{name}_{hex(sel)}'
        results[key] = _format_slice(name, body)
    return results


def _slice_body(instrs, pc_to_idx, dest_pc, max_instr=400):
    """Linear opcode slice from dest_pc to the first block terminator."""
    start = pc_to_idx.get(dest_pc)
    if start is None:
        return []
    out = []
    for k in range(start, min(start + max_instr, len(instrs))):
        nm = instrs[k].name
        out.append(nm)
        if nm in _BLOCK_END:
            break
    return out


def _format_slice(name, ops):
    lines = [f'Function {name}', '\tInstructions:']
    lines += [f'\t\t- {op}' for op in ops]
    return '\n'.join(lines)


if __name__ == '__main__':
    import sys as _s
    bc = open(_s.argv[1]).read() if len(_s.argv) > 1 else _s.stdin.read()
    for k, v in extract_functions_by_selector(bc).items():
        print(k, '->', v[:60].replace('\n', ' '))
