"""
Bytecode → per-function CFG strings in the EXACT format the Nova-Solidity
model was trained on (evm_cfg_builder output, operands stripped):

    Function owner()
        Attributes:
            -payable
            -view

        Basic Blocks:
        - @0x3b-0x42
            Instructions:
            - JUMPDEST
            - PUSH2
            - JUMP
            Incoming basic_block:
            - <cfg BasicBlock@0x1a-0x2a>
            Outgoing basic_block:
            - <cfg BasicBlock@0x75-0x9d>

Usage:
    from bytecode_to_cfg import extract_function_cfgs
    cfgs = extract_function_cfgs("6080604052...")
    for name, cfg_text in cfgs.items():
        print(name, "->", cfg_text[:80])

Install:
    pip install evm_cfg_builder
"""

from __future__ import annotations
import os
import sys
from typing import Optional

# ── Use the locally patched evm_cfg_builder clone ────────────────────────────
# Works on both Mac (/Users/eyerusalemhawoltu/Desktop/Decompliler/) and
# HPC (/scratch/eh3115/Decompliler/Decompliler/) because the relative path
# from nova/ up one level is always the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ECB_DIR lets a baseline run point at a *different* evm_cfg_builder checkout
# (e.g. a pristine clone) without disturbing the production package.
_LOCAL_ECB    = os.environ.get('ECB_DIR', os.path.join(_PROJECT_ROOT, 'evm_cfg_builder'))
if os.path.isdir(_LOCAL_ECB) and _LOCAL_ECB not in sys.path:
    sys.path.insert(0, _LOCAL_ECB)   # shadows the pip-installed version
# ─────────────────────────────────────────────────────────────────────────────

# Functions to skip (dispatcher, internal glue)
_SKIP_NAMES = {'_dispatcher', '_fallback'}


def _fmt_block_ref(bb) -> str:
    """<cfg BasicBlock@0xSTART-0xEND>"""
    return f"<cfg BasicBlock@{hex(bb.start.pc)}-{hex(bb.end.pc)}>"


def _jump_target(bb, all_blocks: dict):
    """
    For a block ending in JUMP or JUMPI, find the destination PC by scanning
    backwards for the last PUSH instruction (static jump target).
    Returns the target BasicBlock or None.
    """
    for instr in reversed(bb.instructions):
        if instr.name.startswith('PUSH') and instr.operand is not None:
            target_pc = instr.operand
            return all_blocks.get(target_pc)
    return None


def _reachable_blocks(entry_block, all_blocks: dict) -> list:
    """
    BFS from entry_block.
    Uses recorded edges first; falls back to manually resolving JUMP/JUMPI
    targets from instruction operands when edges are missing (happens with
    Solidity 0.8.x after the evm_cfg_builder crash patch).
    """
    visited = {}   # pc → block
    queue   = [entry_block]
    while queue:
        bb = queue.pop(0)
        pc = bb.start.pc
        if pc in visited:
            continue
        visited[pc] = bb

        successors = list(bb.all_outgoing_basic_blocks)

        # If no edges recorded, resolve them manually
        if not successors:
            end_op = bb.end.name
            if end_op in ('JUMP', 'JUMPI'):
                tgt = _jump_target(bb, all_blocks)
                if tgt:
                    successors.append(tgt)
            if end_op == 'JUMPI':
                # false branch = block starting at the very next PC
                next_pc = bb.end.pc + 1
                if next_pc in all_blocks:
                    successors.append(all_blocks[next_pc])

        for ob in successors:
            if ob.start.pc not in visited:
                queue.append(ob)

    return sorted(visited.values(), key=lambda b: b.start.pc)


def _format_function(func, cfg_obj=None) -> str:
    """
    Produce the training-format CFG string for one Function object.
    Operands are intentionally omitted to match the training distribution.
    """
    lines = [f"Function {func.name}"]
    lines.append("\tAttributes:")
    for attr in sorted(func.attributes):
        lines.append(f"\t\t-{attr}")
    lines.append("")                       # blank line between attrs and blocks
    lines.append("\tBasic Blocks:")

    # func.basic_blocks is now populated by enhance_cfgs_with_bfs().
    # Fall back to our own BFS only if the library somehow returned nothing.
    blocks = sorted(func.basic_blocks, key=lambda b: b.start.pc)
    if not blocks and cfg_obj:
        entry_pc = func.start_addr
        if entry_pc in cfg_obj._basic_blocks:
            blocks = _reachable_blocks(cfg_obj._basic_blocks[entry_pc], cfg_obj._basic_blocks)

    for bb in blocks:
        lines.append(f"\t- @{hex(bb.start.pc)}-{hex(bb.end.pc)}")
        lines.append("\t\tInstructions:")
        for instr in bb.instructions:
            lines.append(f"\t\t- {instr.name}")   # ← no operand value

        lines.append("\t\tIncoming basic_block:")
        for ib in bb.all_incoming_basic_blocks:
            lines.append(f"\t\t- {_fmt_block_ref(ib)}")

        lines.append("\t\tOutgoing basic_block:")
        out_bbs = list(bb.all_outgoing_basic_blocks)
        if not out_bbs and cfg_obj:
            # Manually resolve when evm_cfg_builder didn't record the edge
            all_blocks = cfg_obj._basic_blocks
            end_op = bb.end.name
            if end_op in ('JUMP', 'JUMPI'):
                tgt = _jump_target(bb, all_blocks)
                if tgt:
                    out_bbs.append(tgt)
            if end_op == 'JUMPI':
                next_pc = bb.end.pc + 1
                if next_pc in all_blocks:
                    out_bbs.append(all_blocks[next_pc])
        for ob in out_bbs:
            lines.append(f"\t\t- {_fmt_block_ref(ob)}")

    return "\n".join(lines)


def extract_function_cfgs(
    bytecode_hex: str,
    skip_dispatcher: bool = True,
) -> dict[str, str]:
    """
    Parse runtime bytecode and return:
        { function_name_or_label : cfg_text }

    cfg_text is in the exact format Nova-Solidity was trained on.

    Args:
        bytecode_hex    : hex string with or without '0x' prefix
        skip_dispatcher : skip _dispatcher and _fallback (default True)

    Returns dict of { name → cfg_text }. Raises ImportError if
    evm_cfg_builder is not installed.
    """
    try:
        from evm_cfg_builder.cfg import CFG
    except ImportError:
        raise ImportError(
            "evm_cfg_builder not installed.\n"
            "Run:  pip install evm_cfg_builder"
        )

    bytecode_hex = bytecode_hex.strip()
    if bytecode_hex.lower().startswith("0x"):
        bytecode_hex = bytecode_hex[2:]

    # Strip any whitespace/newlines inside the hex string
    bytecode_hex = ''.join(bytecode_hex.split())

    # Validate hex
    try:
        bytes.fromhex(bytecode_hex)
    except ValueError as e:
        raise ValueError(f"Invalid hex bytecode: {e}")

    try:
        cfg = CFG(bytecode_hex, remove_metadata=True)
        # Augment every function's block list using aggressive BFS.
        # This fixes Solidity ≥ 0.8 where all function stubs jump into a
        # shared ABI decoder, leaving VSA with only 1-block entry stubs.
        # Guarded so a pristine (unpatched) evm_cfg_builder still runs.
        if hasattr(cfg, 'enhance_cfgs_with_bfs'):
            cfg.enhance_cfgs_with_bfs()
    except Exception as e:
        raise ValueError(f"evm_cfg_builder failed to parse bytecode: {e}")

    results: dict[str, str] = {}
    for func in cfg.functions:
        name = func.name or f"func_{hex(func.start_addr)}"
        # Only skip the dispatcher (routing logic), keep fallback/receive
        if skip_dispatcher and name == '_dispatcher':
            continue
        if not func.basic_blocks:
            continue

        cfg_text = _format_function(func, cfg_obj=cfg)

        # Deduplicate names (shouldn't happen but just in case)
        key = name
        if key in results:
            key = f"{name}_{hex(func.start_addr)}"
        results[key] = cfg_text

    return results


def get_cfg_for_function(
    bytecode_hex: str,
    func_name: Optional[str] = None,
) -> str:
    """
    Convenience wrapper — returns the CFG text for one function.

    If func_name is None, returns the first non-dispatcher function.
    Raises ValueError if not found.
    """
    cfgs = extract_function_cfgs(bytecode_hex)
    if not cfgs:
        raise ValueError("No functions found in bytecode.")

    if func_name is None:
        return next(iter(cfgs.values()))

    if func_name in cfgs:
        return cfgs[func_name]

    # Partial match
    matches = [k for k in cfgs if func_name in k]
    if len(matches) == 1:
        return cfgs[matches[0]]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous name '{func_name}'. Matches: {matches}")

    raise ValueError(
        f"Function '{func_name}' not found.\nAvailable: {list(cfgs.keys())}"
    )


# ── Quick CLI test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json, sys

    ap = argparse.ArgumentParser(description="EVM bytecode → CFG extractor")
    ap.add_argument("bytecode", nargs="?", help="Hex bytecode string")
    ap.add_argument("--file",  help="Read bytecode from file")
    ap.add_argument("--func",  help="Show only this function")
    ap.add_argument("--list",  action="store_true", help="List function names only")
    ap.add_argument("--json",  action="store_true", help="Output as JSON")
    args = ap.parse_args()

    if args.file:
        with open(args.file) as f:
            raw = f.read().strip()
    elif args.bytecode:
        raw = args.bytecode
    else:
        ap.error("Provide bytecode as argument or --file <path>")

    cfgs = extract_function_cfgs(raw)

    if args.list:
        for n in cfgs: print(n)
        sys.exit(0)

    if args.json:
        print(json.dumps(cfgs, indent=2))
        sys.exit(0)

    targets = {args.func: cfgs[args.func]} if args.func and args.func in cfgs else cfgs
    for name, text in targets.items():
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        print(text)
