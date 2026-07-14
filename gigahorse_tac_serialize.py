"""
gigahorse_tac_serialize.py — turn a Gigahorse out/ dir into per-function TAC text.

Joins the relational CSVs (InFunction, TAC_Block, TAC_Op, TAC_Def, TAC_Use,
TAC_Variable_Value, HighLevelFunctionName) into a readable three-address-code
listing per function, analogous to the CFG text representation, so it can feed
the model.

Usage: python gigahorse_tac_serialize.py <out_dir>   (prints functions)
       or import serialize_tac(out_dir) -> {func_name: tac_text}
"""
import os, sys
from collections import defaultdict

def _load_pairs(path):
    rows = []
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                rows.append(line.rstrip('\n').split('\t'))
    return rows

def _hx(x):
    try: return int(x, 16)
    except Exception: return 0

def serialize_tac(out_dir):
    blk_func = {b: fn for b, fn in _load_pairs(os.path.join(out_dir, 'InFunction.csv'))}
    func_blocks = defaultdict(list)
    for b, fn in blk_func.items():
        func_blocks[fn].append(b)

    stmt_block = {s: b for s, b in _load_pairs(os.path.join(out_dir, 'TAC_Block.csv'))}
    block_stmts = defaultdict(list)
    for s, b in stmt_block.items():
        block_stmts[b].append(s)

    op = {s: o for s, o in _load_pairs(os.path.join(out_dir, 'TAC_Op.csv'))}
    defv = {}
    for r in _load_pairs(os.path.join(out_dir, 'TAC_Def.csv')):
        if len(r) >= 2: defv[r[0]] = r[1]
    uses = defaultdict(list)
    for r in _load_pairs(os.path.join(out_dir, 'TAC_Use.csv')):
        if len(r) >= 3: uses[r[0]].append((int(r[2]), r[1]))
    const = {v: val for v, val in _load_pairs(os.path.join(out_dir, 'TAC_Variable_Value.csv'))}
    hln = {}
    for r in _load_pairs(os.path.join(out_dir, 'HighLevelFunctionName.csv')):
        if len(r) >= 2: hln[r[0]] = r[1]

    out = {}
    for fn in sorted(func_blocks, key=_hx):
        name = hln.get(fn, fn)
        if name in ('__function_selector__',):
            continue
        # canonicalise variable names per function: v1, v2, ... (constants kept as values)
        vmap, vc = {}, [0]
        def rv(v):
            if v in const:
                return const[v]
            if v not in vmap:
                vc[0] += 1; vmap[v] = f'v{vc[0]}'
            return vmap[v]
        lines = [f"function {name}"]
        for b in sorted(func_blocks[fn], key=_hx):
            lines.append(f"  block:")
            for s in sorted(block_stmts.get(b, []), key=_hx):
                o = op.get(s, '?')
                us = [rv(v) for _, v in sorted(uses.get(s, []))]
                d = defv.get(s)
                if d and d in const:
                    lines.append(f"    {rv(d)} = {const[d]}")
                elif d:
                    lines.append(f"    {rv(d)} = {o} {' '.join(us)}".rstrip())
                else:
                    lines.append(f"    {o} {' '.join(us)}".rstrip())
        out[name] = '\n'.join(lines)
    return out


if __name__ == '__main__':
    funcs = serialize_tac(sys.argv[1])
    print(f"{len(funcs)} functions\n")
    for name, tac in list(funcs.items())[:3]:
        print('=' * 60)
        print(tac[:1200])
        print()
