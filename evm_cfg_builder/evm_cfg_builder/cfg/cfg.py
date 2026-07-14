import logging
import re
from typing import Optional, Union, Tuple, List, Dict, Set

from pyevmasm import disassemble_all, Instruction


def _patch_push0() -> None:
    """Register PUSH0 (0x5f, Shanghai hardfork) in pyevmasm's tables.

    The bundled pyevmasm only knows opcodes up to the Istanbul fork, so it
    decodes 0x5f as INVALID.  Solidity >= 0.8.20 emits PUSH0 by default, which
    corrupted basic-block boundaries and made the dispatcher unrecoverable
    (functions resolved to only `_fallback`).  We inject PUSH0 — a 0-operand
    instruction that pushes the constant 0 — into every fork table so the
    disassembler decodes it correctly.
    """
    try:
        from pyevmasm.evmasm import instruction_tables
    except Exception:  # pragma: no cover
        return
    entry = ("PUSH0", 0, 0, 1, 2, "Place value 0 (zero) on stack.")
    for _tbl in instruction_tables.values():
        try:
            if 0x5F not in _tbl._instruction_list:
                _tbl._instruction_list[0x5F] = entry
                # invalidate the cached name->opcode map (name-mangled attr)
                setattr(_tbl, "_InstructionTable__name_to_opcode", None)
        except Exception:  # pragma: no cover
            continue


_patch_push0()

from evm_cfg_builder.cfg.basic_block import BasicBlock
from evm_cfg_builder.cfg.function import Function

def _load_known_hashes() -> dict:
    """
    Load selector→signature mapping.
    Tries known_hashes.json first (fast, no Python compile limit).
    Falls back to the .py dict literal (older installs).
    Returns empty dict if neither is available.
    """
    import json, os
    _dir = os.path.dirname(__file__)               # cfg/
    _kh_dir = os.path.join(_dir, '..', 'known_hashes')

    # ── 1. JSON (preferred, works on Python 3.10+) ─────────────────────────
    json_path = os.path.join(_kh_dir, 'known_hashes.json')
    if os.path.isfile(json_path):
        try:
            with open(json_path) as f:
                raw = json.load(f)
            # JSON keys are strings; original dict had int keys
            return {int(k): v for k, v in raw.items()}
        except Exception:
            pass

    # ── 2. .py fallback (Python ≤ 3.9 or pre-JSON install) ─────────────────
    try:
        from evm_cfg_builder.known_hashes.known_hashes import known_hashes as _kh
        return _kh
    except (ImportError, OverflowError, SyntaxError):
        pass

    return {}

known_hashes: dict = _load_known_hashes()


logger = logging.getLogger("evm-cfg-builder")

BASIC_BLOCK_END = [
    "STOP",
    "SELFDESTRUCT",
    "RETURN",
    "REVERT",
    "INVALID",
    "SUICIDE",
    "JUMP",
    "JUMPI",
]


def convert_bytecode(bytecode: Optional[Union[str, bytes]]) -> Optional[bytes]:
    """
    Convert the bytecode to bytes
    Remove trailing \n
    Remove '0x'
    Replace library call to 'AAA.AAA'
    Args:
        bytecode (str|bytes)
    Return:
        (bytes)
    """
    if bytecode is not None:
        if isinstance(bytecode, str):
            for library_found in re.findall(r"__.{36}__", bytecode):
                logger.info("Replace library %s by %s", library_found, "A" * 40)
            bytecode = re.sub(r"__.{36}__", "A" * 40, bytecode)

            bytecode = bytecode.replace("\n", "")
            if bytecode.startswith("0x"):
                bytecode = bytes.fromhex(bytecode[2:])
            else:
                bytecode = bytes.fromhex(bytecode)
        else:
            for library_found in re.findall(b"__.{36}__", bytecode):
                logger.info("Replace library %s by %s", library_found, "A" * 40)
            bytecode = re.sub(b"__.{36}__", b"A" * 40, bytecode)
            if bytecode.startswith(b"0x"):
                bytecode = bytes.fromhex(bytecode[2:].decode().replace("\n", ""))

    return bytecode


class CFG:
    """Implements the control flow graph (CFG) of an EVM bytecode."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        bytecode: Optional[Union[str, bytes]] = None,
        remove_metadata: bool = True,
        analyze: bool = True,
        optimization_enabled: bool = True,
        compute_cfgs: bool = True,
    ) -> None:
        """Initialize an EVM CFG.

        :param bytecode: The EVM bytecode
        :type bytecode: None, str, bytes
        :param remove_metadata: Automatically remove metadata
        :type remove_metadata: bool
        :param analyze: Automatically analyze the bytecode
        :type analyze: bool
        """
        self._functions: Dict[int, Function] = {}
        # __basic_blocks is a dict that matches
        # an address to the basic block
        # The address can be the first or the last
        # instructions
        self._basic_blocks: Dict[int, BasicBlock] = {}
        self._instructions: Dict[int, Instruction] = {}

        self._optimization_enabled = optimization_enabled

        assert isinstance(bytecode, (type(None), str, bytes))

        self._bytecode = convert_bytecode(bytecode)

        if remove_metadata:
            self.remove_metadata()
        if analyze:
            self.create_functions()
            if compute_cfgs:
                self.create_cfgs()

    def __repr__(self) -> str:
        return f"<CFG: {len(self.functions)} Functions, {len(self.basic_blocks)} Basic Blocks>"

    @property
    def bytecode(self) -> Optional[bytes]:
        return self._bytecode

    @bytecode.setter
    def bytecode(self, bytecode: Optional[Union[str, bytes]]) -> None:
        assert isinstance(bytecode, (type(None), str, bytes))

        bytecode = convert_bytecode(bytecode)

        self.clear()
        self._bytecode = bytecode

    @property
    def basic_blocks(self) -> List[BasicBlock]:
        """
        Return the list of basic_block
        """
        bbs = self._basic_blocks.values()
        return list(set(bbs))

    @property
    def entry_point(self) -> BasicBlock:
        """
        Return the entry point of the cfg (the basic block at 0x0)
        """
        return self._basic_blocks[0]

    @property
    def functions(self) -> List[Function]:
        """
        Return the list of functions
        """
        return list(self._functions.values())

    @property
    def instructions(self) -> List[Instruction]:
        """
        Return the list of instructions
        """
        return list(self._instructions.values())

    def get_instruction_at(self, addr: int) -> Instruction:
        """Return the instruction at the provided address.

        :param addr: Address of instruction
        :type addr: int
        """
        return self._instructions.get(addr)

    def get_basic_block_at(self, addr: int) -> Optional[BasicBlock]:
        """Return the basic block at the provided address.

        The address is either the starting or ending instruction of the
        basic block.

        :param addr: Address of basic block start or end
        :type addr: int
        :return: BasicBlock, None -- the requested basic block
        """
        return self._basic_blocks.get(addr)

    def get_function_at(self, addr: int) -> Optional[Function]:
        """Return the function at the provided address.

        :param addr: Address of the function
        :type addr: int
        :return: Function, None -- the requested function
        """
        return self._functions.get(addr)

    def create_functions(self) -> None:
        """
        Create the functions. The CFGs are not computed
        :return:
        """
        self.compute_basic_blocks()
        self.compute_functions(self._basic_blocks[0], True)
        self.add_function(Function(Function.DISPATCHER_ID, 0, self._basic_blocks[0], self))

        for function in self.functions:
            if function.hash_id in known_hashes:
                function.name = known_hashes[function.hash_id]

    def create_cfgs(self) -> None:
        """
        Compute the CFGs
        :return:
        """
        # pylint: disable=import-outside-toplevel
        from evm_cfg_builder.value_analysis.value_set_analysis import StackValueAnalysis

        for function in self.functions:

            vsa = StackValueAnalysis(
                self, function.entry, function.hash_id, self._optimization_enabled
            )
            bbs = vsa.analyze()

            function.basic_blocks = [self._basic_blocks[bb] for bb in bbs]

            if function.hash_id != Function.DISPATCHER_ID:
                function.check_payable()
                function.check_view()
                function.check_pure()

    def clear(self) -> None:
        self._functions = {}
        self._basic_blocks = {}
        self._instructions = {}
        self._bytecode = bytes()

    def remove_metadata(self) -> None:
        """
        Init bytecode contains metadata that needs to be removed
        see http://solidity.readthedocs.io/en/v0.4.24/metadata.html#encoding-of-the-metadata-hash-in-the-bytecode
        """
        if self.bytecode:
            self.bytecode = re.sub(
                bytes(
                    r"\xa1\x65\x62\x7a\x7a\x72\x30\x58\x20[\x00-\xff]{32}\x00\x29".encode("charmap")
                ),
                b"",
                self.bytecode,
            )

    def compute_basic_blocks(self) -> None:
        """
            Split instructions into BasicBlock
        Args:
            self: CFG
        Returns:
            None
        """
        # Do nothing if basic_blocks already exist
        if self._basic_blocks:
            return

        bb = BasicBlock()

        for instruction in disassemble_all(self.bytecode):
            self._instructions[instruction.pc] = instruction

            if instruction.name == "JUMPDEST":
                # JUMPDEST indicates a new BasicBlock. Set the end pc
                # of the current block, and switch to a new one.
                if bb.instructions:
                    self._basic_blocks[bb.end.pc] = bb

                bb = BasicBlock()

                self._basic_blocks[instruction.pc] = bb

            bb.add_instruction(instruction)

            if bb.start.pc == instruction.pc:
                self._basic_blocks[instruction.pc] = bb

            if bb.end.name in BASIC_BLOCK_END:
                self._basic_blocks[bb.end.pc] = bb
                bb = BasicBlock()

    def compute_functions(self, block: "BasicBlock", is_entry_block: bool = False) -> None:
        """
        Create function from basic block
        The heuristic skips the first basic block(s) as they are generated by solc.
        The heuristic checks for:
        - If the basic block contains CALLVALUE: Solidity 0.5.2 added a general 'payable'
        to contract (https://github.com/ethereum/solidity/releases/tag/v0.5.2). In that case nothing is executed
        (including no fallback function)
        - If the basic block contains CALLDATASIZE (checked by is_jump_to_function), the fallback function is executed
        (this was added at some point in Solidity 0.4.x, it might not be present in old versions
        """
        if is_entry_block:
            if block.ends_with_jumpi():
                ins = [i.name for i in block.instructions]
                if "CALLVALUE" in ins:
                    # last_push
                    assert len(block.instructions) > 2
                    push = block.instructions[-2]
                    assert push.name.startswith("PUSH")
                    destination = push.operand
                    # Guard: skip if destination is not a known block start
                    if destination not in self._basic_blocks:
                        return
                    true_branch = self._basic_blocks[destination]
                    self.compute_functions(true_branch)
                    return

        function_start, function_hash = is_jump_to_function(block)
        if function_start:
            # The dispatcher can be a tree and not a list of comparison.
            # If GT is in the basic block, we are branching to a branch of
            # the dispatcher tree rather than directly calling the function.
            if "GT" in [i.name for i in block.instructions]:
                # Guard: skip if the branch target is not in known blocks
                if function_start not in self._basic_blocks:
                    if block.ends_with_jumpi():
                        false_pc = block.end.pc + 1
                        if false_pc in self._basic_blocks:
                            self.compute_functions(self._basic_blocks[false_pc])
                    return
                next_branch = self._basic_blocks[function_start]
                self.compute_functions(next_branch)

            else:
                assert function_hash
                # Guard: Solidity 0.8.x can produce jump targets that land
                # in the middle of PUSH data bytes (not a real block start).
                if function_start not in self._basic_blocks:
                    # Still try the false branch so we don't miss other functions
                    if block.ends_with_jumpi():
                        false_pc = block.end.pc + 1
                        if false_pc in self._basic_blocks:
                            self.compute_functions(self._basic_blocks[false_pc])
                    return
                new_function = Function(
                    function_hash, function_start, self._basic_blocks[function_start], self
                )

                self._functions[function_start] = new_function

            if block.ends_with_jumpi():
                false_pc = block.end.pc + 1
                if false_pc in self._basic_blocks:
                    false_branch = self._basic_blocks[false_pc]
                    self.compute_functions(false_branch)

    def add_function(self, func: Function) -> None:
        assert isinstance(func, Function)
        self._functions[func.start_addr] = func

    def compute_simple_edges(self, key: int) -> None:
        for bb in self._basic_blocks.values():

            if bb.end.name == "JUMPI":
                dst = self._basic_blocks[bb.end.pc + 1]
                bb.add_outgoing_basic_block(dst, key)
                dst.add_incoming_basic_block(bb, key)

            # A bb can be split in the middle if it has a JUMPDEST
            # Because another edge can target the JUMPDEST
            if bb.end.name not in BASIC_BLOCK_END:
                try:
                    dst = self._basic_blocks[bb.end.pc + 1 + bb.end.operand_size]
                except KeyError:
                    continue
                assert dst.start.name == "JUMPDEST"
                bb.add_outgoing_basic_block(dst, key)
                dst.add_incoming_basic_block(bb, key)

    def compute_reachability(self, entry_point: "BasicBlock", key: int) -> None:
        bbs_saw = [entry_point]

        bbs_to_explore = [entry_point]
        while bbs_to_explore:
            bb = bbs_to_explore.pop()
            for son in bb.outgoing_basic_blocks(key):
                if not son in bbs_saw:
                    bbs_saw.append(son)
                    bbs_to_explore.append(son)

        for bb in bbs_saw:
            bb.reacheable.append(key)

        # clean son/fathers that are created by compute_simple_edges
        # but are not reacheable
        for bb in self._basic_blocks.values():
            if not bb in bbs_saw:
                if key in bb.incoming_basic_blocks_as_dict.keys():
                    bb.incoming_basic_blocks_as_dict.pop(key)
                if key in bb.outgoing_basic_blocks_as_dict.keys():
                    bb.outgoing_basic_blocks_as_dict.pop(key)

    # ── BFS-based CFG enhancement (Solidity 0.8.x fix) ────────────────────────

    def _static_jump_target(self, bb: "BasicBlock") -> Optional["BasicBlock"]:
        """
        For a block ending in JUMP or JUMPI, find the static target by
        scanning backwards for the last PUSH instruction.
        Returns the target BasicBlock or None.
        """
        for instr in reversed(bb.instructions):
            if instr.name.startswith("PUSH") and instr.operand is not None:
                return self._basic_blocks.get(instr.operand)
        return None

    def _scan_selector_dispatch(self) -> Dict[int, int]:
        """
        Scan every basic block for the Solidity ABI dispatch pattern:

            PUSH4  <4-byte selector>
            EQ
            PUSH2  <body entry PC>
            JUMPI

        Returns ``{selector: body_entry_pc}``.

        Works for both the inline dispatcher (older Solidity) and the shared
        ABI decoder (Solidity ≥ 0.8.x where all stubs jump to one decoder).
        """
        dispatch_map: Dict[int, int] = {}
        for bb in self._basic_blocks.values():
            instrs = bb.instructions
            for idx in range(len(instrs) - 3):
                i0, i1, i2, i3 = instrs[idx], instrs[idx + 1], instrs[idx + 2], instrs[idx + 3]
                if (
                    i0.name == "PUSH4"
                    and i1.name == "EQ"
                    and i2.name.startswith("PUSH")
                    and i3.name == "JUMPI"
                    and i2.operand is not None
                    and i2.operand in self._basic_blocks
                ):
                    dispatch_map[i0.operand] = i2.operand
        return dispatch_map

    def _bfs_blocks(
        self,
        entry_block: "BasicBlock",
        stop_at: Optional[Set[int]] = None,
        key: Optional[int] = None,
    ) -> List["BasicBlock"]:
        """
        BFS from *entry_block*, collecting all reachable BasicBlocks.

        Edge resolution order:
          1. Function-specific edges recorded by VSA (``key`` must be provided).
             Falls back to *all* recorded edges when ``key`` is None.
          2. Static resolution: last PUSH before a JUMP/JUMPI → target block.
          3. JUMPI false branch: block starting at (JUMPI_pc + 1).

        *stop_at*: set of block-start PCs that belong to **other** functions.
                   The BFS will not enter those blocks.

        *key*: the function hash / ``Function.hash_id``.  Pass this to use
               only edges discovered by VSA for *this* function, which avoids
               accidentally bleeding into other functions via shared blocks.
        """
        if stop_at is None:
            stop_at = set()

        visited: Dict[int, "BasicBlock"] = {}
        queue: List["BasicBlock"] = [entry_block]

        while queue:
            bb = queue.pop(0)
            pc = bb.start.pc
            if pc in visited:
                continue
            visited[pc] = bb

            # 1. Function-specific VSA edges (preferred — no cross-function bleed)
            if key is not None:
                successors = list(bb.outgoing_basic_blocks(key))
            else:
                successors = list(bb.all_outgoing_basic_blocks)

            # 2 & 3. Static fallback — only for blocks VSA never analyzed.
            #
            # If VSA reached this block under `key` (key in bb.reacheable) but
            # found no outgoing edges, that means the JUMP target is dynamic
            # (e.g. a return-jump where the return address was pushed by the
            # caller at runtime).  Guessing via the last PUSH would be wrong —
            # it would follow an unrelated data push and bleed into other
            # functions.  We only apply static resolution when VSA never ran
            # on this block at all.
            if not successors:
                vsa_analyzed = key is not None and key in bb.reacheable
                if not vsa_analyzed:
                    end_op = bb.end.name
                    if end_op in ("JUMP", "JUMPI"):
                        tgt = self._static_jump_target(bb)
                        if tgt:
                            successors.append(tgt)
                    if end_op == "JUMPI":
                        next_pc = bb.end.pc + 1
                        if next_pc in self._basic_blocks:
                            successors.append(self._basic_blocks[next_pc])

            for succ in successors:
                succ_pc = succ.start.pc
                if succ_pc in stop_at or succ_pc in visited:
                    continue
                queue.append(succ)

        return sorted(visited.values(), key=lambda b: b.start.pc)

    def enhance_cfgs_with_bfs(self) -> None:
        """
        Augment each function's ``basic_blocks`` list using aggressive BFS.

        Call this **after** :meth:`create_cfgs` (or after ``CFG.__init__``
        with default arguments).  Essential for Solidity ≥ 0.8 where every
        function stub jumps into a *shared* ABI decoder, which causes VSA to
        stop after just the one-instruction entry stub.

        Algorithm
        ---------
        1. Scan all basic blocks for ``PUSH4 <sel> EQ PUSH2 <dst> JUMPI`` to
           build a *selector → body-entry* map.
        2. For each function:
           a. BFS from the stub entry, stopping at other stubs **and** other
              body entry PCs (keeps function bodies separated).
           b. If the ABI-decoder scan found a dedicated body entry for this
              function, also BFS from that entry and merge the two sets.
        3. Replace ``function.basic_blocks`` only when BFS found **more**
           blocks than VSA (so well-analyzed functions are never downgraded).
        """
        # Step 1 — map selector → body_entry_pc from dispatch patterns
        dispatch_map = self._scan_selector_dispatch()

        # All function stub entry PCs (boundaries we don't cross into)
        stub_entries: Set[int] = {
            f.start_addr
            for f in self.functions
            if f.hash_id not in (Function.DISPATCHER_ID, Function.FALLBACK_ID)
        }

        # All known body entry PCs
        all_body_pcs: Set[int] = set(dispatch_map.values())

        for function in self.functions:
            if function.hash_id in (Function.DISPATCHER_ID, Function.FALLBACK_ID):
                continue

            entry_bb = self._basic_blocks.get(function.start_addr)
            if not entry_bb:
                continue

            # Build stop-set: other stubs + other function bodies
            this_body_pc  = dispatch_map.get(function.hash_id)
            other_stubs   = stub_entries - {function.start_addr}
            other_bodies  = all_body_pcs - ({this_body_pc} if this_body_pc else set())
            stop_at       = other_stubs | other_bodies
            fn_key        = function.hash_id   # use function-specific VSA edges

            # BFS from stub entry using function-specific edges (avoids cross-function bleed)
            bfs_result: Dict[int, "BasicBlock"] = {
                bb.start.pc: bb
                for bb in self._bfs_blocks(entry_bb, stop_at=stop_at, key=fn_key)
            }

            # Also BFS from the actual body entry if the dispatch scan found one
            if this_body_pc and this_body_pc in self._basic_blocks:
                body_bb = self._basic_blocks[this_body_pc]
                for bb in self._bfs_blocks(body_bb, stop_at=stop_at, key=fn_key):
                    bfs_result.setdefault(bb.start.pc, bb)

            bfs_blocks = sorted(bfs_result.values(), key=lambda b: b.start.pc)

            # Only upgrade when BFS found more blocks than VSA did
            if len(bfs_blocks) > len(function.basic_blocks):
                function.basic_blocks = bfs_blocks
                # Re-derive attributes for the full block set
                function._attributes = []  # type: ignore[attr-defined]
                function.check_payable()
                function.check_view()
                function.check_pure()

    def output_to_dot(self, base_filename: str) -> None:

        with open(f"{base_filename}-FULL_GRAPH.dot", "w", encoding="utf-8") as f:
            f.write("digraph{\n")
            for basic_block in self.basic_blocks:
                instructions_ = [f"{hex(ins.pc)}:{str(ins)}" for ins in basic_block.instructions]
                instructions = "\n".join(instructions_)

                f.write(f'{basic_block.start.pc}[label="{instructions}"]\n')

                for son in basic_block.all_outgoing_basic_blocks:
                    f.write(f"{basic_block.start.pc} -> {son.start.pc}\n")

            f.write("\n}")


def is_jump_to_function(block: BasicBlock) -> Tuple[Optional[int], Optional[int]]:
    """
        Heuristic:
        Recent solc version add a first check if calldatasize <4 and jump in fallback
    Args:
        block (BasicBlock)
    Returns:
        (int): function hash, or None
    """
    has_calldata_size = False
    last_pushed_value: Optional[int] = None
    previous_last_pushed_value: Optional[int] = None
    for i in block.instructions:
        if i.name == "CALLDATASIZE":
            has_calldata_size = True

        if i.name.startswith("PUSH"):
            previous_last_pushed_value = last_pushed_value
            last_pushed_value = i.operand

    if block.ends_with_jumpi() and has_calldata_size:
        return last_pushed_value, -1

    if block.ends_with_jumpi() and previous_last_pushed_value:
        return last_pushed_value, previous_last_pushed_value

    return None, None
