# Re-export the public API so `from evm_cfg_builder.cfg import CFG` works,
# matching the interface expected by bytecode_to_cfg.py.
from .cfg import CFG, convert_bytecode
from .basic_block import BasicBlock
from .function import Function

__all__ = ["CFG", "convert_bytecode", "BasicBlock", "Function"]
