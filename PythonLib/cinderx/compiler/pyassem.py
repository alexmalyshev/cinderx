# Portions copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe

"""A flow graph representation for Python bytecode"""
from __future__ import annotations

import sys
from ast import AST
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum

try:
    # pyre-ignore[21]: No _inline_cache_entries
    from opcode import _inline_cache_entries
except ImportError:
    _inline_cache_entries = None
from types import CodeType
from typing import Callable, ClassVar, Generator, Optional, Sequence

from . import opcode_cinder, opcodes
from .consts import (
    CO_ASYNC_GENERATOR,
    CO_COROUTINE,
    CO_GENERATOR,
    CO_NEWLOCALS,
    CO_OPTIMIZED,
    CO_SUPPRESS_JIT,
)
from .flow_graph_optimizer import FlowGraphOptimizer, FlowGraphOptimizer312
from .opcodebase import Opcode


MAX_COPY_SIZE = 4


def sign(a):
    if not isinstance(a, float):
        raise TypeError(f"Must be a real number, not {type(a)}")
    if a != a:
        return 1.0  # NaN case
    return 1.0 if str(a)[0] != "-" else -1.0


def cast_signed_byte_to_unsigned(i):
    if i < 0:
        i = 255 + i + 1
    return i


FVC_MASK = 0x3
FVC_NONE = 0x0
FVC_STR = 0x1
FVC_REPR = 0x2
FVC_ASCII = 0x3
FVS_MASK = 0x4
FVS_HAVE_SPEC = 0x4


UNCONDITIONAL_JUMP_OPCODES = (
    "JUMP_ABSOLUTE",
    "JUMP_FORWARD",
    "JUMP",
    "JUMP_BACKWARD",
    "JUMP_BACKWARD_NO_INTERRUPT",
)

SCOPE_EXIT_OPCODES = (
    "RETURN_VALUE",
    "RETURN_CONST",
    "RETURN_PRIMITIVE",
    "RAISE_VARARGS",
    "RERAISE",
)

SETUP_OPCODES = ("SETUP_FINALLY", "SETUP_WITH", "SETUP_CLEANUP")


@dataclass(frozen=True, slots=True)
class SrcLocation:
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int

    def __repr__(self) -> str:
        return f"SrcLocation({self.lineno}, {self.end_lineno}, {self.col_offset}, {self.end_col_offset})"


NO_LOCATION = SrcLocation(-1, -1, -1, -1)


class Instruction:
    __slots__ = ("opname", "oparg", "target", "ioparg", "loc", "exc_handler")

    def __init__(
        self,
        opname: str,
        oparg: object,
        ioparg: int = 0,
        loc: AST | SrcLocation = NO_LOCATION,
        target: Block | None = None,
        exc_handler: Block | None = None,
    ):
        self.opname = opname
        self.oparg = oparg
        self.loc = loc
        self.ioparg = ioparg
        self.target = target
        self.exc_handler = exc_handler

    @property
    def lineno(self) -> int:
        return self.loc.lineno

    def __repr__(self):
        args = [
            f"{self.opname!r}",
            f"{self.oparg!r}",
            f"{self.ioparg!r}",
            f"{self.loc!r}",
        ]
        if self.target is not None:
            args.append(f"{self.target!r}")
        if self.exc_handler is not None:
            args.append(f"{self.exc_handler!r}")

        return f"Instruction({', '.join(args)})"

    def is_jump(self, opcode: Opcode) -> bool:
        op = opcode.opmap[self.opname]
        return opcode.has_jump(op)

    def copy(self) -> Instruction:
        return Instruction(self.opname, self.oparg, self.ioparg, self.loc, self.target)


class CompileScope:
    START_MARKER = "compile-scope-start-marker"
    __slots__ = "blocks"

    def __init__(self, blocks):
        self.blocks = blocks


class FlowGraph:
    def __init__(self):
        self.block_count = 0
        # List of blocks in the order they should be output for linear
        # code. As we deal with structured code, this order corresponds
        # to the order of source level constructs. (The original
        # implementation from Python2 used a complex ordering algorithm
        # which more buggy and erratic than useful.)
        self.ordered_blocks = []
        # Current block being filled in with instructions.
        self.current = None
        self.entry = Block("entry")
        self.startBlock(self.entry)

        # Source line number to use for next instruction.
        self.loc: AST | SrcLocation = SrcLocation(0, 0, 0, 0)
        # First line of this code block. This field is expected to be set
        # externally and serve as a reference for generating all other
        # line numbers in the code block. (If it's not set, it will be
        # deduced).
        self.firstline = 0
        # Line number of first instruction output. Used to deduce .firstline
        # if it's not set explicitly.
        self.first_inst_lineno = 0
        # If non-zero, do not emit bytecode
        self.do_not_emit_bytecode = 0

    def blocks_in_reverse_allocation_order(self):
        yield from sorted(self.ordered_blocks, key=lambda b: b.alloc_id, reverse=True)

    @contextmanager
    def new_compile_scope(self) -> Generator[CompileScope, None, None]:
        prev_current = self.current
        prev_ordered_blocks = self.ordered_blocks
        prev_line_no = self.first_inst_lineno
        try:
            self.ordered_blocks = []
            self.current = self.newBlock(CompileScope.START_MARKER)
            yield CompileScope(self.ordered_blocks)
        finally:
            self.current = prev_current
            self.ordered_blocks = prev_ordered_blocks
            self.first_inst_lineno = prev_line_no

    def apply_from_scope(self, scope: CompileScope):
        # link current block with the block from out of order result
        block: Block = scope.blocks[0]
        assert block.prev is not None
        assert block.prev.label == CompileScope.START_MARKER
        block.prev = None

        self.current.addNext(block)
        self.ordered_blocks.extend(scope.blocks)
        self.current = scope.blocks[-1]

    def startBlock(self, block: Block) -> None:
        if self._debug:
            if self.current:
                print("end", repr(self.current))
                print("    next", self.current.next)
                print("    prev", self.current.prev)
                print("   ", self.current.get_children())
            print(repr(block))
        block.bid = self.block_count
        self.block_count += 1
        self.current = block
        if block and block not in self.ordered_blocks:
            self.ordered_blocks.append(block)

    def nextBlock(self, block=None, label=""):
        if self.do_not_emit_bytecode:
            return
        # XXX think we need to specify when there is implicit transfer
        # from one block to the next.  might be better to represent this
        # with explicit JUMP_ABSOLUTE instructions that are optimized
        # out when they are unnecessary.
        #
        # I think this strategy works: each block has a child
        # designated as "next" which is returned as the last of the
        # children.  because the nodes in a graph are emitted in
        # reverse post order, the "next" block will always be emitted
        # immediately after its parent.
        # Worry: maintaining this invariant could be tricky
        if block is None:
            block = self.newBlock(label=label)

        # Note: If the current block ends with an unconditional control
        # transfer, then it is techically incorrect to add an implicit
        # transfer to the block graph. Doing so results in code generation
        # for unreachable blocks.  That doesn't appear to be very common
        # with Python code and since the built-in compiler doesn't optimize
        # it out we don't either.
        self.current.addNext(block)
        self.startBlock(block)

    def newBlock(self, label: str = "") -> Block:
        return Block(label)

    _debug = 0

    def _enable_debug(self):
        self._debug = 1

    def _disable_debug(self):
        self._debug = 0

    def emit_with_loc(self, opcode: str, oparg: object, loc: AST | SrcLocation) -> None:
        if isinstance(oparg, Block):
            if not self.do_not_emit_bytecode:
                self.current.addOutEdge(oparg)
                self.current.emit(Instruction(opcode, 0, 0, loc, target=oparg))
            return

        ioparg = self.convertArg(opcode, oparg)

        if not self.do_not_emit_bytecode:
            self.current.emit(Instruction(opcode, oparg, ioparg, loc))

    def emit(self, opcode: str, oparg: object = 0) -> None:
        self.emit_with_loc(opcode, oparg, self.loc)

    def emit_noline(self, opcode: str, oparg: object = 0) -> None:
        self.emit_with_loc(opcode, oparg, NO_LOCATION)

    def emitWithBlock(self, opcode: str, oparg: object, target: Block):
        if not self.do_not_emit_bytecode:
            self.current.addOutEdge(target)
            self.current.emit(Instruction(opcode, oparg, target=target))

    def set_pos(self, node: AST | SrcLocation) -> None:
        if not self.first_inst_lineno:
            self.first_inst_lineno = node.lineno
        self.loc = node

    def convertArg(self, opcode: str, oparg: object) -> int:
        if isinstance(oparg, int):
            return oparg
        raise ValueError(f"invalid oparg {oparg!r} for {opcode!r}")

    def getBlocksInOrder(self):
        """Return the blocks in the order they should be output."""
        return self.ordered_blocks

    def getBlocks(self):
        return self.ordered_blocks

    def getRoot(self):
        """Return nodes appropriate for use with dominator"""
        return self.entry

    def getContainedGraphs(self):
        result = []
        for b in self.getBlocks():
            result.extend(b.getContainedGraphs())
        return result


class Block:
    allocated_block_count: ClassVar[int] = 0

    def __init__(self, label=""):
        self.insts: list[Instruction] = []
        self.outEdges = set()
        self.label: str = label
        self.bid: int | None = None
        self.next: Block | None = None
        self.prev: Block | None = None
        self.returns: bool = False
        self.offset: int = 0
        self.preserve_lasti: False  # used if block is an exception handler
        self.seen: bool = False  # visited during stack depth calculation
        self.startdepth: int = -1
        self.is_exit: bool = False
        self.has_fallthrough: bool = True
        self.num_predecessors: int = 0
        self.alloc_id: int = Block.allocated_block_count
        Block.allocated_block_count += 1

    def __repr__(self):
        data = []
        data.append(f"id={self.bid}")
        data.append(f"startdepth={self.startdepth}")
        if self.next:
            data.append(f"next={self.next.bid}")
        extras = ", ".join(data)
        if self.label:
            return f"<block {self.label} {extras}>"
        else:
            return f"<block {extras}>"

    def __str__(self):
        insts = map(str, self.insts)
        insts = "\n".join(insts)
        return f"<block label={self.label} bid={self.bid} startdepth={self.startdepth}: {insts}>"

    def emit(self, instr: Instruction) -> None:
        # TODO(T128853358): The RETURN_PRIMITIVE logic should live in the Static flow graph.
        if instr.opname in ("RETURN_VALUE", "RETURN_PRIMITIVE"):
            self.returns = True

        self.insts.append(instr)

    def getInstructions(self):
        return self.insts

    def addOutEdge(self, block):
        self.outEdges.add(block)

    def addNext(self, block):
        assert self.next is None, self.next
        self.next = block
        assert block.prev is None, block.prev
        block.prev = self

    def removeNext(self):
        assert self.next is not None
        next = self.next
        next.prev = None
        self.next = None

    def has_return(self):
        # TODO(T128853358): The RETURN_PRIMITIVE logic should live in the Static flow graph.
        return self.insts and self.insts[-1].opname in (
            "RETURN_VALUE",
            "RETURN_PRIMITIVE",
        )

    def get_children(self):
        return list(self.outEdges) + ([self.next] if self.next is not None else [])

    def getContainedGraphs(self):
        """Return all graphs contained within this block.

        For example, a MAKE_FUNCTION block will contain a reference to
        the graph for the function body.
        """
        contained = []
        for inst in self.insts:
            if len(inst) == 1:
                continue
            op = inst[1]
            if hasattr(op, "graph"):
                contained.append(op.graph)
        return contained

    def copy(self):
        # Cannot copy block if it has fallthrough, since a block can have only one
        # fallthrough predecessor
        assert not self.has_fallthrough
        result = Block()
        result.insts = [instr.copy() for instr in self.insts]
        result.is_exit = self.is_exit
        result.has_fallthrough = False
        return result


# flags for code objects

# the FlowGraph is transformed in place; it exists in one of these states
ACTIVE = "ACTIVE"  # accepting calls to .emit()
CLOSED = "CLOSED"  # closed to new instructions
CONSTS_CLOSED = "CONSTS_CLOSED"  # closed to new consts
OPTIMIZED = "OPTIMIZED"  # optimizations have been run
ORDERED = "ORDERED"  # basic block ordering is set
FINAL = "FINAL"  # all optimization and normalization of flow graph is done
FLAT = "FLAT"  # flattened
DONE = "DONE"


class IndexedSet:
    """Container that behaves like a `set` that assigns stable dense indexes
    to each element. Put another way: This behaves like a `list` where you
    check `x in <list>` before doing any insertion to avoid duplicates. But
    contrary to the list this does not require an O(n) member check."""

    __delitem__ = None

    def __init__(self, iterable=()):
        self.keys = {}
        for item in iterable:
            self.get_index(item)

    def __add__(self, iterable):
        result = IndexedSet()
        for item in self.keys.keys():
            result.get_index(item)
        for item in iterable:
            result.get_index(item)
        return result

    def __contains__(self, item):
        return item in self.keys

    def __iter__(self):
        # This relies on `dict` maintaining insertion order.
        return iter(self.keys.keys())

    def __len__(self):
        return len(self.keys)

    def get_index(self, item):
        """Return index of name in collection, appending if necessary"""
        assert type(item) is str
        idx = self.keys.get(item)
        if idx is not None:
            return idx
        idx = len(self.keys)
        self.keys[item] = idx
        return idx

    def index(self, item):
        assert type(item) is str
        idx = self.keys.get(item)
        if idx is not None:
            return idx
        raise ValueError()

    def update(self, iterable):
        for item in iterable:
            self.get_index(item)


class PyFlowGraph(FlowGraph):

    super_init = FlowGraph.__init__
    flow_graph_optimizer: type[FlowGraphOptimizer] = FlowGraphOptimizer
    opcode = opcodes.opcode

    def __init__(
        self,
        name: str,
        filename: str,
        scope,
        flags: int = 0,
        args: Sequence[str] = (),
        kwonlyargs=(),
        starargs=(),
        optimized: int = 0,
        klass: bool = False,
        docstring: str | None = None,
        firstline: int = 0,
        posonlyargs: int = 0,
    ) -> None:
        self.super_init()
        self.name = name
        self.filename = filename
        self.scope = scope
        self.docstring = None
        self.args = args
        self.kwonlyargs = kwonlyargs
        self.posonlyargs = posonlyargs
        self.starargs = starargs
        self.klass = klass
        self.stacksize = 0
        self.docstring = docstring
        self.flags = flags
        if optimized:
            self.setFlag(CO_OPTIMIZED | CO_NEWLOCALS)
        self.consts = {}
        self.names = IndexedSet()
        # Free variables found by the symbol table scan, including
        # variables used only in nested scopes, are included here.
        if scope is not None:
            self.freevars = IndexedSet(scope.get_free_vars())
            self.cellvars = IndexedSet(scope.get_cell_vars())
        else:
            self.freevars = IndexedSet([])
            self.cellvars = IndexedSet([])
        # The closure list is used to track the order of cell
        # variables and free variables in the resulting code object.
        # The offsets used by LOAD_CLOSURE/LOAD_DEREF refer to both
        # kinds of variables.
        self.closure = self.cellvars + self.freevars
        varnames = IndexedSet()
        varnames.update(args)
        varnames.update(kwonlyargs)
        varnames.update(starargs)
        self.varnames = varnames
        self.stage = ACTIVE
        self.firstline = firstline
        self.first_inst_lineno = 0
        # Add any extra consts that were requested to the const pool
        self.extra_consts = []
        self.initializeConsts()
        self.fast_vars = set()
        self.gen_kind = None
        self.lnotab: LineAddrTable = LineAddrTable()
        self.insts: list[Instruction] = []
        if flags & CO_COROUTINE:
            self.gen_kind = 1
        elif flags & CO_ASYNC_GENERATOR:
            self.gen_kind = 2
        elif flags & CO_GENERATOR:
            self.gen_kind = 0

    def emit_gen_start(self) -> None:
        if self.gen_kind is not None:
            self.emit_noline("GEN_START", self.gen_kind)

    def setFlag(self, flag: int) -> None:
        self.flags |= flag

    def checkFlag(self, flag: int) -> int | None:
        if self.flags & flag:
            return 1

    def initializeConsts(self) -> None:
        # Docstring is first entry in co_consts for normal functions
        # (Other types of code objects deal with docstrings in different
        # manner, e.g. lambdas and comprehensions don't have docstrings,
        # classes store them as __doc__ attribute.
        if self.name == "<lambda>":
            self.consts[self.get_const_key(None)] = 0
        elif not self.name.startswith("<") and not self.klass:
            if self.docstring is not None:
                self.consts[self.get_const_key(self.docstring)] = 0
            else:
                self.consts[self.get_const_key(None)] = 0

    def convertArg(self, opcode: str, oparg: object) -> int:
        assert self.stage in {ACTIVE, CLOSED}, self.stage

        if self.do_not_emit_bytecode and opcode in self._quiet_opcodes:
            # return -1 so this errors if it ever ends up in non-dead-code due
            # to a bug.
            return -1

        conv = self._converters.get(opcode)
        if conv is not None:
            return conv(self, oparg)

        return super().convertArg(opcode, oparg)

    def finalize(self) -> None:
        """Perform final optimizations and normalization of flow graph."""
        assert self.stage == ACTIVE, self.stage
        self.stage = CLOSED

        for block in self.ordered_blocks:
            self.normalize_basic_block(block)
        for block in self.blocks_in_reverse_allocation_order():
            self.extend_block(block)
        self.optimizeCFG()
        self.duplicate_exits_without_lineno()

        self.stage = CONSTS_CLOSED
        self.trim_unused_consts()
        self.propagate_line_numbers()
        self.firstline = self.firstline or self.first_inst_lineno or 1
        self.guarantee_lineno_for_exits()

        self.stage = ORDERED
        self.normalize_jumps()
        self.stage = FINAL

    def getCode(self):
        """Get a Python code object"""
        self.finalize()
        assert self.stage == FINAL, self.stage

        self.computeStackDepth()
        self.flattenGraph()

        assert self.stage == FLAT, self.stage
        bytecode = self.make_byte_code()
        linetable = self.make_line_table()
        exception_table = self.make_exception_table()
        assert self.stage == DONE, self.stage
        return self.new_code_object(bytecode, linetable, exception_table)

    def dump(self, io=None):
        if io:
            save = sys.stdout
            sys.stdout = io
        pc = 0
        for block in self.getBlocks():
            print(repr(block))
            for instr in block.getInstructions():
                opname = instr.opname
                if instr.target is None:
                    print("\t", f"{pc:3} {instr.lineno} {opname} {instr.oparg}")
                elif instr.target.label:
                    print(
                        "\t",
                        f"{pc:3} {instr.lineno} {opname} {instr.target.bid} ({instr.target.label})",
                    )
                else:
                    print("\t", f"{pc:3} {instr.lineno} {opname} {instr.target.bid}")
                pc += self.opcode.CODEUNIT_SIZE
        if io:
            sys.stdout = save

    def push_block(self, worklist: list[Block], block: Block, depth: int):
        assert (
            block.startdepth < 0 or block.startdepth >= depth
        ), f"{block!r}: {block.startdepth} vs {depth}"
        if block.startdepth < depth:
            block.startdepth = depth
            worklist.append(block)

    def stackdepth_walk(self, block):
        maxdepth = 0
        worklist = []
        self.push_block(worklist, block, 0 if self.gen_kind is None else 1)
        while worklist:
            block = worklist.pop()
            next = block.next
            depth = block.startdepth
            assert depth >= 0

            for instr in block.getInstructions():
                delta = self.opcode.stack_effect_raw(instr.opname, instr.oparg, False)
                new_depth = depth + delta
                if new_depth > maxdepth:
                    maxdepth = new_depth

                assert depth >= 0, instr

                op = self.opcode.opmap[instr.opname]
                if self.opcode.has_jump(op) or instr.opname == "SETUP_FINALLY":
                    delta = self.opcode.stack_effect_raw(
                        instr.opname, instr.oparg, True
                    )

                    target_depth = depth + delta
                    if target_depth > maxdepth:
                        maxdepth = target_depth

                    assert target_depth >= 0

                    self.push_block(worklist, instr.target, target_depth)

                depth = new_depth

                # TODO(T128853358): The RETURN_PRIMITIVE logic should live in the Static flow graph.
                if (
                    instr.opname in SCOPE_EXIT_OPCODES
                    or instr.opname in UNCONDITIONAL_JUMP_OPCODES
                ):
                    # Remaining code is dead
                    next = None
                    break

            # TODO(dinoviehland): we could save the delta we came up with here and
            # reapply it on subsequent walks rather than having to walk all of the
            # instructions again.
            if next:
                self.push_block(worklist, next, depth)

        return maxdepth

    def computeStackDepth(self):
        """Compute the max stack depth.

        Find the flow path that needs the largest stack.  We assume that
        cycles in the flow graph have no net effect on the stack depth.
        """
        assert self.stage == FINAL, self.stage
        for block in self.getBlocksInOrder():
            # We need to get to the first block which actually has instructions
            if block.getInstructions():
                self.stacksize = self.stackdepth_walk(block)
                break

    def instrsize(self, opname: str, oparg: int):
        if oparg <= 0xFF:
            return 1
        elif oparg <= 0xFFFF:
            return 2
        elif oparg <= 0xFFFFFF:
            return 3
        else:
            return 4

    def flattenGraph(self):
        """Arrange the blocks in order and resolve jumps"""
        assert self.stage == FINAL, self.stage
        # This is an awful hack that could hurt performance, but
        # on the bright side it should work until we come up
        # with a better solution.
        #
        # The issue is that in the first loop blocksize() is called
        # which calls instrsize() which requires i_oparg be set
        # appropriately. There is a bootstrap problem because
        # i_oparg is calculated in the second loop.
        #
        # So we loop until we stop seeing new EXTENDED_ARGs.
        # The only EXTENDED_ARGs that could be popping up are
        # ones in jump instructions.  So this should converge
        # fairly quickly.
        extended_arg_recompile = True
        while extended_arg_recompile:
            extended_arg_recompile = False
            self.insts = insts = []
            pc = 0
            for b in self.getBlocksInOrder():
                b.offset = pc

                for inst in b.getInstructions():
                    insts.append(inst)
                    pc += self.instrsize(inst.opname, inst.ioparg)

            pc = 0
            for inst in insts:
                pc += self.instrsize(inst.opname, inst.ioparg)
                op = self.opcode.opmap[inst.opname]
                if self.opcode.has_jump(op):
                    oparg = inst.ioparg
                    target = inst.target

                    offset = target.offset
                    if op in self.opcode.hasjrel:
                        offset -= pc

                    offset = abs(offset)

                    if self.instrsize(inst.opname, oparg) != self.instrsize(
                        inst.opname, offset
                    ):
                        extended_arg_recompile = True

                    inst.ioparg = offset

        self.stage = FLAT

    # TODO(T128853358): pull out all converters for static opcodes into
    # StaticPyFlowGraph

    def _convert_LOAD_CONST(self, arg: object) -> int:
        getCode = getattr(arg, "getCode", None)
        if getCode is not None:
            arg = getCode()
        key = self.get_const_key(arg)
        res = self.consts.get(key, self)
        if res is self:
            res = self.consts[key] = len(self.consts)
        return res

    def get_const_key(self, value: object):
        if isinstance(value, float):
            return type(value), value, sign(value)
        elif isinstance(value, complex):
            return type(value), value, sign(value.real), sign(value.imag)
        elif isinstance(value, (tuple, frozenset)):
            return (
                type(value),
                value,
                type(value)(self.get_const_key(const) for const in value),
            )

        return type(value), value

    def _convert_LOAD_FAST(self, arg: object) -> int:
        self.fast_vars.add(arg)
        if isinstance(arg, int):
            return arg
        return self.varnames.get_index(arg)

    def _convert_LOAD_LOCAL(self, arg: object) -> int:
        self.fast_vars.add(arg)
        assert isinstance(arg, tuple), "invalid oparg {arg!r}"
        return self._convert_LOAD_CONST((self.varnames.get_index(arg[0]), arg[1]))

    def _convert_NAME(self, arg: object) -> int:
        return self.names.get_index(arg)

    def _convert_LOAD_SUPER(self, arg: object) -> int:
        assert isinstance(arg, tuple), "invalid oparg {arg!r}"
        return self._convert_LOAD_CONST((self._convert_NAME(arg[0]), arg[1]))

    def _convert_LOAD_SUPER_ATTR(self, arg: object) -> int:
        assert isinstance(arg, tuple), "invalid oparg {arg!r}"
        op, name, zero_args = arg
        arg = self._convert_LOAD_CONST((self._convert_NAME(name), zero_args))
        mask = {
            "LOAD_SUPER_ATTR": 2,
            "LOAD_ZERO_SUPER_ATTR": 0,
            "LOAD_SUPER_METHOD": 3,
            "LOAD_ZERO_SUPER_METHOD": 1,
        }
        return (arg << 2) | mask[op]

    def _convert_DEREF(self, arg: object) -> int:
        # Sometimes, both cellvars and freevars may contain the same var
        # (e.g., for class' __class__). In this case, prefer freevars.
        if arg in self.freevars:
            return self.freevars.get_index(arg) + len(self.cellvars)
        return self.closure.get_index(arg)

    # similarly for other opcodes...
    _converters = {
        "LOAD_CLASS": _convert_LOAD_CONST,
        "LOAD_CONST": _convert_LOAD_CONST,
        "INVOKE_FUNCTION": _convert_LOAD_CONST,
        "INVOKE_METHOD": _convert_LOAD_CONST,
        "INVOKE_NATIVE": _convert_LOAD_CONST,
        "LOAD_FIELD": _convert_LOAD_CONST,
        "STORE_FIELD": _convert_LOAD_CONST,
        "CAST": _convert_LOAD_CONST,
        "TP_ALLOC": _convert_LOAD_CONST,
        "BUILD_CHECKED_MAP": _convert_LOAD_CONST,
        "BUILD_CHECKED_LIST": _convert_LOAD_CONST,
        "PRIMITIVE_LOAD_CONST": _convert_LOAD_CONST,
        "LOAD_FAST": _convert_LOAD_FAST,
        "STORE_FAST": _convert_LOAD_FAST,
        "DELETE_FAST": _convert_LOAD_FAST,
        "LOAD_LOCAL": _convert_LOAD_LOCAL,
        "STORE_LOCAL": _convert_LOAD_LOCAL,
        "LOAD_NAME": _convert_NAME,
        "LOAD_FROM_DICT_OR_DEREF": _convert_NAME,
        "LOAD_FROM_DICT_OR_GLOBALS": _convert_NAME,
        "LOAD_CLOSURE": lambda self, arg: self.closure.get_index(arg),
        "COMPARE_OP": lambda self, arg: self.opcode.CMP_OP.index(arg),
        "LOAD_GLOBAL": _convert_NAME,
        "STORE_GLOBAL": _convert_NAME,
        "DELETE_GLOBAL": _convert_NAME,
        "CONVERT_NAME": _convert_NAME,
        "STORE_NAME": _convert_NAME,
        "STORE_ANNOTATION": _convert_NAME,
        "DELETE_NAME": _convert_NAME,
        "IMPORT_NAME": _convert_NAME,
        "IMPORT_FROM": _convert_NAME,
        "STORE_ATTR": _convert_NAME,
        "LOAD_ATTR": _convert_NAME,
        "DELETE_ATTR": _convert_NAME,
        "LOAD_METHOD": _convert_NAME,
        "LOAD_DEREF": _convert_DEREF,
        "STORE_DEREF": _convert_DEREF,
        "DELETE_DEREF": _convert_DEREF,
        "LOAD_CLASSDEREF": _convert_DEREF,
        "REFINE_TYPE": _convert_LOAD_CONST,
        "LOAD_METHOD_SUPER": _convert_LOAD_SUPER,
        "LOAD_ATTR_SUPER": _convert_LOAD_SUPER,
        "LOAD_SUPER_ATTR": _convert_LOAD_SUPER_ATTR,
        "LOAD_ZERO_SUPER_ATTR": _convert_LOAD_SUPER_ATTR,
        "LOAD_SUPER_METHOD": _convert_LOAD_SUPER_ATTR,
        "LOAD_ZERO_SUPER_METHOD": _convert_LOAD_SUPER_ATTR,
        "LOAD_TYPE": _convert_LOAD_CONST,
    }

    # Converters which add an entry to co_consts
    _const_converters = {
        _convert_LOAD_CONST,
        _convert_LOAD_LOCAL,
        _convert_LOAD_SUPER,
        _convert_LOAD_SUPER_ATTR,
    }
    # Opcodes which reference an entry in co_consts
    _const_opcodes = set()
    for op, converter in _converters.items():
        if converter in _const_converters:
            _const_opcodes.add(op)

    # Opcodes which do not add names to co_consts/co_names/co_varnames in dead code (self.do_not_emit_bytecode)
    _quiet_opcodes = {
        "LOAD_GLOBAL",
        "LOAD_CONST",
        "IMPORT_NAME",
        "STORE_ATTR",
        "LOAD_ATTR",
        "DELETE_ATTR",
        "LOAD_METHOD",
        "STORE_FAST",
        "LOAD_FAST",
    }

    def make_byte_code(self) -> bytes:
        assert self.stage == FLAT, self.stage

        code = bytearray()

        def addCode(opcode: int, oparg: int) -> None:
            # T190611021: Currently we still emit some of the pseudo ops, once we
            # get zero-cost exceptions in this can go away.
            code.append(opcode & 0xFF)
            code.append(oparg)

        for t in self.insts:
            oparg = t.ioparg
            assert 0 <= oparg <= 0xFFFFFFFF, oparg
            if oparg > 0xFFFFFF:
                addCode(self.opcode.EXTENDED_ARG, (oparg >> 24) & 0xFF)
            if oparg > 0xFFFF:
                addCode(self.opcode.EXTENDED_ARG, (oparg >> 16) & 0xFF)
            if oparg > 0xFF:
                addCode(self.opcode.EXTENDED_ARG, (oparg >> 8) & 0xFF)
            addCode(self.opcode.opmap[t.opname], oparg & 0xFF)
            self.emit_inline_cache(t.opname, addCode)

        self.stage = DONE
        return bytes(code)

    def emit_inline_cache(
        self, opcode: str, addCode: Callable[[int, int], None]
    ) -> None:
        pass

    def make_line_table(self) -> bytes:
        lnotab = LineAddrTable()
        lnotab.setFirstLine(self.firstline)

        prev_offset = offset = 0
        for t in self.insts:
            if lnotab.current_line != t.lineno and t.lineno:
                lnotab.nextLine(t.lineno, prev_offset, offset)
                prev_offset = offset

            offset += self.instrsize(t.opname, t.ioparg) * self.opcode.CODEUNIT_SIZE

        # Since the linetable format writes the end offset of bytecodes, we can't commit the
        # last write until all the instructions are iterated over.
        lnotab.emitCurrentLine(prev_offset, offset)
        return lnotab.getTable()

    def make_exception_table(self) -> bytes:
        # New in 3.12
        return b""

    def new_code_object(
        self, code: bytes, lnotab: bytes, exception_table: bytes
    ) -> CodeType:
        assert self.stage == DONE, self.stage
        if (self.flags & CO_NEWLOCALS) == 0:
            nlocals = len(self.fast_vars)
        else:
            nlocals = len(self.varnames)

        firstline = self.firstline
        # For module, .firstline is initially not set, and should be first
        # line with actual bytecode instruction (skipping docstring, optimized
        # out instructions, etc.)
        if not firstline:
            firstline = self.first_inst_lineno
        # If no real instruction, fallback to 1
        if not firstline:
            firstline = 1

        consts = self.getConsts()
        consts = consts + tuple(self.extra_consts)
        return self.make_code(nlocals, code, consts, firstline, lnotab, exception_table)

    def make_code(
        self, nlocals, code, consts, firstline: int, lnotab, exception_table=None
    ) -> CodeType:
        return CodeType(
            len(self.args),
            self.posonlyargs,
            len(self.kwonlyargs),
            nlocals,
            self.stacksize,
            self.flags,
            code,
            consts,
            tuple(self.names),
            tuple(self.varnames),
            self.filename,
            self.name,
            firstline,
            lnotab,
            tuple(self.freevars),
            tuple(self.cellvars),
        )

    def getConsts(self):
        """Return a tuple for the const slot of the code object"""
        # Just return the constant value, removing the type portion. Order by const index.
        return tuple(
            const[1] for const, idx in sorted(self.consts.items(), key=lambda o: o[1])
        )

    def propagate_line_numbers(self):
        """Propagate line numbers to instructions without."""
        for block in self.ordered_blocks:
            if not block.insts:
                continue
            prev_loc = NO_LOCATION
            for instr in block.insts:
                if instr.loc == NO_LOCATION:
                    instr.loc = prev_loc
                else:
                    prev_loc = instr.loc
            if block.has_fallthrough and block.next.num_predecessors == 1:
                assert block.next.insts
                next_instr = block.next.insts[0]
                if next_instr.loc == NO_LOCATION:
                    next_instr.loc = prev_loc
            last_instr = block.insts[-1]
            if last_instr.is_jump(self.opcode) and last_instr.opname not in {
                # Only actual jumps, not exception handlers
                "SETUP_ASYNC_WITH",
                "SETUP_WITH",
                "SETUP_FINALLY",
            }:
                target = last_instr.target
                if target.num_predecessors == 1:
                    assert target.insts
                    next_instr = target.insts[0]
                    if next_instr.loc == NO_LOCATION:
                        next_instr.loc = prev_loc

    def guarantee_lineno_for_exits(self):
        assert self.firstline > 0
        loc = SrcLocation(self.firstline, self.firstline, 0, 0)
        for block in self.ordered_blocks:
            if not block.insts:
                continue
            last_instr = block.insts[-1]
            if last_instr.loc == NO_LOCATION:
                # TODO(T128853358): The RETURN_PRIMITIVE logic should live in the Static flow graph.
                if last_instr.opname in ("RETURN_VALUE", "RETURN_PRIMITIVE"):
                    for instr in block.insts:
                        assert instr.loc == NO_LOCATION
                        instr.loc = loc
            else:
                loc = last_instr.loc

    def duplicate_exits_without_lineno(self):
        """
        PEP 626 mandates that the f_lineno of a frame is correct
        after a frame terminates. It would be prohibitively expensive
        to continuously update the f_lineno field at runtime,
        so we make sure that all exiting instruction (raises and returns)
        have a valid line number, allowing us to compute f_lineno lazily.
        We can do this by duplicating the exit blocks without line number
        so that none have more than one predecessor. We can then safely
        copy the line number from the sole predecessor block.
        """
        # Copy all exit blocks without line number that are targets of a jump.
        append_after = {}
        for block in self.blocks_in_reverse_allocation_order():
            if block.insts and (last := block.insts[-1]).is_jump(self.opcode):
                if last.opname in {"SETUP_ASYNC_WITH", "SETUP_WITH", "SETUP_FINALLY"}:
                    continue
                target = last.target
                assert target.insts
                if (
                    target.is_exit
                    and target.insts[0].lineno < 0
                    and target.num_predecessors > 1
                ):
                    new_target = target.copy()
                    new_target.insts[0].loc = last.loc
                    last.target = new_target
                    target.num_predecessors -= 1
                    new_target.num_predecessors = 1
                    new_target.next = target.next
                    target.next = new_target
                    new_target.prev = target
                    new_target.bid = self.block_count
                    self.block_count += 1
                    append_after.setdefault(target, []).append(new_target)
        for after, to_append in append_after.items():
            idx = self.ordered_blocks.index(after) + 1
            self.ordered_blocks[idx:idx] = reversed(to_append)

    def normalize_jumps(self):
        assert self.stage == ORDERED, self.stage

        seen_blocks = set()

        for block in self.ordered_blocks:
            seen_blocks.add(block.bid)
            if not block.insts:
                continue
            last = block.insts[-1]
            if last.opname == "JUMP_ABSOLUTE" and last.target.bid not in seen_blocks:
                last.opname = "JUMP_FORWARD"
            elif last.opname == "JUMP_FORWARD" and last.target.bid in seen_blocks:
                last.opname = "JUMP_ABSOLUTE"

    def optimizeCFG(self):
        """Optimize a well-formed CFG."""
        assert self.stage == CLOSED, self.stage

        optimizer = self.flow_graph_optimizer(self)
        for block in self.ordered_blocks:
            optimizer.optimize_basic_block(block)
            optimizer.clean_basic_block(block, -1)

        for block in self.blocks_in_reverse_allocation_order():
            self.extend_block(block)

        prev_block = None
        for block in self.ordered_blocks:
            prev_lineno = -1
            if prev_block and prev_block.insts:
                prev_lineno = prev_block.insts[-1].lineno
            optimizer.clean_basic_block(block, prev_lineno)
            prev_block = block if block.has_fallthrough else None

        self.eliminate_empty_basic_blocks()
        self.remove_unreachable_basic_blocks()

        # Delete jump instructions made redundant by previous step. If a non-empty
        # block ends with a jump instruction, check if the next non-empty block
        # reached through normal flow control is the target of that jump. If it
        # is, then the jump instruction is redundant and can be deleted.
        maybe_empty_blocks = False
        for block in self.ordered_blocks:
            if not block.insts:
                continue
            last = block.insts[-1]
            if last.opname not in {"JUMP_ABSOLUTE", "JUMP_FORWARD"}:
                continue
            if last.target == block.next:
                block.has_fallthrough = True
                last.opname = "NOP"
                last.oparg = last.ioparg = 0
                last.target = None
                optimizer.clean_basic_block(block, -1)
                maybe_empty_blocks = True

        if maybe_empty_blocks:
            self.eliminate_empty_basic_blocks()

        self.stage = OPTIMIZED

    def eliminate_empty_basic_blocks(self):
        for block in self.ordered_blocks:
            next_block = block.next
            if next_block:
                while not next_block.insts and next_block.next:
                    next_block = next_block.next
                block.next = next_block
        for block in self.ordered_blocks:
            if not block.insts:
                continue
            last = block.insts[-1]
            if last.is_jump(self.opcode):
                target = last.target
                while not target.insts and target.next:
                    target = target.next
                last.target = target

    def remove_unreachable_basic_blocks(self):
        # mark all reachable blocks
        reachable_blocks = set()
        worklist = [self.entry]
        while worklist:
            entry = worklist.pop()
            if entry.bid in reachable_blocks:
                continue
            reachable_blocks.add(entry.bid)
            for instruction in entry.getInstructions():
                target = instruction.target
                if target is not None:
                    worklist.append(target)
                    target.num_predecessors += 1

            if entry.has_fallthrough:
                worklist.append(entry.next)
                entry.next.num_predecessors += 1

        self.ordered_blocks = [
            block for block in self.ordered_blocks if block.bid in reachable_blocks
        ]
        prev = None
        for block in self.ordered_blocks:
            block.prev = prev
            if prev is not None:
                prev.next = block
            prev = block

    def normalize_basic_block(self, block: Block) -> None:
        """Sets the `fallthrough` and `exit` properties of a block, and ensures that the targets of
        any jumps point to non-empty blocks by following the next pointer of empty blocks.
        """
        for instr in block.getInstructions():
            # TODO(T128853358): The RETURN_PRIMITIVE logic should live in the Static flow graph.
            if instr.opname in SCOPE_EXIT_OPCODES:
                block.is_exit = True
                block.has_fallthrough = False
                continue
            elif instr.opname in UNCONDITIONAL_JUMP_OPCODES:
                block.has_fallthrough = False
            elif not instr.is_jump(self.opcode):
                continue
            while not instr.target.insts:
                instr.target = instr.target.next

    def extend_block(self, block: Block) -> None:
        """If this block ends with an unconditional jump to an exit block,
        then remove the jump and extend this block with the target.
        """
        if len(block.insts) == 0:
            return
        last = block.insts[-1]
        if last.opname not in ("JUMP_ABSOLUTE", "JUMP_FORWARD"):
            return
        target = last.target
        assert target is not None
        if not target.is_exit:
            return
        if len(target.insts) > MAX_COPY_SIZE:
            return
        last = block.insts[-1]
        last.opname = "NOP"
        last.oparg = last.ioparg = 0
        last.target = None
        for instr in target.insts:
            block.insts.append(instr.copy())
        block.next = None
        block.is_exit = True
        block.has_fallthrough = False

    def trim_unused_consts(self) -> None:
        """Remove trailing unused constants."""
        assert self.stage == CONSTS_CLOSED, self.stage

        max_const_index = 0
        for block in self.ordered_blocks:
            for instr in block.insts:
                if (
                    instr.opname in self._const_opcodes
                    and instr.ioparg > max_const_index
                ):
                    max_const_index = instr.ioparg
        self.consts = {
            key: index for key, index in self.consts.items() if index <= max_const_index
        }


class PyFlowGraphCinder(PyFlowGraph):
    opcode = opcode_cinder.opcode

    def make_code(
        self, nlocals, code, consts, firstline: int, lnotab, exception_table=None
    ) -> CodeType:
        if self.scope is not None and self.scope.suppress_jit:
            self.setFlag(CO_SUPPRESS_JIT)
        return super().make_code(nlocals, code, consts, firstline, lnotab)


class PyFlowGraph312(PyFlowGraph):
    flow_graph_optimizer = FlowGraphOptimizer312

    def __init__(
        self,
        name: str,
        filename: str,
        scope,
        flags: int = 0,
        args=(),
        kwonlyargs=(),
        starargs=(),
        optimized: int = 0,
        klass: bool = False,
        docstring: Optional[str] = None,
        firstline: int = 0,
        posonlyargs: int = 0,
        qualname: Optional[str] = None,
    ) -> None:
        super().__init__(
            name,
            filename,
            scope,
            flags,
            args,
            kwonlyargs,
            starargs,
            optimized,
            klass,
            docstring,
            firstline,
            posonlyargs,
        )
        self.qualname = qualname or name

    def emit_gen_start(self) -> None:
        # This is handled with the prefix instructions in finalize
        pass

    def compute_except_handlers(self) -> set[Block]:
        except_handlers: set[Block] = set()
        for block in self.ordered_blocks:
            for instr in block.insts:
                if instr.opname in SETUP_OPCODES:
                    except_handlers.add(instr.target)
                    break

        return except_handlers

    def push_cold_blocks_to_end(self, except_handlers: set[Block]) -> None:
        warm = self.compute_warm()

        # If we have a cold block with fallthrough to a warm block, add
        # an explicit jump instead of fallthrough
        for block in list(self.ordered_blocks):
            if block not in warm and block.has_fallthrough and block.next in warm:
                explicit_jump = self.newBlock("explicit_jump")
                explicit_jump.bid = self.block_count
                self.block_count += 1
                self.current = explicit_jump

                self.emit("JUMP_BACKWARD", block.next)
                self.ordered_blocks.insert(
                    self.ordered_blocks.index(block) + 1, explicit_jump
                )

                explicit_jump.next = block.next
                explicit_jump.has_fallthrough = False
                block.next = explicit_jump

        to_end = [block for block in self.ordered_blocks if block not in warm]

        self.ordered_blocks = [block for block in self.ordered_blocks if block in warm]

        self.ordered_blocks.extend(to_end)

    def compute_warm(self) -> set[Block]:
        """Compute the set of 'warm' blocks, which are blocks that are reachable
        through normal control flow. 'Cold' blocks are those that are not
        directly reachable and may be moved to the end of the block list
        for optimization purposes."""

        stack = [self.entry]
        visited = set(stack)
        warm: set[Block] = set()

        while stack:
            cur = stack.pop()
            warm.add(cur)
            next = cur.next
            if next is not None and cur.has_fallthrough and next not in visited:
                stack.append(next)
                visited.add(next)

            for instr in cur.insts:
                target = instr.target
                if (
                    target is not None
                    and instr.is_jump(self.opcode)
                    and target not in visited
                ):
                    stack.append(target)
                    visited.add(target)
        return warm

    def optimizeCFG(self) -> None:
        super().optimizeCFG()

        except_handlers = self.compute_except_handlers()

        self.push_cold_blocks_to_end(except_handlers)

    def finalize(self):
        if self.cellvars or self.freevars or self.gen_kind is not None:
            to_insert = []
            if self.gen_kind is not None:
                firstline = self.firstline or self.first_inst_lineno or 1
                to_insert.append(
                    Instruction(
                        "RETURN_GENERATOR",
                        0,
                        loc=SrcLocation(firstline, firstline, -1, -1),
                    )
                )
                to_insert.append(Instruction("POP_TOP", 0))

            if self.freevars:
                to_insert.append(Instruction("COPY_FREE_VARS", len(self.freevars)))

            if self.cellvars:
                # varnames come first
                offset = len(self.varnames)
                for i in range(len(self.cellvars)):
                    to_insert.append(Instruction("MAKE_CELL", offset + i))

            self.entry.insts[0:0] = to_insert

        super().finalize()

    def instrsize(self, opname: str, oparg: int):
        opcode_index = opcodes.opcode.opmap[opname]
        # pyre-ignore[16]: no _inline_cache_entries
        if opcode_index >= len(_inline_cache_entries):
            # T190611021: This should never happen as we should remove pseudo
            # instructions, but we are still missing some functionality
            # like zero-cost exceptions so we emit things like END_FINALLY
            base_size = 0
        else:
            # pyre-ignore[16]: no _inline_cache_entries
            base_size = _inline_cache_entries[opcode_index]
        if oparg <= 0xFF:
            return 1 + base_size
        elif oparg <= 0xFFFF:
            return 2 + base_size
        elif oparg <= 0xFFFFFF:
            return 3 + base_size
        else:
            return 4 + base_size

    def normalize_jumps(self):
        assert self.stage == ORDERED, self.stage

        seen_blocks = set()

        for block in self.ordered_blocks:
            seen_blocks.add(block.bid)

            if not block.insts:
                continue

            last = block.insts[-1]
            if last.opname == "JUMP":
                is_forward = last.target.bid not in seen_blocks
                last.opname = "JUMP_FORWARD" if is_forward else "JUMP_BACKWARD"

    def emit_inline_cache(
        self, opcode: str, addCode: Callable[[int, int], None]
    ) -> None:
        opcode_index = opcodes.opcode.opmap[opcode]
        # pyre-ignore[16]: no _inline_cache_entries
        if opcode_index < len(_inline_cache_entries):
            # pyre-ignore[16]: no _inline_cache_entries
            base_size = _inline_cache_entries[opcode_index]
        else:
            base_size = 0
        for _i in range(base_size):
            addCode(0, 0)

    def make_line_table(self) -> bytes:
        lpostab = LinePositionTable(self.firstline)

        loc = NO_LOCATION
        size = 0
        for t in self.insts:
            if t.loc != loc:
                lpostab.emit_location(loc, size)
                loc = t.loc
                size = 0

            # The size is in terms of code units
            size += self.instrsize(t.opname, t.ioparg)

        # Since the linetable format writes the end offset of bytecodes, we can't commit the
        # last write until all the instructions are iterated over.
        lpostab.emit_location(loc, size)
        return lpostab.getTable()

    def make_exception_table(self) -> bytes:
        exception_table = ExceptionTable()
        ioffset = 0
        handler = None
        start = -1
        for instr in self.insts:
            if instr.exc_handler and (
                not handler or instr.exc_handler.offset != handler.offset
            ):
                if handler:
                    exception_table.emit_entry(start, ioffset, handler)
                start = ioffset
                handler = instr.exc_handler
            ioffset += self.instrsize(instr.opname, instr.ioparg)
        if handler:
            exception_table.emit_entry(start, ioffset, handler)
        return exception_table.getTable()

    def make_code(
        self, nlocals, code, consts, firstline, lnotab, exception_table
    ) -> CodeType:
        # pyre-ignore[19]: Too many arguments (this is right for 3.12)
        return CodeType(
            len(self.args),
            self.posonlyargs,
            len(self.kwonlyargs),
            nlocals,
            self.stacksize,
            self.flags,
            code,
            consts,
            tuple(self.names),
            tuple(self.varnames),
            self.filename,
            self.name,
            self.qualname,
            firstline,
            lnotab,
            exception_table,
            tuple(self.freevars),
            tuple(self.cellvars),
        )

    def _convert_LOAD_ATTR(self, arg: object) -> int:
        # 3.12 uses the low-bit to indicate that the LOAD_ATTR is
        # part of a LOAD_ATTR/CALL sequence which loads two values,
        # the first being NULL or the object instance and the 2nd
        # being the method to be called.
        if isinstance(arg, tuple):
            return (self.names.get_index(arg[0]) << 1) | arg[1]

        return self.names.get_index(arg) << 1

    _converters = {
        **PyFlowGraph._converters,
        "LOAD_ATTR": _convert_LOAD_ATTR,
    }


class LineAddrTable:
    """linetable / lnotab

    This class builds the linetable, which is documented in
    Objects/lnotab_notes.txt. Here's a brief recap:

    For each new lineno after the first one, two bytes are added to the
    linetable.  (In some cases, multiple two-byte entries are added.)  The first
    byte is the distance in bytes between the instruction for the current lineno
    and the next lineno.  The second byte is offset in line numbers.  If either
    offset is greater than 255, multiple two-byte entries are added -- see
    lnotab_notes.txt for the delicate details.

    """

    def __init__(self) -> None:
        self.current_line = 0
        self.prev_line = 0
        self.linetable = []

    def setFirstLine(self, lineno: int) -> None:
        self.current_line = lineno
        self.prev_line = lineno

    def nextLine(self, lineno: int, start: int, end: int) -> None:
        assert lineno
        self.emitCurrentLine(start, end)

        if self.current_line >= 0:
            self.prev_line = self.current_line
        self.current_line = lineno

    def emitCurrentLine(self, start: int, end: int) -> None:
        # compute deltas
        addr_delta = end - start
        if not addr_delta:
            return
        if self.current_line < 0:
            line_delta = -128
        else:
            line_delta = self.current_line - self.prev_line
            while line_delta < -127 or 127 < line_delta:
                if line_delta < 0:
                    k = -127
                else:
                    k = 127
                self.push_entry(0, k)
                line_delta -= k

        while addr_delta > 254:
            self.push_entry(254, line_delta)
            line_delta = -128 if self.current_line < 0 else 0
            addr_delta -= 254

        assert -128 <= line_delta and line_delta <= 127
        self.push_entry(addr_delta, line_delta)

    def getTable(self) -> bytes:
        return bytes(self.linetable)

    def push_entry(self, addr_delta, line_delta):
        self.linetable.append(addr_delta)
        self.linetable.append(cast_signed_byte_to_unsigned(line_delta))


class CodeLocationInfoKind(IntEnum):
    SHORT0 = 0
    ONE_LINE0 = 10
    ONE_LINE1 = 11
    ONE_LINE2 = 12
    NO_COLUMNS = 13
    LONG = 14
    NONE = 15


class LinePositionTable:
    """Generates the Python 3.12 and later position table which tracks
    line numbers as well as column information."""

    def __init__(self, firstline: int) -> None:
        self.linetable = bytearray()
        self.lineno = firstline

    # https://github.com/python/cpython/blob/3.12/Python/assemble.c#L170
    # https://github.com/python/cpython/blob/3.12/Objects/locations.md
    def emit_location(self, loc: AST | SrcLocation, size: int) -> None:
        if size == 0:
            return

        while size > 8:
            self.write_entry(loc, 8)
            size -= 8

        self.write_entry(loc, size)

    def write_entry(self, loc: AST | SrcLocation, size: int) -> None:
        if loc.lineno < 0:
            return self.write_entry_no_location(size)

        line_delta = loc.lineno - self.lineno
        column = loc.col_offset
        end_column = loc.end_col_offset
        assert isinstance(end_column, int)
        assert column >= -1
        assert end_column >= -1
        if column < 0 or end_column < 0:
            if loc.end_lineno == loc.lineno or loc.end_lineno == -1:
                self.write_no_column(size, line_delta)
                self.lineno = loc.lineno
                return
        elif loc.end_lineno == loc.lineno:
            if (
                line_delta == 0
                and column < 80
                and end_column - column < 16
                and end_column >= column
            ):
                return self.write_short_form(size, column, end_column)
            if 0 <= line_delta < 3 and column < 128 and end_column < 128:
                self.write_one_line_form(size, line_delta, column, end_column)
                self.lineno = loc.lineno
                return

        self.write_long_form(loc, size)
        self.lineno = loc.lineno

    def write_short_form(self, size: int, column: int, end_column: int) -> None:
        assert size > 0 and size <= 8
        column_low_bits = column & 7
        column_group = column >> 3
        assert column < 80
        assert end_column >= column
        assert end_column - column < 16
        self.write_first_byte(CodeLocationInfoKind.SHORT0 + column_group, size)
        # Start column / end column
        self.write_byte((column_low_bits << 4) | (end_column - column))

    def write_one_line_form(
        self, size: int, line_delta: int, column: int, end_column: int
    ) -> None:
        assert size > 0 and size <= 8
        assert line_delta >= 0 and line_delta < 3
        assert column < 128
        assert end_column < 128
        # Start line delta
        self.write_first_byte(CodeLocationInfoKind.ONE_LINE0 + line_delta, size)
        self.write_byte(column)  # Start column
        self.write_byte(end_column)  # End column

    def write_no_column(self, size: int, line_delta: int) -> None:
        self.write_first_byte(CodeLocationInfoKind.NO_COLUMNS, size)
        self.write_signed_varint(line_delta)  # Start line delta

    def write_long_form(self, loc: AST | SrcLocation, size: int) -> None:
        end_lineno = loc.end_lineno
        end_col_offset = loc.end_col_offset

        assert size > 0 and size <= 8
        assert end_lineno is not None and end_col_offset is not None
        assert end_lineno >= loc.lineno

        self.write_first_byte(CodeLocationInfoKind.LONG, size)
        self.write_signed_varint(loc.lineno - self.lineno)  # Start line delta
        self.write_varint(end_lineno - loc.lineno)  # End line delta
        self.write_varint(loc.col_offset + 1)  # Start column
        self.write_varint(end_col_offset + 1)  # End column

    def write_entry_no_location(self, size: int) -> None:
        self.write_first_byte(CodeLocationInfoKind.NONE, size)

    def write_varint(self, value: int) -> None:
        while value >= 64:
            self.linetable.append(0x40 | (value & 0x3F))
            value >>= 6

        self.linetable.append(value)

    def write_signed_varint(self, value: int) -> None:
        if value < 0:
            uval = ((-value) << 1) | 1
        else:
            uval = value << 1

        self.write_varint(uval)

    def write_first_byte(self, code: int, length: int) -> None:
        assert code & 0x0F == code
        self.linetable.append(0x80 | (code << 3) | (length - 1))

    def write_byte(self, code: int) -> None:
        self.linetable.append(code & 0xFF)

    def getTable(self) -> bytes:
        return bytes(self.linetable)


class ExceptionTable:
    """Generates the Python 3.12+ exception table."""

    # https://github.com/python/cpython/blob/3.12/Objects/exception_handling_notes.txt

    def __init__(self):
        self.exception_table = bytearray()

    def getTable(self) -> bytes:
        return bytes(self.exception_table)

    def write_byte(self, byte: int) -> None:
        self.exception_table.append(byte)

    def emit_item(self, value: int, msb: int) -> None:
        assert (msb | 128) == 128
        assert value >= 0 and value < (1 << 30)
        CONTINUATION_BIT = 64

        if value > (1 << 24):
            self.write_byte((value >> 24) | CONTINUATION_BIT | msb)
            msb = 0
        for i in (18, 12, 6):
            if value >= (1 << i):
                v = (value >> i) & 0x3F
                self.write_byte(v | CONTINUATION_BIT | msb)
                msb = 0
        self.write_byte((value & 0x3F) | msb)

    def emit_entry(self, start: int, end: int, handler: Block) -> None:
        size = end - start
        assert end > start
        target = handler.offset
        depth = handler.startdepth - 1
        if handler.preserve_lasti:
            depth = depth - 1
        assert depth >= 0
        depth_lasti = (depth << 1) | int(handler.preserve_lasti)
        self.emit_item(start, (1 << 7))
        self.emit_item(size, 0)
        self.emit_item(target, 0)
        self.emit_item(depth_lasti, 0)
