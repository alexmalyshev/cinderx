# Portions copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

# operation flags
OP_ASSIGN = "OP_ASSIGN"
OP_DELETE = "OP_DELETE"
OP_APPLY = "OP_APPLY"

SC_LOCAL = 1
SC_GLOBAL_IMPLICIT = 2
SC_GLOBAL_EXPLICIT = 3
SC_FREE = 4
SC_CELL = 5
SC_TYPE_PARAM = 9
SC_UNKNOWN = 6


CO_OPTIMIZED = 0x0001
CO_NEWLOCALS = 0x0002
CO_VARARGS = 0x0004
CO_VARKEYWORDS = 0x0008
CO_NESTED = 0x0010
CO_GENERATOR = 0x0020
CO_NOFREE = 0x0040
CO_COROUTINE = 0x0080
CO_GENERATOR_ALLOWED = 0
CO_ITERABLE_COROUTINE = 0x0100
CO_ASYNC_GENERATOR = 0x0200
CO_FUTURE_DIVISION = 0x20000
CO_FUTURE_ABSOLUTE_IMPORT = 0x40000
CO_FUTURE_WITH_STATEMENT = 0x80000
CO_FUTURE_PRINT_FUNCTION = 0x100000
CO_FUTURE_UNICODE_LITERALS = 0x200000
CO_FUTURE_BARRY_AS_BDFL = 0x400000
CO_FUTURE_GENERATOR_STOP = 0x800000
CO_FUTURE_ANNOTATIONS = 0x1000000
CI_CO_STATICALLY_COMPILED = 0x4000000
CO_SUPPRESS_JIT = 0x40000000

PyCF_MASK_OBSOLETE: int = CO_NESTED
PyCF_SOURCE_IS_UTF8 = 0x0100
PyCF_DONT_IMPLY_DEDENT = 0x0200
PyCF_ONLY_AST = 0x0400
PyCF_IGNORE_COOKIE = 0x0800
PyCF_TYPE_COMMENTS = 0x1000
PyCF_ALLOW_TOP_LEVEL_AWAIT = 0x2000
PyCF_COMPILE_MASK: int = (
    PyCF_ONLY_AST
    | PyCF_ALLOW_TOP_LEVEL_AWAIT
    | PyCF_TYPE_COMMENTS
    | PyCF_DONT_IMPLY_DEDENT
)

PyCF_MASK: int = (
    CO_FUTURE_DIVISION
    | CO_FUTURE_ABSOLUTE_IMPORT
    | CO_FUTURE_WITH_STATEMENT
    | CO_FUTURE_PRINT_FUNCTION
    | CO_FUTURE_UNICODE_LITERALS
    | CO_FUTURE_BARRY_AS_BDFL
    | CO_FUTURE_GENERATOR_STOP
    | CO_FUTURE_ANNOTATIONS
    | CI_CO_STATICALLY_COMPILED
)
