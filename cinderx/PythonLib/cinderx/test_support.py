# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

import abc
import ctypes
import dis
import importlib
import multiprocessing
import os.path
import sys
import tempfile
import unittest

from contextlib import contextmanager
from pathlib import Path

from typing import Callable, Coroutine, Generator, Sequence, TypeVar

import cinderx
import cinderx.jit

try:
    import cinder

    def hasCinderX() -> bool:
        return True

except ImportError:

    def hasCinderX() -> bool:
        return False


# String encoding to use for subprocesses.
ENCODING: str = sys.stdout.encoding or sys.getdefaultencoding()

# Hack to allow subprocesses to find where the cinderx module is.
CINDERX_PATH: str = os.path.dirname(os.path.dirname(cinderx.__file__))


def get_cinderjit_xargs() -> list[str]:
    args = []
    for k, v in sys._xoptions.items():
        if not k.startswith("jit"):
            continue
        elif v is True:
            args.extend(["-X", k])
        else:
            args.extend(["-X", f"{k}={v}"])
    return args


TYield = TypeVar("TYield")
TSend = TypeVar("TSend")
TReturn = TypeVar("TReturn")


def get_await_stack(
    coro: Coroutine[TYield, TSend, TReturn],
) -> list[Coroutine[TYield, TSend, TReturn]]:
    """Return the chain of coroutines reachable from coro via its awaiter"""

    stack = []
    awaiter = cinder._get_coro_awaiter(coro)
    while awaiter is not None:
        stack.append(awaiter)

        # pyre-ignore[1001]
        awaiter = cinder._get_coro_awaiter(awaiter)
    return stack


def verify_stack(
    testcase: unittest.TestCase, stack: Sequence[str], expected: Sequence[str]
) -> None:
    n = len(expected)
    frames = stack[-n:]
    testcase.assertEqual(len(frames), n, "Callstack had less frames than expected")

    for actual, exp in zip(frames, expected):
        testcase.assertTrue(
            actual.endswith(exp),
            f"The actual frame {actual} doesn't refer to the expected function {exp}",
        )


def compiles_after_one_call() -> bool:
    """
    Check if CinderX will automatically compile functions after they are called once.
    """
    return cinderx.jit.get_compile_after_n_calls() == 0


def skip_if_jit(reason: str) -> Callable[[Callable[..., None]], Callable[..., None]]:
    return unittest.skipIf(cinderx.jit.is_enabled(), reason)


def skip_unless_jit(
    reason: str,
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    return unittest.skipUnless(cinderx.jit.is_enabled(), reason)


def skip_unless_lazy_imports(
    reason: str = "Depends on Lazy Imports being enabled",
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    return unittest.skipUnless(hasattr(importlib, "set_lazy_imports"), reason)


TRet = TypeVar("TRet")


def failUnlessJITCompiled(func: Callable[..., TRet]) -> Callable[..., TRet]:
    """
    Fail a test if the JIT is enabled but the test body wasn't JIT-compiled.
    """
    if not cinderx.jit.is_enabled():
        return func

    try:
        # force_compile raises a RuntimeError if compilation fails. If it does,
        # defer raising an exception to when the decorated function runs.
        cinderx.jit.force_compile(func)
    except RuntimeError as re:
        if re.args == ("PYJIT_RESULT_NOT_ON_JITLIST",):
            # We generally only run tests with a jitlist under
            # Tools/scripts/jitlist_bisect.py. In that case, we want to allow
            # the decorated function to run under the interpreter to determine
            # if it's the function the JIT is handling incorrectly.
            return func

        # re is cleared at the end of the except block but we need the value
        # when wrapper() is eventually called.
        exc: RuntimeError = re

        def wrapper(*args: ...) -> None:
            raise RuntimeError(
                f"JIT compilation of {func.__qualname__} failed with {exc}"
            )

        return wrapper

    return func


def fail_if_deopt(func: Callable[..., TRet]) -> Callable[..., TRet]:
    """
    Raise a RuntimeException if _any_ deopts occur during execution of the
    wrapped function. Note deopts occuring in nested function calls will also
    trigger this. Also, execution will run to completion - it won't stop at the
    point a deopt occurs.
    """

    if not cinderx.jit.is_enabled():
        return func

    def wrapper(*args: ..., **kwargs: ...) -> TRet:
        cinderx.jit.get_and_clear_runtime_stats()
        r = func(*args, **kwargs)
        # pyre-ignore[6]
        if len(deopts := cinderx.jit.get_and_clear_runtime_stats()["deopt"]):
            raise RuntimeError(f"Deopt occured {deopts}")
        return r

    wrapper.inner_function = func

    return wrapper


def is_asan_build() -> bool:
    try:
        ctypes.pythonapi.__asan_init
        return True
    except AttributeError:
        return False


# This is long because ASAN + JIT + subprocess + the Python compiler can be
# pretty slow in CI.
SUBPROCESS_TIMEOUT_SEC = 100 if is_asan_build() else 5


@contextmanager
def temp_sys_path() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        _orig_sys_modules = sys.modules
        sys.modules = _orig_sys_modules.copy()
        _orig_sys_path = sys.path[:]
        sys.path.insert(0, tmpdir)
        try:
            yield Path(tmpdir)
        finally:
            sys.path[:] = _orig_sys_path
            sys.modules = _orig_sys_modules


def run_in_subprocess(func: Callable[..., TRet]) -> Callable[..., TRet]:
    """
    Run a function in a subprocess.  This enables modifying process state in a
    test without affecting other test functions.
    """

    queue: multiprocessing.Queue = multiprocessing.Queue()

    def wrapper(queue: multiprocessing.Queue, *args: object) -> None:
        result = func(*args)
        queue.put(result, timeout=SUBPROCESS_TIMEOUT_SEC)

    def wrapped(*args: object) -> TRet:
        p = multiprocessing.Process(target=wrapper, args=(queue, *args))
        p.start()
        value = queue.get(timeout=SUBPROCESS_TIMEOUT_SEC)
        p.join(timeout=SUBPROCESS_TIMEOUT_SEC)
        return value

    return wrapped


class AssertBytecodeContainsMixin(abc.ABC):
    @abc.abstractmethod
    def assertIn(
        self, expected: object, actual: Sequence[object], msg: str | None = None
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def assertTrue(self, expr: object, msg: str | None = None) -> None:
        raise NotImplementedError

    def assertBytecodeContains(
        self,
        func: object,
        expected_opcode: str,
        expected_oparg: int | None = None,
    ) -> None:
        try:
            # pyre-ignore[16] - for things wrapped by fail_if_deopt()
            inner_function = func.inner_function
        except AttributeError:
            pass
        else:
            func = inner_function

        bytecode_instructions = dis.get_instructions(func)

        if expected_oparg is None:
            opcodes = [instr.opname for instr in bytecode_instructions]
            self.assertIn(
                expected_opcode,
                opcodes,
                f"{expected_opcode} opcode should be present in {func.__name__} bytecode",
            )
        else:
            matching_instructions = [
                instr
                for instr in bytecode_instructions
                if instr.opname == expected_opcode and instr.arg == expected_oparg
            ]
            self.assertTrue(
                len(matching_instructions) > 0,
                f"{expected_opcode} opcode with oparg {expected_oparg} should be present in {func.__name__} bytecode",
            )
