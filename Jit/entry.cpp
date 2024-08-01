// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/entry.h"

#include "cinderx/Common/extra-py-flags.h"
#include "cinderx/Interpreter/interpreter.h"

#include "cinderx/Jit/pyjit.h"

#include <Python.h>

#if PY_VERSION_HEX < 0x030C0000
#include "cinderx/Shadowcode/shadowcode.h"
#include "pycore_interp.h"
#endif

namespace jit {
namespace {

void initFunctionObjectForStaticOrNonJIT(PyFunctionObject* func) {
  // Check that func hasn't already been initialized.
  JIT_DCHECK(
      func->vectorcall ==
          reinterpret_cast<vectorcallfunc>(
              Ci_JIT_lazyJITInitFuncObjectVectorcall),
      "Double initializing function {}",
      repr(func->func_qualname));
  if (((PyCodeObject*)func->func_code)->co_flags & CI_CO_STATICALLY_COMPILED) {
    func->vectorcall =
        reinterpret_cast<vectorcallfunc>(Ci_StaticFunction_Vectorcall);
  } else {
    func->vectorcall = reinterpret_cast<vectorcallfunc>(_PyFunction_Vectorcall);
  }
}

unsigned int countCalls(PyCodeObject* code) {
#if PY_VERSION_HEX < 0x030C0000
  // The interpreter will only increment up to the shadowcode threshold
  // PYSHADOW_INIT_THRESHOLD. After that, it will stop incrementing. If someone
  // sets -X jit-auto above the PYSHADOW_INIT_THRESHOLD, we still have to keep
  // counting.
  unsigned int ncalls = code->co_mutable->ncalls;
  if (ncalls > PYSHADOW_INIT_THRESHOLD) {
    ncalls++;
    code->co_mutable->ncalls = ncalls;
  }
  return ncalls;
#else
  UPGRADE_ASSERT(CHANGED_PYCODEOBJECT);
  return 0;
#endif
}

PyObject* autoJITFuncObjectVectorcall(
    PyFunctionObject* func,
    PyObject** stack,
    Py_ssize_t nargsf,
    PyObject* kwnames) {
  auto code = reinterpret_cast<PyCodeObject*>(func->func_code);
  auto func_obj = reinterpret_cast<PyObject*>(func);

  // Interpret function as usual until it passes the call count threshold.
  if (countCalls(code) <= getConfig().auto_jit_threshold) {
    return _PyFunction_Vectorcall(func_obj, stack, nargsf, kwnames);
  }

  _PyJIT_Result result = _PyJIT_CompileFunction(func);
  if (result == PYJIT_RESULT_PYTHON_EXCEPTION) {
    return nullptr;
  } else if (result != PYJIT_RESULT_OK) {
    func->vectorcall = reinterpret_cast<vectorcallfunc>(
        Ci_JIT_lazyJITInitFuncObjectVectorcall);
    initFunctionObjectForStaticOrNonJIT(func);
  }
  JIT_DCHECK(
      func->vectorcall !=
          reinterpret_cast<vectorcallfunc>(autoJITFuncObjectVectorcall),
      "Auto-JIT left function as auto-JIT'able on {}",
      repr(func->func_qualname));
  return func->vectorcall(func_obj, stack, nargsf, kwnames);
}

} // namespace

void initFunctionObjectForJIT(PyFunctionObject* func) {
  JIT_DCHECK(
      !_PyJIT_IsCompiled(func),
      "Function {} is already compiled",
      repr(func->func_qualname));
  if (_PyJIT_IsAutoJITEnabled()) {
    func->vectorcall =
        reinterpret_cast<vectorcallfunc>(autoJITFuncObjectVectorcall);
    return;
  }
  func->vectorcall =
      reinterpret_cast<vectorcallfunc>(Ci_JIT_lazyJITInitFuncObjectVectorcall);
  if (!_PyJIT_RegisterFunction(func)) {
    initFunctionObjectForStaticOrNonJIT(func);
  }
}

} // namespace jit

PyObject* Ci_JIT_lazyJITInitFuncObjectVectorcall(
    PyFunctionObject* func,
    PyObject** stack,
    Py_ssize_t nargsf,
    PyObject* kwnames) {
  if (!_PyJIT_IsEnabled()) {
    jit::initFunctionObjectForStaticOrNonJIT(func);
  } else {
    _PyJIT_Result result = _PyJIT_CompileFunction(func);
    if (result == PYJIT_RESULT_PYTHON_EXCEPTION) {
      return nullptr;
    } else if (result != PYJIT_RESULT_OK) {
      jit::initFunctionObjectForStaticOrNonJIT(func);
    }
  }
  JIT_DCHECK(
      func->vectorcall !=
          reinterpret_cast<vectorcallfunc>(
              Ci_JIT_lazyJITInitFuncObjectVectorcall),
      "Lazy JIT left function as lazy-JIT'able on {}",
      jit::repr(func->func_qualname));
  return func->vectorcall((PyObject*)func, stack, nargsf, kwnames);
}
