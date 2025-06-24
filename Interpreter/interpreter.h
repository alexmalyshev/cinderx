// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <Python.h>

#include "cinderx/Common/extra-py-flags.h"
#if PY_VERSION_HEX >= 0x030D0000
#include "internal/pycore_function.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

#if PY_VERSION_HEX < 0x030C0000
PyObject* _Py_HOT_FUNCTION
Ci_EvalFrame(PyThreadState* tstate, PyFrameObject* f, int throwflag);
#else
PyObject* _Py_HOT_FUNCTION Ci_EvalFrame(
    PyThreadState* tstate,
    struct _PyInterpreterFrame* f,
    int throwflag);
#endif

/*
 * General vectorcall entry point to a function compiled by the Static Python
 * compiler.  The function will be executed in the interpreter.
 */
PyObject* Ci_StaticFunction_Vectorcall(
    PyObject* func,
    PyObject* const* stack,
    size_t nargsf,
    PyObject* kwnames);

/*
 * Optimized form of Ci_StaticFunction_Vectorcall, where all arguments are
 * guaranteed to have the correct type and do not use `kwnames`.
 */
PyObject* Ci_PyFunction_CallStatic(
    PyFunctionObject* func,
    PyObject* const* args,
    size_t nargsf,
    PyObject* kwnames);

/*
 * Get the default vectorcall entrypoint for Python functions.
 */
static inline vectorcallfunc getDefaultInterpretedVectorcall(
    [[maybe_unused]] const PyFunctionObject* func) {
#if PY_VERSION_HEX >= 0x030D0000
  return PyVectorcall_Function((PyObject*)func);
#else
  return _PyFunction_Vectorcall;
#endif
}

/*
 * Get the appropriate entry point that will execute a function object in the
 * interpreter.
 *
 * This is a different function for Static Python functions versus "normal"
 * Python functions.
 */
static inline vectorcallfunc getInterpretedVectorcall(
    const PyFunctionObject* func) {
#ifdef ENABLE_INTERPRETER
  const PyCodeObject* code = (const PyCodeObject*)(func->func_code);
  if (code->co_flags & CI_CO_STATICALLY_COMPILED) {
    return Ci_StaticFunction_Vectorcall;
  }
#endif

  return getDefaultInterpretedVectorcall(func);
}

void Ci_InitOpcodes();

extern bool Ci_DelayAdaptiveCode;
extern uint64_t Ci_AdaptiveThreshold;

#ifdef __cplusplus
}
#endif
