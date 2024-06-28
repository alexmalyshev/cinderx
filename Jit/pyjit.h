// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <Python.h>
#include "frameobject.h"

#include "cinderx/Jit/pyjit_result.h"
#include "cinderx/Jit/pyjit_typeslots.h"

#ifdef __cplusplus
#include "cinderx/Jit/hir/preload.h"
#endif

#ifndef Py_LIMITED_API
#ifdef __cplusplus
extern "C" {
#endif

/*
 * This defines the global public API for the JIT that is consumed by the
 * runtime.
 *
 * These methods assume that the GIL is held unless it is explicitly stated
 * otherwise.
 */

/*
 * Initialize any global state required by the JIT.
 *
 * This must be called before attempting to use the JIT.
 *
 * Returns 0 on success, -1 on error, or -2 if we just printed the jit args.
 */
PyAPI_FUNC(int) _PyJIT_Initialize(void);

/*
 * Enable the global JIT.
 *
 * _PyJIT_Initialize must be called before calling this.
 *
 * Returns 1 if the JIT is enabled and 0 otherwise.
 */
PyAPI_FUNC(int) _PyJIT_Enable(void);

/*
 * Disable the global JIT.
 */
PyAPI_FUNC(void) _PyJIT_Disable(void);

/*
 * Returns 1 if JIT compilation is enabled and 0 otherwise.
 */
PyAPI_FUNC(int) _PyJIT_IsEnabled(void);

/*
 * Returns 1 if auto-JIT is enabled and 0 otherwise.
 */
PyAPI_FUNC(int) _PyJIT_IsAutoJITEnabled(void);

/*
 * Get the number of calls needed to mark a function for compilation by AutoJIT.
 * Returns 0 when AutoJIT is disabled.
 */
PyAPI_FUNC(unsigned) _PyJIT_AutoJITThreshold(void);

/*
 * Get the number of calls needed to profile hot functions when using AutoJIT.
 * This is added on top of the normal threshold.  Returns 0 when AutoJIT is
 * disabled.
 */
PyAPI_FUNC(unsigned) _PyJIT_AutoJITProfileThreshold(void);

/*
 * JIT compile func and patch its entry point.
 *
 * On success, positional only calls to func will use the JIT compiled version.
 *
 * Returns PYJIT_RESULT_OK on success.
 */
PyAPI_FUNC(_PyJIT_Result) _PyJIT_CompileFunction(PyFunctionObject* func);

/*
 * Registers a function with the JIT to be compiled in the future.
 *
 * The JIT will still be informed by _PyJIT_CompileFunction before the
 * function executes for the first time.  The JIT can choose to compile
 * the function at some future point.  Currently the JIT will compile
 * the function before it shuts down to make sure all eligable functions
 * were compiled.
 *
 * The JIT will not keep the function alive, instead it will be informed
 * that the function is being de-allocated via _PyJIT_UnregisterFunction
 * before the function goes away.
 *
 * Returns 1 if the function is registered with JIT or is already compiled,
 * and 0 otherwise.
 */
PyAPI_FUNC(int) _PyJIT_RegisterFunction(PyFunctionObject* func);

/*
 * Informs the JIT that a type, function, or code object is being created,
 * modified, or destroyed.
 */
PyAPI_FUNC(void) _PyJIT_TypeCreated(PyTypeObject* type);
PyAPI_FUNC(void) _PyJIT_TypeModified(PyTypeObject* type);
PyAPI_FUNC(void) _PyJIT_TypeNameModified(PyTypeObject* type);
PyAPI_FUNC(void) _PyJIT_TypeDestroyed(PyTypeObject* type);
PyAPI_FUNC(void) _PyJIT_FuncModified(PyFunctionObject* func);
PyAPI_FUNC(void) _PyJIT_FuncDestroyed(PyFunctionObject* func);
PyAPI_FUNC(void) _PyJIT_CodeDestroyed(PyCodeObject* code);

/*
 * Clean up any resources allocated by the JIT.
 *
 * This is intended to be called at interpreter shutdown in Py_Finalize.
 *
 * Returns 0 on success or -1 on error.
 */
PyAPI_FUNC(int) _PyJIT_Finalize(void);

/* Dict-watching callbacks, invoked by dictobject.c when appropriate. */

/*
 * Gets the global cache for the given builtins and globals dictionaries and
 * key.  The global that is pointed to will automatically be updated as
 * builtins and globals change.  The value that is pointed to will be NULL if
 * the dictionaries can no longer be tracked or if the value is no longer
 * defined, in which case the dictionaries need to be consulted.  This will
 * return NULL if the required tracking cannot be initialized.
 */
PyAPI_FUNC(PyObject**)
    _PyJIT_GetGlobalCache(PyObject* builtins, PyObject* globals, PyObject* key);

/*
 * Gets the cache for the given dictionary and key.  The value that is pointed
 * to will automatically be updated as the dictionary changes.  The value that
 * is pointed to will be NULL if the dictionaries can no longer be tracked or if
 * the value is no longer defined, in which case the dictionaries need to be
 * consulted.  This will return NULL if the required tracking cannot be
 * initialized.
 */
PyAPI_FUNC(PyObject**) _PyJIT_GetDictCache(PyObject* dict, PyObject* key);

/*
 * Send into/resume a suspended JIT generator and return the result.
 */
PyAPI_FUNC(PyObject*) _PyJIT_GenSend(
    PyGenObject* gen,
    PyObject* arg,
    int exc,
    PyFrameObject* f,
    PyThreadState* tstate,
    int finish_yield_from);

/*
 * Materialize the frame for gen. Returns a borrowed reference.
 */
PyAPI_FUNC(PyFrameObject*) _PyJIT_GenMaterializeFrame(PyGenObject* gen);

/*
 * Visit owned references in a JIT-backed generator object.
 */
PyAPI_FUNC(int)
    _PyJIT_GenVisitRefs(PyGenObject* gen, visitproc visit, void* arg);

/*
 * Release any JIT-related data in a PyGenObject.
 */
PyAPI_FUNC(void) _PyJIT_GenDealloc(PyGenObject* gen);

/*
 * Return current sub-iterator from JIT generator or NULL if there is none.
 */
PyAPI_FUNC(PyObject*) _PyJIT_GenYieldFromValue(PyGenObject* gen);

/*
 * Specifies the offset from a JITed function entry point where the re-entry
 * point for calling with the correct bound args lives */
#define JITRT_CALL_REENTRY_OFFSET (-6)

/*
 * Fixes the JITed function entry point up to be the re-entry point after
 * binding the args */
#define JITRT_GET_REENTRY(entry) \
  ((vectorcallfunc)(((char*)entry) + JITRT_CALL_REENTRY_OFFSET))

/*
 * Specifies the offset from a JITed function entry point where the static
 * entry point lives */
#if defined(_M_X64) || defined(_M_AMD64) || defined(__x86_64__)
#define JITRT_STATIC_ENTRY_OFFSET (-11)
#else
/* Without JIT support there's no entry offset */
#define JITRT_STATIC_ENTRY_OFFSET (0)
#endif

/*
 * Fixes the JITed function entry point up to be the static entry point after
 * binding the args */
#define JITRT_GET_STATIC_ENTRY(entry) \
  ((vectorcallfunc)(((char*)entry) + JITRT_STATIC_ENTRY_OFFSET))

/*
 * Fixes the JITed function entry point up to be the static entry point after
 * binding the args */
#define JITRT_GET_NORMAL_ENTRY_FROM_STATIC(entry) \
  ((vectorcallfunc)(((char*)entry) - JITRT_STATIC_ENTRY_OFFSET))

/*
 * Checks if the given function is JITed.

 * Returns 1 if the function is JITed, 0 if not.
 */
PyAPI_FUNC(int) _PyJIT_IsCompiled(PyFunctionObject* func);

/*
 * Returns a borrowed reference to the globals for the top-most Python function
 * associated with tstate.
 */
PyAPI_FUNC(PyObject*) _PyJIT_GetGlobals(PyThreadState* tstate);

/*
 * Returns a borrowed reference to the builtins for the top-most Python function
 * associated with tstate.
 */
PyAPI_FUNC(PyObject*) _PyJIT_GetBuiltins(PyThreadState* tstate);

/*
 * Check if a code object should be profiled for type information.
 */
PyAPI_FUNC(int) _PyJIT_IsProfilingCandidate(PyCodeObject* code);

/*
 * Count the number of code objects currently marked as profiling candidates.
 */
PyAPI_FUNC(unsigned) _PyJIT_NumProfilingCandidates(void);

/*
 * Mark a code object as a good candidate for type profiling.
 */
PyAPI_FUNC(void) _PyJIT_MarkProfilingCandidate(PyCodeObject* code);

/*
 * Unmark a code object as a good candidate for type profiling
 */
PyAPI_FUNC(void) _PyJIT_UnmarkProfilingCandidate(PyCodeObject* code);

/*
 * Record a type profile for the current instruction.
 */
PyAPI_FUNC(void) _PyJIT_ProfileCurrentInstr(
    PyFrameObject* frame,
    PyObject** stack_top,
    int opcode,
    int oparg);

/*
 * Record profiled instructions for the given code object upon exit from a
 * frame, some of which may not have had their types recorded.
 */
PyAPI_FUNC(void)
    _PyJIT_CountProfiledInstrs(PyCodeObject* code, Py_ssize_t count);

/*
 * Get and clear, or just clear, information about the recorded type profiles.
 */
PyAPI_FUNC(PyObject*) _PyJIT_GetAndClearTypeProfiles(void);
PyAPI_FUNC(void) _PyJIT_ClearTypeProfiles(void);

/*
 * Returns a borrowed reference to the top-most frame of tstate.
 *
 * When shadow frame mode is active, calling this function will materialize
 * PyFrameObjects for any jitted functions on the call stack.
 */
PyAPI_FUNC(PyFrameObject*) _PyJIT_GetFrame(PyThreadState* tstate);

/*
 * Set output format for function disassembly. E.g. with -X jit-disas-funcs.
 */
PyAPI_FUNC(void) _PyJIT_SetDisassemblySyntaxATT(void);
PyAPI_FUNC(int) _PyJIT_IsDisassemblySyntaxIntel(void);

PyAPI_FUNC(void) _PyJIT_SetProfileNewInterpThreads(int);
PyAPI_FUNC(int) _PyJIT_GetProfileNewInterpThreads(void);

PyAPI_FUNC(int) _PyPerfTrampoline_IsPreforkCompilationEnabled(void);
PyAPI_FUNC(void) _PyPerfTrampoline_CompilePerfTrampolinePreFork(void);

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus
namespace jit {

/*
 * JIT compile func or code object, only if a preloader is available.
 *
 * Re-entrant compile that is safe to call from within compilation, because it
 * will only use an already-created preloader, it will not preload, and
 * therefore it cannot raise a Python exception.
 *
 * Returns PYJIT_RESULT_NO_PRELOADER if no preloader is available.
 */
_PyJIT_Result tryCompilePreloaded(BorrowedRef<> unit);

/*
 * Load the preloader for a given function or code object, if it exists.
 */
hir::Preloader* lookupPreloader(BorrowedRef<> unit);

/*
 * Check if a function or code object has been preloaded.
 */
bool isPreloaded(BorrowedRef<> unit);

/*
 * Preload given function and its compilation dependencies.
 *
 * Dependencies are functions that this function statically invokes (so we want
 * to ensure they are compiled first so we can emit a direct x64 call), and any
 * functions we can detect that this function may call, so they can potentially
 * be inlined. Exposed for test use.
 *
 */
bool preloadFuncAndDeps(BorrowedRef<PyFunctionObject> func);

using PreloaderMap = std::
    unordered_map<BorrowedRef<PyCodeObject>, std::unique_ptr<hir::Preloader>>;

/*
 * RAII device for isolating preloaders state. Exposed for test use.
 */
class IsolatedPreloaders {
 public:
  IsolatedPreloaders();
  ~IsolatedPreloaders();

 private:
  PreloaderMap orig_preloaders_;
};

} // namespace jit
#endif

#endif /* Py_LIMITED_API */
