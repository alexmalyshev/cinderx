// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <Python.h>

#include "cinderx/Common/ref.h"
#include "cinderx/Common/util.h"
#include "cinderx/Jit/compiler.h"
#include "cinderx/Jit/containers.h"
#include "cinderx/Jit/elf/note.h"
#include "cinderx/Jit/hir/preload.h"
#include "cinderx/Jit/pyjit_result.h"

#include <memory>
#include <vector>

namespace jit {

// Lookup key for compiled functions in Context: a code object and the globals
// and builtins dicts it was JIT-compiled with.
struct CompilationKey {
  // These three are borrowed references; the values are kept alive by strong
  // references in the corresponding jit::CodeRuntime.
  PyObject* code;
  PyObject* builtins;
  PyObject* globals;

  CompilationKey(PyObject* code, PyObject* builtins, PyObject* globals)
      : code(code), builtins(builtins), globals(globals) {}

  constexpr bool operator==(const CompilationKey& other) const = default;
};

} // namespace jit

template <>
struct std::hash<jit::CompilationKey> {
  std::size_t operator()(const jit::CompilationKey& key) const {
    std::hash<PyObject*> hasher;
    return jit::combineHash(
        hasher(key.code), hasher(key.globals), hasher(key.builtins));
  }
};

namespace jit {

/*
 * A jit::Context encapsulates all the state managed by an instance of the JIT.
 */
class Context {
 public:
  /*
   * Will deopt all compiled functions back to the interpreter.
   */
  ~Context();

  /*
   * JIT compile function/code-object from a Preloader.
   *
   * Patches func entrypoint if a func is provided.
   *
   * Will return PYJIT_RESULT_OK if the function/code object was already
   * compiled.
   */
  _PyJIT_Result compilePreloader(
      BorrowedRef<PyFunctionObject> func,
      const hir::Preloader& preloader);

  /*
   * Attach already-compiled code to the given function, if it exists.
   *
   * Intended for (but not limited to) use with nested functions after the JIT
   * is disabled.
   *
   * Will return PYJIT_RESULT_OK if the given function already had compiled code
   * attached.
   */
  _PyJIT_Result attachCompiledCode(BorrowedRef<PyFunctionObject> func);

  /*
   * Callbacks invoked by the runtime when a PyFunctionObject is modified or
   * destroyed.
   */
  void funcModified(BorrowedRef<PyFunctionObject> func);
  void funcDestroyed(BorrowedRef<PyFunctionObject> func);

  /*
   * Return whether or not this context compiled the supplied function.
   */
  bool didCompile(BorrowedRef<PyFunctionObject> func);

  /*
   * Look up the compiled function object for a given Python function object.
   */
  CompiledFunction* lookupFunc(BorrowedRef<PyFunctionObject> func);

  /*
   * Returns the number of functions inlined into a specified JIT-compiled
   * function.
   *
   * Returns -1 if an error occurred.
   */
  int numInlinedFunctions(BorrowedRef<PyFunctionObject> func);

  /*
   * Return a stats object on the functions inlined into a specified
   * JIT-compiled function.
   *
   * Will return nullptr if the supplied function has not been JIT-compiled.
   */
  Ref<> inlinedFunctionsStats(BorrowedRef<PyFunctionObject> func);

  /*
   * Return the HIR opcode counts for a JIT-compiled function, or nullptr if the
   * function has not been JIT-compiled.
   */
  const hir::OpcodeCounts* hirOpcodeCounts(BorrowedRef<PyFunctionObject> func);

  /*
   * Print the HIR for func to stdout if it was JIT-compiled.
   * This function is a no-op if func was not JIT-compiled.
   *
   * Returns -1 if an error occurred or 0 otherwise.
   */
  int printHIR(BorrowedRef<PyFunctionObject> func);

  /*
   * Print the disassembled code for func to stdout if it was JIT-compiled.
   * This function is a no-op if func was not JIT-compiled.
   *
   * Returns -1 if an error occurred or 0 otherwise.
   */
  int disassemble(BorrowedRef<PyFunctionObject> func);

  /*
   * Get a range over all function objects that have been compiled.
   */
  const UnorderedSet<BorrowedRef<PyFunctionObject>>& compiledFuncs();

  /*
   * Set and hold a reference to the cinderjit Python module.
   */
  void setCinderJitModule(Ref<> mod);

  /*
   * Clear cache of compiled code such that subsequent compilations are always
   * full rather than just re-binding pre-compiled code. Only intended to be
   * used during multithreaded_compile_test.
   */
  void clearCache();

 private:
  struct CompilationResult {
    CompiledFunction* compiled;
    _PyJIT_Result result;
  };

  CompilationResult compilePreloader(const hir::Preloader& preloader);

  CompiledFunction* lookupCode(
      BorrowedRef<PyCodeObject> code,
      BorrowedRef<PyDictObject> builtins,
      BorrowedRef<PyDictObject> globals);

  /*
   * Reset a function's entry point if it was JIT-compiled.
   */
  void deoptFunc(BorrowedRef<PyFunctionObject> func);

  /*
   * Record per-function metadata for a newly compiled function and set the
   * function's entrypoint.
   */
  void finalizeFunc(
      BorrowedRef<PyFunctionObject> func,
      const CompiledFunction& compiled);

  /* General purpose jit compiler */
  Compiler jit_compiler_;

  /* Set of which functions have JIT-compiled entrypoints. */
  UnorderedSet<BorrowedRef<PyFunctionObject>> compiled_funcs_;

  /*
   * Map of all compiled code objects, keyed by their address and also their
   * builtins and globals objects.
   */
  UnorderedMap<CompilationKey, std::unique_ptr<CompiledFunction>>
      compiled_codes_;

  /*
   * Code which is being kept alive in case it was in use when
   * clearCache was called. Only intended to be used during
   * multithreaded_compile_test.
   */
  std::vector<std::unique_ptr<CompiledFunction>> orphaned_compiled_codes_;

  Ref<> cinderjit_module_;
};

/*
 * An AotContext is like the JIT context, but it holds onto state for
 * ahead-of-time compiled functions.
 */
class AotContext {
 public:
  struct FuncState {
    elf::CodeNoteData note;
    BorrowedRef<PyFunctionObject> func;
    std::span<const std::byte> compiled_code;

    vectorcallfunc normalEntry() const {
      return reinterpret_cast<vectorcallfunc>(const_cast<std::byte*>(
          compiled_code.data() + note.normal_entry_offset));
    }
  };

  /*
   * Initialize the context with the handle to the AOT bundle created by
   * dlopen().
   */
  void init(void* bundle_handle);

  /* Clean up the context object. */
  void destroy();

  /*
   * Register a new function whose metadata has been parsed out of the AOT
   * bundle.
   */
  void registerFunc(const elf::Note& note);

  /* Look up the state associated with a given Python function. */
  const FuncState* lookupFuncState(BorrowedRef<PyFunctionObject> func);

 private:
  // The handle to the AOT bundle created by dlopen().
  void* bundle_handle_{nullptr};

  jit::UnorderedMap<std::string, FuncState> funcs_;
};

extern AotContext g_aot_ctx;

} // namespace jit
