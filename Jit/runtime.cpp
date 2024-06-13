// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/runtime.h"

#include "cinderx/Common/watchers.h"
#include "internal/pycore_interp.h"

#include "cinderx/Jit/type_deopt_patchers.h"

#include <sys/mman.h>

#include <memory>

namespace jit {

const int64_t CodeRuntime::kPyCodeOffset =
    RuntimeFrameState::codeOffset() + CodeRuntime::frameStateOffset();

void CodeRuntime::releaseReferences() {
  references_.clear();
}

void CodeRuntime::addReference(Ref<>&& obj) {
  JIT_CHECK(obj != nullptr, "Can't own a reference to nullptr");
  references_.emplace(std::move(obj));
}

void CodeRuntime::addReference(BorrowedRef<> obj) {
  // Serialize as we modify the ref-count to obj which may be widely accessible.
  ThreadedCompileSerialize guard;
  return addReference(Ref<>::create(obj));
}

void Builtins::init() {
  ThreadedCompileSerialize guard;
  if (is_initialized_) {
    return;
  }
  // we want to check the exact function address, rather than relying on
  // modules which can be mutated.  First find builtins, which we have
  // to do a search for because PyEval_GetBuiltins() returns the
  // module dict.
  PyObject* mods = _PyInterpreterState_GET()->modules_by_index;
  PyModuleDef* builtins = nullptr;
  for (Py_ssize_t i = 0; i < PyList_GET_SIZE(mods); i++) {
    PyObject* cur = PyList_GET_ITEM(mods, i);
    if (cur == Py_None) {
      continue;
    }
    PyModuleDef* def = PyModule_GetDef(cur);
    if (def == nullptr) {
      PyErr_Clear();
      continue;
    }
    if (std::strcmp(def->m_name, "builtins") == 0) {
      builtins = def;
      break;
    }
  }
  JIT_CHECK(builtins != nullptr, "could not find builtins module");

  auto add = [this](const std::string& name, PyMethodDef* meth) {
    cfunc_to_name_[meth] = name;
    name_to_cfunc_[name] = meth;
  };
  // Find all free functions.
  for (PyMethodDef* fdef = builtins->m_methods; fdef->ml_name != nullptr;
       fdef++) {
    add(fdef->ml_name, fdef);
  }
  // Find all methods on types.
  PyTypeObject* types[] = {
      &PyDict_Type,
      &PyList_Type,
      &PyTuple_Type,
      &PyUnicode_Type,
  };
  for (auto type : types) {
    for (PyMethodDef* fdef = type->tp_methods; fdef->ml_name != nullptr;
         fdef++) {
      add(fmt::format("{}.{}", type->tp_name, fdef->ml_name), fdef);
    }
  }
  // Only mark as initialized after everything is done to avoid concurrent
  // reads of an unfinished map.
  is_initialized_ = true;
}

std::optional<std::string> Builtins::find(PyMethodDef* meth) const {
  auto result = cfunc_to_name_.find(meth);
  if (result == cfunc_to_name_.end()) {
    return std::nullopt;
  }
  return result->second;
}

std::optional<PyMethodDef*> Builtins::find(const std::string& name) const {
  auto result = name_to_cfunc_.find(name);
  if (result == name_to_cfunc_.end()) {
    return std::nullopt;
  }
  return result->second;
}

Runtime* Runtime::s_runtime_{nullptr};

void Runtime::shutdown() {
  s_runtime_->globalCaches().clear();
  delete s_runtime_;
  s_runtime_ = nullptr;
}

void Runtime::mlockProfilerDependencies() {
  for (auto& codert : code_runtimes_) {
    PyCodeObject* code = codert.frameState()->code().get();
    ::mlock(code, sizeof(PyCodeObject));
    ::mlock(code->co_qualname, Py_SIZE(code->co_qualname));
  }
  code_runtimes_.mlock();
}

ProfileRuntime& Runtime::profileRuntime() {
  return profile_runtime_;
}

GlobalCacheManager& Runtime::globalCaches() {
  return global_caches_;
}

Ref<> Runtime::pageInProfilerDependencies() {
  ThreadedCompileSerialize guard;
  Ref<> qualnames = Ref<>::steal(PyList_New(0));
  if (qualnames == nullptr) {
    return nullptr;
  }
  // We want to force the OS to page in the memory on the
  // code_rt->code->qualname path and keep the compiler from optimizing away
  // the code to do so. There are probably more efficient ways of doing this
  // but perf isn't a major concern.
  for (auto& code_rt : code_runtimes_) {
    BorrowedRef<> qualname = code_rt.frameState()->code()->co_qualname;
    if (qualname == nullptr) {
      continue;
    }
    if (PyList_Append(qualnames, qualname) < 0) {
      return nullptr;
    }
  }
  return qualnames;
}

void** Runtime::findFunctionEntryCache(PyFunctionObject* function) {
  auto result = function_entry_caches_.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(function),
      std::forward_as_tuple());
  addReference(reinterpret_cast<PyObject*>(function));
  if (result.second) {
    result.first->second.ptr = pointer_caches_.allocate();
    if (_PyClassLoader_HasPrimitiveArgs((PyCodeObject*)function->func_code)) {
      result.first->second.arg_info =
          Ref<_PyTypedArgsInfo>::steal(_PyClassLoader_GetTypedArgsInfo(
              (PyCodeObject*)function->func_code, 1));
    }
  }
  return result.first->second.ptr;
}

bool Runtime::hasFunctionEntryCache(PyFunctionObject* function) const {
  return function_entry_caches_.find(function) != function_entry_caches_.end();
}

_PyTypedArgsInfo* Runtime::findFunctionPrimitiveArgInfo(
    PyFunctionObject* function) {
  auto cache = function_entry_caches_.find(function);
  if (cache == function_entry_caches_.end()) {
    return nullptr;
  }
  return cache->second.arg_info.get();
}

std::size_t Runtime::addDeoptMetadata(DeoptMetadata&& deopt_meta) {
  // Serialize as the deopt data is shared across compile threads.
  ThreadedCompileSerialize guard;
  deopt_metadata_.emplace_back(std::move(deopt_meta));
  return deopt_metadata_.size() - 1;
}

DeoptMetadata& Runtime::getDeoptMetadata(std::size_t id) {
  JIT_CHECK(
      g_threaded_compile_context.canAccessSharedData(),
      "getDeoptMetadata() called in unsafe context");
  return deopt_metadata_[id];
}

void Runtime::recordDeopt(std::size_t idx, PyObject* guilty_value) {
  DeoptStat& stat = deopt_stats_[idx];
  stat.count++;
  if (guilty_value != nullptr) {
    stat.types.recordType(Py_TYPE(guilty_value));
  }
}

const DeoptStats& Runtime::deoptStats() const {
  return deopt_stats_;
}

void Runtime::clearDeoptStats() {
  deopt_stats_.clear();
}

InlineCacheStats Runtime::getAndClearLoadMethodCacheStats() {
  InlineCacheStats stats;
  for (auto& cache : load_method_caches_) {
    if (cache.cacheStats() == nullptr) {
      // Cache stat may not have been initialized if LoadMethodCached instr was
      // optimized away.
      continue;
    }
    stats.push_back(*cache.cacheStats());
    cache.clearCacheStats();
  }
  return stats;
}

InlineCacheStats Runtime::getAndClearLoadTypeMethodCacheStats() {
  InlineCacheStats stats;
  for (auto& cache : load_type_method_caches_) {
    if (cache.cacheStats() == nullptr) {
      // Cache stat may not have been initialized if LoadTypeMethod instr
      // was optimized away.
      continue;
    }
    stats.push_back(*cache.cacheStats());
    cache.clearCacheStats();
  }
  return stats;
}

void Runtime::setGuardFailureCallback(Runtime::GuardFailureCallback cb) {
  guard_failure_callback_ = cb;
}

void Runtime::guardFailed(const DeoptMetadata& deopt_meta) {
  if (guard_failure_callback_) {
    guard_failure_callback_(deopt_meta);
  }
}

void Runtime::clearGuardFailureCallback() {
  guard_failure_callback_ = nullptr;
}

void Runtime::addReference(Ref<>&& obj) {
  JIT_CHECK(obj != nullptr, "Can't own a reference to nullptr");
  // Serialize as we modify the globally accessible references_ object.
  ThreadedCompileSerialize guard;
  references_.emplace(std::move(obj));
}

void Runtime::addReference(BorrowedRef<> obj) {
  // Serialize as we modify the ref-count to obj which may be widely accessible.
  ThreadedCompileSerialize guard;
  return addReference(Ref<>::create(obj));
}

void Runtime::releaseReferences() {
  for (auto& code_rt : code_runtimes_) {
    code_rt.releaseReferences();
  }
  references_.clear();
  type_deopt_patchers_.clear();
}

std::optional<std::string> symbolize(const void* func) {
  if (!g_symbolize_funcs) {
    return std::nullopt;
  }
  std::optional<std::string_view> mangled_name =
      Runtime::get()->symbolize(func);
  if (!mangled_name.has_value()) {
    return std::nullopt;
  }
  return jit::demangle(std::string{*mangled_name});
}

void Runtime::watchType(
    BorrowedRef<PyTypeObject> type,
    TypeDeoptPatcher* patcher) {
  ThreadedCompileSerialize guard;
  Ci_Watchers_WatchType(type);
  type_deopt_patchers_[type].emplace_back(patcher);
}

void Runtime::notifyTypeModified(
    BorrowedRef<PyTypeObject> lookup_type,
    BorrowedRef<PyTypeObject> new_type) {
  notifyICsTypeChanged(lookup_type);

  ThreadedCompileSerialize guard;
  auto it = type_deopt_patchers_.find(lookup_type);
  if (it == type_deopt_patchers_.end()) {
    return;
  }

  std::vector<TypeDeoptPatcher*> remaining_patchers;
  for (TypeDeoptPatcher* patcher : it->second) {
    if (!patcher->maybePatch(new_type)) {
      remaining_patchers.emplace_back(patcher);
    }
  }

  if (remaining_patchers.empty()) {
    type_deopt_patchers_.erase(it);
    // don't unwatch type; shadowcode may still be watching it
  } else {
    it->second = std::move(remaining_patchers);
  }
}

} // namespace jit
