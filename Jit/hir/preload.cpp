// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/preload.h"

#include "cinderx/Common/dict.h"
#include "cinderx/Common/extra-py-flags.h"
#include "cinderx/Common/util.h"
#include "cinderx/Interpreter/cinder_opcode.h"
#include "cinderx/Jit/bytecode.h"
#include "cinderx/StaticPython/classloader.h"
#include "cinderx/StaticPython/strictmoduleobject.h"
#include "cinderx/StaticPython/vtable_builder.h"
#include "cinderx/module_state.h"

#include <utility>

namespace jit::hir {

namespace {

static OwnedType resolve_type_descr(BorrowedRef<> descr) {
  int optional, exact;
  auto type = Ref<PyTypeObject>::steal(
      _PyClassLoader_ResolveType(descr, &optional, &exact));

  return {
      std::move(type), static_cast<bool>(optional), static_cast<bool>(exact)};
}

static FieldInfo resolve_field_descr(BorrowedRef<PyTupleObject> descr) {
  int field_type;
  Py_ssize_t offset = _PyClassLoader_ResolveFieldOffset(descr, &field_type);

  JIT_CHECK(offset != -1, "failed to resolve field {}", repr(descr));

  return {
      offset,
      prim_type_to_type(field_type),
      PyTuple_GET_ITEM(descr, PyTuple_GET_SIZE(descr) - 1)};
}

static void _fill_primitive_arg_types_helper(
    BorrowedRef<_PyTypedArgsInfo> prim_args_info,
    ArgToType& map) {
  for (Py_ssize_t i = 0; i < Py_SIZE(prim_args_info.get()); i++) {
    map.emplace(
        prim_args_info->tai_args[i].tai_argnum,
        prim_type_to_type(prim_args_info->tai_args[i].tai_primitive_type));
  }
}

static void fill_primitive_arg_types_func(
    BorrowedRef<PyFunctionObject> func,
    ArgToType& map) {
  auto prim_args_info =
      Ref<_PyTypedArgsInfo>::steal(_PyClassLoader_GetTypedArgsInfo(
          reinterpret_cast<PyCodeObject*>(func->func_code), 1));
  _fill_primitive_arg_types_helper(prim_args_info, map);
}

static void fill_primitive_arg_types_thunk(
    BorrowedRef<PyObject> thunk,
    ArgToType& map,
    PyObject* container) {
  auto prim_args_info = Ref<_PyTypedArgsInfo>::steal(
      _PyClassLoader_GetTypedArgsInfoFromThunk(thunk, container, 1));

  _fill_primitive_arg_types_helper(prim_args_info, map);
}

static void fill_primitive_arg_types_builtin(
    BorrowedRef<> callable,
    ArgToType& map) {
  Ci_PyTypedMethodDef* def = _PyClassLoader_GetTypedMethodDef(callable);
  JIT_CHECK(def != nullptr, "expected typed method def");
  for (Py_ssize_t i = 0; def->tmd_sig[i] != nullptr; i++) {
    const Ci_Py_SigElement* elem = def->tmd_sig[i];
    int code = Ci_Py_SIG_TYPE_MASK(elem->se_argtype);
    Type typ = prim_type_to_type(code);
    if (typ <= TPrimitive) {
      map.emplace(i, typ);
    }
  }
}

static std::unique_ptr<NativeTarget> resolve_native_target(
    BorrowedRef<> native_descr,
    BorrowedRef<> signature) {
  auto target = std::make_unique<NativeTarget>();
  void* raw_ptr = _PyClassloader_LookupSymbol(
      PyTuple_GET_ITEM(native_descr.get(), 0),
      PyTuple_GET_ITEM(native_descr.get(), 1));

  JIT_CHECK(
      raw_ptr != nullptr, "invalid address for native function: {}", raw_ptr);

  target->callable = raw_ptr;

  Py_ssize_t siglen = PyTuple_GET_SIZE(signature.get());
  auto return_type_code = _PyClassLoader_ResolvePrimitiveType(
      PyTuple_GET_ITEM(signature.get(), siglen - 1));
  target->return_type = prim_type_to_type(return_type_code);
  JIT_DCHECK(
      target->return_type <= TCInt,
      "native function return type must be a primitive");

  // Fill in the primitive arg type map in the target (index -> Type)
  ArgToType& primitive_arg_types = target->primitive_arg_types;
  for (Py_ssize_t i = 0; i < siglen - 1; i++) {
    int arg_type_code = _PyClassLoader_ResolvePrimitiveType(
        PyTuple_GET_ITEM(signature.get(), i));
    Type typ = prim_type_to_type(arg_type_code);
    JIT_DCHECK(typ <= TCInt, "native function arg type must be a primitive");

    primitive_arg_types.emplace(i, typ);
  }

  return target;
}

PreloaderManager s_manager;

} // namespace

std::unique_ptr<InvokeTarget> Preloader::resolve_target_descr(
    BorrowedRef<> descr,
    int opcode) {
  auto target = std::make_unique<InvokeTarget>();
  PyObject* container;
  auto callable =
      Ref<>::steal(_PyClassLoader_ResolveFunction(descr, &container));
  if (callable == nullptr) {
    JIT_LOG(
        "unknown invoke target {} during preloading {}",
        repr(descr),
        fullname());
    return nullptr;
  }

  int optional, exact, func_flags;
  auto return_pytype =
      Ref<PyTypeObject>::steal(_PyClassLoader_ResolveReturnType(
          callable, &optional, &exact, &func_flags));

  target->container_is_immutable = _PyClassLoader_IsImmutable(container);
  if (return_pytype != nullptr) {
    if (func_flags & Ci_FUNC_FLAGS_COROUTINE) {
      // TODO properly handle coroutine returns awaitable type
      target->return_type = TObject;
    } else {
      OwnedType preloaded_type{
          std::move(return_pytype),
          static_cast<bool>(optional),
          static_cast<bool>(exact)};
      target->return_type = preloaded_type.toHir();
    }
  }
  target->is_statically_typed = _PyClassLoader_IsStaticCallable(callable);
  PyMethodDef* def;
  Ci_PyTypedMethodDef* tmd;
  bool is_thunk = false;
  if (PyFunction_Check(callable)) {
    target->is_function = true;
  } else if (_PyClassLoader_IsPatchedThunk(callable)) {
    is_thunk = true;
  } else if ((def = _PyClassLoader_GetMethodDef(callable)) != nullptr) {
    target->is_builtin = true;
    target->builtin_c_func = reinterpret_cast<void*>(def->ml_meth);
    if (def->ml_flags == METH_NOARGS) {
      target->builtin_expected_nargs = 1;
    } else if (def->ml_flags == METH_O) {
      target->builtin_expected_nargs = 2;
    } else if ((tmd = _PyClassLoader_GetTypedMethodDef(callable))) {
      target->builtin_returns_error_code = (tmd->tmd_ret == Ci_Py_SIG_ERROR);
      target->builtin_returns_void = (tmd->tmd_ret == Ci_Py_SIG_VOID);
      target->builtin_c_func = tmd->tmd_meth;
    }
  }
  target->callable = std::move(callable);

  if (opcode == LOAD_METHOD_STATIC) {
    target->slot = _PyClassLoader_ResolveMethod(descr);
    JIT_CHECK(target->slot != -1, "method lookup failed: {}", repr(descr));
  } else { // the rest of this only used by INVOKE_FUNCTION currently
    target->uses_runtime_func =
        target->is_function && usesRuntimeFunc(target->func()->func_code);
    if (!target->container_is_immutable) {
      target->indirect_ptr = _PyClassLoader_ResolveIndirectPtr(descr);
      if (target->indirect_ptr == nullptr) {
        if (PyErr_Occurred()) {
          PyErr_WriteUnraisable(descr);
        }
        JIT_ABORT("indirect_ptr null for {} (stale bytecode?)", repr(descr));
      }
    }
  }

  if (target->is_statically_typed) {
    if (target->is_function) {
      fill_primitive_arg_types_func(
          target->func(), target->primitive_arg_types);
    } else {
      fill_primitive_arg_types_builtin(
          target->callable, target->primitive_arg_types);
    }
  }

  if (is_thunk) {
    fill_primitive_arg_types_thunk(
        target->callable.get(), target->primitive_arg_types, container);
  }

  return target;
}

BorrowedRef<PyFunctionObject> InvokeTarget::func() const {
  JIT_CHECK(is_function, "not a PyFunctionObject");
  return reinterpret_cast<PyFunctionObject*>(callable.get());
}

Type Preloader::type(BorrowedRef<> descr) const {
  return preloadedType(descr).toHir();
}

int Preloader::primitiveTypecode(BorrowedRef<> descr) const {
  return _PyClassLoader_GetTypeCode(pyType(descr));
}

BorrowedRef<PyTypeObject> Preloader::pyType(BorrowedRef<> descr) const {
  auto const& preloader_type = preloadedType(descr);
  JIT_CHECK(!preloader_type.optional, "unexpected optional type");
  return preloader_type.type;
}

const OwnedType& Preloader::preloadedType(BorrowedRef<> descr) const {
  return map_get(types_, descr);
}

const FieldInfo& Preloader::fieldInfo(BorrowedRef<> descr) const {
  return map_get(fields_, descr);
}

const InvokeTarget& Preloader::invokeFunctionTarget(BorrowedRef<> descr) const {
  return *(map_get(func_targets_, descr));
}

const InvokeTarget& Preloader::invokeMethodTarget(BorrowedRef<> descr) const {
  return *(map_get(meth_targets_, descr));
}

const NativeTarget& Preloader::invokeNativeTarget(BorrowedRef<> target) const {
  return *(map_get(native_targets_, target));
}

Type Preloader::checkArgType(long local_idx) const {
  return map_get(check_arg_types_, local_idx, TObject);
}

PyObject** Preloader::getGlobalCache(BorrowedRef<> name_obj) const {
  JIT_DCHECK(
      canCacheGlobals(),
      "trying to get a globals cache with unwatchable builtins and/or globals");
  JIT_CHECK(PyUnicode_CheckExact(name_obj), "Name must be a str");
  BorrowedRef<PyUnicodeObject> name{name_obj};
  return cinderx::getModuleState()->cacheManager()->getGlobalCache(
      builtins_, globals_, name);
}

bool Preloader::canCacheGlobals() const {
  return hasOnlyUnicodeKeys(builtins_) && hasOnlyUnicodeKeys(globals_);
}

BorrowedRef<> Preloader::global(int name_idx) const {
  BorrowedRef<> name = map_get(global_names_, name_idx, nullptr);
  if (name != nullptr && canCacheGlobals()) {
    return *getGlobalCache(name);
  }
  return nullptr;
}

std::unique_ptr<Function> Preloader::makeFunction() const {
  // We touch refcounts of Python objects here, so must serialize
  ThreadedCompileSerialize guard;
  auto irfunc = std::make_unique<Function>();
  irfunc->fullname = fullname_;
  irfunc->setCode(code_);
  irfunc->builtins.reset(builtins_);
  irfunc->globals.reset(globals_);
  irfunc->prim_args_info.reset(prim_args_info_);
  irfunc->return_type = return_type_;
  irfunc->has_primitive_args = has_primitive_args_;
  irfunc->has_primitive_first_arg = has_primitive_first_arg_;
  for (auto& [local, preloaded_type] : check_arg_pytypes_) {
    irfunc->typed_args.emplace_back(
        local,
        preloaded_type.type,
        preloaded_type.optional,
        preloaded_type.exact,
        preloaded_type.toHir());
  }
  return irfunc;
}

BorrowedRef<> Preloader::constArg(BytecodeInstruction& bc_instr) const {
  return PyTuple_GET_ITEM(code_->co_consts, bc_instr.oparg());
}

bool Preloader::preload() {
  bool is_static = code_->co_flags & CI_CO_STATICALLY_COMPILED;
  if (is_static && !preloadStatic()) {
    return false;
  }

  jit::BytecodeInstructionBlock bc_instrs{code_};
  for (auto bc_instr : bc_instrs) {
    switch (bc_instr.opcode()) {
      case LOAD_GLOBAL: {
        if (!canCacheGlobals()) {
          break;
        }
        PyObject* names = code_->co_names;
        Py_ssize_t names_len = PyTuple_Size(names);
        int name_idx = loadGlobalIndex(bc_instr.oparg());
        JIT_CHECK(
            name_idx < names_len,
            "Preloaded LOAD_GLOBAL with index {} for names tuple of length {}",
            name_idx,
            names_len);

        BorrowedRef<> name = PyTuple_GET_ITEM(names, name_idx);
        JIT_CHECK(name != nullptr, "name cannot be null");
        // Make sure the cached value has been loaded and any side effects of
        // loading it (e.g. lazy imports) have been exercised before we create
        // the GlobalCache; otherwise GlobalCache initialization can
        // self-destroy due to side effects of PyDict_GetItem and cause a
        // use-after-free.
        PyObject* global_value = PyDict_GetItem(globals_, name);
        if (!global_value) {
          // It's extremely unlikely that builtins dict could ever contain a
          // lazy import that needs warming up, but since it is technically
          // possible, we may as well go ahead and warm that up too if the key
          // isn't in globals.
          PyDict_GetItem(builtins_, name);
        }
        if (PyErr_Occurred()) {
          return false;
        }
        // The above dict fetches may have had side effects that mean globals
        // are no longer cacheable, so recheck that.
        if (canCacheGlobals()) {
          // We also initialize the GlobalCache here so we don't have to
          // thread-serialize initializing it later (it calls PyDict_GetItem,
          // which can cause data races in multithreaded compile.)
          getGlobalCache(name);
          global_names_.emplace(name_idx, name);
        }
        break;
      }
      case BUILD_CHECKED_LIST:
      case BUILD_CHECKED_MAP: {
        BorrowedRef<> descr = PyTuple_GetItem(constArg(bc_instr), 0);
        OwnedType collection_type = resolve_type_descr(descr);
        if (collection_type.type == nullptr) {
          JIT_LOG(
              "unknown collection type descr {} during preloading of {}",
              repr(descr),
              fullname());
          return false;
        }
        types_.emplace(descr, std::move(collection_type));
        break;
      }
      case CAST:
      case LOAD_CLASS:
      case REFINE_TYPE:
      case TP_ALLOC: {
        BorrowedRef<> descr = constArg(bc_instr);
        OwnedType alloc_type = resolve_type_descr(descr);
        if (alloc_type.type == nullptr) {
          JIT_LOG(
              "unknown {} type descr {} during preloading of {}",
              bc_instr.opcode(),
              repr(descr),
              fullname());
          return false;
        }
        types_.emplace(descr, std::move(alloc_type));
        break;
      }
      case LOAD_FIELD:
      case STORE_FIELD: {
        BorrowedRef<PyTupleObject> descr(constArg(bc_instr));
        fields_.emplace(descr, resolve_field_descr(descr));
        break;
      }
      case LOAD_METHOD_STATIC:
      case INVOKE_FUNCTION:
      case INVOKE_METHOD: {
        BorrowedRef<> descr = PyTuple_GetItem(constArg(bc_instr), 0);
        auto& map = bc_instr.opcode() == INVOKE_FUNCTION ? func_targets_
                                                         : meth_targets_;
        std::unique_ptr<InvokeTarget> target =
            resolve_target_descr(descr, bc_instr.opcode());
        if (target) {
          map.emplace(descr, std::move(target));
          break;
        } else {
          return false;
        }
      }
      case INVOKE_NATIVE: {
        BorrowedRef<> target_descr = PyTuple_GetItem(constArg(bc_instr), 0);
        BorrowedRef<> signature = PyTuple_GetItem(constArg(bc_instr), 1);
        native_targets_.emplace(
            target_descr, resolve_native_target(target_descr, signature));
        break;
      }
    }
  }

  if (has_primitive_args_) {
    prim_args_info_ = Ref<_PyTypedArgsInfo>::steal(
        _PyClassLoader_GetTypedArgsInfo(code_, true));
  }
  return true;
}

bool Preloader::preloadStatic() {
  BorrowedRef<> ret_type_descr = _PyClassLoader_GetCodeReturnTypeDescr(code_);
  if (ret_type_descr == nullptr) {
    // Special case where a module's code object is being preloaded.  It will
    // not have argument or return type descrs (and they cannot be added as they
    // interfere with the "<import-from>" list!).
    if (isModuleCodeObject()) {
      return true;
    }

    JIT_LOG(
        "Statically typed function {} has no return type descr, co_consts "
        "is {}",
        fullname(),
        repr(code_->co_consts));
    return false;
  }

  OwnedType ret_type = resolve_type_descr(ret_type_descr);
  if (ret_type.type == nullptr) {
    JIT_LOG(
        "unknown return type descr {} during preloading of {}",
        repr(ret_type_descr),
        fullname());
    return false;
  }

  return_type_ = ret_type.toHir();

  BorrowedRef<PyTupleObject> checks = reinterpret_cast<PyTupleObject*>(
      _PyClassLoader_GetCodeArgumentTypeDescrs(code_));

  for (int i = 0; i < PyTuple_GET_SIZE(checks); i += 2) {
    long local = PyLong_AsLong(PyTuple_GET_ITEM(checks, i));
    if (local < 0) {
#if PY_VERSION_HEX < 0x030C0000
      // A negative value for local indicates that it's a cell
      JIT_CHECK(
          code_->co_cell2arg != nullptr,
          "no cell2arg but negative local {}",
          local);
      long arg = code_->co_cell2arg[-1 * (local + 1)];
      JIT_CHECK(
          arg != CO_CELL_NOT_AN_ARG, "cell not an arg for local {}", local);
      local = arg;
#else
      JIT_ABORT(
          "In Static Python function {}, hit negative local {} at index {}, "
          "arguments checks tuple is {}",
          fullname(),
          local,
          i,
          repr(checks));
#endif
    }
    OwnedType preloaded_type =
        resolve_type_descr(PyTuple_GET_ITEM(checks, i + 1));
    if (preloaded_type.type == nullptr) {
      JIT_LOG(
          "unknown type descr {} during preloading of {}",
          repr(PyTuple_GET_ITEM(checks, i + 1)),
          fullname());
      return false;
    }
    JIT_CHECK(
        preloaded_type.type != reinterpret_cast<PyTypeObject*>(&PyObject_Type),
        "shouldn't generate type checks for object");
    Type type = preloaded_type.toHir();
    check_arg_types_.emplace(local, type);
    check_arg_pytypes_.emplace(local, std::move(preloaded_type));
    if (type <= TPrimitive) {
      has_primitive_args_ = true;
      if (local == 0) {
        has_primitive_first_arg_ = true;
      }
    }
  }

  return true;
}

// Check if a code object is for the top-level code in a module.
bool Preloader::isModuleCodeObject() const {
  return fullname().ends_with("<module>") || fullname() == "__main__:__main__";
}

void PreloaderManager::add(
    BorrowedRef<PyCodeObject> code,
    std::unique_ptr<Preloader> preloader) {
  auto [_, inserted] = preloaders_.emplace(code, std::move(preloader));
  JIT_CHECK(
      inserted,
      "Trying to create a duplicate preloader for {}",
      PyUnicode_AsUTF8(code->co_qualname));
}

Preloader* PreloaderManager::find(BorrowedRef<PyCodeObject> code) {
  auto it = preloaders_.find(code);
  return it != preloaders_.end() ? it->second.get() : nullptr;
}

Preloader* PreloaderManager::find(BorrowedRef<PyFunctionObject> func) {
  BorrowedRef<PyCodeObject> code = func->func_code;
  return find(code);
}

bool PreloaderManager::empty() const {
  return preloaders_.empty();
}

void PreloaderManager::clear() {
  preloaders_.clear();
}

void PreloaderManager::swap(PreloaderMap& replacement) {
  // Should never be called from within the actual multi-threaded compile;
  // it's not safe to mess with the global preloaders map in that context.
  JIT_CHECK(
      !getThreadedCompileContext().compileRunning(),
      "cannot preload single func from within multi-threaded compile");
  preloaders_.swap(replacement);
}

PreloaderManager& preloaderManager() {
  return s_manager;
}

IsolatedPreloaders::IsolatedPreloaders() {
  swap();
}

IsolatedPreloaders::~IsolatedPreloaders() {
  swap();
}

void IsolatedPreloaders::swap() {
  s_manager.swap(orig_preloaders_);
}

} // namespace jit::hir
