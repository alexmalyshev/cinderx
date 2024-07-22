/* Copyright (c) Meta Platforms, Inc. and affiliates. */

#include "cinderx/StaticPython/classloader.h"

#include <Python.h>

#include "descrobject.h"
#include "dictobject.h"
#include "object.h"
#include "pycore_object.h" // PyHeapType_CINDER_EXTRA
#include "pyerrors.h"
#include "pyport.h"
#include "structmember.h"

#if PY_VERSION_HEX < 0x030C0000
#include "cinder/exports.h"
#endif
#include "cinderx/Upgrade/upgrade_stubs.h"  // @donotremove

#include "cinderx/CachedProperties/cached_properties.h"
#include "cinderx/Common/extra-py-flags.h"  // @donotremove
#include "cinderx/Common/py-portability.h"
#include "cinderx/Jit/entry.h"
#include "cinderx/Jit/pyjit.h"
#include "cinderx/StaticPython/descrs.h"
#include "cinderx/StaticPython/errors.h"
#include "cinderx/StaticPython/thunks.h"
#include "cinderx/StaticPython/strictmoduleobject.h"

#include <dlfcn.h>

// This is a dict containing a mapping of lib name to "handle"
// as returned by `dlopen()`.
// Dict[str, int]
static PyObject* dlopen_cache;

// This is a dict containing a mapping of (lib_name, symbol_name) to
// the raw address as returned by `dlsym()`.
// Dict[Tuple[str, str], int]
static PyObject* dlsym_cache;

static PyObject* rettype_check(
    PyTypeObject* cls,
    PyObject* ret,
    _PyClassLoader_RetTypeInfo* rt_info);

int used_in_vtable(PyObject* value);

PyObject* rettype_cb(_PyClassLoader_Awaitable* awaitable, PyObject* result) {
  if (result == NULL) {
    return NULL;
  }
  return rettype_check(
      Py_TYPE(awaitable),
      result,
      (_PyClassLoader_RetTypeInfo*)awaitable->state);
}

extern int _PyObject_GetMethod(PyObject*, PyObject*, PyObject**);

static int rettype_check_traverse(
    _PyClassLoader_RetTypeInfo* op,
    visitproc visit,
    void* arg) {
  visit((PyObject*)op->rt_expected, arg);
  return 0;
}

static int rettype_check_clear(_PyClassLoader_RetTypeInfo* op) {
  Py_CLEAR(op->rt_expected);
  Py_CLEAR(op->rt_name);
  return 0;
}

static int classloader_is_property_tuple(PyTupleObject* name) {
  if (PyTuple_GET_SIZE(name) != 2) {
    return 0;
  }
  PyObject* property_method_name = PyTuple_GET_ITEM(name, 1);
  if (!PyUnicode_Check(property_method_name)) {
    return 0;
  }
  return _PyUnicode_EqualToASCIIString(property_method_name, "fget") ||
      _PyUnicode_EqualToASCIIString(property_method_name, "fset");
}

PyObject* classloader_get_func_name(PyObject* name) {
  if (PyTuple_Check(name) &&
      classloader_is_property_tuple((PyTupleObject*)name)) {
    return PyTuple_GET_ITEM(name, 0);
  }
  return name;
}

static PyObject* rettype_check(
    PyTypeObject* cls,
    PyObject* ret,
    _PyClassLoader_RetTypeInfo* rt_info) {
  if (ret == NULL) {
    return NULL;
  }

  int type_code = _PyClassLoader_GetTypeCode(rt_info->rt_expected);
  int overflow = 0;
  if (type_code != TYPED_OBJECT) {
    size_t int_val;
    switch (type_code) {
      case TYPED_BOOL:
        if (PyBool_Check(ret)) {
          return ret;
        }
        break;
      case TYPED_INT8:
      case TYPED_INT16:
      case TYPED_INT32:
      case TYPED_INT64:
      case TYPED_UINT8:
      case TYPED_UINT16:
      case TYPED_UINT32:
      case TYPED_UINT64:
        if (PyLong_Check(ret)) {
          if (_PyClassLoader_OverflowCheck(ret, type_code, &int_val)) {
            return ret;
          }
          overflow = 1;
        }
        break;
      default:
        PyErr_SetString(
            PyExc_RuntimeError, "unsupported primitive return type");
        Py_DECREF(ret);
        return NULL;
    }
  }

  if (overflow ||
      !(_PyObject_TypeCheckOptional(
          ret,
          rt_info->rt_expected,
          rt_info->rt_optional,
          rt_info->rt_exact))) {
    /* The override returned an incompatible value, report error */
    const char* msg;
    PyObject* exc_type = CiExc_StaticTypeError;
    if (overflow) {
      exc_type = PyExc_OverflowError;
      msg =
          "unexpected return type from %s%s%U, expected %s, got out-of-range %s (%R)";
    } else if (rt_info->rt_optional) {
      msg =
          "unexpected return type from %s%s%U, expected Optional[%s], "
          "got %s";
    } else {
      msg = "unexpected return type from %s%s%U, expected %s, got %s";
    }

    PyErr_Format(
        exc_type,
        msg,
        cls ? cls->tp_name : "",
        cls ? "." : "",
        classloader_get_func_name(rt_info->rt_name),
        rt_info->rt_expected->tp_name,
        Py_TYPE(ret)->tp_name,
        ret);

    Py_DECREF(ret);
    return NULL;
  }
  return ret;
}

static _PyClassLoader_StaticCallReturn return_to_native(
    PyObject* val,
    PyTypeObject* ret_type) {
  _PyClassLoader_StaticCallReturn ret;
  int type_code = _PyClassLoader_GetTypeCode(ret_type);
  if (val != NULL && type_code != TYPED_OBJECT) {
    ret.rax = (void*)_PyClassLoader_Unbox(val, type_code);
  } else {
    ret.rax = (void*)(uint64_t)val;
  }
  ret.rdx = (void*)(uint64_t)(val != NULL);
  return ret;
}

static const _PyClassLoader_StaticCallReturn StaticError = {0, 0};

static int hydrate_args(
    PyCodeObject* code,
    Py_ssize_t arg_count,
    void** args,
    PyObject** call_args,
    PyObject** free_args) {
  _PyTypedArgsInfo* typed_arg_info = _PyClassLoader_GetTypedArgsInfo(code, 1);
  PyObject** extra_args = (PyObject**)args[5];
  for (Py_ssize_t i = 0, cur_arg = 0; i < arg_count; i++) {
    void* original;
    if (i < 5) {
      original = args[i]; // skip the v-table state
    } else {
      original = extra_args[i - 3];
    }

    if (cur_arg < Py_SIZE(typed_arg_info) &&
        typed_arg_info->tai_args[cur_arg].tai_argnum == i) {
      call_args[i] = _PyClassLoader_Box(
          (uint64_t)original,
          typed_arg_info->tai_args[cur_arg].tai_primitive_type);
      if (call_args[i] == NULL) {
        for (Py_ssize_t free_arg = 0; free_arg < i; free_arg++) {
          Py_CLEAR(free_args[free_arg]);
        }
        return -1;
      }
      free_args[i] = call_args[i];
      cur_arg++;
    } else {
      free_args[i] = NULL;
      call_args[i] = (PyObject*)original;
    }
  }
  return 0;
}

static void free_hydrated_args(PyObject** free_args, Py_ssize_t arg_count) {
  for (Py_ssize_t i = 0; i < arg_count; i++) {
    Py_XDECREF(free_args[i]);
  }
}

static Py_ssize_t get_original_argcount(PyObject** callable);

PyTypeObject* resolve_function_rettype(
    PyObject* funcobj,
    int* optional,
    int* exact,
    int* func_flags);

_PyClassLoader_StaticCallReturn
invoke_from_native(PyObject* original, PyObject* func, void** args) {
  PyCodeObject* code = (PyCodeObject*)((PyFunctionObject*)original)->func_code;
  Py_ssize_t arg_count = code->co_argcount;
  PyObject* call_args[arg_count];
  PyObject* free_args[arg_count];

  if (hydrate_args(code, arg_count, args, call_args, free_args) < 0) {
    return StaticError;
  }

  PyObject* res =
      ((PyFunctionObject*)func)->vectorcall(func, call_args, arg_count, NULL);
  free_hydrated_args(free_args, arg_count);

  int optional, exact, func_flags;
  PyTypeObject* type =
      resolve_function_rettype(original, &optional, &exact, &func_flags);
  return return_to_native(res, type);
}

// Defines a helper thunk with the same layout as our JIT generated static
// entry points.  This has the static entry point at the start, and then 11
// instructions in we have the vectorcall entry point.  We install the static
// entry points into the v-table and can easily switch to the vector call
// form when we're invoking from the interpreter or somewhere that can't use
// the native calling convention.
#if defined(_M_X64) || defined(_M_AMD64) || defined(__x86_64__)
#define VTABLE_THUNK(name, _arg1_type)                                            \
  __attribute__((naked)) void name##_dont_bolt(void) {                            \
    __asm__(                                                                    \
                /* static_entry: */                                                 \
                /* we explicitly encode the jmp forward to static_entry_impl so */  \
                /* that we always get the 2 byte version.  The 0xEB is the jump, */ \
                /* the 14 is the length to jump, which is based upon the size of */ \
                /* jmp to the vectorcall entrypoint */                              \
                ".byte 0xEB\n"                                                      \
                ".byte 14\n"                                                        \
                "nop\n"                                                             \
                "nop\n"                                                             \
                "nop\n"                                                             \
                "nop\n"                                                             \
                "nop\n"                                                             \
                "nop\n"                                                             \
                "nop\n"                                                             \
                "nop\n"                                                             \
                "nop\n"                                                             \
                                                                                    \
                /* vector_entry: */                                                 \
                "jmp " #name "_vectorcall\n"                                        \
                                                                                    \
                /* static_entry_impl: */                                            \
                "push %rbp\n"                                                       \
                "mov %rsp, %rbp\n"                                                  \
                "push %rsp\n"                                                       \
                /* We want to push the arguments passed natively onto the stack */  \
                /* so that we can recover them in hydrate_args.  So we push them */ \
                /* onto the stack and then move the address of them into rsi */     \
                /* which will make them available as the 2nd argument.  Note we */  \
                /* don't need to push rdi as it's the state argument which we're */ \
                /* passing in anyway */                                             \
                "push %r9\n"                                                        \
                "push %r8\n"                                                        \
                "push %rcx\n"                                                       \
                "push %rdx\n"                                                       \
                "push %rsi\n"                                                       \
                "mov %rsp, %rsi\n"                                                  \
                "call " #name "_native\n"                                           \
                /* We don't know if we're returning a floating point value or not */\
                /* so we assume we are, and always populate the xmm registers */    \
                /* even if we don't need to */                                      \
                "movq %rax, %xmm0\n"                                                \
                "movq %rdx, %xmm1\n"                                                \
                "leave\n"                                                           \
                "ret\n"                                                             \
        ); \
  }
#else
#define VTABLE_THUNK(name, arg1_type)                     \
  PyObject* name##_dont_bolt(                             \
      arg1_type* state, PyObject** args, size_t nargsf) { \
    return name##_vectorcall(state, args, nargsf);        \
  }
#endif

__attribute__((__used__)) PyObject* type_vtable_coroutine_property_vectorcall(
    _PyClassLoader_TypeCheckState* state,
    PyObject* const* args,
    size_t nargsf) {
  PyObject* self = args[0];
  PyObject* descr = state->tcs_value;
  PyObject* name = state->tcs_rt.rt_name;
  PyObject* coro;
  int eager;

  /* we have to perform the descriptor checks at runtime because the
   * descriptor type can be modified preventing us from being able to have
   * more optimized fast paths */
  if (!PyDescr_IsData(descr)) {
    PyObject** dictptr = _PyObject_GetDictPtr(self);
    if (dictptr != NULL) {
      PyObject* dict = *dictptr;
      if (dict != NULL) {
        coro = PyDict_GetItem(dict, PyTuple_GET_ITEM(name, 0));
        if (coro != NULL) {
          Py_INCREF(coro);
          eager = 0;
          goto done;
        }
      }
    }
  }

  if (Py_TYPE(descr)->tp_descr_get != NULL) {
    PyObject* self = args[0];
    PyObject* get =
        Py_TYPE(descr)->tp_descr_get(descr, self, (PyObject*)Py_TYPE(self));
    if (get == NULL) {
      return NULL;
    }

    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    coro = _PyObject_Vectorcall(get, args + 1, (nargs - 1), NULL);
    Py_DECREF(get);
  } else {
    coro = _PyObject_Vectorcall(descr, args, nargsf, NULL);
  }

  eager = Ci_PyWaitHandle_CheckExact(coro);
  if (eager) {
    Ci_PyWaitHandleObject* handle = (Ci_PyWaitHandleObject*)coro;
    if (handle->wh_waiter == NULL) {
      if (rettype_check(
              Py_TYPE(descr),
              handle->wh_coro_or_result,
              (_PyClassLoader_RetTypeInfo*)state)) {
        return coro;
      }
      Ci_PyWaitHandle_Release(coro);
      return NULL;
    }
  }
done:
  return _PyClassLoader_NewAwaitableWrapper(
      coro, eager, (PyObject*)state, rettype_cb, NULL);
}

__attribute__((__used__)) PyObject* type_vtable_classmethod_vectorcall(
    PyObject* state,
    PyObject* const* args,
    Py_ssize_t nargsf);

__attribute__((__used__)) _PyClassLoader_StaticCallReturn
type_vtable_coroutine_property_native(
    _PyClassLoader_TypeCheckState* state,
    void** args) {
  PyFunctionObject* original =
      (PyFunctionObject*)state->tcs_rt.rt_base.mt_original;

  PyCodeObject* code = (PyCodeObject*)original->func_code;
  Py_ssize_t arg_count = code->co_argcount;
  PyObject* call_args[arg_count];
  PyObject* free_args[arg_count];

  if (hydrate_args(code, arg_count, args, call_args, free_args) < 0) {
    return StaticError;
  }
  _PyClassLoader_StaticCallReturn res;
  res.rax =
      type_vtable_coroutine_property_vectorcall(state, call_args, arg_count);
  res.rdx = (void*)(uint64_t)(res.rax != NULL);
  free_hydrated_args(free_args, arg_count);

  return res;
}

VTABLE_THUNK(type_vtable_coroutine_property, _PyClassLoader_TypeCheckState)

__attribute__((__used__)) PyObject*
type_vtable_coroutine_classmethod_vectorcall(
    _PyClassLoader_TypeCheckState* state,
    PyObject* const* args,
    size_t nargsf) {
  PyObject* callable = PyTuple_GET_ITEM(state->tcs_value, 0);
  PyObject* coro;
#if PY_VERSION_HEX < 0x030C0000
  Py_ssize_t awaited = nargsf & Ci_Py_AWAITED_CALL_MARKER;
#else
  UPGRADE_ASSERT(AWAITED_FLAG)
  Py_ssize_t awaited = 0;
#endif

  if (Py_TYPE(callable) == &PyClassMethod_Type) {
    // We need to do some special set up for class methods when invoking.
    coro = type_vtable_classmethod_vectorcall(state->tcs_value, args, nargsf);
  } else if (Py_TYPE(callable)->tp_descr_get != NULL) {
    PyObject* self = args[0];
    PyObject* get = Py_TYPE(callable)->tp_descr_get(
        callable, self, (PyObject*)Py_TYPE(self));
    if (get == NULL) {
      return NULL;
    }

    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    coro = _PyObject_Vectorcall(get, args + 1, (nargs - 1), NULL);
    Py_DECREF(get);
  } else {
    // In this case, we have a patched class method, and the self has been
    // handled via descriptors already.
    coro = _PyObject_Vectorcall(
        callable,
        args + 1,
        (PyVectorcall_NARGS(nargsf) - 1) | PY_VECTORCALL_ARGUMENTS_OFFSET |
            awaited,
        NULL);
  }

  if (coro == NULL) {
    return NULL;
  }

  int eager = Ci_PyWaitHandle_CheckExact(coro);
  if (eager) {
    Ci_PyWaitHandleObject* handle = (Ci_PyWaitHandleObject*)coro;
    if (handle->wh_waiter == NULL) {
      if (rettype_check(
              Py_TYPE(callable),
              handle->wh_coro_or_result,
              (_PyClassLoader_RetTypeInfo*)state)) {
        return coro;
      }
      Ci_PyWaitHandle_Release(coro);
      return NULL;
    }
  }

  return _PyClassLoader_NewAwaitableWrapper(
      coro, eager, (PyObject*)state, rettype_cb, NULL);
}

__attribute__((__used__)) _PyClassLoader_StaticCallReturn
type_vtable_coroutine_classmethod_native(
    _PyClassLoader_TypeCheckState* state,
    void** args,
    Py_ssize_t nargsf) {
  PyObject* original = state->tcs_rt.rt_base.mt_original;
  assert(Py_TYPE(original) == &PyClassMethod_Type);
  PyFunctionObject* callable =
      (PyFunctionObject*)Ci_PyClassMethod_GetFunc(original);

  PyCodeObject* code = (PyCodeObject*)callable->func_code;
  Py_ssize_t arg_count = code->co_argcount;
  PyObject* call_args[arg_count];
  PyObject* free_args[arg_count];

  if (hydrate_args(code, arg_count, args, call_args, free_args) < 0) {
    return StaticError;
  }

  int optional, exact, func_flags;
  PyTypeObject* type = resolve_function_rettype(
      (PyObject*)callable, &optional, &exact, &func_flags);

  _PyClassLoader_StaticCallReturn res = return_to_native(
      type_vtable_coroutine_classmethod_vectorcall(state, call_args, arg_count),
      type);
  free_hydrated_args(free_args, arg_count);
  return res;
}

VTABLE_THUNK(type_vtable_coroutine_classmethod, _PyClassLoader_TypeCheckState)

__attribute__((__used__)) PyObject* type_vtable_coroutine_vectorcall(
    _PyClassLoader_TypeCheckState* state,
    PyObject* const* args,
    size_t nargsf) {
  PyObject* coro;
  PyObject* callable = state->tcs_value;
  if (PyFunction_Check(callable)) {
    coro = _PyObject_Vectorcall(callable, args, nargsf, NULL);
  } else if (Py_TYPE(callable) == &PyClassMethod_Type) {
    // We need to do some special set up for class methods when invoking.
    callable = Ci_PyClassMethod_GetFunc(state->tcs_value);
    coro = _PyObject_Vectorcall(callable, args, nargsf, NULL);
  } else if (Py_TYPE(callable)->tp_descr_get != NULL) {
    PyObject* self = args[0];
    PyObject* get = Py_TYPE(callable)->tp_descr_get(
        callable, self, (PyObject*)Py_TYPE(self));
    if (get == NULL) {
      return NULL;
    }

    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    coro = _PyObject_Vectorcall(get, args + 1, (nargs - 1), NULL);
    Py_DECREF(get);
  } else {
    // self isn't passed if we're not a descriptor
    coro = _PyObject_Vectorcall(callable, args + 1, nargsf - 1, NULL);
  }
  if (coro == NULL) {
    return NULL;
  }

  int eager = Ci_PyWaitHandle_CheckExact(coro);
  if (eager) {
    Ci_PyWaitHandleObject* handle = (Ci_PyWaitHandleObject*)coro;
    if (handle->wh_waiter == NULL) {
      if (rettype_check(
              Py_TYPE(callable),
              handle->wh_coro_or_result,
              (_PyClassLoader_RetTypeInfo*)state)) {
        return coro;
      }
      Ci_PyWaitHandle_Release(coro);
      return NULL;
    }
  }

  return _PyClassLoader_NewAwaitableWrapper(
      coro, eager, (PyObject*)state, rettype_cb, NULL);
}

__attribute__((__used__)) _PyClassLoader_StaticCallReturn
type_vtable_coroutine_native(
    _PyClassLoader_TypeCheckState* state,
    void** args) {
  PyFunctionObject* original =
      (PyFunctionObject*)state->tcs_rt.rt_base.mt_original;

  PyCodeObject* code = (PyCodeObject*)original->func_code;
  Py_ssize_t arg_count = code->co_argcount;
  PyObject* call_args[arg_count];
  PyObject* free_args[arg_count];

  if (hydrate_args(code, arg_count, args, call_args, free_args) < 0) {
    return StaticError;
  }
  _PyClassLoader_StaticCallReturn res;
  res.rax = type_vtable_coroutine_vectorcall(state, call_args, arg_count);
  res.rdx = (void*)(uint64_t)(res.rax != NULL);
  free_hydrated_args(free_args, arg_count);
  return res;
}

VTABLE_THUNK(type_vtable_coroutine, _PyClassLoader_TypeCheckState)

__attribute__((__used__)) PyObject* type_vtable_nonfunc_property_vectorcall(
    _PyClassLoader_TypeCheckState* state,
    PyObject** args,
    size_t nargsf) {
  PyObject* self = args[0];
  PyObject* descr = state->tcs_value;
  PyObject* name = state->tcs_rt.rt_name;
  PyObject* res;

  /* we have to perform the descriptor checks at runtime because the
   * descriptor type can be modified preventing us from being able to have
   * more optimized fast paths */
  if (!PyDescr_IsData(descr)) {
    PyObject** dictptr = _PyObject_GetDictPtr(self);
    if (dictptr != NULL) {
      PyObject* dict = *dictptr;
      if (dict != NULL) {
        res = PyDict_GetItem(dict, PyTuple_GET_ITEM(name, 0));
        if (res != NULL) {
          Py_INCREF(res);
          goto done;
        }
      }
    }
  }

  if (Py_TYPE(descr)->tp_descr_get != NULL) {
    PyObject* self = args[0];
    PyObject* get =
        Py_TYPE(descr)->tp_descr_get(descr, self, (PyObject*)Py_TYPE(self));
    if (get == NULL) {
      return NULL;
    }

    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    res = _PyObject_Vectorcall(
        get, args + 1, (nargs - 1) | PY_VECTORCALL_ARGUMENTS_OFFSET, NULL);
    Py_DECREF(get);
    goto done;
  }
  res = _PyObject_Vectorcall(descr, args, nargsf, NULL);
done:
  return rettype_check(Py_TYPE(self), res, (_PyClassLoader_RetTypeInfo*)state);
}

__attribute__((__used__)) _PyClassLoader_StaticCallReturn
type_vtable_nonfunc_property_native(
    _PyClassLoader_TypeCheckState* state,
    void** args) {
  PyObject* original = state->tcs_rt.rt_base.mt_original;
  Py_ssize_t arg_count = get_original_argcount(&original);
  if (arg_count < 0) {
    return StaticError;
  }
  PyObject* call_args[arg_count];
  PyObject* free_args[arg_count];
  // We can have a property-like object which doesn't have an original function,
  // for example a typed descriptor with a value.  TODO: Can one of those be a
  // primitive?
  if (PyFunction_Check(original)) {
    PyCodeObject* code =
        (PyCodeObject*)((PyFunctionObject*)original)->func_code;

    if (hydrate_args(code, arg_count, args, call_args, free_args) < 0) {
      return StaticError;
    }
  } else {
    for (Py_ssize_t i = 0; i < arg_count; i++) {
      call_args[i] = (PyObject*)args[i];
      free_args[i] = 0;
    }
  }
  PyObject* obj =
      type_vtable_nonfunc_property_vectorcall(state, call_args, arg_count);
  free_hydrated_args(free_args, arg_count);
  return return_to_native(
      obj, ((_PyClassLoader_RetTypeInfo*)state)->rt_expected);
}

VTABLE_THUNK(type_vtable_nonfunc_property, _PyClassLoader_TypeCheckState)

__attribute__((__used__)) PyObject* type_vtable_nonfunc_vectorcall(
    _PyClassLoader_TypeCheckState* state,
    PyObject** args,
    size_t nargsf) {
  PyObject* self = args[0];
  PyObject* descr = state->tcs_value;
  PyObject* name = state->tcs_rt.rt_name;
  PyObject* res;
  /* we have to perform the descriptor checks at runtime because the
   * descriptor type can be modified preventing us from being able to have
   * more optimized fast paths */
  if (!PyDescr_IsData(descr)) {
    PyObject** dictptr = _PyObject_GetDictPtr(self);
    if (dictptr != NULL) {
      PyObject* dict = *dictptr;
      if (dict != NULL) {
        PyObject* value = PyDict_GetItem(dict, name);
        if (value != NULL) {
          /* descriptor was overridden by instance value */
          Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
          res = _PyObject_Vectorcall(value, args + 1, nargs - 1, NULL);
          goto done;
        }
      }
    }
  }

  if (Py_TYPE(descr)->tp_descr_get != NULL) {
    PyObject* self = args[0];
    PyObject* get =
        Py_TYPE(descr)->tp_descr_get(descr, self, (PyObject*)Py_TYPE(self));
    if (get == NULL) {
      return NULL;
    }

    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    res = _PyObject_Vectorcall(
        get, args + 1, (nargs - 1) | PY_VECTORCALL_ARGUMENTS_OFFSET, NULL);
    Py_DECREF(get);
    goto done;
  }
  res = _PyObject_Vectorcall(descr, args + 1, nargsf - 1, NULL);
done:
  return rettype_check(Py_TYPE(self), res, (_PyClassLoader_RetTypeInfo*)state);
}

__attribute__((__used__)) _PyClassLoader_StaticCallReturn
type_vtable_nonfunc_native(_PyClassLoader_TypeCheckState* state, void** args) {
  PyObject* original = state->tcs_rt.rt_base.mt_original;
  Py_ssize_t arg_count = get_original_argcount(&original);
  if (arg_count < 0) {
    return StaticError;
  }

  PyCodeObject* code = (PyCodeObject*)((PyFunctionObject*)original)->func_code;
  PyObject* call_args[arg_count];
  PyObject* free_args[arg_count];

  if (hydrate_args(code, arg_count, args, call_args, free_args) < 0) {
    return StaticError;
  }
  PyObject* obj = type_vtable_nonfunc_vectorcall(state, call_args, arg_count);
  free_hydrated_args(free_args, arg_count);
  return return_to_native(
      obj, ((_PyClassLoader_RetTypeInfo*)state)->rt_expected);
}

VTABLE_THUNK(type_vtable_nonfunc, _PyClassLoader_TypeCheckState)

__attribute__((__used__)) PyObject*
type_vtable_descr_vectorcall(PyObject* descr, PyObject** args, size_t nargsf) {
  return _PyObject_Vectorcall(descr, args, nargsf, NULL);
}

__attribute__((__used__)) _PyClassLoader_StaticCallReturn
type_vtable_descr_native(PyObject* descr, void** args) {
  _PyClassLoader_StaticCallReturn res;
  res.rax = type_vtable_descr_vectorcall(descr, (PyObject**)args, 1);
  res.rdx = (void*)(uint64_t)(res.rax != NULL);
  return res;
}

VTABLE_THUNK(type_vtable_descr, PyObject)

__attribute__((__used__)) PyObject* vtable_static_function_vectorcall(
    PyObject* state,
    PyObject** args,
    Py_ssize_t nargsf) {
  return Ci_PyFunction_CallStatic((PyFunctionObject*)state, args, nargsf, NULL);
}

__attribute__((__used__)) _PyClassLoader_StaticCallReturn
vtable_static_function_native(PyObject* state, void** args) {
  return invoke_from_native(state, state, args);
}

VTABLE_THUNK(vtable_static_function, PyObject)

__attribute__((__used__)) _PyClassLoader_StaticCallReturn
vtable_arg_thunk_ret_primitive_non_jitted_native(PyObject* state, void** args) {
  PyFunctionObject* func = (PyFunctionObject*)PyTuple_GET_ITEM(state, 0);
  PyTypeObject* ret_type = (PyTypeObject*)PyTuple_GET_ITEM(state, 1);
  Py_ssize_t arg_count = ((PyCodeObject*)func->func_code)->co_argcount;

  _PyClassLoader_StaticCallReturn res;
  PyObject* call_args[arg_count];
  PyObject* free_args[arg_count];

  if (hydrate_args(
          (PyCodeObject*)func->func_code,
          arg_count,
          args,
          call_args,
          free_args)) {
    res.rdx = NULL;
    return res;
  }

  PyObject* obj = func->vectorcall((PyObject*)func, call_args, arg_count, NULL);
  free_hydrated_args(free_args, arg_count);
  if (obj != NULL) {
    res.rax =
        (void*)_PyClassLoader_Unbox(obj, _PyClassLoader_GetTypeCode(ret_type));
  }
  res.rdx = (void*)(uint64_t)(obj != NULL);
  return res;
}

__attribute__((__used__)) PyObject*
vtable_arg_thunk_ret_primitive_non_jitted_vectorcall(
    PyObject* state,
    PyObject** args,
    Py_ssize_t nargsf) {
  PyFunctionObject* func = (PyFunctionObject*)PyTuple_GET_ITEM(state, 0);

  return func->vectorcall((PyObject*)func, args, nargsf, NULL);
}

VTABLE_THUNK(vtable_arg_thunk_ret_primitive_non_jitted, PyObject)

__attribute__((__used__)) void* vtable_arg_thunk_vectorcall_only_vectorcall(
    PyObject* state,
    PyObject** args,
    Py_ssize_t nargsf) {
  return PyObject_Vectorcall(state, args, nargsf, NULL);
}

__attribute__((__used__)) void* vtable_arg_thunk_vectorcall_only_native(
    PyObject* state,
    void** args) {
  PyErr_SetString(PyExc_RuntimeError, "unsupported native call");
  return NULL;
}

VTABLE_THUNK(vtable_arg_thunk_vectorcall_only, PyObject)

PyObject* _PyClassLoader_InvokeMethod(
    _PyType_VTable* vtable,
    Py_ssize_t slot,
    PyObject** args,
    Py_ssize_t nargsf) {
  vectorcallfunc func =
      JITRT_GET_NORMAL_ENTRY_FROM_STATIC(vtable->vt_entries[slot].vte_entry);
  PyObject* state = vtable->vt_entries[slot].vte_state;
  return func(state, args, nargsf, NULL);
}

__attribute__((__used__)) PyObject* type_vtable_func_overridable_vectorcall(
    _PyClassLoader_TypeCheckState* state,
    PyObject** args,
    size_t nargsf) {
  PyObject* self = args[0];
  PyObject** dictptr = _PyObject_GetDictPtr(self);
  PyObject* dict = dictptr != NULL ? *dictptr : NULL;
  PyObject* res;
  if (dict != NULL) {
    /* ideally types using INVOKE_METHOD are defined w/o out dictionaries,
     * which allows us to avoid this lookup.  If they're not then we'll
     * fallback to supporting looking in the dictionary */
    PyObject* name = state->tcs_rt.rt_name;
    PyObject* callable = PyDict_GetItem(dict, name);
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    if (callable != NULL) {
      res = _PyObject_Vectorcall(
          callable,
          args + 1,
          (nargs - 1) | PY_VECTORCALL_ARGUMENTS_OFFSET,
          NULL);
      goto done;
    }
  }

  res = _PyObject_Vectorcall(state->tcs_value, (PyObject**)args, nargsf, NULL);

done:
  return rettype_check(Py_TYPE(self), res, (_PyClassLoader_RetTypeInfo*)state);
}

__attribute__((__used__)) _PyClassLoader_StaticCallReturn
type_vtable_func_overridable_native(
    _PyClassLoader_TypeCheckState* state,
    void** args) {
  PyFunctionObject* func =
      (PyFunctionObject*)((_PyClassLoader_MethodThunk*)state)->mt_original;
  Py_ssize_t arg_count = ((PyCodeObject*)func->func_code)->co_argcount;
  PyObject* call_args[arg_count];
  PyObject* free_args[arg_count];

  if (hydrate_args(
          (PyCodeObject*)func->func_code,
          arg_count,
          args,
          call_args,
          free_args) < 0) {
    return StaticError;
  }

  PyObject* obj = type_vtable_func_overridable_vectorcall(
      state, call_args, ((PyCodeObject*)func->func_code)->co_argcount);
  free_hydrated_args(free_args, ((PyCodeObject*)func->func_code)->co_argcount);
  return return_to_native(
      obj, ((_PyClassLoader_RetTypeInfo*)state)->rt_expected);
}

VTABLE_THUNK(type_vtable_func_overridable, _PyClassLoader_TypeCheckState)

static inline int is_static_entry(vectorcallfunc func) {
  return func == (vectorcallfunc)Ci_StaticFunction_Vectorcall;
}

void set_entry_from_func(_PyType_VTableEntry* entry, PyFunctionObject* func) {
  assert(_PyClassLoader_IsStaticFunction((PyObject*)func));
  if (is_static_entry(func->vectorcall)) {
    /* this will always be invoked statically via the v-table */
    entry->vte_entry = (vectorcallfunc)vtable_static_function_dont_bolt;
  } else {
    assert(_PyJIT_IsCompiled(func));
    entry->vte_entry = JITRT_GET_STATIC_ENTRY(func->vectorcall);
  }
}

/**
    This vectorcall entrypoint pulls out the function, slot index and replaces
    its own entrypoint in the v-table with optimized static vectorcall. (It also
    calls the underlying function and returns the value while doing so).
*/
__attribute__((__used__)) PyObject* type_vtable_func_lazyinit_vectorcall(
    PyObject* state,
    PyObject** args,
    Py_ssize_t nargsf) {
  /* state is (vtable, index, function) */
  _PyType_VTable* vtable = (_PyType_VTable*)PyTuple_GET_ITEM(state, 0);
  long index = PyLong_AS_LONG(PyTuple_GET_ITEM(state, 1));
  PyFunctionObject* func = (PyFunctionObject*)PyTuple_GET_ITEM(state, 2);

  PyObject* res =
      func->vectorcall((PyObject*)func, (PyObject**)args, nargsf, NULL);
  if (vtable->vt_entries[index].vte_state == state) {
    vtable->vt_entries[index].vte_state = (PyObject*)func;
    set_entry_from_func(&vtable->vt_entries[index], func);
    Py_INCREF(func);
    Py_DECREF(state);
  }
  return res;
}

__attribute__((__used__)) _PyClassLoader_StaticCallReturn
type_vtable_func_lazyinit_native(PyObject* state, void** args) {
  PyFunctionObject* func = (PyFunctionObject*)PyTuple_GET_ITEM(state, 2);
  PyCodeObject* code = (PyCodeObject*)((PyFunctionObject*)func)->func_code;
  Py_ssize_t arg_count = code->co_argcount;
  PyObject* call_args[arg_count];
  PyObject* free_args[arg_count];

  if (hydrate_args(code, arg_count, args, call_args, free_args) < 0) {
    return StaticError;
  }

  PyObject* res =
      type_vtable_func_lazyinit_vectorcall(state, call_args, arg_count);
  free_hydrated_args(free_args, arg_count);
  int optional, exact, func_flags;
  PyTypeObject* type =
      resolve_function_rettype((PyObject*)func, &optional, &exact, &func_flags);
  return return_to_native(res, type);
}

VTABLE_THUNK(type_vtable_func_lazyinit, PyObject)

__attribute__((__used__)) PyObject* type_vtable_staticmethod_vectorcall(
    PyObject* method,
    PyObject** args,
    Py_ssize_t nargsf) {
  PyObject* func = Ci_PyStaticMethod_GetFunc(method);

  return _PyObject_Vectorcall(func, ((PyObject**)args) + 1, nargsf - 1, NULL);
}

__attribute__((__used__)) _PyClassLoader_StaticCallReturn
type_vtable_staticmethod_native(PyObject* method, void** args) {
  PyObject* func = Ci_PyStaticMethod_GetFunc(method);
  PyCodeObject* code = (PyCodeObject*)((PyFunctionObject*)func)->func_code;
  Py_ssize_t arg_count =
      code->co_argcount + 1; // hydrate self and then we'll drop it
  PyObject* call_args[arg_count];
  PyObject* free_args[arg_count];

  if (hydrate_args(code, arg_count, args, call_args, free_args) < 0) {
    return StaticError;
  }

  PyObject* res =
      type_vtable_staticmethod_vectorcall(method, call_args, arg_count);
  free_hydrated_args(free_args, arg_count);
  int optional, exact, func_flags;
  PyTypeObject* type =
      resolve_function_rettype((PyObject*)func, &optional, &exact, &func_flags);
  return return_to_native(res, type);
}

VTABLE_THUNK(type_vtable_staticmethod, PyObject)

__attribute__((__used__)) PyObject*
type_vtable_staticmethod_overridable_vectorcall(
    PyObject* thunk,
    PyObject** args,
    Py_ssize_t nargsf) {
  PyObject* method = ((_PyClassLoader_TypeCheckState*)thunk)->tcs_value;
  PyObject* func = Ci_PyStaticMethod_GetFunc(method);

  return _PyObject_Vectorcall(func, ((PyObject**)args) + 1, nargsf - 1, NULL);
}

__attribute__((__used__)) _PyClassLoader_StaticCallReturn
type_vtable_staticmethod_overridable_native(PyObject* thunk, void** args) {
  PyObject* original = Ci_PyStaticMethod_GetFunc(
      ((_PyClassLoader_MethodThunk*)thunk)->mt_original);
  PyObject* method = ((_PyClassLoader_TypeCheckState*)thunk)->tcs_value;
  PyObject* func = Ci_PyStaticMethod_GetFunc(method);
  return invoke_from_native(original, func, args);
}

VTABLE_THUNK(type_vtable_staticmethod_overridable, PyObject)

#define _PyClassMethod_Check(op) (Py_TYPE(op) == &PyClassMethod_Type)

__attribute__((__used__)) PyObject* type_vtable_classmethod_vectorcall(
    PyObject* state,
    PyObject* const* args,
    Py_ssize_t nargsf) {
  PyObject* classmethod = PyTuple_GET_ITEM(state, 0);
  PyTypeObject* decltype = (PyTypeObject*)PyTuple_GET_ITEM(state, 1);
  PyObject* func = Ci_PyClassMethod_GetFunc(classmethod);
  if (!PyObject_TypeCheck(args[0], decltype)) {
    return _PyObject_Vectorcall(func, args, nargsf, NULL);
  }

  Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
  PyObject* stack[nargs];
  stack[0] = (PyObject*)Py_TYPE(args[0]);
  for (Py_ssize_t i = 1; i < nargs; i++) {
    stack[i] = args[i];
  }
  return _PyObject_Vectorcall(func, stack, nargsf, NULL);
}

__attribute__((__used__)) _PyClassLoader_StaticCallReturn
type_vtable_classmethod_native(PyObject* state, void** args) {
  PyObject* classmethod = PyTuple_GET_ITEM(state, 0);
  PyTypeObject* decltype = (PyTypeObject*)PyTuple_GET_ITEM(state, 1);
  PyObject* func = Ci_PyClassMethod_GetFunc(classmethod);
  PyCodeObject* code = (PyCodeObject*)((PyFunctionObject*)func)->func_code;
  Py_ssize_t arg_count = code->co_argcount;
  PyObject* call_args[arg_count];
  PyObject* free_args[arg_count];

  if (hydrate_args(code, arg_count, args, call_args, free_args) < 0) {
    return StaticError;
  }
  if (PyObject_TypeCheck(call_args[0], decltype)) {
    call_args[0] = (PyObject*)Py_TYPE(call_args[0]);
  }

  PyObject* res =
      ((PyFunctionObject*)func)->vectorcall(func, call_args, arg_count, NULL);
  free_hydrated_args(free_args, arg_count);

  int optional, exact, func_flags;
  PyTypeObject* type =
      resolve_function_rettype(func, &optional, &exact, &func_flags);
  return return_to_native(res, type);
}

VTABLE_THUNK(type_vtable_classmethod, PyObject)

__attribute__((__used__)) PyObject*
type_vtable_classmethod_overridable_vectorcall(
    _PyClassLoader_TypeCheckState* state,
    PyObject** args,
    size_t nargsf) {
  PyObject* clsmethod = PyTuple_GET_ITEM(state->tcs_value, 0);
  if (_PyClassMethod_Check(clsmethod)) {
    return type_vtable_classmethod_vectorcall(state->tcs_value, args, nargsf);
  }
  // Invoked via an instance, we need to check its dict to see if the
  // classmethod was overridden.
  PyObject* self = args[0];
  PyObject** dictptr = _PyObject_GetDictPtr(self);
  PyObject* dict = dictptr != NULL ? *dictptr : NULL;
  PyObject* res;
  if (dict != NULL) {
    /* ideally types using INVOKE_METHOD are defined w/o out dictionaries,
     * which allows us to avoid this lookup.  If they're not then we'll
     * fallback to supporting looking in the dictionary */
    PyObject* name = state->tcs_rt.rt_name;
    PyObject* callable = PyDict_GetItem(dict, name);
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    if (callable != NULL) {
      res = _PyObject_Vectorcall(
          callable,
          args + 1,
          (nargs - 1) | PY_VECTORCALL_ARGUMENTS_OFFSET,
          NULL);
      return rettype_check(
          Py_TYPE(self), res, (_PyClassLoader_RetTypeInfo*)state);
    }
  }

  return _PyObject_Vectorcall(clsmethod, args, nargsf, NULL);
}

__attribute__((__used__)) _PyClassLoader_StaticCallReturn
type_vtable_classmethod_overridable_native(
    _PyClassLoader_TypeCheckState* state,
    void** args) {
  PyObject* original = state->tcs_rt.rt_base.mt_original;
  PyFunctionObject* func =
      (PyFunctionObject*)Ci_PyClassMethod_GetFunc(original);

  PyCodeObject* code = (PyCodeObject*)func->func_code;
  Py_ssize_t arg_count = code->co_argcount;
  PyObject* call_args[arg_count];
  PyObject* free_args[arg_count];

  if (hydrate_args(code, arg_count, args, call_args, free_args) < 0) {
    return StaticError;
  }
  PyObject* obj = type_vtable_classmethod_overridable_vectorcall(
      state, call_args, arg_count);
  free_hydrated_args(free_args, arg_count);
  return return_to_native(
      obj, ((_PyClassLoader_RetTypeInfo*)state)->rt_expected);
}

VTABLE_THUNK(type_vtable_classmethod_overridable, _PyClassLoader_TypeCheckState)

__attribute__((__used__)) _PyClassLoader_StaticCallReturn
type_vtable_func_missing_native(PyObject* state, void** args) {
  PyFunctionObject* original = (PyFunctionObject*)PyTuple_GET_ITEM(state, 3);
  PyCodeObject* code = (PyCodeObject*)original->func_code;
  PyObject* call_args[code->co_argcount];
  PyObject* free_args[code->co_argcount];

  if (hydrate_args(code, code->co_argcount, args, call_args, free_args)) {
    return StaticError;
  }

  PyObject* self = call_args[0];
  PyObject* name = PyTuple_GET_ITEM(state, 0);
  PyErr_Format(
      PyExc_AttributeError,
      "'%s' object has no attribute %R",
      Py_TYPE(self)->tp_name,
      name);
  free_hydrated_args(free_args, code->co_argcount);
  return StaticError;
}

__attribute__((__used__)) void* type_vtable_func_missing_vectorcall(
    PyObject* state,
    PyObject** args,
    Py_ssize_t nargsf) {
  PyObject* self = args[0];
  PyObject* name = PyTuple_GET_ITEM(state, 0);
  PyErr_Format(
      PyExc_AttributeError,
      "'%s' object has no attribute %R",
      Py_TYPE(self)->tp_name,
      name);
  return NULL;
}

VTABLE_THUNK(type_vtable_func_missing, PyObject)

/**
    This does the initialization of the vectorcall entrypoint for the v-table
   for static functions. It'll set the entrypoint to type_vtable_func_lazyinit
   if the functions entry point hasn't yet been initialized.

    If it has been initialized and is being handled by the interpreter loop
   it'll go through the single Ci_PyFunction_CallStatic entry point. Otherwise
   it'll just use the function entry point, which should be JITed.
*/
// TODO: Drop name argument
static int type_vtable_set_opt_slot(
    PyTypeObject* tp,
    PyObject* name,
    _PyType_VTable* vtable,
    Py_ssize_t slot,
    PyObject* value) {
  vectorcallfunc entry = ((PyFunctionObject*)value)->vectorcall;
  if (entry == (vectorcallfunc)Ci_JIT_lazyJITInitFuncObjectVectorcall) {
    /* entry point isn't initialized yet, we want to run it once, and
     * then update our own entry point */
    int optional, exact, func_flags;
    PyTypeObject* ret_type = (PyTypeObject*)_PyClassLoader_ResolveReturnType(
        value, &optional, &exact, &func_flags);
    int type_code = _PyClassLoader_GetTypeCode(ret_type);

    PyObject* state = PyTuple_New(type_code != TYPED_OBJECT ? 4 : 3);
    if (state == NULL) {
      return -1;
    }
    PyTuple_SET_ITEM(state, 0, (PyObject*)vtable);
    Py_INCREF(vtable);
    PyObject* new_index = PyLong_FromSize_t(slot);
    if (new_index == NULL) {
      Py_DECREF(state);
      return -1;
    }
    PyTuple_SET_ITEM(state, 1, new_index);
    PyTuple_SET_ITEM(state, 2, value);
    if (type_code != TYPED_OBJECT) {
      PyTuple_SET_ITEM(state, 3, (PyObject*)ret_type);
      Py_INCREF(ret_type);
    }
    Py_INCREF(value);
    Py_XDECREF(vtable->vt_entries[slot].vte_state);
    vtable->vt_entries[slot].vte_state = state;
    vtable->vt_entries[slot].vte_entry =
        (vectorcallfunc)type_vtable_func_lazyinit_dont_bolt;
  } else if (entry == (vectorcallfunc)_PyFunction_Vectorcall) {
    // non-JITed function, it could return a primitive in which case we need a
    // stub to unbox the value.
    int optional, exact, func_flags;
    PyTypeObject* ret_type = (PyTypeObject*)_PyClassLoader_ResolveReturnType(
        value, &optional, &exact, &func_flags);
    int type_code = _PyClassLoader_GetTypeCode(ret_type);

    if (type_code != TYPED_OBJECT) {
      PyObject* tuple = PyTuple_New(2);
      if (tuple == NULL) {
        return -1;
      }
      PyTuple_SET_ITEM(tuple, 0, value);
      Py_INCREF(value);
      PyTuple_SET_ITEM(tuple, 1, (PyObject*)ret_type);
      Py_INCREF(ret_type);
      vtable->vt_entries[slot].vte_state = tuple;
      vtable->vt_entries[slot].vte_entry =
          (vectorcallfunc)vtable_arg_thunk_ret_primitive_non_jitted_dont_bolt;
    } else {
      Py_XDECREF(vtable->vt_entries[slot].vte_state);
      vtable->vt_entries[slot].vte_state = value;
      set_entry_from_func(&vtable->vt_entries[slot], (PyFunctionObject*)value);
      Py_INCREF(value);
    }
  } else {
    Py_XDECREF(vtable->vt_entries[slot].vte_state);
    vtable->vt_entries[slot].vte_state = value;
    set_entry_from_func(&vtable->vt_entries[slot], (PyFunctionObject*)value);
    Py_INCREF(value);
  }
  return 0;
}


static PyObject* g_missing_fget = NULL;
static PyObject* g_missing_fset = NULL;

static PyObject* classloader_get_property_missing_fget() {
  if (g_missing_fget == NULL) {
    PyObject* mod = PyImport_ImportModule("_static");
    if (mod == NULL) {
      return NULL;
    }
    PyObject* func = PyObject_GetAttrString(mod, "_property_missing_fget");
    Py_DECREF(mod);
    if (func == NULL) {
      return NULL;
    }
    g_missing_fget = func;
  }
  return g_missing_fget;
}

static PyObject* classloader_maybe_unwrap_callable(PyObject* func) {
  if (func != NULL) {
    PyObject* res;
    if (Py_TYPE(func) == &PyStaticMethod_Type) {
      res = Ci_PyStaticMethod_GetFunc(func);
      Py_INCREF(res);
      return res;
    } else if (Py_TYPE(func) == &PyClassMethod_Type) {
      res = Ci_PyClassMethod_GetFunc(func);
      Py_INCREF(res);
      return res;
    } else if (Py_TYPE(func) == &PyProperty_Type) {
      Ci_propertyobject* prop = (Ci_propertyobject*)func;
      // A "callable" usually refers to the read path
      res = prop->prop_get;
      Py_INCREF(res);
      return res;
    }
  }
  return NULL;
}

static PyObject* classloader_get_property_missing_fset() {
  if (g_missing_fset == NULL) {
    PyObject* mod = PyImport_ImportModule("_static");
    if (mod == NULL) {
      return NULL;
    }
    PyObject* func = PyObject_GetAttrString(mod, "_property_missing_fset");
    Py_DECREF(mod);
    if (func == NULL) {
      return NULL;
    }
    g_missing_fset = func;
  }
  return g_missing_fset;
}

static PyObject* classloader_ensure_specials_cache(PyTypeObject* type) {
  _PyType_VTable* vtable = _PyClassLoader_EnsureVtable(type, 0);
  if (vtable == NULL) {
    return NULL;
  }
  PyObject* specials = vtable->vt_specials;
  if (specials == NULL) {
    specials = vtable->vt_specials = PyDict_New();
    if (specials == NULL) {
      return NULL;
    }
  }

  return specials;
}

/* Stores a newly created special thunk in the special thunk cache.  If it fails
 * to store decref the thunk and return NULL */
static PyObject* classloader_cache_new_special(
    PyTypeObject* type,
    PyObject* name,
    PyObject* special) {
  if (type == NULL) {
    return special;
  }
  PyObject* specials = classloader_ensure_specials_cache(type);
  if (specials == NULL) {
    return NULL;
  }

  if (PyDict_SetItem(specials, name, special)) {
    Py_DECREF(special);
    return NULL;
  }
  return special;
}

static PyObject* classloader_get_property_fget(
    PyTypeObject* type,
    PyObject* name,
    PyObject* property) {
  if (Py_TYPE(property) == &PyProperty_Type) {
    PyObject* func = ((Ci_propertyobject*)property)->prop_get;
    if (func == NULL) {
      func = classloader_get_property_missing_fget();
    }
    Py_XINCREF(func);
    return func;
  } else if (Py_TYPE(property) == &PyCachedPropertyWithDescr_Type) {
    _Py_CachedPropertyThunk* thunk = _Py_CachedPropertyThunk_New(property);
    if (thunk == NULL) {
      return NULL;
    }

    return classloader_cache_new_special(type, name, (PyObject*)thunk);
  } else if (Py_TYPE(property) == &PyAsyncCachedPropertyWithDescr_Type) {
    _Py_AsyncCachedPropertyThunk* thunk = _Py_AsyncCachedPropertyThunk_New(property);
    if (thunk == NULL) {
      return NULL;
    }

    return classloader_cache_new_special(type, name, (PyObject*)thunk);
  } else if (Py_TYPE(property) == &_PyTypedDescriptorWithDefaultValue_Type) {
    PyObject *thunk = _PyClassLoader_TypedDescriptorThunkGet_New(property);
    if (thunk == NULL) {
      return NULL;
    }
    return classloader_cache_new_special(type, name, thunk);
  } else {
    PyObject *thunk = _PyClassLoader_PropertyThunkGet_New(property);
    if (thunk == NULL) {
      return NULL;
    }
    return classloader_cache_new_special(type, name, thunk);
  }
}

static PyObject* classloader_get_property_fset(
    PyTypeObject* type,
    PyObject* name,
    PyObject* property) {
  if (Py_TYPE(property) == &PyProperty_Type) {
    PyObject* func = ((Ci_propertyobject*)property)->prop_set;
    if (func == NULL) {
      func = classloader_get_property_missing_fset();
    }
    Py_XINCREF(func);
    return func;
  } else if (
      Py_TYPE(property) == &PyCachedPropertyWithDescr_Type ||
      Py_TYPE(property) == &PyAsyncCachedPropertyWithDescr_Type) {
    PyObject* func = classloader_get_property_missing_fset();
    Py_XINCREF(func);
    return func;
  } else if (Py_TYPE(property) == &_PyTypedDescriptorWithDefaultValue_Type) {
    PyObject *thunk = _PyClassLoader_TypedDescriptorThunkSet_New(property);
    if (thunk == NULL) {
      return NULL;
    }
    return classloader_cache_new_special(type, name, thunk);
  } else {
    PyObject *thunk = _PyClassLoader_PropertyThunkSet_New(property);
    if (thunk == NULL) {
      return NULL;
    }
    return classloader_cache_new_special(type, name, thunk);
  }
}

static PyObject* classloader_get_property_method(
    PyTypeObject* type,
    PyObject* property,
    PyTupleObject* name) {
  PyObject* fname = PyTuple_GET_ITEM(name, 1);
  if (_PyUnicode_EqualToASCIIString(fname, "fget")) {
    return classloader_get_property_fget(type, (PyObject*)name, property);
  } else if (_PyUnicode_EqualToASCIIString(fname, "fset")) {
    return classloader_get_property_fset(type, (PyObject*)name, property);
  }
  PyErr_Format(
      PyExc_RuntimeError, "bad property method name %R in classloader", fname);
  return NULL;
}

PyTypeObject* resolve_function_rettype(
    PyObject* funcobj,
    int* optional,
    int* exact,
    int* func_flags) {
  assert(PyFunction_Check(funcobj));
  PyFunctionObject* func = (PyFunctionObject*)funcobj;
  if (((PyCodeObject*)func->func_code)->co_flags & CO_COROUTINE) {
    *func_flags |= Ci_FUNC_FLAGS_COROUTINE;
  }
  return _PyClassLoader_ResolveType(
      _PyClassLoader_GetReturnTypeDescr(func), optional, exact);
}

PyObject* _PyClassLoader_GetReturnTypeDescr(PyFunctionObject* func) {
  return _PyClassLoader_GetCodeReturnTypeDescr((PyCodeObject*)func->func_code);
}

PyObject* _PyClassLoader_GetCodeReturnTypeDescr(PyCodeObject* code) {
  // last element of consts is ((arg_checks, ...), ret_type)
  PyObject* static_type_info =
      PyTuple_GET_ITEM(code->co_consts, PyTuple_GET_SIZE(code->co_consts) - 1);

  return PyTuple_GET_ITEM(static_type_info, 1);
}

PyObject* _PyClassLoader_GetCodeArgumentTypeDescrs(PyCodeObject* code) {
  // last element of consts is ((arg_checks, ...), ret_type)
  PyObject* static_type_info =
      PyTuple_GET_ITEM(code->co_consts, PyTuple_GET_SIZE(code->co_consts) - 1);

  return PyTuple_GET_ITEM(static_type_info, 0);
}

static int _PyClassLoader_TypeCheckState_traverse(
    _PyClassLoader_TypeCheckState* op,
    visitproc visit,
    void* arg) {
  rettype_check_traverse((_PyClassLoader_RetTypeInfo*)op, visit, arg);
  visit(op->tcs_value, arg);
  visit(op->tcs_rt.rt_base.mt_original, arg);
  return 0;
}

static int _PyClassLoader_TypeCheckState_clear(
    _PyClassLoader_TypeCheckState* op) {
  rettype_check_clear((_PyClassLoader_RetTypeInfo*)op);
  Py_CLEAR(op->tcs_value);
  Py_CLEAR(op->tcs_rt.rt_base.mt_original);
  return 0;
}

static void _PyClassLoader_TypeCheckState_dealloc(
    _PyClassLoader_TypeCheckState* op) {
  PyObject_GC_UnTrack((PyObject*)op);
  rettype_check_clear((_PyClassLoader_RetTypeInfo*)op);
  Py_XDECREF(op->tcs_value);
  Py_XDECREF(op->tcs_rt.rt_base.mt_original);
  PyObject_GC_Del((PyObject*)op);
}

PyTypeObject _PyType_TypeCheckState = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0) "vtable_state_obj",
    sizeof(_PyClassLoader_TypeCheckState),
    .tp_dealloc = (destructor)_PyClassLoader_TypeCheckState_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_BASETYPE |
        _Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_traverse = (traverseproc)_PyClassLoader_TypeCheckState_traverse,
    .tp_clear = (inquiry)_PyClassLoader_TypeCheckState_clear,
};

static void _PyClassLoader_MethodThunk_dealloc(_PyClassLoader_MethodThunk* op) {
  PyObject_GC_UnTrack((PyObject*)op);
  Py_XDECREF(op->mt_original);
  PyObject_GC_Del((PyObject*)op);
}

static int _PyClassLoader_MethodThunk_traverse(
    _PyClassLoader_MethodThunk* op,
    visitproc visit,
    void* arg) {
  visit(op->mt_original, arg);
  return 0;
}

static int _PyClassLoader_MethodThunk_clear(_PyClassLoader_MethodThunk* op) {
  Py_CLEAR(op->mt_original);
  return 0;
}

PyTypeObject _PyType_MethodThunk = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0) "vtable_method_thunk",
    sizeof(_PyClassLoader_MethodThunk),
    .tp_dealloc = (destructor)_PyClassLoader_MethodThunk_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_BASETYPE,
    .tp_traverse = (traverseproc)_PyClassLoader_MethodThunk_traverse,
    .tp_clear = (inquiry)_PyClassLoader_MethodThunk_clear,
};

static int type_vtable_setslot_typecheck(
    PyTypeObject* decltype,
    PyObject* ret_type,
    int optional,
    int exact,
    int func_flags,
    PyObject* name,
    _PyType_VTable* vtable,
    Py_ssize_t slot,
    PyObject* value,
    PyObject* original) {
  _PyClassLoader_TypeCheckState* state =
      PyObject_GC_New(_PyClassLoader_TypeCheckState, &_PyType_TypeCheckState);
  if (state == NULL) {
    return -1;
  }
  state->tcs_value = value;
  Py_INCREF(value);
  state->tcs_rt.rt_name = name;
  Py_INCREF(name);
  state->tcs_rt.rt_expected = (PyTypeObject*)ret_type;
  Py_INCREF(ret_type);
  state->tcs_rt.rt_optional = optional;
  state->tcs_rt.rt_exact = exact;
  state->tcs_rt.rt_base.mt_original = original;
  Py_INCREF(original);

  Py_XDECREF(vtable->vt_entries[slot].vte_state);
  vtable->vt_entries[slot].vte_state = (PyObject*)state;
  if (func_flags & Ci_FUNC_FLAGS_COROUTINE) {
    if (func_flags & Ci_FUNC_FLAGS_CLASSMETHOD) {
      PyObject* tuple = PyTuple_New(2);
      if (tuple == NULL) {
        Py_DECREF(state);
        return -1;
      }
      PyTuple_SET_ITEM(tuple, 0, value);
      PyTuple_SET_ITEM(tuple, 1, (PyObject*)decltype);
      Py_INCREF(decltype);
      state->tcs_value = tuple;
      vtable->vt_entries[slot].vte_entry =
          (vectorcallfunc)type_vtable_coroutine_classmethod_dont_bolt;
    } else if (
        PyTuple_Check(name) &&
        classloader_is_property_tuple((PyTupleObject*)name)) {
      vtable->vt_entries[slot].vte_entry =
          (vectorcallfunc)type_vtable_coroutine_property_dont_bolt;
    } else {
      vtable->vt_entries[slot].vte_entry =
          (vectorcallfunc)type_vtable_coroutine_dont_bolt;
    }
  } else if (
      PyTuple_Check(name) &&
      classloader_is_property_tuple((PyTupleObject*)name)) {
    vtable->vt_entries[slot].vte_entry =
        (vectorcallfunc)type_vtable_nonfunc_property_dont_bolt;
  } else if (PyFunction_Check(value)) {
    vtable->vt_entries[slot].vte_entry =
        (vectorcallfunc)type_vtable_func_overridable_dont_bolt;
  } else if (func_flags & Ci_FUNC_FLAGS_CLASSMETHOD) {
    PyObject* tuple = PyTuple_New(2);
    if (tuple == NULL) {
      Py_DECREF(state);
      return -1;
    }
    PyTuple_SET_ITEM(tuple, 0, value);
    PyTuple_SET_ITEM(tuple, 1, (PyObject*)decltype);
    Py_INCREF(decltype);
    state->tcs_value = tuple;
    vtable->vt_entries[slot].vte_entry =
        (vectorcallfunc)type_vtable_classmethod_overridable_dont_bolt;
  } else {
    vtable->vt_entries[slot].vte_entry =
        (vectorcallfunc)type_vtable_nonfunc_dont_bolt;
  }
  return 0;
}

/**
    As the name suggests, this creates v-tables for all subclasses of the given
   type (recursively).
*/
static int type_init_subclass_vtables(PyTypeObject* target_type) {
  /* TODO: This can probably be a lot more efficient.  If a type
   * hasn't been fully loaded yet we can probably propagate the
   * parent dict down, and either initialize the slot to the parent
   * slot (if not overridden) or initialize the slot to the child slot.
   * We then only need to populate the child dict w/ its members when
   * a member is accessed from the child type.  When we init the child
   * we can check if it's dict sharing with its parent. */
  PyObject* ref;
  PyObject* subclasses = target_type->tp_subclasses;
  if (subclasses != NULL) {
    Py_ssize_t i = 0;
    while (PyDict_Next(subclasses, &i, NULL, &ref)) {
      assert(PyWeakref_CheckRef(ref));
      ref = PyWeakref_GET_OBJECT(ref);
      if (ref == Py_None) {
        continue;
      }

      PyTypeObject* subtype = (PyTypeObject*)ref;
      if (subtype->tp_cache != NULL) {
        /* already inited */
        continue;
      }

      _PyType_VTable* vtable = _PyClassLoader_EnsureVtable(subtype, 1);
      if (vtable == NULL) {
        return -1;
      }
    }
  }
  return 0;
}

static void _PyClassLoader_UpdateDerivedSlot(
    PyTypeObject* type,
    PyObject* name,
    Py_ssize_t index,
    PyObject* state,
    vectorcallfunc func) {
  /* Update any derived types which don't have slots */
  PyObject* ref;
  PyObject* subclasses = type->tp_subclasses;
  if (subclasses != NULL) {
    Py_ssize_t i = 0;
    while (PyDict_Next(subclasses, &i, NULL, &ref)) {
      assert(PyWeakref_CheckRef(ref));
      ref = PyWeakref_GET_OBJECT(ref);
      if (ref == Py_None) {
        continue;
      }

      PyTypeObject* subtype = (PyTypeObject*)ref;
      PyObject* override = PyDict_GetItem(_PyType_GetDict(subtype), name);
      if (override != NULL) {
        /* subtype overrides the value */
        continue;
      }

      assert(subtype->tp_cache != NULL);
      _PyType_VTable* subvtable = (_PyType_VTable*)subtype->tp_cache;
      Py_XDECREF(subvtable->vt_entries[index].vte_state);
      subvtable->vt_entries[index].vte_state = state;
      Py_INCREF(state);
      subvtable->vt_entries[index].vte_entry = func;

      _PyClassLoader_UpdateDerivedSlot(subtype, name, index, state, func);
    }
  }
}

static int thunktraverse(_Py_StaticThunk* op, visitproc visit, void* arg) {
  rettype_check_traverse((_PyClassLoader_RetTypeInfo*)op, visit, arg);
  Py_VISIT(op->thunk_tcs.tcs_value);
  Py_VISIT((PyObject*)op->thunk_cls);
  return 0;
}

static int thunkclear(_Py_StaticThunk* op) {
  rettype_check_clear((_PyClassLoader_RetTypeInfo*)op);
  Py_CLEAR(op->thunk_tcs.tcs_value);
  Py_CLEAR(op->thunk_cls);
  return 0;
}

static void thunkdealloc(_Py_StaticThunk* op) {
  PyObject_GC_UnTrack((PyObject*)op);
  rettype_check_clear((_PyClassLoader_RetTypeInfo*)op);
  Py_XDECREF(op->thunk_tcs.tcs_value);
  Py_XDECREF(op->thunk_cls);
  PyObject_GC_Del((PyObject*)op);
}

static void set_thunk_type_error(_Py_StaticThunk* thunk, const char* msg) {
  PyObject* name = thunk->thunk_tcs.tcs_rt.rt_name;
  if (thunk->thunk_cls != NULL) {
    name = PyUnicode_FromFormat("%s.%U", thunk->thunk_cls->tp_name, name);
  }
  PyErr_Format(CiExc_StaticTypeError, msg, name);
  if (thunk->thunk_cls != NULL) {
    Py_DECREF(name);
  }
}

PyObject* thunk_vectorcall(
    _Py_StaticThunk* thunk,
    PyObject* const* args,
    size_t nargsf,
    PyObject* kwnames) {
  PyObject* func = thunk->thunk_tcs.tcs_value;
  if (func == NULL) {
    set_thunk_type_error(thunk, "%U has been deleted");
    return NULL;
  }
  if (thunk->thunk_flags & Ci_FUNC_FLAGS_CLASSMETHOD) {
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    if (nargs == 0) {
      set_thunk_type_error(thunk, "%U must be invoked with >= 1 arguments");
      return NULL;
    }

    if (thunk->thunk_flags & Ci_FUNC_FLAGS_COROUTINE) {
      return type_vtable_coroutine_vectorcall(
          (_PyClassLoader_TypeCheckState*)thunk, args, nargs);
    }
    PyObject* res = _PyObject_Vectorcall(func, args + 1, nargs - 1, kwnames);
    return rettype_check(
        thunk->thunk_cls, res, (_PyClassLoader_RetTypeInfo*)thunk);
  }

  if (!(thunk->thunk_flags & Ci_FUNC_FLAGS_STATICMETHOD) &&
      !PyFunction_Check(func)) {
    PyObject* callable;
    if (Py_TYPE(func)->tp_descr_get != NULL) {
      PyObject* self = args[0];
      callable =
          Py_TYPE(func)->tp_descr_get(func, self, (PyObject*)Py_TYPE(self));
      if (callable == NULL) {
        return NULL;
      }
    } else {
      Py_INCREF(func);
      callable = func;
    }

    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    PyObject* res =
        _PyObject_Vectorcall(callable, args + 1, (nargs - 1), kwnames);
    Py_DECREF(callable);

    if (thunk->thunk_flags & Ci_FUNC_FLAGS_COROUTINE) {
      return _PyClassLoader_NewAwaitableWrapper(
          res, 0, (PyObject*)thunk, rettype_cb, NULL);
    }
    return rettype_check(
        thunk->thunk_cls, res, (_PyClassLoader_RetTypeInfo*)thunk);
  }

#if PY_VERSION_HEX < 0x030C0000
  if (thunk->thunk_flags & Ci_FUNC_FLAGS_COROUTINE) {
    PyObject* coro = _PyObject_Vectorcall(
        func, args, nargsf & ~Ci_Py_AWAITED_CALL_MARKER, kwnames);

    return _PyClassLoader_NewAwaitableWrapper(
        coro, 0, (PyObject*)thunk, rettype_cb, NULL);
  }

  PyObject* res = _PyObject_Vectorcall(
      func, args, nargsf & ~Ci_Py_AWAITED_CALL_MARKER, kwnames);
  return rettype_check(
      thunk->thunk_cls, res, (_PyClassLoader_RetTypeInfo*)thunk);
#else
  UPGRADE_ASSERT(AWAITED_FLAG)
  return NULL;
#endif
}

int get_func_or_special_callable(
    PyTypeObject* type,
    PyObject* name,
    PyObject** result);

int _PyClassLoader_InitTypeForPatching(PyTypeObject* type) {
  if (!(type->tp_flags & Ci_Py_TPFLAGS_IS_STATICALLY_DEFINED)) {
    return 0;
  }
  _PyType_VTable* vtable = (_PyType_VTable*)type->tp_cache;
  if (vtable != NULL && vtable->vt_original != NULL) {
    return 0;
  }
  if (_PyClassLoader_EnsureVtable(type, 0) == NULL) {
    return -1;
  }
  vtable = (_PyType_VTable*)type->tp_cache;

  PyObject *name, *slot, *clsitem;
  PyObject* slotmap = vtable->vt_slotmap;
  PyObject* origitems = vtable->vt_original = PyDict_New();

  Py_ssize_t i = 0;
  while (PyDict_Next(slotmap, &i, &name, &slot)) {
    if (get_func_or_special_callable(type, name, &clsitem)) {
      return -1;
    }
    if (clsitem != NULL) {
      if (PyDict_SetItem(origitems, name, clsitem)) {
        Py_DECREF(clsitem);
        goto error;
      }
      Py_DECREF(clsitem);
    }
  }
  return 0;
error:
  vtable->vt_original = NULL;
  Py_DECREF(origitems);
  return -1;
}

static PyObject* classloader_get_static_type(const char* name) {
  PyObject* mod = PyImport_ImportModule("__static__");
  if (mod == NULL) {
    return NULL;
  }
  PyObject* type = PyObject_GetAttrString(mod, name);
  Py_DECREF(mod);
  return type;
}

PyObject* _PyClassLoader_ResolveReturnType(
    PyObject* func,
    int* optional,
    int* exact,
    int* func_flags) {
  *optional = *exact = *func_flags = 0;
  PyTypeObject* res = NULL;
  if (PyFunction_Check(func)) {
    if (_PyClassLoader_IsStaticFunction(func)) {
      res = resolve_function_rettype(func, optional, exact, func_flags);
    } else {
      res = &PyBaseObject_Type;
    }
  } else if (Py_TYPE(func) == &PyStaticMethod_Type) {
    PyObject* static_func = Ci_PyStaticMethod_GetFunc(func);
    if (_PyClassLoader_IsStaticFunction(static_func)) {
      res = resolve_function_rettype(static_func, optional, exact, func_flags);
    }
    *func_flags |= Ci_FUNC_FLAGS_STATICMETHOD;
  } else if (Py_TYPE(func) == &PyClassMethod_Type) {
    PyObject* static_func = Ci_PyClassMethod_GetFunc(func);
    if (_PyClassLoader_IsStaticFunction(static_func)) {
      res = resolve_function_rettype(static_func, optional, exact, func_flags);
    }
    *func_flags |= Ci_FUNC_FLAGS_CLASSMETHOD;
  } else if (Py_TYPE(func) == &PyProperty_Type) {
    Ci_propertyobject* property = (Ci_propertyobject*)func;
    PyObject* fget = property->prop_get;
    if (_PyClassLoader_IsStaticFunction(fget)) {
      res = resolve_function_rettype(fget, optional, exact, func_flags);
    }
  } else if (Py_TYPE(func) == &_PyType_CachedPropertyThunk) {
    PyObject* target = _Py_CachedPropertyThunk_GetFunc(func);
    if (_PyClassLoader_IsStaticFunction(target)) {
      res = resolve_function_rettype(target, optional, exact, func_flags);
    }
  } else if (Py_TYPE(func) == &_PyType_AsyncCachedPropertyThunk) {
    PyObject* target = _Py_AsyncCachedPropertyThunk_GetFunc(func);
    if (_PyClassLoader_IsStaticFunction(target)) {
      res = resolve_function_rettype(target, optional, exact, func_flags);
    }
  } else if (Py_TYPE(func) == &PyCachedPropertyWithDescr_Type) {
    PyCachedPropertyDescrObject* property = (PyCachedPropertyDescrObject*)func;
    if (_PyClassLoader_IsStaticFunction(property->func)) {
      res =
          resolve_function_rettype(property->func, optional, exact, func_flags);
    }
  } else if (Py_TYPE(func) == &PyAsyncCachedPropertyWithDescr_Type) {
    PyAsyncCachedPropertyDescrObject* property =
        (PyAsyncCachedPropertyDescrObject*)func;
    if (_PyClassLoader_IsStaticFunction(property->func)) {
      res =
          resolve_function_rettype(property->func, optional, exact, func_flags);
    }
  } else if (Py_TYPE(func) == &_PyType_TypedDescriptorThunk) {
    _Py_TypedDescriptorThunk* thunk = (_Py_TypedDescriptorThunk*)func;
    if (thunk->is_setter) {
      res = &_PyNone_Type;
      Py_INCREF(res);
    } else {
      _PyTypedDescriptorWithDefaultValue* td =
          (_PyTypedDescriptorWithDefaultValue*)
              thunk->typed_descriptor_thunk_target;
      if (PyTuple_CheckExact(td->td_type)) {
        res = _PyClassLoader_ResolveType(
            td->td_type, &td->td_optional, &td->td_exact);
        *optional = td->td_optional;
        *exact = td->td_exact;
      } else { // Already resolved.
        assert(PyType_CheckExact(td->td_type));
        res = (PyTypeObject*)td->td_type;
        *optional = td->td_optional;
      }
      if (res == NULL) {
        return NULL;
      }
    }
  } else if (Py_TYPE(func) == &_PyTypedDescriptorWithDefaultValue_Type) {
    _PyTypedDescriptorWithDefaultValue* td =
        (_PyTypedDescriptorWithDefaultValue*)func;
    if (PyTuple_CheckExact(td->td_type)) {
      res = _PyClassLoader_ResolveType(
          td->td_type, &td->td_optional, &td->td_exact);
      *optional = td->td_optional;
      *exact = td->td_exact;
    } else { // Already resolved.
      assert(PyType_CheckExact(td->td_type));
      res = (PyTypeObject*)td->td_type;
      *optional = td->td_optional;
      *exact = td->td_exact;
    }
    if (res == NULL) {
      return NULL;
    }
  } else if (Py_TYPE(func) == &_PyType_StaticThunk) {
    _Py_StaticThunk* sthunk = (_Py_StaticThunk*)func;
    res = sthunk->thunk_tcs.tcs_rt.rt_expected;
    *optional = sthunk->thunk_tcs.tcs_rt.rt_optional;
    *exact = sthunk->thunk_tcs.tcs_rt.rt_exact;
    Py_INCREF(res);
  } else {
    Ci_PyTypedMethodDef* tmd = _PyClassLoader_GetTypedMethodDef(func);
    *optional = 0;
    if (tmd != NULL) {
      switch (tmd->tmd_ret) {
        case Ci_Py_SIG_VOID:
        case Ci_Py_SIG_ERROR: {
          // The underlying C implementations of these functions don't
          // produce a Python object at all, but we ensure (in
          // _PyClassLoader_ConvertRet and in JIT HIR builder) that
          // when we call them we produce a None.
          *exact = 0;
          res = (PyTypeObject*)&_PyNone_Type;
          break;
        }
        case Ci_Py_SIG_STRING: {
          *exact = 0;
          res = &PyUnicode_Type;
          break;
        }
        case Ci_Py_SIG_INT8: {
          *exact = 1;
          return classloader_get_static_type("int8");
        }
        case Ci_Py_SIG_INT16: {
          *exact = 1;
          return classloader_get_static_type("int16");
        }
        case Ci_Py_SIG_INT32: {
          *exact = 1;
          return classloader_get_static_type("int32");
        }
        case Ci_Py_SIG_INT64: {
          *exact = 1;
          return classloader_get_static_type("int64");
        }
        case Ci_Py_SIG_UINT8: {
          *exact = 1;
          return classloader_get_static_type("uint8");
        }
        case Ci_Py_SIG_UINT16: {
          *exact = 1;
          return classloader_get_static_type("uint16");
        }
        case Ci_Py_SIG_UINT32: {
          *exact = 1;
          return classloader_get_static_type("uint32");
        }
        case Ci_Py_SIG_UINT64: {
          *exact = 1;
          return classloader_get_static_type("uint64");
        }
        default: {
          *exact = 0;
          res = &PyBaseObject_Type;
        }
      }
      Py_INCREF(res);
    } else if (Py_TYPE(func) == &PyMethodDescr_Type) {
      // We emit invokes to untyped builtin methods; just assume they
      // return object.
      *exact = 0;
      res = &PyBaseObject_Type;
      Py_INCREF(res);
    }
  }
  return (PyObject*)res;
}

int get_func_or_special_callable(
    PyTypeObject* type,
    PyObject* name,
    PyObject** result) {
  PyObject* dict = _PyType_GetDict(type);
  if (PyTuple_CheckExact(name)) {
    if (classloader_is_property_tuple((PyTupleObject*)name)) {
      _PyType_VTable* vtable = (_PyType_VTable*)type->tp_cache;
      if (vtable != NULL) {
        PyObject* specials = vtable->vt_specials;
        if (specials != NULL) {
          *result = PyDict_GetItem(specials, name);
          if (*result != NULL) {
            Py_INCREF(*result);
            return 0;
          }
        }
      }

      PyObject* property = PyDict_GetItem(dict, PyTuple_GET_ITEM(name, 0));
      if (property == NULL) {
        *result = NULL;
        return 0;
      }
      *result =
          classloader_get_property_method(type, property, (PyTupleObject*)name);
      if (*result == NULL) {
        return -1;
      }
      return 0;
    }
  }
  *result = PyDict_GetItem(dict, name);
  Py_XINCREF(*result);
  return 0;
}

int _PyClassLoader_IsPatchedThunk(PyObject* obj) {
  if (obj != NULL && Py_TYPE(obj) == &_PyType_StaticThunk) {
    return 1;
  }
  return 0;
}

int is_static_type(PyTypeObject* type);

/*
    Looks up through parent classes to find a member specified by the name. If a
   parent class attribute has been patched, that is ignored, i.e it goes through
   the originally defined members.
*/
int _PyClassLoader_GetStaticallyInheritedMember(
    PyTypeObject* type,
    PyObject* name,
    PyObject** result) {
  PyObject *mro = type->tp_mro, *base;

  for (Py_ssize_t i = 1; i < PyTuple_GET_SIZE(mro); i++) {
    PyTypeObject* next = (PyTypeObject*)PyTuple_GET_ITEM(type->tp_mro, i);
    if (!is_static_type(next)) {
      continue;
    }
    if (next->tp_cache != NULL &&
        ((_PyType_VTable*)next->tp_cache)->vt_original != NULL) {
      /* if we've initialized originals it contains all of our possible slot
       * values including special callables. */
      base =
          PyDict_GetItem(((_PyType_VTable*)next->tp_cache)->vt_original, name);
      if (base == NULL) {
        continue;
      }
      assert(used_in_vtable(base));
      Py_INCREF(base);
      *result = base;
      return 0;
    } else if (_PyType_GetDict(next) == NULL) {
      continue;
    } else if (get_func_or_special_callable(next, name, &base)) {
      return -1;
    }

    if (base != NULL) {
      *result = base;
      return 0;
    }
  }
  *result = NULL;
  return 0;
}

static PyObject* g_fget = NULL;
static PyObject* g_fset = NULL;

PyObject* get_descr_tuple(PyObject* name, PyObject* accessor) {
  PyObject* getter_tuple = PyTuple_New(2);
  Py_INCREF(name);
  PyTuple_SET_ITEM(getter_tuple, 0, name);
  Py_INCREF(accessor);
  PyTuple_SET_ITEM(getter_tuple, 1, accessor);
  return getter_tuple;
}

PyObject* get_property_getter_descr_tuple(PyObject* name) {
  if (g_fget == NULL) {
    g_fget = PyUnicode_FromStringAndSize("fget", 4);
  }
  return get_descr_tuple(name, g_fget);
}

PyObject* get_property_setter_descr_tuple(PyObject* name) {
  if (g_fset == NULL) {
    g_fset = PyUnicode_FromStringAndSize("fset", 4);
  }
  return get_descr_tuple(name, g_fset);
}

static void
update_thunk(_Py_StaticThunk* thunk, PyObject* previous, PyObject* new_value) {
  Py_CLEAR(thunk->thunk_tcs.tcs_value);
  if (new_value != NULL) {
    PyObject* unwrapped_new = classloader_maybe_unwrap_callable(new_value);
    if (unwrapped_new != NULL) {
      thunk->thunk_tcs.tcs_value = unwrapped_new;
    } else {
      thunk->thunk_tcs.tcs_value = new_value;
      Py_INCREF(new_value);
    }
  }
  PyObject* funcref;
  if (new_value == previous) {
    funcref = previous;
  } else {
    funcref = (PyObject*)thunk;
  }
  PyObject* unwrapped = classloader_maybe_unwrap_callable(funcref);
  if (unwrapped != NULL) {
    thunk->thunk_funcref = unwrapped;
    Py_DECREF(unwrapped);
  } else {
    thunk->thunk_funcref = funcref;
  }
}

/* Static types have a slot containing all final methods in their inheritance
   chain. This function returns the contents of that slot by looking up the MRO,
   if it exists.
 */
static PyObject* get_final_method_names(PyTypeObject* type) {
  PyObject* mro = type->tp_mro;
  if (mro == NULL) {
    return NULL;
  }
  Py_ssize_t n = PyTuple_GET_SIZE(mro);
  for (Py_ssize_t i = 0; i < n; i++) {
    PyObject* mro_type = PyTuple_GET_ITEM(mro, i);
    if (((PyTypeObject*)mro_type)->tp_flags &
        Ci_Py_TPFLAGS_IS_STATICALLY_DEFINED) {
      _Py_IDENTIFIER(__final_method_names__);
      PyObject* final_method_names_string =
          _PyUnicode_FromId(&PyId___final_method_names__);
      PyObject* final_method_names = _PyObject_GenericGetAttrWithDict(
          mro_type,
          final_method_names_string,
          /*dict=*/NULL,
          /*suppress=*/1);
      return final_method_names;
    }
  }
  return NULL;
}

int _PyClassLoader_IsFinalMethodOverridden(
    PyTypeObject* base_type,
    PyObject* members_dict) {
  PyObject* final_method_names = get_final_method_names(base_type);
  if (final_method_names == NULL) {
    return 0;
  }
  if (!PyTuple_Check(final_method_names)) {
    PyErr_Format(
        CiExc_StaticTypeError,
        "The __final_method_names__ slot for type %R is not a tuple.",
        final_method_names);
    Py_DECREF(final_method_names);
    return -1;
  }
  Py_ssize_t member_pos = 0;
  PyObject *key, *value;
  while (PyDict_Next(members_dict, &member_pos, &key, &value)) {
    for (Py_ssize_t final_method_index = 0;
         final_method_index < PyTuple_GET_SIZE(final_method_names);
         final_method_index++) {
      PyObject* current_final_method_name =
          PyTuple_GET_ITEM(final_method_names, final_method_index);
      int compare_result = PyUnicode_Compare(key, current_final_method_name);
      if (compare_result == 0) {
        PyErr_Format(
            CiExc_StaticTypeError,
            "%R overrides a final method in the static base class %R",
            key,
            base_type);
        Py_DECREF(final_method_names);
        return -1;
      } else if (compare_result == -1 && PyErr_Occurred()) {
        return -1;
      }
    }
  }
  Py_DECREF(final_method_names);
  return 0;
}

static int check_if_final_method_overridden(
    PyTypeObject* type,
    PyObject* name) {
  PyTypeObject* base_type = type->tp_base;
  if (base_type == NULL) {
    return 0;
  }
  PyObject* final_method_names = get_final_method_names(base_type);
  if (final_method_names == NULL) {
    return 0;
  }
  if (!PyTuple_Check(final_method_names)) {
    PyErr_Format(
        CiExc_StaticTypeError,
        "The __final_method_names__ slot for type %R is not a tuple.",
        final_method_names);
    Py_DECREF(final_method_names);
    return -1;
  }
  for (Py_ssize_t final_method_index = 0;
       final_method_index < PyTuple_GET_SIZE(final_method_names);
       final_method_index++) {
    PyObject* current_final_method_name =
        PyTuple_GET_ITEM(final_method_names, final_method_index);
    int compare_result = PyUnicode_Compare(name, current_final_method_name);
    if (compare_result == 0) {
      PyErr_Format(
          CiExc_StaticTypeError,
          "%R overrides a final method in the static base class %R",
          name,
          base_type);
      Py_DECREF(final_method_names);
      return -1;
    } else if (compare_result == -1 && PyErr_Occurred()) {
      Py_DECREF(final_method_names);
      return -1;
    }
  }
  Py_DECREF(final_method_names);
  return 0;
}

/* UpdateModuleName will be called on any patching of a name in a StrictModule.
 */
int _PyClassLoader_UpdateModuleName(
    Ci_StrictModuleObject* mod,
    PyObject* name,
    PyObject* new_value) {
  if (mod->static_thunks != NULL) {
    _Py_StaticThunk* thunk =
        (_Py_StaticThunk*)PyDict_GetItem(mod->static_thunks, name);
    if (thunk != NULL) {
      PyObject* previous = PyDict_GetItem(mod->originals, name);
      update_thunk(thunk, previous, new_value);
    }
  }
  return 0;
}

int populate_getter_and_setter(
    PyTypeObject* type,
    PyObject* name,
    PyObject* new_value) {
  PyObject* getter_value = new_value == NULL
      ? NULL
      : classloader_get_property_fget(type, name, new_value);
  PyObject* setter_value = new_value == NULL
      ? NULL
      : classloader_get_property_fset(type, name, new_value);

  PyObject* getter_tuple = get_property_getter_descr_tuple(name);
  PyObject* setter_tuple = get_property_setter_descr_tuple(name);

  int result = 0;
  if (_PyClassLoader_UpdateSlot(type, (PyObject*)getter_tuple, getter_value)) {
    result = -1;
  }
  Py_DECREF(getter_tuple);
  Py_XDECREF(getter_value);

  if (_PyClassLoader_UpdateSlot(type, (PyObject*)setter_tuple, setter_value)) {
    result = -1;
  }
  Py_DECREF(setter_tuple);
  Py_XDECREF(setter_value);

  return result;
}

static int classloader_get_original_static_def(
    PyTypeObject* tp,
    PyObject* name,
    PyObject** original) {
  _PyType_VTable* vtable = (_PyType_VTable*)tp->tp_cache;
  *original = NULL;
  if (is_static_type(tp)) {
    if (vtable->vt_original != NULL) {
      *original = PyDict_GetItem(vtable->vt_original, name);
      if (*original != NULL) {
        Py_INCREF(*original);
        return 0;
      }
    } else if (get_func_or_special_callable(tp, name, original)) {
      return -1;
    }
    // If a static type has a non-static member (for instance, due to having a
    // decorated method) we need to keep looking up the MRO for a static base.
    if (*original == NULL || !used_in_vtable(*original)) {
      Py_CLEAR(*original);
    }
  }

  if (*original == NULL) {
    // The member was actually defined in one of the parent classes, so try to
    // look it up from there.
    // TODO: It might be possible to avoid the type-check in this situation,
    // because while `tp` was patched, the parent Static classes may not be.
    if (_PyClassLoader_GetStaticallyInheritedMember(tp, name, original)) {
      return -1;
    }
  }
  return 0;
}

static int type_vtable_setslot(
    PyTypeObject* tp,
    PyObject* name,
    Py_ssize_t slot,
    PyObject* value,
    PyObject* original);

/* The UpdateSlot method will always get called by `tp_setattro` when one of a
   type's attribute gets changed, and serves as an entry point for handling
   modifications to vtables. */
int _PyClassLoader_UpdateSlot(
    PyTypeObject* type,
    PyObject* name,
    PyObject* new_value) {
  /* This check needs to be happen before we look into the vtable, as non-static
     subclasses of static classes won't necessarily have vtables already
     constructed. */
  if (check_if_final_method_overridden(type, name)) {
    return -1;
  }
  _PyType_VTable* vtable = (_PyType_VTable*)type->tp_cache;
  if (vtable == NULL) {
    return 0;
  }

  PyObject* slotmap = vtable->vt_slotmap;
  PyObject* slot = PyDict_GetItem(slotmap, name);
  if (slot == NULL) {
    return 0;
  }

  PyObject* original;
  if (classloader_get_original_static_def(type, name, &original)) {
    return -1;
  }

  /* we need to search in the MRO if we don't contain the
   * item directly or we're currently deleting the current value */
  if (new_value == NULL) {
    /* We need to look for an item explicitly declared in our parent if we're
     * inheriting. Note we don't care about static vs non-static, and we don't
     * want to look at the original values either.  The new value is simply
     * whatever the currently inherited value is. */
    PyObject* mro = type->tp_mro;

    for (Py_ssize_t i = 1; i < PyTuple_GET_SIZE(mro); i++) {
      PyTypeObject* next = (PyTypeObject*)PyTuple_GET_ITEM(type->tp_mro, i);
      PyObject* next_dict = _PyType_GetDict(next);
      if (next_dict == NULL) {
        continue;
      }
      new_value = PyDict_GetItem(next_dict, name);
      if (new_value != NULL) {
        break;
      }
    }
  }

  /* update the value that exists in our thunks for performing indirections
   * necessary for patched INVOKE_FUNCTION calls */
  if (vtable->vt_thunks != NULL) {
    _Py_StaticThunk* thunk =
        (_Py_StaticThunk*)PyDict_GetItem(vtable->vt_thunks, name);
    if (thunk != NULL) {
      update_thunk(thunk, original, new_value);
    }
  }

  assert(original != NULL);

  int cur_optional = 0, cur_exact = 0, cur_func_flags = 0;
  PyObject* cur_type = _PyClassLoader_ResolveReturnType(
      original, &cur_optional, &cur_exact, &cur_func_flags);
  assert(cur_type != NULL);

  // if this is a property slot, also update the getter and setter slots
  if (Py_TYPE(original) == &PyProperty_Type ||
      Py_TYPE(original) == &PyCachedPropertyWithDescr_Type ||
      Py_TYPE(original) == &PyAsyncCachedPropertyWithDescr_Type ||
      Py_TYPE(original) == &_PyTypedDescriptorWithDefaultValue_Type) {
    if (new_value) {
      // If we have a new value, and it's not a descriptor, we can type-check it
      // at the time of assignment.
      PyTypeObject* new_value_type = Py_TYPE(new_value);
      if (new_value_type->tp_descr_get == NULL &&
          !_PyObject_TypeCheckOptional(
              new_value, (PyTypeObject*)cur_type, cur_optional, cur_exact)) {
        PyErr_Format(
            CiExc_StaticTypeError,
            "Cannot assign a %s, because %s.%U is expected to be a %s",
            Py_TYPE(new_value)->tp_name,
            type->tp_name,
            name,
            ((PyTypeObject*)cur_type)->tp_name);
        Py_DECREF(cur_type);
        Py_DECREF(original);
        return -1;
      }
    }
    if (populate_getter_and_setter(type, name, new_value) < 0) {
      Py_DECREF(original);
      return -1;
    }
  }
  Py_DECREF(cur_type);

  Py_ssize_t index = PyLong_AsSsize_t(slot);

  if (type_vtable_setslot(type, name, index, new_value, original)) {
    Py_DECREF(original);
    return -1;
  }

  Py_DECREF(original);

  /* propagate slot update to derived classes that don't override
   * the function (but first, ensure they have initialized vtables) */
  if (type_init_subclass_vtables(type) != 0) {
    return -1;
  }
  _PyClassLoader_UpdateDerivedSlot(
      type,
      name,
      index,
      vtable->vt_entries[index].vte_state,
      vtable->vt_entries[index].vte_entry);
  return 0;
}

/**
    Sets the vtable slot entry for the given method name to the correct type of
   vectorcall. We specialize where possible, but also have a generic fallback
   which checks whether the actual return type matches the declared one (if
   any).
*/
static int type_vtable_setslot(
    PyTypeObject* tp,
    PyObject* name,
    Py_ssize_t slot,
    PyObject* value,
    PyObject* original) {
  _PyType_VTable* vtable = (_PyType_VTable*)tp->tp_cache;
  assert(original != NULL);

  if (original == value) {
    if (tp->tp_dictoffset == 0) {
      // These cases mean that the type instances don't have a __dict__ slot,
      // meaning our compile time type-checks are valid (nothing's been patched)
      // meaning we can omit return type checks at runtime.
      if (_PyClassLoader_IsStaticFunction(value)) {
        return type_vtable_set_opt_slot(tp, name, vtable, slot, value);
      } else if (
          Py_TYPE(value) == &PyStaticMethod_Type &&
          _PyClassLoader_IsStaticFunction(Ci_PyStaticMethod_GetFunc(value))) {
        Py_XSETREF(vtable->vt_entries[slot].vte_state, value);
        vtable->vt_entries[slot].vte_entry =
            (vectorcallfunc)&type_vtable_staticmethod_dont_bolt;
        Py_INCREF(value);
        return 0;
      } else if (
          Py_TYPE(value) == &PyClassMethod_Type &&
          _PyClassLoader_IsStaticFunction(Ci_PyClassMethod_GetFunc(value))) {
        PyObject* tuple = PyTuple_New(2);
        if (tuple == NULL) {
          return -1;
        }
        PyTuple_SET_ITEM(tuple, 0, value);
        PyTuple_SET_ITEM(tuple, 1, (PyObject*)tp);
        Py_INCREF(tp);
        Py_XSETREF(vtable->vt_entries[slot].vte_state, tuple);
        vtable->vt_entries[slot].vte_entry =
            (vectorcallfunc)type_vtable_classmethod_dont_bolt;
        Py_INCREF(value);
        return 0;
      } else if (Py_TYPE(value) == &PyMethodDescr_Type) {
        Py_XSETREF(vtable->vt_entries[slot].vte_state, value);
        vtable->vt_entries[slot].vte_entry =
            (vectorcallfunc)&vtable_arg_thunk_vectorcall_only_dont_bolt;
        Py_INCREF(value);
        return 0;
      }
    }

    if (Py_TYPE(value) == &_PyType_CachedPropertyThunk ||
        Py_TYPE(value) == &_PyType_TypedDescriptorThunk) {
      Py_XSETREF(vtable->vt_entries[slot].vte_state, value);
      vtable->vt_entries[slot].vte_entry =
          (vectorcallfunc)type_vtable_descr_dont_bolt;
      Py_INCREF(value);
      return 0;
    }
  }

  int optional = 0, exact = 0, func_flags = 0;
  PyObject* ret_type = _PyClassLoader_ResolveReturnType(
      original, &optional, &exact, &func_flags);

  if (ret_type == NULL) {
    PyErr_Format(
        PyExc_RuntimeError,
        "missing type annotation on static compiled method %R of %s",
        name,
        tp->tp_name);
    return -1;
  }

  if (value == NULL) {
    PyObject* missing_state = PyTuple_New(4);
    if (missing_state == NULL) {
      Py_DECREF(ret_type);
      return -1;
    }

    PyObject* func_name = classloader_get_func_name(name);
    PyTuple_SET_ITEM(missing_state, 0, func_name);
    PyTuple_SET_ITEM(missing_state, 1, (PyObject*)tp);
    PyObject* optional_obj = optional ? Py_True : Py_False;
    PyTuple_SET_ITEM(missing_state, 2, optional_obj);
    PyTuple_SET_ITEM(missing_state, 3, original);
    Py_INCREF(func_name);
    Py_INCREF(tp);
    Py_INCREF(optional_obj);
    Py_INCREF(original);

    Py_XDECREF(vtable->vt_entries[slot].vte_state);
    vtable->vt_entries[slot].vte_state = missing_state;
    vtable->vt_entries[slot].vte_entry =
        (vectorcallfunc)type_vtable_func_missing_dont_bolt;
    Py_DECREF(ret_type);
    return 0;
  }

  int res = type_vtable_setslot_typecheck(
      tp,
      ret_type,
      optional,
      exact,
      func_flags,
      name,
      vtable,
      slot,
      value,
      original);
  Py_DECREF(ret_type);
  return res;
}

static Py_ssize_t get_original_argcount(PyObject** callable) {
  Py_ssize_t arg_count;
  PyObject* original = *callable;
  if (!PyFunction_Check(original)) {
    if (_PyClassMethod_Check(original)) {
      *callable = Ci_PyClassMethod_GetFunc(original);
      if (!PyFunction_Check(*callable)) {
        PyErr_SetString(PyExc_RuntimeError, "Not a function in a class method");
        return -1;
      }
      arg_count = ((PyCodeObject*)((PyFunctionObject*)*callable)->func_code)
                      ->co_argcount;
    } else if (Py_TYPE(original) == &PyStaticMethod_Type) {
      *callable = Ci_PyStaticMethod_GetFunc(original);
      if (!PyFunction_Check(*callable)) {
        PyErr_SetString(PyExc_RuntimeError, "Not a function in a class method");
        return -1;
      }
      // static method doesn't take self, but it's passed as an argument in an
      // INVOKE_METHOD.
      arg_count = ((PyCodeObject*)((PyFunctionObject*)*callable)->func_code)
                      ->co_argcount +
          1;
    } else if (Py_TYPE(original) == &_PyType_CachedPropertyThunk) {
      arg_count = 1;
      *callable =
          ((PyCachedPropertyDescrObject*)((_Py_CachedPropertyThunk*)original)
               ->propthunk_target)
              ->func;
    } else if (Py_TYPE(original) == &_PyType_TypedDescriptorThunk) {
      // TODO: Test setter case?
      if (((_Py_TypedDescriptorThunk*)original)->is_setter) {
        arg_count = 2;
      } else {
        arg_count = 1;
      }
      *callable =
          ((_Py_TypedDescriptorThunk*)original)->typed_descriptor_thunk_target;
    } else {
      PyErr_Format(PyExc_RuntimeError, "Not a function: %R", original);
      return -1;
    }
  } else {
    arg_count =
        ((PyCodeObject*)((PyFunctionObject*)*callable)->func_code)->co_argcount;
  }
  return arg_count;
}

/**
    This is usually what we use as the initial entrypoint in v-tables. Then,
    when a method is called, this traverses the MRO, finds the correct callable,
    and updates the vtable entry with the correct one (and then calls the
    callable). All following method invokes directly hit the actual callable,
    because the v-table has been updated.
*/
static _PyClassLoader_StaticCallReturn type_vtable_lazyinit_impl(
    PyObject* info,
    void** args,
    Py_ssize_t nargsf,
    int is_native) {
  PyTypeObject* type = (PyTypeObject*)PyTuple_GET_ITEM(info, 1);
  PyObject* name = PyTuple_GET_ITEM(info, 0);
  _PyType_VTable* vtable = (_PyType_VTable*)type->tp_cache;
  PyObject* mro = type->tp_mro;
  Py_ssize_t slot = PyLong_AsSsize_t(PyDict_GetItem(vtable->vt_slotmap, name));

  assert(vtable != NULL);
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(mro); i++) {
    PyObject* value = NULL;
    PyTypeObject* cur_type = (PyTypeObject*)PyTuple_GET_ITEM(mro, i);
    if (get_func_or_special_callable(cur_type, name, &value)) {
      return StaticError;
    }
    if (value != NULL) {
      PyObject* original = NULL;
      if (classloader_get_original_static_def(type, name, &original)) {
        Py_DECREF(value);
        return StaticError;
      }
      if (type_vtable_setslot(type, name, slot, value, original)) {
        Py_XDECREF(original);
        Py_DECREF(value);
        return StaticError;
      }

      _PyClassLoader_StaticCallReturn res;
      if (is_native) {
        PyObject* callable = original;
        Py_ssize_t arg_count = get_original_argcount(&callable);
        if (arg_count < 0) {
          return StaticError;
        }

        PyObject* obj_res;
        if (PyFunction_Check(callable)) {
          PyCodeObject* code =
              (PyCodeObject*)((PyFunctionObject*)callable)->func_code;
          PyObject* call_args[arg_count];
          PyObject* free_args[arg_count];

          if (hydrate_args(code, arg_count, args, call_args, free_args) < 0) {
            return StaticError;
          }

          obj_res =
              _PyClassLoader_InvokeMethod(vtable, slot, call_args, arg_count);
          free_hydrated_args(free_args, arg_count);
          if (obj_res != NULL) {
            int optional = 0, exact = 0, func_flags = 0, type_code;
            PyTypeObject* type = resolve_function_rettype(
                callable, &optional, &exact, &func_flags);
            if (type != NULL &&
                (type_code = _PyClassLoader_GetTypeCode(type)) !=
                    TYPED_OBJECT) {
              res.rax = (void*)_PyClassLoader_Unbox(obj_res, type_code);
            } else {
              res.rax = obj_res;
            }
          } else {
            res.rax = NULL;
          }
          res.rdx = (void*)(uint64_t)(obj_res != NULL);
        } else {
          assert(arg_count < 5);
          res.rax = _PyClassLoader_InvokeMethod(
              vtable, slot, (PyObject**)args, arg_count);
          res.rdx = (void*)(uint64_t)(res.rax != NULL);
        }

      } else {
        res.rax =
            _PyClassLoader_InvokeMethod(vtable, slot, (PyObject**)args, nargsf);
        res.rdx = (void*)(uint64_t)(res.rax != NULL);
      }

      Py_XDECREF(original);
      Py_DECREF(value);
      return res;
    }
  }

  PyErr_Format(
      CiExc_StaticTypeError, "'%s' has no attribute %U", type->tp_name, name);
  return StaticError;
}

__attribute__((__used__)) PyObject* type_vtable_lazyinit_vectorcall(
    PyObject* thunk,
    PyObject** args,
    Py_ssize_t nargsf) {
  return (PyObject*)type_vtable_lazyinit_impl(thunk, (void**)args, nargsf, 0)
      .rax;
}

__attribute__((__used__)) _PyClassLoader_StaticCallReturn
type_vtable_lazyinit_native(PyObject* thunk, void** args) {
  return type_vtable_lazyinit_impl(thunk, args, 0, 1);
}

VTABLE_THUNK(type_vtable_lazyinit, PyObject)

/**
    For every slot in the vtable slotmap, this sets the vectorcall entrypoint
    to `type_vtable_lazyinit`.
*/
int _PyClassLoader_ReinitVtable(PyTypeObject* type, _PyType_VTable* vtable) {
  PyObject *name, *slot;
  PyObject* slotmap = vtable->vt_slotmap;
  Py_ssize_t i = 0;
  while (PyDict_Next(slotmap, &i, &name, &slot)) {
    Py_ssize_t index = PyLong_AsSsize_t(slot);
    PyObject* tuple = PyTuple_New(2);
    if (tuple == NULL) {
      return -1;
    }

    PyTuple_SET_ITEM(tuple, 0, name);
    Py_INCREF(name);
    PyTuple_SET_ITEM(tuple, 1, (PyObject*)type);
    Py_INCREF(type);
    vtable->vt_entries[index].vte_state = tuple;
    vtable->vt_entries[index].vte_entry =
        (vectorcallfunc)type_vtable_lazyinit_dont_bolt;
  }
  return 0;
}

int used_in_vtable_worker(PyObject* value) {
  // we'll emit invokes to untyped builtin methods
  if (Py_TYPE(value) == &PyMethodDescr_Type) {
    return 1;
  } else if (Py_TYPE(value) == &_PyType_CachedPropertyThunk) {
    return used_in_vtable_worker(_Py_CachedPropertyThunk_GetFunc(value));
  } else if (Py_TYPE(value) == &_PyType_AsyncCachedPropertyThunk) {
    return used_in_vtable_worker(_Py_AsyncCachedPropertyThunk_GetFunc(value));
  }
  if (Py_TYPE(value) == &_PyTypedDescriptorWithDefaultValue_Type) {
    return 1;
  }
  if (Py_TYPE(value) == &_PyType_TypedDescriptorThunk) {
    return 1;
  }
  return _PyClassLoader_IsStaticCallable(value);
}

int used_in_vtable(PyObject* value) {
  if (used_in_vtable_worker(value)) {
    return 1;
  } else if (
      Py_TYPE(value) == &PyStaticMethod_Type &&
      used_in_vtable_worker(Ci_PyStaticMethod_GetFunc(value))) {
    return 1;
  } else if (
      Py_TYPE(value) == &PyClassMethod_Type &&
      used_in_vtable_worker(Ci_PyClassMethod_GetFunc(value))) {
    return 1;
  } else if (Py_TYPE(value) == &PyProperty_Type) {
    PyObject* func = ((Ci_propertyobject*)value)->prop_get;
    if (func != NULL && used_in_vtable_worker(func)) {
      return 1;
    }
    func = ((Ci_propertyobject*)value)->prop_set;
    if (func != NULL && used_in_vtable_worker(func)) {
      return 1;
    }
  } else if (Py_TYPE(value) == &PyCachedPropertyWithDescr_Type) {
    PyObject* func = ((PyCachedPropertyDescrObject*)value)->func;
    if (used_in_vtable_worker(func)) {
      return 1;
    }
  } else if (Py_TYPE(value) == &PyAsyncCachedPropertyWithDescr_Type) {
    PyObject* func = ((PyAsyncCachedPropertyDescrObject*)value)->func;
    if (used_in_vtable_worker(func)) {
      return 1;
    }
  }

  return 0;
}

// Steals a reference to the `getter_tuple` and `setter_tuple` objects.
int update_property_slot(
    PyObject* slotmap,
    int* slot_index,
    PyObject* getter_tuple,
    PyObject* setter_tuple) {
  PyObject* getter_index = PyLong_FromLong((*slot_index)++);
  int err = PyDict_SetItem(slotmap, getter_tuple, getter_index);
  Py_DECREF(getter_index);
  Py_DECREF(getter_tuple);
  if (err) {
    Py_DECREF(setter_tuple);
    return -1;
  }
  PyObject* setter_index = PyLong_FromLong((*slot_index)++);
  err = PyDict_SetItem(slotmap, setter_tuple, setter_index);
  Py_DECREF(setter_index);
  Py_DECREF(setter_tuple);
  if (err) {
    return -1;
  }
  return 0;
}
/**
    Merges the slot map of our bases with our own members, initializing the
    map with the members which are defined in the current type but not the
    base type. Also, skips non-static callables that exist in tp_dict,
    because we cannot invoke against those anyway.
*/
int _PyClassLoader_UpdateSlotMap(PyTypeObject* self, PyObject* slotmap) {
  PyObject *key, *value;
  Py_ssize_t i;

  /* Add indexes for anything that is new in our class */
  int slot_index = PyDict_Size(slotmap);
  i = 0;
  while (PyDict_Next(_PyType_GetDict(self), &i, &key, &value)) {
    if (PyDict_GetItem(slotmap, key) || !used_in_vtable(value)) {
      /* we either share the same slot, or this isn't a static function,
       * so it doesn't need a slot */
      continue;
    }
    PyObject* index = PyLong_FromLong(slot_index++);
    int err = PyDict_SetItem(slotmap, key, index);
    Py_DECREF(index);
    if (err) {
      return -1;
    }
    PyTypeObject* val_type = Py_TYPE(value);
    if (val_type == &PyProperty_Type ||
        val_type == &PyCachedPropertyWithDescr_Type ||
        val_type == &PyAsyncCachedPropertyWithDescr_Type) {
      PyObject* getter_index = PyLong_FromLong(slot_index++);
      PyObject* getter_tuple = get_property_getter_descr_tuple(key);
      err = PyDict_SetItem(slotmap, getter_tuple, getter_index);
      Py_DECREF(getter_index);
      Py_DECREF(getter_tuple);
      if (err) {
        return -1;
      }
      PyObject* setter_index = PyLong_FromLong(slot_index++);
      PyObject* setter_tuple = get_property_setter_descr_tuple(key);
      err = PyDict_SetItem(slotmap, setter_tuple, setter_index);
      Py_DECREF(setter_index);
      Py_DECREF(setter_tuple);
      if (err) {
        return -1;
      }
    } else if (Py_TYPE(value) == &_PyTypedDescriptorWithDefaultValue_Type) {
      PyObject* getter_tuple = get_property_getter_descr_tuple(key);
      PyObject* setter_tuple = get_property_setter_descr_tuple(key);
      if (update_property_slot(
              slotmap, &slot_index, getter_tuple, setter_tuple) < 0) {
        return -1;
      }
    }
  }
  return 0;
}

int is_static_type(PyTypeObject* type) {
  return (type->tp_flags &
          (Ci_Py_TPFLAGS_IS_STATICALLY_DEFINED |
           Ci_Py_TPFLAGS_GENERIC_TYPE_INST)) ||
      !(type->tp_flags & Py_TPFLAGS_HEAPTYPE);
}

/**
    Creates a vtable for a type. Goes through the MRO, and recursively creates
   v-tables for any static base classes if needed.
*/
_PyType_VTable* _PyClassLoader_EnsureVtable(
    PyTypeObject* self,
    int init_subclasses) {
  _PyType_VTable* vtable = (_PyType_VTable*)self->tp_cache;
  PyObject* slotmap = NULL;
  PyObject* mro;

  if (self == &PyBaseObject_Type) {
    // We don't create a vtable for `object`. If we try to do that, all
    // subclasses of `object` (which is all classes), will need to have a
    // v-table of their own, and that's too much memory usage for almost no
    // benefit (since most classes are not Static). Also, none of the attributes
    // on `object` are interesting enough to invoke against.
    PyErr_SetString(
        PyExc_RuntimeError, "cannot initialize vtable for builtins.object");
    return NULL;
  }
  if (vtable != NULL) {
    return vtable;
  }

  mro = self->tp_mro;
  Py_ssize_t mro_size = PyTuple_GET_SIZE(mro);
  if (mro_size > 1) {
    /* TODO: Non-type objects in mro? */
    /* TODO: Multiple inheritance */

    /* Get the size of the next element which is a static class
     * in our mro, we'll build on it.  We don't care about any
     * non-static classes because we don't generate invokes to them */
    PyTypeObject* next;
    for (Py_ssize_t i = 1; i < mro_size; i++) {
      next = (PyTypeObject*)PyTuple_GET_ITEM(mro, i);
      if (is_static_type(next)) {
        break;
      }
    }

    assert(PyType_Check(next));
    assert(is_static_type(next));
    if (next != &PyBaseObject_Type) {
      _PyType_VTable* base_vtable = (_PyType_VTable*)next->tp_cache;
      if (base_vtable == NULL) {
        base_vtable = _PyClassLoader_EnsureVtable(next, 0);

        if (base_vtable == NULL) {
          return NULL;
        }

        if (init_subclasses && type_init_subclass_vtables(next)) {
          return NULL;
        }

        if (self->tp_cache != NULL) {
          /* we have recursively initialized the current v-table,
           * no need to continue with initialization now */
          return (_PyType_VTable*)self->tp_cache;
        }
      }

      PyObject* next_slotmap = base_vtable->vt_slotmap;
      assert(next_slotmap != NULL);

      slotmap = PyDict_Copy(next_slotmap);

      if (slotmap == NULL) {
        return NULL;
      }
    }
  }

  if (slotmap == NULL) {
    slotmap = _PyDict_NewPresized(PyDict_Size(_PyType_GetDict(self)));
  }

  if (slotmap == NULL) {
    return NULL;
  }

  if (is_static_type(self)) {
    if (_PyClassLoader_UpdateSlotMap(self, slotmap)) {
      Py_DECREF(slotmap);
      return NULL;
    }
  }

  /* finally allocate the vtable, which will have empty slots initially */
  Py_ssize_t slot_count = PyDict_Size(slotmap);
  vtable = PyObject_GC_NewVar(_PyType_VTable, &_PyType_VTableType, slot_count);

  if (vtable == NULL) {
    Py_DECREF(slotmap);
    return NULL;
  }
  vtable->vt_size = slot_count;
  vtable->vt_thunks = NULL;
  vtable->vt_original = NULL;
  vtable->vt_specials = NULL;
  vtable->vt_slotmap = slotmap;
  vtable->vt_typecode = TYPED_OBJECT;
  self->tp_cache = (PyObject*)vtable;
  memset(&vtable->vt_entries[0], 0, sizeof(_PyType_VTableEntry) * slot_count);

  if (_PyClassLoader_ReinitVtable(self, vtable)) {
    self->tp_cache = NULL;
    Py_DECREF(vtable);
    return NULL;
  }

  PyObject_GC_Track(vtable);

  if (init_subclasses && type_init_subclass_vtables(self)) {
    return NULL;
  }

  return vtable;
}

static int clear_vtables_recurse(PyTypeObject* type) {
  PyObject* subclasses = type->tp_subclasses;
  PyObject* ref;
  if (type->tp_cache != NULL) {
    // If the type has a type code we need to preserve it, but we'll clear
    // everything else
    int type_code = ((_PyType_VTable*)type->tp_cache)->vt_typecode;
    Py_CLEAR(type->tp_cache);
    if (type_code != TYPED_OBJECT) {
      _PyType_VTable* vtable = _PyClassLoader_EnsureVtable(type, 0);
      if (vtable != NULL) {
        vtable->vt_typecode = type_code;
      }
    }
  }
  if (subclasses != NULL) {
    Py_ssize_t i = 0;
    while (PyDict_Next(subclasses, &i, NULL, &ref)) {
      assert(PyWeakref_CheckRef(ref));
      ref = PyWeakref_GET_OBJECT(ref);
      if (ref == Py_None) {
        continue;
      }

      assert(PyType_Check(ref));
      if (clear_vtables_recurse((PyTypeObject*)ref)) {
        return -1;
      }
    }
  }
  return 0;
}

int _PyClassLoader_ClearVtables() {
  /* Recursively clear all vtables.
   *
   * This is really only intended for use in tests to avoid state pollution.
   */
  return clear_vtables_recurse(&PyBaseObject_Type);
}

/*
    Fetches the member held at the path defined by a type descriptor.
    e.g: ("mymod", "MyClass", "my_member")

    When container is not NULL, populates it with the `PyTypeObject` of the
   container. When containerkey is not NULL, populates it with the member name.
   This could be a tuple in the case of properties, such as ("my_member",
   "fget").

    The lookup is done from `sys.modules` (tstate->interp->modules), and if a
   module is not found, this function will import it.
*/
static PyObject* classloader_get_member(
    PyObject* path,
    Py_ssize_t items,
    PyObject** container,
    PyObject** containerkey) {
  if (container) {
    *container = NULL;
  }
  if (containerkey) {
    *containerkey = NULL;
  }

  if (PyTuple_GET_SIZE(path) != 2) {
    PyErr_Format(
      CiExc_StaticTypeError,
      "bad descriptor: %R",
      path
    );
    return NULL;
  }

  PyObject *container_obj = _PyClassLoader_ResolveContainer(PyTuple_GET_ITEM(path, 0));
  if (container_obj == NULL) {
    return NULL;
  }

  PyObject *attr_name = PyTuple_GET_ITEM(path, 1);
  if (containerkey) {
    *containerkey = attr_name;
  }

  PyObject *attr;
  if (PyType_Check(container_obj)) {
    PyObject *type_dict = ((PyTypeObject*)container_obj)->tp_dict;
     if (!PyTuple_CheckExact(attr_name)) {
      attr = PyDict_GetItem(type_dict, attr_name);
      if (attr == NULL) {
        PyErr_Format(
            CiExc_StaticTypeError,
            "bad name provided for class loader, %R doesn't exist in type %s",
            attr_name,
            ((PyTypeObject *)container_obj)->tp_name);
          goto error;
      }
      Py_INCREF(attr);
    } else if (get_func_or_special_callable((PyTypeObject*)container_obj, attr_name, &attr) < 0) {
        goto error;
    }
  } else {
    attr = _PyClassLoader_GetModuleAttr(container_obj, attr_name);
  }

  if (attr == NULL) {
    goto error;
  }

  if (container) {
    *container = container_obj;
  } else {
    Py_DECREF(container_obj);
  }

  return attr;
error:
  Py_DECREF(container_obj);
  return NULL;
}

/**
    This function is called when a member on a previously unseen
    class is encountered.

    Given a type descriptor to a callable, this function:
    - Ensures that the containing class has a v-table.
    - Adds an entry to the global `classloader_cache`
      (so future slot index lookups are faster)
    - Initializes v-tables for all subclasses of the containing class
*/
static int classloader_init_slot(PyObject* path) {
  /* path is "mod.submod.Class.func", start search from
   * sys.modules */
  PyObject *classloader_cache = _PyClassLoader_GetCache();
  if (classloader_cache == NULL) {
    return -1;
  }

  PyObject* target_type = _PyClassLoader_ResolveContainer(PyTuple_GET_ITEM(path, 0));
  if (target_type == NULL) {
    return -1;
  } else if (_PyClassLoader_VerifyType(target_type, path)) {
    Py_XDECREF(target_type);
    return -1;
  }

  /* Now we need to update or make the v-table for this type */
  _PyType_VTable* vtable = _PyClassLoader_EnsureVtable((PyTypeObject *)target_type, 0);
  if (vtable == NULL) {
    Py_XDECREF(target_type);
    return -1;
  }

  PyObject* slot_map = vtable->vt_slotmap;
  PyObject* slot_name = PyTuple_GET_ITEM(path, PyTuple_GET_SIZE(path) - 1);
  PyObject* new_index = PyDict_GetItem(slot_map, slot_name);
  if (new_index == NULL) {
    PyErr_Format(
        PyExc_RuntimeError,
        "unable to resolve v-table slot %R in %s is_static: %s",
        slot_name,
        ((PyTypeObject *)target_type)->tp_name,
        is_static_type((PyTypeObject *)target_type) ? "true" : "false");
    Py_DECREF(target_type);
    return -1;
  }
  assert(new_index != NULL);

  if (PyDict_SetItem(classloader_cache, path, new_index) ||
      type_init_subclass_vtables((PyTypeObject *)target_type)) {
    Py_DECREF(target_type);
    return -1;
  }

  Py_DECREF(target_type);
  return 0;
}

/**
    Returns a slot index given a "path" (type descr tuple) to a method.
    e.g ("my_mod", "MyClass", "my_method")
*/
Py_ssize_t _PyClassLoader_ResolveMethod(PyObject* path) {
  PyObject *classloader_cache = _PyClassLoader_GetCache();
  if (classloader_cache == NULL) {
    return -1;
  }

  /* TODO: Should we gracefully handle when there are two
   * classes with the same name? */
  PyObject* slot_index_obj = PyDict_GetItem(classloader_cache, path);
  if (slot_index_obj == NULL) {
    if (classloader_init_slot(path)) {
      return -1;
    }
    slot_index_obj = PyDict_GetItem(classloader_cache, path);
  }
  return PyLong_AS_LONG(slot_index_obj);
}

_Py_StaticThunk* get_or_make_thunk(
    PyObject* func,
    PyObject* original,
    PyObject* container,
    PyObject* name) {
  PyObject* thunks = NULL;
  PyTypeObject* type = NULL;
  if (PyType_Check(container)) {
    type = (PyTypeObject*)container;
    _PyType_VTable* vtable = (_PyType_VTable*)type->tp_cache;
    if (vtable->vt_thunks == NULL) {
      vtable->vt_thunks = PyDict_New();
      if (vtable->vt_thunks == NULL) {
        return NULL;
      }
    }
    thunks = vtable->vt_thunks;
  } else if (Ci_StrictModule_Check(container)) {
    Ci_StrictModuleObject* mod = (Ci_StrictModuleObject*)container;
    if (mod->static_thunks == NULL) {
      mod->static_thunks = PyDict_New();
      if (mod->static_thunks == NULL) {
        return NULL;
      }
    }
    thunks = mod->static_thunks;
  }
  _Py_StaticThunk* thunk = (_Py_StaticThunk*)PyDict_GetItem(thunks, name);
  if (thunk != NULL) {
    Py_INCREF(thunk);
    return thunk;
  }
  thunk = PyObject_GC_New(_Py_StaticThunk, &_PyType_StaticThunk);
  if (thunk == NULL) {
    return NULL;
  }

  PyObject* func_name = classloader_get_func_name(name);
  thunk->thunk_tcs.tcs_rt.rt_name = func_name;
  Py_INCREF(func_name);
  thunk->thunk_cls = type;
  Py_XINCREF(type);
  thunk->thunk_vectorcall = (vectorcallfunc)&thunk_vectorcall;
  thunk->thunk_tcs.tcs_value = NULL;

  update_thunk(thunk, original, func);

  thunk->thunk_tcs.tcs_rt.rt_expected =
      (PyTypeObject*)_PyClassLoader_ResolveReturnType(
          original,
          &thunk->thunk_tcs.tcs_rt.rt_optional,
          &thunk->thunk_tcs.tcs_rt.rt_exact,
          &thunk->thunk_flags);

  if (Ci_StrictModule_Check(container)) {
    // Treat functions in modules as static, we don't want to peel off the first
    // argument.
    thunk->thunk_flags |= Ci_FUNC_FLAGS_STATICMETHOD;
  }
  if (thunk->thunk_tcs.tcs_rt.rt_expected == NULL) {
    Py_DECREF(thunk);
    return NULL;
  }
  if (PyDict_SetItem(thunks, name, (PyObject*)thunk)) {
    Py_DECREF(thunk);
    return NULL;
  }
  return thunk;
}

PyObject* _PyClassLoader_ResolveFunction(PyObject* path, PyObject** container) {
  PyObject* containerkey;
  PyObject* func = classloader_get_member(
      path, PyTuple_GET_SIZE(path), container, &containerkey);

  PyObject* original = NULL;
  if (container != NULL && *container != NULL) {
    assert(containerkey != NULL);
    if (PyType_Check(*container)) {
      PyTypeObject* type = (PyTypeObject*)*container;
      if (type->tp_cache != NULL) {
        PyObject* originals = ((_PyType_VTable*)type->tp_cache)->vt_original;
        if (originals != NULL) {
          original = PyDict_GetItem(originals, containerkey);
        }
      }
    } else if (Ci_StrictModule_Check(*container)) {
      original = Ci_StrictModule_GetOriginal(*container, containerkey);
    }
  }
  if (original == func) {
    original = NULL;
  }

  if (original != NULL) {
    PyObject* res =
        (PyObject*)get_or_make_thunk(func, original, *container, containerkey);
    Py_DECREF(func);
    assert(res != NULL);
    return res;
  }

  if (func != NULL) {
    if (Py_TYPE(func) == &PyStaticMethod_Type) {
      PyObject* res = Ci_PyStaticMethod_GetFunc(func);
      Py_INCREF(res);
      Py_DECREF(func);
      func = res;
    } else if (Py_TYPE(func) == &PyClassMethod_Type) {
      PyObject* res = Ci_PyClassMethod_GetFunc(func);
      Py_INCREF(res);
      Py_DECREF(func);
      func = res;
    }
  }

  return func;
}

PyObject** _PyClassLoader_ResolveIndirectPtr(PyObject* path) {
  PyObject* container;
  PyObject* name;
  PyObject* func =
      classloader_get_member(path, PyTuple_GET_SIZE(path), &container, &name);
  if (func == NULL) {
    return NULL;
  }

  // for performance reason should only be used on mutable containers
  assert(!_PyClassLoader_IsImmutable(container));

  PyObject** cache = NULL;
  int use_thunk = 0;
  if (PyType_Check(container)) {
    _PyType_VTable* vtable =
        _PyClassLoader_EnsureVtable((PyTypeObject*)container, 1);
    if (vtable == NULL) {
      goto error;
    }
    use_thunk = 1;
  } else if (Ci_StrictModule_Check(container)) {
    use_thunk = 1;
  } else if (PyModule_Check(container)) {
    /* modules have no special translation on things we invoke, so
     * we just rely upon the normal JIT dict watchers */
    PyObject* dict = Ci_MaybeStrictModule_Dict(container);
    if (dict != NULL) {
      cache = _PyJIT_GetDictCache(dict, name);
    }
  }
  if (use_thunk) {
    /* we pass func in for original here.  Either the thunk will already exist
     * in which case the value has been patched, or it won't yet exist in which
     * case func is the original function in the type. */
    _Py_StaticThunk* thunk = get_or_make_thunk(func, func, container, name);
    if (thunk == NULL) {
      goto error;
    }

    cache = &thunk->thunk_funcref;
    Py_DECREF(thunk);
  }

error:
  Py_DECREF(container);
  Py_DECREF(func);
  return cache;
}

int _PyClassLoader_IsImmutable(PyObject* container) {
  if (PyType_Check(container)) {
    PyTypeObject* type = (PyTypeObject*)container;
#if PY_VERSION_HEX < 0x030C0000
    if (type->tp_flags & Ci_Py_TPFLAGS_FROZEN ||
        !(type->tp_flags & Py_TPFLAGS_HEAPTYPE)) {
      return 1;
    }
#else
    if (type->tp_flags & Py_TPFLAGS_IMMUTABLETYPE ||
        !(type->tp_flags & Py_TPFLAGS_HEAPTYPE)) {
      return 1;
    }
#endif
  }

  if (Ci_StrictModule_CheckExact(container) &&
      ((Ci_StrictModuleObject*)container)->global_setter == NULL) {
    return 1;
  }
  return 0;
}

PyMethodDescrObject* _PyClassLoader_ResolveMethodDef(PyObject* path) {
  PyTypeObject* target_type;
  PyObject* cur = classloader_get_member(
      path, PyTuple_GET_SIZE(path), (PyObject**)&target_type, NULL);

  if (cur == NULL) {
    assert(target_type == NULL);
    return NULL;
  } else if (
      _PyClassLoader_VerifyType((PyObject*)target_type, path) ||
      target_type->tp_flags & Py_TPFLAGS_BASETYPE) {
    Py_XDECREF(target_type);
    Py_DECREF(cur);
    return NULL;
  }

  Py_DECREF(target_type);
  if (Py_TYPE(cur) == &PyMethodDescr_Type) {
    return (PyMethodDescrObject*)cur;
  }

  Py_DECREF(cur);
  return NULL;
}

int _PyClassLoader_AddSubclass(PyTypeObject* base, PyTypeObject* type) {
  if (base->tp_cache == NULL) {
    /* nop if base class vtable isn't initialized */
    return 0;
  }

  _PyType_VTable* vtable = _PyClassLoader_EnsureVtable(type, 0);
  if (vtable == NULL) {
    return -1;
  }
  return 0;
}

int _PyClassLoader_NotifyDictChange(PyDictObject* dict, PyObject* key) {
  return _PyClassLoader_CheckModuleChange(dict, key);
}

static int classloader_init_field(PyObject* path, int* field_type) {
  /* path is "mod.submod.Class.func", start search from
   * sys.modules */
  PyObject* cur =
      classloader_get_member(path, PyTuple_GET_SIZE(path), NULL, NULL);
  if (cur == NULL) {
    return -1;
  }

  if (Py_TYPE(cur) == &PyMemberDescr_Type) {
    if (field_type != NULL) {
      switch (((PyMemberDescrObject*)cur)->d_member->type) {
        case T_BYTE:
          *field_type = TYPED_INT8;
          break;
        case T_SHORT:
          *field_type = TYPED_INT16;
          break;
        case T_INT:
          *field_type = TYPED_INT32;
          break;
        case T_LONG:
          *field_type = TYPED_INT64;
          break;
        case T_UBYTE:
          *field_type = TYPED_UINT8;
          break;
        case T_USHORT:
          *field_type = TYPED_UINT16;
          break;
        case T_UINT:
          *field_type = TYPED_UINT32;
          break;
        case T_ULONG:
          *field_type = TYPED_UINT64;
          break;
        case T_BOOL:
          *field_type = TYPED_BOOL;
          break;
        case T_DOUBLE:
          *field_type = TYPED_DOUBLE;
          break;
        case T_FLOAT:
          *field_type = TYPED_SINGLE;
          break;
        case T_CHAR:
          *field_type = TYPED_CHAR;
          break;
        case T_OBJECT_EX:
          *field_type = TYPED_OBJECT;
          break;
        default:
          Py_DECREF(cur);
          PyErr_Format(PyExc_ValueError, "unknown static type: %S", path);
          return -1;
      }
    }
    Py_DECREF(cur);
    Py_ssize_t offset = ((PyMemberDescrObject*)cur)->d_member->offset;
    return offset;
  } else if (Py_TYPE(cur) == &_PyTypedDescriptor_Type) {
    if (field_type != NULL) {
      *field_type = TYPED_OBJECT;
      assert(((_PyTypedDescriptor*)cur)->td_offset % sizeof(Py_ssize_t) == 0);
    }
    Py_DECREF(cur);
    return ((_PyTypedDescriptor*)cur)->td_offset;
  } else if (Py_TYPE(cur) == &_PyTypedDescriptorWithDefaultValue_Type) {
    if (field_type != NULL) {
      *field_type = TYPED_OBJECT;
      assert(
          ((_PyTypedDescriptorWithDefaultValue*)cur)->td_offset %
              sizeof(Py_ssize_t) ==
          0);
    }
    Py_DECREF(cur);
    return ((_PyTypedDescriptorWithDefaultValue*)cur)->td_offset;
  }

  Py_DECREF(cur);
  PyErr_Format(CiExc_StaticTypeError, "bad field for class loader %R", path);
  return -1;
}

/* Resolves the offset for a given field, returning -1 on failure with an error
 * set or the field offset.  Path is a tuple in the form
 * ('module', 'class', 'field_name')
 */
Py_ssize_t _PyClassLoader_ResolveFieldOffset(PyObject* path, int* field_type) {
  PyObject *classloader_cache = _PyClassLoader_GetCache();
  if (classloader_cache == NULL) {
  }

  /* TODO: Should we gracefully handle when there are two
   * classes with the same name? */
  PyObject* slot_index_obj = PyDict_GetItem(classloader_cache, path);
  if (slot_index_obj != NULL) {
    PyObject* offset = PyTuple_GET_ITEM(slot_index_obj, 0);
    if (field_type != NULL) {
      PyObject* type = PyTuple_GET_ITEM(slot_index_obj, 1);
      *field_type = PyLong_AS_LONG(type);
    }
    return PyLong_AS_LONG(offset);
  }

  int tmp_field_type = 0;
  Py_ssize_t slot_index = classloader_init_field(path, &tmp_field_type);
  if (slot_index < 0) {
    return -1;
  }
  slot_index_obj = PyLong_FromLong(slot_index);
  if (slot_index_obj == NULL) {
    return -1;
  }

  PyObject* field_type_obj = PyLong_FromLong(tmp_field_type);
  if (field_type_obj == NULL) {
    Py_DECREF(slot_index);
    return -1;
  }

  PyObject* cache = PyTuple_New(2);
  if (cache == NULL) {
    Py_DECREF(slot_index_obj);
    Py_DECREF(field_type_obj);
    return -1;
  }
  PyTuple_SET_ITEM(cache, 0, slot_index_obj);
  PyTuple_SET_ITEM(cache, 1, field_type_obj);

  if (PyDict_SetItem(classloader_cache, path, cache)) {
    Py_DECREF(cache);
    return -1;
  }

  Py_DECREF(cache);
  if (field_type != NULL) {
    *field_type = tmp_field_type;
  }

  return slot_index;
}

#define GENINST_GET_PARAM(self, i) \
  (((_PyGenericTypeInst*)Py_TYPE(self))->gti_inst[i].gtp_type)

void _PyClassLoader_ArgError(
    PyObject* func_name,
    int arg,
    int type_param,
    const Ci_Py_SigElement* sig_elem,
    PyObject* ctx) {
  const char* expected = "?";
  int argtype = sig_elem->se_argtype;
  if (argtype & Ci_Py_SIG_TYPE_PARAM) {
    expected =
        ((PyTypeObject*)GENINST_GET_PARAM(ctx, Ci_Py_SIG_TYPE_MASK(argtype)))
            ->tp_name;

  } else {
    switch (Ci_Py_SIG_TYPE_MASK(argtype)) {
      case Ci_Py_SIG_OBJECT:
        PyErr_Format(
            CiExc_StaticTypeError, "%U() argument %d is missing", func_name, arg);
        return;
      case Ci_Py_SIG_STRING:
        expected = "str";
        break;
      case Ci_Py_SIG_SSIZE_T:
        expected = "int";
        break;
    }
  }

  PyErr_Format(
      CiExc_StaticTypeError,
      "%U() argument %d expected %s",
      func_name,
      arg,
      expected);
}

const Ci_Py_SigElement Ci_Py_Sig_T0 = {Ci_Py_SIG_TYPE_PARAM_IDX(0)};
const Ci_Py_SigElement Ci_Py_Sig_T1 = {Ci_Py_SIG_TYPE_PARAM_IDX(1)};
const Ci_Py_SigElement Ci_Py_Sig_T0_Opt = {
    Ci_Py_SIG_TYPE_PARAM_IDX(0) | Ci_Py_SIG_OPTIONAL,
    Py_None};
const Ci_Py_SigElement Ci_Py_Sig_T1_Opt = {
    Ci_Py_SIG_TYPE_PARAM_IDX(1) | Ci_Py_SIG_OPTIONAL,
    Py_None};
const Ci_Py_SigElement Ci_Py_Sig_Object = {Ci_Py_SIG_OBJECT};
const Ci_Py_SigElement Ci_Py_Sig_Object_Opt = {
    Ci_Py_SIG_OBJECT | Ci_Py_SIG_OPTIONAL,
    Py_None};
const Ci_Py_SigElement Ci_Py_Sig_String = {Ci_Py_SIG_STRING};
const Ci_Py_SigElement Ci_Py_Sig_String_Opt = {
    Ci_Py_SIG_STRING | Ci_Py_SIG_OPTIONAL,
    Py_None};

const Ci_Py_SigElement Ci_Py_Sig_SSIZET = {Ci_Py_SIG_SSIZE_T};
const Ci_Py_SigElement Ci_Py_Sig_SIZET = {Ci_Py_SIG_SIZE_T};
const Ci_Py_SigElement Ci_Py_Sig_INT8 = {Ci_Py_SIG_INT8};
const Ci_Py_SigElement Ci_Py_Sig_INT16 = {Ci_Py_SIG_INT16};
const Ci_Py_SigElement Ci_Py_Sig_INT32 = {Ci_Py_SIG_INT32};
const Ci_Py_SigElement Ci_Py_Sig_INT64 = {Ci_Py_SIG_INT64};
const Ci_Py_SigElement Ci_Py_Sig_UINT8 = {Ci_Py_SIG_UINT8};
const Ci_Py_SigElement Ci_Py_Sig_UINT16 = {Ci_Py_SIG_UINT16};
const Ci_Py_SigElement Ci_Py_Sig_UINT32 = {Ci_Py_SIG_UINT32};
const Ci_Py_SigElement Ci_Py_Sig_UINT64 = {Ci_Py_SIG_UINT64};

_PyTypedArgsInfo* _PyClassLoader_GetTypedArgsInfo(
    PyCodeObject* code,
    int only_primitives) {
  PyObject* checks = _PyClassLoader_GetCodeArgumentTypeDescrs(code);

  int count;
  if (only_primitives) {
    count = 0;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(checks); i += 2) {
      PyObject* type_descr = PyTuple_GET_ITEM(checks, i + 1);
      if (_PyClassLoader_ResolvePrimitiveType(type_descr) != TYPED_OBJECT) {
        count++;
      }
    }
  } else {
    count = PyTuple_GET_SIZE(checks) / 2;
  }

  _PyTypedArgsInfo* arg_checks =
      PyObject_GC_NewVar(_PyTypedArgsInfo, &_PyTypedArgsInfo_Type, count);
  if (arg_checks == NULL) {
    return NULL;
  }

  int checki = 0;
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(checks); i += 2) {
    _PyTypedArgInfo* cur_check = &arg_checks->tai_args[checki];

    PyObject* type_descr = PyTuple_GET_ITEM(checks, i + 1);
    int optional, exact;
    PyTypeObject* ref_type =
        _PyClassLoader_ResolveType(type_descr, &optional, &exact);
    if (ref_type == NULL) {
      return NULL;
    }

    int prim_type = _PyClassLoader_GetTypeCode(ref_type);
    if (prim_type == TYPED_BOOL) {
      cur_check->tai_type = &PyBool_Type;
      cur_check->tai_optional = 0;
      cur_check->tai_exact = 1;
      Py_INCREF(&PyBool_Type);
      Py_DECREF(ref_type);
    } else if (prim_type == TYPED_DOUBLE) {
      cur_check->tai_type = &PyFloat_Type;
      cur_check->tai_optional = 0;
      cur_check->tai_exact = 1;
      Py_INCREF(&PyFloat_Type);
      Py_DECREF(ref_type);
    } else if (prim_type != TYPED_OBJECT) {
      assert(prim_type <= TYPED_INT64);
      cur_check->tai_type = &PyLong_Type;
      cur_check->tai_optional = 0;
      cur_check->tai_exact = 1;
      Py_INCREF(&PyLong_Type);
      Py_DECREF(ref_type);
    } else if (only_primitives) {
      Py_DECREF(ref_type);
      continue;
    } else {
      cur_check->tai_type = ref_type;
      cur_check->tai_optional = optional;
      cur_check->tai_exact = exact;
    }
    cur_check->tai_primitive_type = prim_type;
    cur_check->tai_argnum = PyLong_AsLong(PyTuple_GET_ITEM(checks, i));
    checki++;
  }
  return arg_checks;
}

_PyTypedArgsInfo* _PyClassLoader_GetTypedArgsInfoFromThunk(
    PyObject* thunk,
    PyObject* container,
    int only_primitives) {
  if (!_PyClassLoader_IsPatchedThunk(thunk)) {
    return NULL;
  }
  PyObject* originals = NULL;
  if (PyType_Check(container)) {
    PyObject* vtable = ((PyTypeObject*)container)->tp_cache;
    originals = ((_PyType_VTable*)vtable)->vt_original;
  } else if (Ci_StrictModule_Check(container)) {
    originals = ((Ci_StrictModuleObject*)container)->originals;
  }
  if (!originals) {
    return NULL;
  }
  PyObject* original = PyDict_GetItem(
      originals, ((_Py_StaticThunk*)thunk)->thunk_tcs.tcs_rt.rt_name);
  if (original == NULL) {
    return NULL;
  }
  PyObject* unwrapped = classloader_maybe_unwrap_callable(original);
  if (unwrapped != NULL) {
    original = unwrapped;
  }
  PyObject* code = PyFunction_GetCode(original);
  if (code == NULL) {
    return NULL;
  }
  return _PyClassLoader_GetTypedArgsInfo((PyCodeObject*)code, only_primitives);
}

int _PyClassLoader_HasPrimitiveArgs(PyCodeObject* code) {
  PyObject* checks = _PyClassLoader_GetCodeArgumentTypeDescrs(code);
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(checks); i += 2) {
    PyObject* type_descr = PyTuple_GET_ITEM(checks, i + 1);

    if (_PyClassLoader_ResolvePrimitiveType(type_descr) != TYPED_OBJECT) {
      return 1;
    }
  }
  return 0;
}

static PyObject* invoke_native_helper = NULL;

static inline int import_invoke_native() {
  if (__builtin_expect(invoke_native_helper == NULL, 0)) {
    PyObject* native_utils = PyImport_ImportModule("__static__.native_utils");
    if (native_utils == NULL) {
      return -1;
    }
    invoke_native_helper =
        PyObject_GetAttrString(native_utils, "invoke_native");
    Py_DECREF(native_utils);
    if (invoke_native_helper == NULL) {
      return -1;
    }
  }
  return 0;
}

PyObject* _PyClassloader_InvokeNativeFunction(
    PyObject* lib_name,
    PyObject* symbol_name,
    PyObject* signature,
    PyObject** args,
    Py_ssize_t nargs) {
  if (!PyUnicode_CheckExact(lib_name)) {
    PyErr_Format(
        PyExc_RuntimeError,
        "'lib_name' must be a str, got '%s'",
        Py_TYPE(lib_name)->tp_name);
    return NULL;
  }
  if (!PyUnicode_CheckExact(lib_name)) {
    PyErr_Format(
        PyExc_RuntimeError,
        "'symbol_name' must be a str, got '%s'",
        Py_TYPE(lib_name)->tp_name);
    return NULL;
  }
  if (!PyTuple_CheckExact(signature)) {
    PyErr_Format(
        PyExc_RuntimeError,
        "'signature' must be a tuple of type descriptors",
        Py_TYPE(lib_name)->tp_name);
    return NULL;
  }

  int return_typecode =
      _PyClassLoader_ResolvePrimitiveType(PyTuple_GET_ITEM(signature, nargs));
  if (return_typecode == -1) {
    // exception must be set already
    assert(PyErr_Occurred());
    return NULL;
  }

  // build arg tuple.. this is kinda wasteful, but we're not optimizing for the
  // interpreter here
  PyObject* arguments = PyTuple_New(nargs);
  if (arguments == NULL) {
    return NULL;
  }
  for (int i = 0; i < nargs; i++) {
    PyTuple_SET_ITEM(arguments, i, args[i]);
    Py_INCREF(args[i]);
  }

  if (import_invoke_native() < 0) {
    return NULL;
  }
  PyObject* res = PyObject_CallFunction(
      invoke_native_helper,
      "OOOO",
      lib_name,
      symbol_name,
      signature,
      arguments);

  Py_DECREF(arguments);
  return res;
}

// Returns the size of the dlsym_cache dict (0 if uninitialized)
PyObject* _PyClassloader_SizeOf_DlSym_Cache() {
  if (dlsym_cache == NULL) {
    return PyLong_FromLong(0);
  }
  Py_ssize_t size = PyDict_Size(dlsym_cache);
  return PyLong_FromSsize_t(size);
}

// Returns the size of the dlopen_cache dict (0 if uninitialized)
PyObject* _PyClassloader_SizeOf_DlOpen_Cache() {
  if (dlopen_cache == NULL) {
    return PyLong_FromLong(0);
  }
  Py_ssize_t size = PyDict_Size(dlopen_cache);
  return PyLong_FromSsize_t(size);
}

// Clears the dlsym_cache dict
void _PyClassloader_Clear_DlSym_Cache() {
  if (dlsym_cache != NULL) {
    PyDict_Clear(dlsym_cache);
  }
}

// Clears the dlopen_cache dict
void _PyClassloader_Clear_DlOpen_Cache() {
  if (dlopen_cache != NULL) {
    PyObject *name, *handle;
    Py_ssize_t i = 0;
    while (PyDict_Next(dlopen_cache, &i, &name, &handle)) {
      void* raw_handle = PyLong_AsVoidPtr(handle);
      // Ignore errors - we can't do much even if they occur
      dlclose(raw_handle);
    }

    PyDict_Clear(dlopen_cache);
  }
}

// Thin wrapper over dlopen, returns the handle of the opened lib
static void* classloader_dlopen(PyObject* lib_name) {
  assert(PyUnicode_CheckExact(lib_name));
  const char* raw_lib_name = PyUnicode_AsUTF8(lib_name);
  if (raw_lib_name == NULL) {
    return NULL;
  }
  void* handle = dlopen(raw_lib_name, RTLD_NOW | RTLD_LOCAL);
  if (handle == NULL) {
    PyErr_Format(
        PyExc_RuntimeError,
        "classloader: Could not load library '%s': %s",
        raw_lib_name,
        dlerror());
    return NULL;
  }
  return handle;
}

// Looks up the cached handle to the shared lib of given name. If not found,
// proceeds to load it and populates the cache.
static void* classloader_lookup_sharedlib(PyObject* lib_name) {
  assert(PyUnicode_CheckExact(lib_name));
  PyObject* val = NULL;

  // Ensure cache exists
  if (dlopen_cache == NULL) {
    dlopen_cache = PyDict_New();
    if (dlopen_cache == NULL) {
      return NULL;
    }
  }

  val = PyDict_GetItem(dlopen_cache, lib_name);
  if (val != NULL) {
    // Cache hit
    return PyLong_AsVoidPtr(val);
  }

  // Lookup the lib
  void* handle = classloader_dlopen(lib_name);
  if (handle == NULL) {
    return NULL;
  }

  // Populate the cache with the handle
  val = PyLong_FromVoidPtr(handle);
  if (val == NULL) {
    return NULL;
  }
  int res = PyDict_SetItem(dlopen_cache, lib_name, val);
  Py_DECREF(val);
  if (res < 0) {
    return NULL;
  }
  return handle;
}

// Wrapper over `dlsym`.
static PyObject* classloader_lookup_symbol(
    PyObject* lib_name,
    PyObject* symbol_name) {
  void* handle = classloader_lookup_sharedlib(lib_name);
  if (handle == NULL) {
    assert(PyErr_Occurred());
    return NULL;
  }

  const char* raw_symbol_name = PyUnicode_AsUTF8(symbol_name);
  if (raw_symbol_name == NULL) {
    return NULL;
  }

  void* res = dlsym(handle, raw_symbol_name);
  if (res == NULL) {
    // Technically, `res` could actually have the value `NULL`, but we're
    // in the business of looking up callables, so we raise an exception
    // (NULL cannot be called anyway).
    //
    // To be 100% correct, we could clear existing errors with `dlerror`,
    // call `dlsym` and then call `dlerror` again, to check whether an
    // error occured, but that'll be more work than we need.
    PyErr_Format(
        PyExc_RuntimeError,
        "classloader: unable to lookup '%U' in '%U': %s",
        symbol_name,
        lib_name,
        dlerror());
    return NULL;
  }

  PyObject* symbol = PyLong_FromVoidPtr(res);
  if (symbol == NULL) {
    return NULL;
  }
  return symbol;
}

// Looks up the raw symbol address from the given lib, and returns
// a boxed value of it.
void* _PyClassloader_LookupSymbol(PyObject* lib_name, PyObject* symbol_name) {
  if (!PyUnicode_CheckExact(lib_name)) {
    PyErr_Format(
        CiExc_StaticTypeError,
        "classloader: 'lib_name' must be a str, got '%s'",
        Py_TYPE(lib_name)->tp_name);
    return NULL;
  }
  if (!PyUnicode_CheckExact(symbol_name)) {
    PyErr_Format(
        CiExc_StaticTypeError,
        "classloader: 'symbol_name' must be a str, got '%s'",
        Py_TYPE(symbol_name)->tp_name);
    return NULL;
  }

  // Ensure cache exists
  if (dlsym_cache == NULL) {
    dlsym_cache = PyDict_New();
    if (dlsym_cache == NULL) {
      return NULL;
    }
  }

  PyObject* key = PyTuple_Pack(2, lib_name, symbol_name);
  if (key == NULL) {
    return NULL;
  }

  PyObject* res = PyDict_GetItem(dlsym_cache, key);

  if (res != NULL) {
    Py_DECREF(key);
    return PyLong_AsVoidPtr(res);
  }

  res = classloader_lookup_symbol(lib_name, symbol_name);
  if (res == NULL) {
    Py_DECREF(key);
    return NULL;
  }

  if (PyDict_SetItem(dlsym_cache, key, res) < 0) {
    Py_DECREF(key);
    Py_DECREF(res);
    return NULL;
  }

  void* addr = PyLong_AsVoidPtr(res);
  Py_DECREF(key);
  Py_DECREF(res);
  return addr;
}

static int Ci_populate_type_info(PyObject* arg_info, int argtype) {
  _Py_IDENTIFIER(NoneType);
  _Py_IDENTIFIER(object);
  _Py_IDENTIFIER(str);
  _Py_static_string(__static__int8, "__static__.int8");
  _Py_static_string(__static__int16, "__static__.int16");
  _Py_static_string(__static__int32, "__static__.int32");
  _Py_static_string(__static__int64, "__static__.int64");
  _Py_static_string(__static__uint8, "__static__.uint8");
  _Py_static_string(__static__uint16, "__static__.uint16");
  _Py_static_string(__static__uint32, "__static__.uint32");
  _Py_static_string(__static__uint64, "__static__.uint64");
  _Py_IDENTIFIER(optional);
  _Py_IDENTIFIER(type_param);
  _Py_IDENTIFIER(type);

  if ((argtype & Ci_Py_SIG_OPTIONAL) &&
      _PyDict_SetItemId(arg_info, &PyId_optional, Py_True)) {
    return -1;
  }

  if (argtype & Ci_Py_SIG_TYPE_PARAM) {
    /* indicate the type parameter */
    PyObject* type = PyLong_FromLong(Ci_Py_SIG_TYPE_MASK(argtype));
    if (_PyDict_SetItemId(arg_info, &PyId_type_param, type)) {
      Py_DECREF(type);
      return -1;
    }
    Py_DECREF(type);
  } else {
    PyObject* name;
    switch (argtype & ~Ci_Py_SIG_OPTIONAL) {
      case Ci_Py_SIG_ERROR:
      case Ci_Py_SIG_VOID:
        name = _PyUnicode_FromId(&PyId_NoneType);
        break;
      case Ci_Py_SIG_OBJECT:
        name = _PyUnicode_FromId(&PyId_object);
        break;
      case Ci_Py_SIG_STRING:
        name = _PyUnicode_FromId(&PyId_str);
        break;
      case Ci_Py_SIG_INT8:
        name = _PyUnicode_FromId(&__static__int8);
        break;
      case Ci_Py_SIG_INT16:
        name = _PyUnicode_FromId(&__static__int16);
        break;
      case Ci_Py_SIG_INT32:
        name = _PyUnicode_FromId(&__static__int32);
        break;
      case Ci_Py_SIG_INT64:
        name = _PyUnicode_FromId(&__static__int64);
        break;
      case Ci_Py_SIG_UINT8:
        name = _PyUnicode_FromId(&__static__uint8);
        break;
      case Ci_Py_SIG_UINT16:
        name = _PyUnicode_FromId(&__static__uint16);
        break;
      case Ci_Py_SIG_UINT32:
        name = _PyUnicode_FromId(&__static__uint32);
        break;
      case Ci_Py_SIG_UINT64:
        name = _PyUnicode_FromId(&__static__uint64);
        break;
      default:
        PyErr_SetString(PyExc_RuntimeError, "unknown type");
        return -1;
    }
    if (name == NULL || _PyDict_SetItemId(arg_info, &PyId_type, name)) {
      return -1;
    }
  }
  return 0;
}

PyObject* Ci_PyMethodDef_GetTypedSignature(PyMethodDef* method) {
  _Py_IDENTIFIER(default);
  _Py_IDENTIFIER(type);
  if (!(method->ml_flags & Ci_METH_TYPED)) {
    Py_RETURN_NONE;
  }
  Ci_PyTypedMethodDef* def = (Ci_PyTypedMethodDef*)method->ml_meth;
  PyObject* res = PyDict_New();
  PyObject* args = PyList_New(0);
  if (PyDict_SetItemString(res, "args", args)) {
    Py_DECREF(res);
    return NULL;
  }
  Py_DECREF(args);
  const Ci_Py_SigElement* const* sig = def->tmd_sig;
  while (*sig != NULL) {
    /* each arg is a dictionary */
    PyObject* arg_info = PyDict_New();
    if (arg_info == NULL) {
      Py_DECREF(res);
      return NULL;
    } else if (PyList_Append(args, arg_info)) {
      Py_DECREF(arg_info);
      Py_DECREF(res);
      return NULL;
    }
    Py_DECREF(arg_info); /* kept alive by args list */
    int argtype = (*sig)->se_argtype;
    if (Ci_populate_type_info(arg_info, argtype)) {
      return NULL;
    }

    PyObject* name;
    if ((*sig)->se_name != NULL) {
      name = PyUnicode_FromString((*sig)->se_name);
      if (name == NULL) {
        Py_DECREF(res);
        return NULL;
      } else if (_PyDict_SetItemId(arg_info, &PyId_type, name)) {
        Py_DECREF(name);
        Py_DECREF(res);
        return NULL;
      }
      Py_DECREF(name);
    }

    if ((*sig)->se_default_value != NULL &&
        _PyDict_SetItemId(arg_info, &PyId_default, (*sig)->se_default_value)) {
      Py_DECREF(res);
      return NULL;
    }
    sig++;
  }

  PyObject* ret_info = PyDict_New();
  if (ret_info == NULL || PyDict_SetItemString(res, "return", ret_info) ||
      Ci_populate_type_info(ret_info, def->tmd_ret)) {
    Py_XDECREF(ret_info);
    return NULL;
  }
  Py_DECREF(ret_info);

  return res;
}
