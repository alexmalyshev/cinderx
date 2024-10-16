// Copyright (c) Meta Platforms, Inc. and affiliates.

// This file is processed by UpstreamBorrow.py. To see the generated output:
// buck build --out=- fbcode//cinderx/UpstreamBorrow:gen_borrowed.c

// clang-format off

#include "cinderx/UpstreamBorrow/borrowed.h"

// @Borrow CPP directives from Objects/genobject.c

// Internal dependencies for _PyGen_yf which only exist in 3.12.
// @Borrow function is_resume from Objects/genobject.c [3.12]
// @Borrow function _PyGen_GetCode from Objects/genobject.c [3.12]
// End internal dependencies for _PyGen_yf.

#define _PyGen_yf Cix_PyGen_yf
// @Borrow function _PyGen_yf from Objects/genobject.c

// Internal dependencies for _PyCoro_GetAwaitableIter.
// @Borrow function gen_is_coroutine from Objects/genobject.c
// End internal dependencies for _PyCoro_GetAwaitableIter.

#define _PyCoro_GetAwaitableIter Cix_PyCoro_GetAwaitableIter
// @Borrow function _PyCoro_GetAwaitableIter from Objects/genobject.c

// Internal dependencies for _PyAsyncGenValueWrapperNew.
// @Borrow typedef _PyAsyncGenWrappedValue from Objects/genobject.c
// @Borrow function get_async_gen_state from Objects/genobject.c
// End internal dependencies for _PyAsyncGenValueWrapperNew.

#if PY_VERSION_HEX < 0x030C0000
#define _PyAsyncGenValueWrapperNew Cix_PyAsyncGenValueWrapperNew
#else
// In 3.12 we need a temporary name before wrapping to avoid conflicting with
// the forward declaration in genobject.h.
#define _PyAsyncGenValueWrapperNew __PyAsyncGenValueWrapperNew
#endif
// @Borrow function _PyAsyncGenValueWrapperNew from Objects/genobject.c
#if PY_VERSION_HEX >= 0x030C0000
// In 3.12 _PyAsyncGenValueWrapperNew needs thread-state. As this is used from
// the JIT we could get the value from the thread-state register. This would be
// slightly more efficient, but quite a bit more work and async-generators are
// rare. So we just wrap it up here.
PyObject* Cix_PyAsyncGenValueWrapperNew(PyObject* value) {
  return __PyAsyncGenValueWrapperNew(PyThreadState_GET(), value);
}
#endif

// _Py_IncRefTotal is used by internal functions in 3.12 dictobject.c.
// Pragmatically @Borrow'ing this doesn't seem worth it at this stage. We would
// need UpstreamBorrow.py to somehow not attempt/ignore failure to extract
// _Py_IncRefTotal on non-debug builds where it's deleted by the CPP. All the
// simple solutions I can think of seem just as ugly as manually copying. This
// is made worse by the fact internally _Py_IncRefTotal uses a macro which
// isn't easily visible to us as it's #undef'd after usage. So we'd need a fix
// or to copy that anyway.
#if defined(Py_DEBUG) && PY_VERSION_HEX >= 0x030C0000
#define _Py_IncRefTotal __Py_IncRefTotal
static void _Py_IncRefTotal(PyInterpreterState* interp) {
  interp->object_state.reftotal++;
}

#define _Py_DecRefTotal __Py_DecRefTotal
static void _Py_DecRefTotal(PyInterpreterState* interp) {
  interp->object_state.reftotal--;
}
#endif

// @Borrow CPP directives from Objects/dictobject.c

// These are global singletons and some of the functions we're borrowing
// check for them with pointer equality. Fortunately we are able to get
// the values in init_upstream_borrow().
#if PY_VERSION_HEX < 0x030C0000
static PyObject** empty_values = NULL;
#else
#undef Py_EMPTY_KEYS
static PyDictKeysObject* Py_EMPTY_KEYS = NULL;
#endif

// Internal dependencies for things borrowed from dictobject.c.
// @Borrow function dictkeys_get_index from Objects/dictobject.c [3.12]
// @Borrow function unicode_get_hash from Objects/dictobject.c [3.12]
// @Borrow function unicodekeys_lookup_unicode from Objects/dictobject.c [3.12]
// @Borrow function unicodekeys_lookup_generic from Objects/dictobject.c [3.12]
// @Borrow function dictkeys_generic_lookup from Objects/dictobject.c [3.12]
// Rename to avoid clashing with existing version when statically linking.
#define _Py_dict_lookup __Py_dict_lookup
// @Borrow function _Py_dict_lookup from Objects/dictobject.c [3.12]
// @Borrow function get_dict_state from Objects/dictobject.c
// @Borrow function new_values from Objects/dictobject.c [3.12]
// @Borrow function free_values from Objects/dictobject.c [3.12]
// @Borrow function shared_keys_usable_size from Objects/dictobject.c [3.12]
// @Borrow function free_keys_object from Objects/dictobject.c
// @Borrow function dictkeys_decref from Objects/dictobject.c
// @Borrow function dictkeys_incref from Objects/dictobject.c
// @Borrow function new_dict from Objects/dictobject.c
// @Borrow function new_dict_with_shared_keys from Objects/dictobject.c
// @Borrow function dict_event_name from Objects/dictobject.c [3.12]
// End internal dependencies.

#define _PyObjectDict_SetItem Cix_PyObjectDict_SetItem
// @Borrow function _PyObjectDict_SetItem from Objects/dictobject.c

#define _PyDict_LoadGlobal Cix_PyDict_LoadGlobal
// @Borrow function _PyDict_LoadGlobal from Objects/dictobject.c

#if PY_VERSION_HEX >= 0x030C0000
// Include _PyDict_SendEvent with its original name but weakly as we use
// some static inline functions from CPython headers which depend on this.
__attribute__((weak))
#endif
// We do not rename to a Cix_ function as this is only used from static
// inline functions in CPython headers.
// @Borrow function _PyDict_SendEvent from Objects/dictobject.c [3.12]

// @Borrow function set_attribute_error_context from Objects/object.c

// Wrapper as set_attribute_error_context is declared "static inline".
int
Cix_set_attribute_error_context(PyObject *v, PyObject *name) {
  return set_attribute_error_context(v, name);
}

#if PY_VERSION_HEX >= 0x030C0000
// Internal dependencies for _PyTuple_FromArray.
// Unfortunately these macros can't be borrowed by pulling in the CPP
// directives as they are #undef'd after use.
#define STATE (interp->tuple)
#define FREELIST_FINALIZED (STATE.numfree[0] < 0)
// @Borrow function maybe_freelist_pop from Objects/tupleobject.c [3.12]
// @Borrow function tuple_get_empty from Objects/tupleobject.c [3.12]
// @Borrow function tuple_alloc from Objects/tupleobject.c [3.12]
// End internal dependencies for _PyTuple_FromArray.
#define _PyTuple_FromArray Cix_PyTuple_FromArray
// @Borrow function _PyTuple_FromArray from Objects/tupleobject.c [3.12]
#endif

// Internal dependencies for _PyStaticType_GetState.
// @Borrow function static_builtin_index_is_set from Objects/typeobject.c [3.12]
// @Borrow function static_builtin_index_get from Objects/typeobject.c [3.12]
// @Borrow function static_builtin_state_get from Objects/typeobject.c [3.12]
// End internal dependencies.
#if PY_VERSION_HEX >= 0x030C0000
// Include _PyStaticType_GetState with its original name but weakly as we use
// some static inline functions from CPython headers which depend on this.
__attribute__((weak))
#endif
// @Borrow function _PyStaticType_GetState from Objects/typeobject.c [3.12]
#if PY_VERSION_HEX >= 0x030C0000
static_builtin_state*
Cix_PyStaticType_GetState(PyInterpreterState* interp, PyTypeObject* self) {
  return _PyStaticType_GetState(interp, self);
}
#endif

// These are global singletons used transitively by _Py_union_type_or.
// We initialize them in init_upstream_borrow().
PyTypeObject* Cix_PyUnion_Type = NULL;
#define _PyUnion_Type (*Cix_PyUnion_Type)

#if PY_VERSION_HEX >= 0x030C0000
PyTypeObject* Cix_PyTypeAlias_Type = NULL;
#define _PyTypeAlias_Type (*Cix_PyTypeAlias_Type)
#endif

// Internal dependencies for _Py_union_type_or.
// @Borrow CPP directives from Objects/unionobject.c
// @Borrow typedef unionobject from Objects/unionobject.c
// @Borrow function flatten_args from Objects/unionobject.c [3.10]
// @Borrow function dedup_and_flatten_args from Objects/unionobject.c [3.10]
// @Borrow function is_unionable from Objects/unionobject.c
// @Borrow function is_same from Objects/unionobject.c [3.12]
// @Borrow function contains from Objects/unionobject.c [3.12]
// @Borrow function merge from Objects/unionobject.c [3.12]
// @Borrow function get_types from Objects/unionobject.c [3.12]
// Rename to avoid clashing with existing version when statically linking.
#define make_union Cix_make_union
// @Borrow function make_union from Objects/unionobject.c
// End internal dependencies.
#define _Py_union_type_or Cix_Py_union_type_or
// @Borrow function _Py_union_type_or from Objects/unionobject.c

#if PY_VERSION_HEX >= 0x030C0000
// Internal dependencies for _PyThreadState_*Frame.
// @Borrow CPP directives from Objects/obmalloc.c [3.12]
#define _PyObject_VirtualAlloc __PyObject_VirtualAlloc
// @Borrow function _PyObject_VirtualAlloc from Objects/obmalloc.c [3.12]
#define _PyObject_VirtualFree __PyObject_VirtualFree
// @Borrow function _PyObject_VirtualFree from Objects/obmalloc.c [3.12]
// @Borrow CPP directives from Python/pystate.c [3.12]
// @Borrow function allocate_chunk from Python/pystate.c [3.12]
// @Borrow function push_chunk from Python/pystate.c [3.12]
// End internal dependencies.
#define _PyThreadState_PushFrame Cix_PyThreadState_PushFrame
// @Borrow function _PyThreadState_PushFrame from Python/pystate.c [3.12]
#define _PyThreadState_PopFrame Cix_PyThreadState_PopFrame
// @Borrow function _PyThreadState_PopFrame from Python/pystate.c [3.12]

// Internal dependencies for _PyFrame_ClearExceptCode.
// @Borrow CPP directives from Python/frame.c [3.12]
__attribute__((weak))
// @Borrow function _PyFrame_New_NoTrack from Objects/frameobject.c [3.12]
__attribute__((weak))
// @Borrow function _PyFrame_MakeAndSetFrameObject from Python/frame.c [3.12]
// @Borrow function take_ownership from Python/frame.c [3.12]
#define _PyFrame_ClearLocals __PyFrame_ClearLocals
// @Borrow function _PyFrame_ClearLocals from Python/frame.c [3.12]
// End internal dependencies.
#define _PyFrame_ClearExceptCode Cix_PyFrame_ClearExceptCode
// @Borrow function _PyFrame_ClearExceptCode from Python/frame.c [3.12]

// @Borrow var DE_INSTRUMENT from Python/instrumentation.c [3.12]
uint8_t
Cix_DEINSTRUMENT(uint8_t op) {
  return DE_INSTRUMENT[op];
}

// Internal dependencies for bytecode address to line number mapping.
// @Borrow CPP directives from Objects/codeobject.c [3.12]
// @Borrow function scan_varint from Objects/codeobject.c [3.12]
// @Borrow function scan_signed_varint from Objects/codeobject.c [3.12]
// @Borrow function get_line_delta from Objects/codeobject.c [3.12]
// @Borrow function next_code_delta from Objects/codeobject.c [3.12]
// @Borrow function is_no_line_marker from Objects/codeobject.c [3.12]
// @Borrow function at_end from Objects/codeobject.c [3.12]
// @Borrow function advance from Objects/codeobject.c [3.12]
#define _PyLineTable_InitAddressRange __PyLineTable_InitAddressRange
// @Borrow function _PyLineTable_InitAddressRange from Objects/codeobject.c [3.12]
// End internal dependencies.
#define _PyCode_InitAddressRange Cix_PyCode_InitAddressRange
// @Borrow function _PyCode_InitAddressRange from Objects/codeobject.c [3.12]
#define _PyLineTable_NextAddressRange Cix_PyLineTable_NextAddressRange
// @Borrow function _PyLineTable_NextAddressRange from Objects/codeobject.c [3.12]
#endif

// Internal dependencies for Cix_do_raise.
#define _PyErr_SetRaisedException __PyErr_SetRaisedException
// @Borrow function _PyErr_SetRaisedException from Python/errors.c [3.12]
// End internal dependencies.
// @Borrow function do_raise from Python/ceval.c
int Cix_do_raise(PyThreadState* tstate, PyObject* exc, PyObject* cause) {
  return do_raise(tstate, exc, cause);
}

int init_upstream_borrow(void) {
  PyObject* empty_dict = PyDict_New();
  if (empty_dict == NULL) {
    return -1;
  }
#if PY_VERSION_HEX < 0x030C0000
  empty_values = ((PyDictObject*)empty_dict)->ma_values;
#else
  Py_EMPTY_KEYS = ((PyDictObject*)empty_dict)->ma_keys;
#endif
  Py_DECREF(empty_dict);

  // Initialize the Cix_PyUnion_Type global reference.
  PyObject* unionobj =
      PyNumber_Or((PyObject*)&PyLong_Type, (PyObject*)&PyUnicode_Type);
  if (unionobj != NULL) {
    Cix_PyUnion_Type = Py_TYPE(unionobj);
    Py_DECREF(unionobj);
  }
  if (Cix_PyUnion_Type == NULL) {
    return -1;
  }

#if PY_VERSION_HEX >= 0x030C0000
  // Initialize the Cix_PyTypeAlias_Type global reference.
  PyObject* typing_module = PyImport_ImportModule("typing");
  if (!typing_module) {
    return -1;
  }
  PyObject* type_alias_type =
      PyObject_GetAttrString(typing_module, "TypeAliasType");

  if (!type_alias_type) {
    Py_DECREF(typing_module);
    return -1;
  }
  assert(PyType_Check(type_alias_type));
  Cix_PyTypeAlias_Type = (PyTypeObject*)type_alias_type;
  Py_DECREF(typing_module);
#endif

  return 0;
}
