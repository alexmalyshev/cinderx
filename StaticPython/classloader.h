/* Copyright (c) Meta Platforms, Inc. and affiliates. */
#ifndef Ci_CLASSLOADER_H
#define Ci_CLASSLOADER_H

#include "cinder/exports.h"
#include "cinder/hooks.h"
#include "internal/pycore_moduleobject.h"

#include "cinderx/StaticPython/awaitable.h"
#include "cinderx/StaticPython/generic_type.h"
#include "cinderx/StaticPython/strictmoduleobject.h"
#include "cinderx/StaticPython/type.h"
#include "cinderx/StaticPython/vtable.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_LIMITED_API

#endif

typedef struct _PyClassLoader_StaticCallReturn {
  void* rax;
  void* rdx;
} _PyClassLoader_StaticCallReturn;

typedef struct {
  PyObject_HEAD;
  PyObject* mt_original;
} _PyClassLoader_MethodThunk;

/**
    In order to ensure sanity of types at runtime, we need to check the return
   values of functions and ensure they remain compatible with the declared
   return type (even if the callable is patched).

    This structure helps us do that.
*/
typedef struct {
  _PyClassLoader_MethodThunk rt_base;
  PyTypeObject* rt_expected;
  PyObject* rt_name;
  int rt_optional;
  int rt_exact;
} _PyClassLoader_RetTypeInfo;

typedef struct {
  PyObject_HEAD PyObject* prop_get;
  PyObject* prop_set;
  PyObject* prop_del;
  PyObject* prop_doc;
  int getter_doc;
} Ci_propertyobject;

#define Ci_METH_TYPED 0x0400

/* Flag marks this as optional */
#define Ci_Py_SIG_OPTIONAL 0x01
/* Flag marks this a type param, high bits are type index */
#define Ci_Py_SIG_TYPE_PARAM 0x02
#define Ci_Py_SIG_TYPE_PARAM_IDX(x) ((x << 2) | Ci_Py_SIG_TYPE_PARAM)
#define Ci_Py_SIG_TYPE_PARAM_OPT(x) \
  ((x << 2) | Ci_Py_SIG_TYPE_PARAM | Ci_Py_SIG_OPTIONAL)

typedef struct Ci_Py_SigElement {
  int se_argtype;
  PyObject* se_default_value;
  const char* se_name;
} Ci_Py_SigElement;

typedef struct {
  void* tmd_meth; /* The C function that implements it */
  const Ci_Py_SigElement* const* tmd_sig; /* The function signature */
  int tmd_ret;
} Ci_PyTypedMethodDef;

#define Ci_Py_TYPED_SIGNATURE(name, ret_type, ...)                   \
  static const Ci_Py_SigElement* const name##_sig[] = {__VA_ARGS__}; \
  static Ci_PyTypedMethodDef name##_def = {                          \
      name,                                                          \
      name##_sig,                                                    \
      ret_type,                                                      \
  }

CiAPI_FUNC(PyObject*) Ci_PyMethodDef_GetTypedSignature(PyMethodDef* method);

typedef struct {
  _PyClassLoader_RetTypeInfo tcs_rt;
  PyObject* tcs_value;
} _PyClassLoader_TypeCheckState;

CiAPI_FUNC(Py_ssize_t) _PyClassLoader_ResolveMethod(PyObject* path);
CiAPI_FUNC(Py_ssize_t)
    _PyClassLoader_ResolveFieldOffset(PyObject* path, int* field_type);
CiAPI_FUNC(int) _PyClassLoader_ResolvePrimitiveType(PyObject* descr);
CiAPI_FUNC(int) _PyClassLoader_GetTypeCode(PyTypeObject* type);
CiAPI_FUNC(PyTypeObject*)
    _PyClassLoader_ResolveType(PyObject* descr, int* optional, int* exact);

CiAPI_FUNC(int)
    _PyClassLoader_AddSubclass(PyTypeObject* base, PyTypeObject* type);

CiAPI_FUNC(_PyType_VTable*)
    _PyClassLoader_EnsureVtable(PyTypeObject* self, int init_subclasses);
CiAPI_FUNC(int) _PyClassLoader_ClearVtables(void);
CiAPI_FUNC(void) _PyClassLoader_ClearGenericTypes(void);

CiAPI_FUNC(int) _PyClassLoader_IsPatchedThunk(PyObject* obj);

/* Gets an indirect pointer for a function.  This should be used if
 * the given container is mutable, and the indirect pointer will
 * track changes to the object.  If changes are unable to be tracked
 * the pointer will start pointing at NULL and the function will need
 * to be re-resolved.
 */
CiAPI_FUNC(PyObject**) _PyClassLoader_ResolveIndirectPtr(PyObject* path);

/* Checks to see if the given container is immutable */
CiAPI_FUNC(int) _PyClassLoader_IsImmutable(PyObject* container);

/* Resolves a function and returns the underlying object and the
 * container.  Static functions return the underlying callable */
CiAPI_FUNC(PyObject*)
    _PyClassLoader_ResolveFunction(PyObject* path, PyObject** container);

CiAPI_FUNC(PyObject*) _PyClassLoader_ResolveReturnType(
    PyObject* func,
    int* optional,
    int* exact,
    int* func_flags);

CiAPI_FUNC(PyMethodDescrObject*)
    _PyClassLoader_ResolveMethodDef(PyObject* path);
CiAPI_FUNC(void) _PyClassLoader_ClearCache(void);

CiAPI_FUNC(PyObject*) _PyClassLoader_GetReturnTypeDescr(PyFunctionObject* func);
CiAPI_FUNC(PyObject*) _PyClassLoader_GetCodeReturnTypeDescr(PyCodeObject* code);
CiAPI_FUNC(PyObject*)
    _PyClassLoader_GetCodeArgumentTypeDescrs(PyCodeObject* code);

/* Checks whether any method in the members dict overrides a final method in the
   base type. This API explicitly takes in a base_type and members_dict instead
   of a type object as it is used within `type_new`.
 */
CiAPI_FUNC(int) _PyClassLoader_IsFinalMethodOverridden(
    PyTypeObject* base_type,
    PyObject* members_dict);

#define PRIM_OP_ADD_INT 0
#define PRIM_OP_SUB_INT 1
#define PRIM_OP_MUL_INT 2
#define PRIM_OP_DIV_INT 3
#define PRIM_OP_DIV_UN_INT 4
#define PRIM_OP_MOD_INT 5
#define PRIM_OP_MOD_UN_INT 6
#define PRIM_OP_POW_INT 7
#define PRIM_OP_LSHIFT_INT 8
#define PRIM_OP_RSHIFT_INT 9
#define PRIM_OP_RSHIFT_UN_INT 10
#define PRIM_OP_XOR_INT 11
#define PRIM_OP_OR_INT 12
#define PRIM_OP_AND_INT 13

#define PRIM_OP_ADD_DBL 14
#define PRIM_OP_SUB_DBL 15
#define PRIM_OP_MUL_DBL 16
#define PRIM_OP_DIV_DBL 17
#define PRIM_OP_MOD_DBL 18
#define PRIM_OP_POW_DBL 19
#define PRIM_OP_POW_UN_INT 20

#define PRIM_OP_EQ_INT 0
#define PRIM_OP_NE_INT 1
#define PRIM_OP_LT_INT 2
#define PRIM_OP_LE_INT 3
#define PRIM_OP_GT_INT 4
#define PRIM_OP_GE_INT 5
#define PRIM_OP_LT_UN_INT 6
#define PRIM_OP_LE_UN_INT 7
#define PRIM_OP_GT_UN_INT 8
#define PRIM_OP_GE_UN_INT 9
#define PRIM_OP_EQ_DBL 10
#define PRIM_OP_NE_DBL 11
#define PRIM_OP_LT_DBL 12
#define PRIM_OP_LE_DBL 13
#define PRIM_OP_GT_DBL 14
#define PRIM_OP_GE_DBL 15

#define PRIM_OP_NEG_INT 0
#define PRIM_OP_INV_INT 1
#define PRIM_OP_NEG_DBL 2
#define PRIM_OP_NOT_INT 3

#define FAST_LEN_INEXACT (1 << 4)
#define FAST_LEN_LIST 0
#define FAST_LEN_DICT 1
#define FAST_LEN_SET 2
#define FAST_LEN_TUPLE 3
#define FAST_LEN_ARRAY 4
#define FAST_LEN_STR 5

// At the time of defining these, we needed to remain backwards compatible,
// so SEQ_LIST had to be zero. Therefore, we let the array types occupy the
// higher nibble and the lower nibble is for defining other sequence types
// (top bit of lower nibble is for unchecked flag).
#define SEQ_LIST 0
#define SEQ_TUPLE 1
#define SEQ_LIST_INEXACT 2
#define SEQ_SUBSCR_UNCHECKED (1 << 3)
#define SEQ_REPEAT_INEXACT_SEQ (1 << 4)
#define SEQ_REPEAT_INEXACT_NUM (1 << 5)
#define SEQ_REPEAT_REVERSED (1 << 6)
#define SEQ_REPEAT_PRIMITIVE_NUM (1 << 7)

// For arrays, the constant contains the element type in the higher
// nibble
#define SEQ_ARRAY_INT64 ((TYPED_INT64 << 4) | TYPED_ARRAY)
#define SEQ_REPEAT_FLAGS                                                   \
  (SEQ_REPEAT_INEXACT_SEQ | SEQ_REPEAT_INEXACT_NUM | SEQ_REPEAT_REVERSED | \
   SEQ_REPEAT_PRIMITIVE_NUM)
#define SEQ_CHECKED_LIST (1 << 8)

#define Ci_Py_SIG_INT8 (TYPED_INT8 << 2)
#define Ci_Py_SIG_INT16 (TYPED_INT16 << 2)
#define Ci_Py_SIG_INT32 (TYPED_INT32 << 2)
#define Ci_Py_SIG_INT64 (TYPED_INT64 << 2)
#define Ci_Py_SIG_UINT8 (TYPED_UINT8 << 2)
#define Ci_Py_SIG_UINT16 (TYPED_UINT16 << 2)
#define Ci_Py_SIG_UINT32 (TYPED_UINT32 << 2)
#define Ci_Py_SIG_UINT64 (TYPED_UINT64 << 2)
#define Ci_Py_SIG_OBJECT (TYPED_OBJECT << 2)
#define Ci_Py_SIG_VOID (TYPED_VOID << 2)
#define Ci_Py_SIG_STRING (TYPED_STRING << 2)
#define Ci_Py_SIG_ERROR (TYPED_ERROR << 2)
#define Ci_Py_SIG_SSIZE_T \
  (sizeof(void*) == 8 ? Ci_Py_SIG_INT64 : Ci_Py_SIG_INT32)
#define Ci_Py_SIG_SIZE_T \
  (sizeof(void*) == 8 ? Ci_Py_SIG_UINT64 : Ci_Py_SIG_UINT32)
#define Ci_Py_SIG_TYPE_MASK(x) ((x) >> 2)

#define Ci_FUNC_FLAGS_COROUTINE 0x01
#define Ci_FUNC_FLAGS_CLASSMETHOD 0x02
#define Ci_FUNC_FLAGS_STATICMETHOD 0x04

#ifndef Py_LIMITED_API
typedef struct {
  PyObject_HEAD PyObject* td_name;
  PyObject* td_type; /* tuple type reference or type object once resolved */
  Py_ssize_t td_offset;
  int td_optional;
  int td_exact;
} _PyTypedDescriptor;

CiAPI_DATA(PyTypeObject) _PyTypedDescriptor_Type;

CiAPI_FUNC(PyObject*)
    _PyTypedDescriptor_New(PyObject* name, PyObject* type, Py_ssize_t offset);

CiAPI_DATA(PyTypeObject) _PyTypedDescriptorWithDefaultValue_Type;

typedef struct {
  PyObject_HEAD PyObject* td_name;
  PyObject* td_type; /* tuple type reference or type object once resolved */
  PyObject* td_default; /* the default value to return from the get if offset is
                           null */
  Py_ssize_t td_offset;
  int td_optional;
  int td_exact;
} _PyTypedDescriptorWithDefaultValue;

CiAPI_DATA(PyTypeObject) _PyTypedDescriptor_Type;

CiAPI_FUNC(PyObject*) _PyTypedDescriptorWithDefaultValue_New(
    PyObject* name,
    PyObject* type,
    Py_ssize_t offset,
    PyObject* default_value);

CiAPI_FUNC(int) _PyClassLoader_UpdateModuleName(
    Ci_StrictModuleObject* mod,
    PyObject* name,
    PyObject* new_value);

CiAPI_FUNC(int) _PyClassLoader_UpdateSlot(
    PyTypeObject* type,
    PyObject* name,
    PyObject* new_value);

CiAPI_FUNC(int) _PyClassLoader_InitTypeForPatching(PyTypeObject* type);

CiAPI_FUNC(PyObject*) _PyClassloader_SizeOf_DlSym_Cache(void);
CiAPI_FUNC(PyObject*) _PyClassloader_SizeOf_DlOpen_Cache(void);
CiAPI_FUNC(void) _PyClassloader_Clear_DlSym_Cache(void);
CiAPI_FUNC(void) _PyClassloader_Clear_DlOpen_Cache(void);

CiAPI_FUNC(void*)
    _PyClassloader_LookupSymbol(PyObject* lib_name, PyObject* symbol_name);

CiAPI_FUNC(int)
    _PyClassLoader_AddSubclass(PyTypeObject* base, PyTypeObject* type);

typedef struct {
  int tai_primitive_type;
  PyTypeObject* tai_type;
  int tai_argnum;
  int tai_optional;
  int tai_exact;
} _PyTypedArgInfo;

typedef struct {
  PyObject_VAR_HEAD _PyTypedArgInfo tai_args[1];
} _PyTypedArgsInfo;

CiAPI_DATA(PyTypeObject) _PyTypedArgsInfo_Type;

CiAPI_FUNC(_PyTypedArgsInfo*)
    _PyClassLoader_GetTypedArgsInfo(PyCodeObject* code, int only_primitives);
CiAPI_FUNC(_PyTypedArgsInfo*) _PyClassLoader_GetTypedArgsInfoFromThunk(
    PyObject* thunk,
    PyObject* container,
    int only_primitives);
CiAPI_FUNC(int) _PyClassLoader_HasPrimitiveArgs(PyCodeObject* code);

static inline int
_PyClassLoader_CheckParamType(PyObject* self, PyObject* arg, int index) {
  _PyGenericTypeParam* param =
      &((_PyGenericTypeInst*)Py_TYPE(self))->gti_inst[index];
  return (arg == Py_None && param->gtp_optional) ||
      PyObject_TypeCheck(arg, param->gtp_type);
}

CiAPI_DATA(const Ci_Py_SigElement) Ci_Py_Sig_T0;
CiAPI_DATA(const Ci_Py_SigElement) Ci_Py_Sig_T1;
CiAPI_DATA(const Ci_Py_SigElement) Ci_Py_Sig_T0_Opt;
CiAPI_DATA(const Ci_Py_SigElement) Ci_Py_Sig_T1_Opt;
CiAPI_DATA(const Ci_Py_SigElement) Ci_Py_Sig_Object;
CiAPI_DATA(const Ci_Py_SigElement) Ci_Py_Sig_Object_Opt;
CiAPI_DATA(const Ci_Py_SigElement) Ci_Py_Sig_String;
CiAPI_DATA(const Ci_Py_SigElement) Ci_Py_Sig_String_Opt;

CiAPI_DATA(const Ci_Py_SigElement) Ci_Py_Sig_SSIZET;
CiAPI_DATA(const Ci_Py_SigElement) Ci_Py_Sig_SIZET;
CiAPI_DATA(const Ci_Py_SigElement) Ci_Py_Sig_INT8;
CiAPI_DATA(const Ci_Py_SigElement) Ci_Py_Sig_INT16;
CiAPI_DATA(const Ci_Py_SigElement) Ci_Py_Sig_INT32;
CiAPI_DATA(const Ci_Py_SigElement) Ci_Py_Sig_INT64;
CiAPI_DATA(const Ci_Py_SigElement) Ci_Py_Sig_UINT8;
CiAPI_DATA(const Ci_Py_SigElement) Ci_Py_Sig_UINT16;
CiAPI_DATA(const Ci_Py_SigElement) Ci_Py_Sig_UINT32;
CiAPI_DATA(const Ci_Py_SigElement) Ci_Py_Sig_UINT64;

static inline int
_PyClassLoader_OverflowCheck(PyObject* arg, int type, size_t* value) {
  static uint64_t overflow_masks[] = {
      0xFFFFFFFFFFFFFF00, 0xFFFFFFFFFFFF0000, 0xFFFFFFFF00000000, 0x0};
  static uint64_t soverflow_masks[] = {
      0xFFFFFFFFFFFFFF80,
      0xFFFFFFFFFFFF8000,
      0xFFFFFFFF80000000,
      0x8000000000000000};

  assert(Py_TYPE(arg) == &PyLong_Type);

  if (type & TYPED_INT_SIGNED) {
    Py_ssize_t ival = PyLong_AsSsize_t(arg);
    if (ival == -1 && PyErr_Occurred()) {
      PyErr_Clear();
      return 0;
    }
    if ((ival & soverflow_masks[type >> 1]) &&
        (ival & soverflow_masks[type >> 1]) != soverflow_masks[type >> 1]) {
      return 0;
    }
    *value = (size_t)ival;
  } else {
    size_t ival = PyLong_AsSize_t(arg);
    if (ival == (size_t)-1 && PyErr_Occurred()) {
      PyErr_Clear();
      return 0;
    }

    if (ival & overflow_masks[type >> 1]) {
      return 0;
    }
    *value = (size_t)ival;
  }
  return 1;
}

static inline void* _PyClassLoader_ConvertArg(
    PyObject* ctx,
    const Ci_Py_SigElement* sig_elem,
    Py_ssize_t i,
    Py_ssize_t nargsf,
    PyObject* const* args,
    int* error) {
  PyObject* arg =
      i < PyVectorcall_NARGS(nargsf) ? args[i] : sig_elem->se_default_value;
  int argtype = sig_elem->se_argtype;
  if ((argtype & Ci_Py_SIG_OPTIONAL) && (arg == NULL || arg == Py_None)) {
    return arg;
  } else if (arg == NULL) {
    *error = 1;
  } else if (argtype & Ci_Py_SIG_TYPE_PARAM) {
    if (!_PyClassLoader_CheckParamType(
            ctx, arg, Ci_Py_SIG_TYPE_MASK(argtype))) {
      *error = 1;
    }
    return arg;
  } else {
    switch (argtype & ~(Ci_Py_SIG_OPTIONAL)) {
      case Ci_Py_SIG_OBJECT:
        return arg;
      case Ci_Py_SIG_STRING:
        *error = !PyUnicode_Check(arg);
        return arg;
      case Ci_Py_SIG_UINT8:
      case Ci_Py_SIG_UINT16:
      case Ci_Py_SIG_UINT32:
      case Ci_Py_SIG_INT8:
      case Ci_Py_SIG_INT16:
      case Ci_Py_SIG_INT32:
        if (PyLong_Check(arg)) {
          size_t res;
          if (_PyClassLoader_OverflowCheck(
                  arg, Ci_Py_SIG_TYPE_MASK(argtype), &res)) {
            return (void*)res;
          }
          *error = 1;
          PyErr_SetString(PyExc_OverflowError, "overflow");
        } else {
          *error = 1;
        }
        break;
      case Ci_Py_SIG_INT64:
        if (PyLong_Check(arg)) {
          Py_ssize_t val = PyLong_AsSsize_t(arg);
          if (val == -1 && PyErr_Occurred()) {
            *error = 1;
          }
          return (void*)val;
        } else {
          *error = 1;
        }
        break;
      case Ci_Py_SIG_UINT64:
        if (PyLong_Check(arg)) {
          size_t val = PyLong_AsSize_t(arg);
          if (val == ((size_t)-1) && PyErr_Occurred()) {
            *error = 1;
          }
          return (void*)val;
        } else {
          *error = 1;
        }
        break;
    }
  }
  return NULL;
}

static inline PyObject* _PyClassLoader_ConvertRet(void* value, int ret_type) {
  switch (ret_type) {
    // We could update the compiler so that void returning functions either
    // are only used in void contexts, or explicitly emit a LOAD_CONST None
    // when not used in a void context. For now we just produce None here (and
    // in JIT HIR builder).
    case Ci_Py_SIG_VOID:
      Py_INCREF(Py_None);
      return Py_None;
    case Ci_Py_SIG_INT8:
      return PyLong_FromSsize_t((int8_t)(Py_ssize_t)value);
    case Ci_Py_SIG_INT16:
      return PyLong_FromSsize_t((int16_t)(Py_ssize_t)value);
    case Ci_Py_SIG_INT32:
      return PyLong_FromSsize_t((int32_t)(Py_ssize_t)value);
    case Ci_Py_SIG_INT64:
#if SIZEOF_VOID_P >= 8
      return PyLong_FromSsize_t((int64_t)value);
#elif SIZEOF_LONG_LONG < SIZEOF_VOID_P
#error "sizeof(long long) < sizeof(void*)"
#else
      return PyLong_FromLongLong((long long)value);
#endif
    case Ci_Py_SIG_UINT8:
      return PyLong_FromSize_t((uint8_t)(size_t)value);
    case Ci_Py_SIG_UINT16:
      return PyLong_FromSize_t((uint16_t)(size_t)value);
    case Ci_Py_SIG_UINT32:
      return PyLong_FromSize_t((uint32_t)(size_t)value);
    case Ci_Py_SIG_UINT64:
#if SIZEOF_VOID_P >= 8
      return PyLong_FromSize_t((uint64_t)value);
#elif SIZEOF_LONG_LONG < SIZEOF_VOID_P
#error "sizeof(long long) < sizeof(void*)"
#else
      return PyLong_FromUnsignedLongLong((unsigned long long)value);
#endif
    case Ci_Py_SIG_ERROR:
      if (value) {
        return NULL;
      }
      Py_INCREF(Py_None);
      return Py_None;
    default:
      return (PyObject*)value;
  }
}

CiAPI_FUNC(void) _PyClassLoader_ArgError(
    PyObject* func_name,
    int arg,
    int type_param,
    const Ci_Py_SigElement* sig_elem,
    PyObject* ctx);

static inline int _PyClassLoader_IsStaticFunction(PyObject* obj) {
  if (obj == NULL || !PyFunction_Check(obj)) {
    return 0;
  }
  return ((PyCodeObject*)(((PyFunctionObject*)obj))->func_code)->co_flags &
      CO_STATICALLY_COMPILED;
}

static inline PyMethodDef* _PyClassLoader_GetMethodDef(PyObject* obj) {
  if (obj == NULL) {
    return NULL;
  } else if (PyCFunction_Check(obj)) {
    return ((PyCFunctionObject*)obj)->m_ml;
  } else if (Py_TYPE(obj) == &PyMethodDescr_Type) {
    return ((PyMethodDescrObject*)obj)->d_method;
  }
  return NULL;
}

static inline Ci_PyTypedMethodDef* _PyClassLoader_GetTypedMethodDef(
    PyObject* obj) {
  PyMethodDef* def = _PyClassLoader_GetMethodDef(obj);
  if (def && def->ml_flags & Ci_METH_TYPED) {
    return (Ci_PyTypedMethodDef*)def->ml_meth;
  }
  return NULL;
}

static inline int _PyClassLoader_IsStaticBuiltin(PyObject* obj) {
  return (_PyClassLoader_GetTypedMethodDef(obj) != NULL);
}

static inline int _PyClassLoader_IsStaticCallable(PyObject* obj) {
  return _PyClassLoader_IsStaticFunction(obj) ||
      _PyClassLoader_IsStaticBuiltin(obj);
}

CiAPI_FUNC(int)
    _PyClassLoader_NotifyDictChange(PyDictObject* dict, PyObject* key);

CiAPI_FUNC(PyObject*) _PyClassloader_InvokeNativeFunction(
    PyObject* lib_name,
    PyObject* symbol_name,
    PyObject* signature,
    PyObject** args,
    int64_t nargs);

PyObject* _PyClassLoader_InvokeMethod(
    _PyType_VTable* vtable,
    Py_ssize_t slot,
    PyObject** args,
    Py_ssize_t nargsf);

#endif

#ifdef __cplusplus
}
#endif
#endif /* !Ci_CLASSLOADER_H */
