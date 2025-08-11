// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "cinderx/module_state.h"

#include "internal/pycore_object.h"

#include "cinderx/Common/log.h"

namespace cinderx {

int ModuleState::traverse(visitproc visit, void* arg) {
  Py_VISIT(builtin_next_);
  return 0;
}

int ModuleState::clear() {
  sys_clear_caches_.reset();
  builtin_next_.reset();
  return 0;
}

static ModuleState* s_cinderx_state;

void ModuleState::shutdown() {
  jit_gen_free_list_.reset();
  cache_manager_.reset();
  runtime_.reset();
  symbolizer_.reset();
  JIT_DCHECK(
      this == s_cinderx_state,
      "Global module state pointer inconsistent with this module state {} != "
      "{}",
      reinterpret_cast<void*>(this),
      reinterpret_cast<void*>(s_cinderx_state));
  s_cinderx_state = nullptr;
}

void setModule(PyObject* m) {
  auto state = reinterpret_cast<cinderx::ModuleState*>(PyModule_GetState(m));
  s_cinderx_state = state;
  state->setModule(m);
}

ModuleState* getModuleState() {
  return s_cinderx_state;
}

bool ModuleState::initBuiltinMembers() {
#if PY_VERSION_HEX >= 0x030C0000
  constexpr PyTypeObject* types[] = {
      &PyBool_Type,
      &PyBytes_Type,
      &PyByteArray_Type,
      &PyComplex_Type,
      &PyCode_Type,
      &PyDict_Type,
      &PyFloat_Type,
      &PyFrozenSet_Type,
      &PyList_Type,
      &PyLong_Type,
      &_PyNone_Type,
      &PyProperty_Type,
      &PySet_Type,
      &PyTuple_Type,
      &PyUnicode_Type,
  };

  for (auto type : types) {
    PyObject* mro = type->tp_mro;
    if (mro == nullptr) {
      continue;
    }

    Ref<> type_members = Ref<>::steal(PyDict_New());
    if (type_members == nullptr) {
      return false;
    }
    for (Py_ssize_t i = 0; i < Py_SIZE(mro); i++) {
      PyTypeObject* base =
          reinterpret_cast<PyTypeObject*>(PyTuple_GetItem(mro, i));
      Py_ssize_t cur_mem = 0;
      PyObject *key, *value;
      Ref<> tp_dict = Ref<>::steal(PyType_GetDict(base));
      while (PyDict_Next(tp_dict, &cur_mem, &key, &value)) {
        if (PyDict_Contains(type_members, key)) {
          continue;
        }
        if (PyDict_SetItem(type_members, key, value) < 0) {
          return false;
        }
      }
    }

    builtin_members_.emplace(type, std::move(type_members));
  }
#endif
  return true;
}

} // namespace cinderx

extern "C" {
vectorcallfunc Ci_PyFunction_Vectorcall;
}
