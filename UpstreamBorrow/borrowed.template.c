// This file is processed by UpstreamBorrow.py. To see the generated output:
// buck build --out=- fbcode//cinderx/UpstreamBorrow:gen_borrowed.c

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


// @Borrow function set_attribute_error_context from Objects/object.c

// Wrapper as set_attribute_error_context is declared "static inline".
int
Cix_set_attribute_error_context(PyObject *v, PyObject *name) {
    return set_attribute_error_context(v, name);
}
