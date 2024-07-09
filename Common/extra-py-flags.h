// Copyright (c) Meta Platforms, Inc. and affiliates.

/*
 * A place to put extra CinderX-only flags used on existing Python objects.
 */

#pragma once

// Additional PyCodeObject flags (see Include/code.h)
#define CI_CO_STATICALLY_COMPILED 0x4000000

#if PY_VERSION_HEX >= 0x030C0000

// Stealing one of the old values used by stackless
#define Ci_Py_TPFLAGS_IS_STATICALLY_DEFINED  (1UL << 15)

#endif