// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <Python.h>

#if PY_VERSION_HEX < 0x030D0000
#define PyLong_AsInt _PyLong_AsInt
#endif
