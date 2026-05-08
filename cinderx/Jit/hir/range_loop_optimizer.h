// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/hir/pass.h"

namespace jit::hir {

// Strength-reduce `for n in range(...)` loops into primitive integer
// counter loops.  When the start, stop, and step arguments to range() are
// integer-typed and the resulting range/iterator values are only used by
// the for loop itself, the entire range allocation and iterator protocol
// can be replaced with a simple while loop over an int.
class RangeLoopOptimizer : public Pass {
 public:
  RangeLoopOptimizer() : Pass("RangeLoopOptimizer") {}

  void Run(Function& irfunc) override;

  static std::unique_ptr<RangeLoopOptimizer> Factory() {
    return std::make_unique<RangeLoopOptimizer>();
  }
};

} // namespace jit::hir
