// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/dead_code_elimination.h"

#include "cinderx/Jit/hir/instr_effects.h"
#include "cinderx/Jit/hir/printer.h"

#define TRACE(...) \
  JIT_LOGIF(getConfig().log.debug_dead_code_elimination, __VA_ARGS__)

namespace jit::hir {

namespace {

bool isUseful(Instr& instr) {
  return instr.IsTerminator() || instr.IsSnapshot() ||
      (instr.asDeoptBase() != nullptr && !instr.IsPrimitiveBox()) ||
      (!instr.IsPhi() && memoryEffects(instr).may_store != AEmpty);
}

} // namespace

void DeadCodeElimination::Run(Function& func) {
  TRACE("Running dead code elimination on {}", func.fullname);

  // Find all immediately useful instructions.
  Worklist<Instr*> worklist;
  for (auto& block : func.cfg.blocks) {
    for (Instr& instr : block) {
      if (isUseful(instr)) {
        TRACE("Identified {} as useful", instr);
        worklist.push(&instr);
      }
    }
  }

  // Compute all necessary instructions, based on the initial set of useful
  // instructions.
  std::unordered_set<Instr*> live_set;
  while (!worklist.empty()) {
    auto live_op = worklist.front();
    worklist.pop();
    if (live_set.insert(live_op).second) {
      live_op->visitUses([&](Register*& reg) {
        if (!live_set.contains(reg->instr())) {
          TRACE("Adding '{}' as a live instruction, because {} is used in '{}'",
                *reg->instr(), *reg, *live_op);
          worklist.push(reg->instr());
        }
        return true;
      });
    }
  }

  for (auto& block : func.cfg.blocks) {
    for (auto it = block.begin(); it != block.end();) {
      auto& instr = *it;
      ++it;
      if (!live_set.contains(&instr)) {
        TRACE("Deleting '{}' as it is not live", instr);
        instr.unlink();
        delete &instr;
      }
    }
  }
}

} // namespace jit::hir
