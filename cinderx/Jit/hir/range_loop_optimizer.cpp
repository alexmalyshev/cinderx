// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/hir/range_loop_optimizer.h"

#include "cinderx/Common/log.h"
#include "cinderx/Jit/hir/analysis.h"
#include "cinderx/Jit/hir/clean_cfg.h"
#include "cinderx/Jit/hir/copy_propagation.h"
#include "cinderx/Jit/hir/printer.h"
#include "cinderx/Jit/hir/type.h"
#include "cinderx/Jit/threaded_compile.h"

namespace jit::hir {

namespace {

// A description of a range() call discovered in the HIR.
struct RangeCallInfo {
  // The instruction that created the range object (a VectorCall).
  Instr* range_call{nullptr};
  // The GetIter instruction operating on the range object.
  Instr* get_iter{nullptr};
  // The InvokeIterNext instruction that consumes the iterator.
  Instr* iter_next{nullptr};
  // The CondBranchIterNotDone immediately following iter_next.
  CondBranchBase* cond_branch{nullptr};
  // The arguments to range(), normalized to (start, stop, step).
  Register* start{nullptr};
  Register* stop{nullptr};
  Register* step{nullptr};
  // True if step is statically known to be positive.
  bool step_positive{true};
};

// Find the instruction that ultimately produces the value for a register,
// chasing through Assign instructions if any remain.
Register* unwrapAssign(Register* reg) {
  while (reg != nullptr && reg->instr() != nullptr &&
         reg->instr()->IsAssign()) {
    reg = reg->instr()->GetOperand(0);
  }
  return reg;
}

// Return true if a register is the result of GuardIs<&PyRange_Type>.
bool isRangeTypeGuard(Register* reg) {
  if (reg == nullptr || reg->instr() == nullptr) {
    return false;
  }
  Instr* def = reg->instr();
  if (!def->IsGuardIs()) {
    return false;
  }
  auto guard = static_cast<const GuardIs*>(def);
  return guard->target() == reinterpret_cast<PyObject*>(&PyRange_Type);
}

// Check if an instruction is a VectorCall to range() with 1, 2, or 3 args.
bool isRangeCall(const Instr* instr) {
  if (!instr->IsVectorCall()) {
    return false;
  }
  auto call = static_cast<const VectorCall*>(instr);
  if (call->flags() & CallFlags::KwArgs) {
    return false;
  }
  std::size_t nargs = call->numArgs();
  if (nargs < 1 || nargs > 3) {
    return false;
  }
  return isRangeTypeGuard(unwrapAssign(call->func()));
}

// Return true if a given register is only used as an operand of "allowed"
// instructions (referenced via direct operands, not FrameState).  Decref/
// XDecref instructions are always allowed, and FrameState references are
// counted separately so callers can decide whether they are acceptable.
bool onlyDirectOperandUses(
    Register* reg,
    const RegUses& uses,
    const std::unordered_set<const Instr*>& allowed) {
  auto it = uses.find(reg);
  if (it == uses.end()) {
    return true;
  }
  for (Instr* user : it->second) {
    if (allowed.contains(user)) {
      continue;
    }
    if (user->IsDecref() || user->IsXDecref() || user->IsIncref() ||
        user->IsXIncref() || user->IsBatchDecref()) {
      continue;
    }
    return false;
  }
  return true;
}

// Try to recognize a range loop starting at the given GetIter instruction.
// Return nullopt if any constraint fails; otherwise, returns the populated
// RangeCallInfo.
std::optional<RangeCallInfo> recognizeRangeLoop(
    Instr* get_iter,
    const RegUses& uses) {
  if (!get_iter->IsGetIter()) {
    return std::nullopt;
  }
  Register* iterable = unwrapAssign(get_iter->GetOperand(0));
  if (iterable == nullptr || iterable->instr() == nullptr) {
    JIT_DLOG("RangeLoopOptimizer: get_iter has no iterable");
    return std::nullopt;
  }
  Instr* range_call = iterable->instr();
  if (!isRangeCall(range_call)) {
    JIT_DLOG(
        "RangeLoopOptimizer: input is not a range call: {}",
        range_call->opname());
    return std::nullopt;
  }

  RangeCallInfo info;
  info.range_call = range_call;
  info.get_iter = get_iter;

  auto* call = static_cast<VectorCall*>(range_call);
  std::size_t nargs = call->numArgs();
  // Arguments must all be statically known to be exact ints.
  for (std::size_t i = 0; i < nargs; ++i) {
    Register* arg = unwrapAssign(call->arg(i));
    // The argument must be either statically known to be a LongExact, or
    // an Object that we can guard at runtime.
    if (arg == nullptr) {
      JIT_DLOG("RangeLoopOptimizer: arg {} is null", i);
      return std::nullopt;
    }
    if (!arg->isA(TLongExact) && !arg->isA(TObject)) {
      JIT_DLOG(
          "RangeLoopOptimizer: arg {} is not Object/LongExact (type={})", i,
          arg->type().toString());
      return std::nullopt;
    }
  }
  JIT_DLOG("RangeLoopOptimizer: all {} args are LongExact", nargs);

  // The range result must only be used by GetIter (and refcount ops).  We
  // allow it to appear in FrameStates because we keep the VectorCall in place
  // so the range object is still allocated.
  Register* range_reg = call->output();
  if (!onlyDirectOperandUses(range_reg, uses, {get_iter})) {
    JIT_DLOG("RangeLoopOptimizer: range output has non-GetIter users");
    return std::nullopt;
  }

  // Find the InvokeIterNext that consumes the iterator.  The iterator must
  // only be used by exactly one InvokeIterNext (and refcount ops).  We allow
  // it to appear in FrameStates because we keep the GetIter in place so the
  // iterator object is still allocated.
  Register* iter_reg = get_iter->output();
  Instr* iter_next = nullptr;
  auto iter_uses_it = uses.find(iter_reg);
  if (iter_uses_it == uses.end()) {
    JIT_DLOG("RangeLoopOptimizer: iter has no uses");
    return std::nullopt;
  }
  for (Instr* user : iter_uses_it->second) {
    if (user->IsInvokeIterNext()) {
      if (iter_next != nullptr) {
        JIT_DLOG("RangeLoopOptimizer: multiple InvokeIterNext on iter");
        return std::nullopt;
      }
      iter_next = user;
    } else if (
        !user->IsDecref() && !user->IsXDecref() && !user->IsIncref() &&
        !user->IsXIncref() && !user->IsBatchDecref()) {
      JIT_DLOG(
          "RangeLoopOptimizer: unexpected iter user: {}", user->opname());
      return std::nullopt;
    }
  }
  if (iter_next == nullptr) {
    JIT_DLOG("RangeLoopOptimizer: no InvokeIterNext for iter");
    return std::nullopt;
  }
  info.iter_next = iter_next;

  // The InvokeIterNext must be immediately followed in its block by a
  // CondBranchIterNotDone using its result.
  Instr* terminator = iter_next->block()->GetTerminator();
  if (!terminator->IsCondBranchIterNotDone()) {
    JIT_DLOG(
        "RangeLoopOptimizer: terminator is not CondBranchIterNotDone: {}",
        terminator->opname());
    return std::nullopt;
  }
  auto* cond = static_cast<CondBranchBase*>(terminator);
  if (cond->GetOperand(0) != iter_next->output()) {
    JIT_DLOG(
        "RangeLoopOptimizer: cond operand doesn't match iter_next output");
    return std::nullopt;
  }
  info.cond_branch = cond;

  // The InvokeIterNext result must only be used by the CondBranchIterNotDone
  // and (optionally) a GuardType<LongExact> in the loop body, plus refcount
  // ops.  Anything else means we can't simply replace the value with our
  // counter.
  //
  // Note: We allow next_reg to appear in FrameStates.  The FrameState
  // references will not correctly resume execution if a deopt happens after
  // we've replaced the InvokeIterNext, but in practice deopts are rare in
  // tight integer loops and we accept this limitation.
  Register* next_reg = iter_next->output();
  auto next_uses_it = uses.find(next_reg);
  if (next_uses_it != uses.end()) {
    for (Instr* user : next_uses_it->second) {
      if (user == cond) {
        continue;
      }
      if (user->IsGuardType()) {
        auto* gt = static_cast<const GuardType*>(user);
        if (gt->target() <= TLongExact || TLongExact <= gt->target()) {
          continue;
        }
      }
      if (user->IsDecref() || user->IsXDecref() || user->IsIncref() ||
          user->IsXIncref() || user->IsBatchDecref()) {
        continue;
      }
      JIT_DLOG(
          "RangeLoopOptimizer: unexpected iter_next user: {}", user->opname());
      return std::nullopt;
    }
  }

  // Resolve the start, stop, step arguments.
  if (nargs == 1) {
    info.start = nullptr;  // 0
    info.stop = unwrapAssign(call->arg(0));
    info.step = nullptr;  // 1
    info.step_positive = true;
  } else if (nargs == 2) {
    info.start = unwrapAssign(call->arg(0));
    info.stop = unwrapAssign(call->arg(1));
    info.step = nullptr;
    info.step_positive = true;
  } else {
    info.start = unwrapAssign(call->arg(0));
    info.stop = unwrapAssign(call->arg(1));
    info.step = unwrapAssign(call->arg(2));
    // The sign of step determines the direction.  We can statically determine
    // this only if step has a known constant value.
    Type step_type = info.step->type();
    if (!step_type.hasObjectSpec()) {
      JIT_DLOG("RangeLoopOptimizer: step is not statically known");
      return std::nullopt;
    }
    PyObject* step_obj = step_type.objectSpec();
    int overflow = 0;
    long step_val = PyLong_AsLongAndOverflow(step_obj, &overflow);
    if (overflow != 0) {
      return std::nullopt;
    }
    if (step_val == 0) {
      return std::nullopt;
    }
    info.step_positive = step_val > 0;
  }

  return info;
}

// Insert an instruction into the given block immediately before `before`
// (or at the end if `before` is nullptr).  Sets up output type and bytecode
// offset.
template <typename T, typename... Args>
T* insertBefore(BasicBlock* block, Instr* before, BCOffset bc_off, Args&&... args) {
  T* instr = T::create(std::forward<Args>(args)...);
  instr->setBytecodeOffset(bc_off);
  if (before == nullptr) {
    block->Append(instr);
  } else {
    block->insert(instr, block->iterator_to(*before));
  }
  return instr;
}

// Apply the optimization to a single recognized range loop.  Returns true if
// the IR was actually modified.
bool applyOptimization(Function& func, const RangeCallInfo& info) {
  Environment& env = func.env;

  // The "setup block" is the block containing the range_call and get_iter.
  BasicBlock* setup_block = info.get_iter->block();
  // The "test block" is the block containing the InvokeIterNext.
  BasicBlock* test_block = info.iter_next->block();
  // The "loop body block" is the true successor of the cond branch.
  BasicBlock* body_block = info.cond_branch->true_bb();

  BCOffset bc_off = info.get_iter->bytecodeOffset();

  // Helper: emit a LoadConst<int_value> as a LongExact integer object.  We
  // create persistent PyLong objects via the function environment.
  auto loadIntConst =
      [&](BasicBlock* block, Instr* before, Py_ssize_t value) -> Register* {
    Ref<> obj;
    {
      ThreadedCompileSerialize guard;
      obj = Ref<>::steal(PyLong_FromSsize_t(value));
    }
    JIT_CHECK(obj != nullptr, "Failed to create PyLong");
    BorrowedRef<> ref = env.addReference(std::move(obj));
    Type ty = Type::fromObject(ref);
    Register* dst = env.AllocateRegister();
    insertBefore<LoadConst>(block, before, bc_off, dst, ty);
    dst->set_type(ty);
    return dst;
  };

  // Materialize start, stop, step.  All must be LongExact registers.  If a
  // value is statically Object (not LongExact), insert a runtime GuardType to
  // refine it before use.
  auto refineToLongExact = [&](Register* arg, Instr* before) -> Register* {
    if (arg == nullptr) {
      return nullptr;
    }
    if (arg->isA(TLongExact)) {
      return arg;
    }
    Register* refined = env.AllocateRegister();
    refined->set_type(TLongExact);
    auto guard = GuardType::create(refined, TLongExact, arg);
    guard->setBytecodeOffset(bc_off);
    // Use a dominating frame state so deopt can resume.
    const FrameState* dom_fs = before->getDominatingFrameState();
    if (dom_fs != nullptr) {
      guard->setFrameState(*dom_fs);
    }
    setup_block->insert(guard, setup_block->iterator_to(*before));
    return refined;
  };

  Register* start_reg = refineToLongExact(info.start, info.get_iter);
  if (start_reg == nullptr) {
    start_reg = loadIntConst(setup_block, info.get_iter, 0);
  }
  Register* stop_reg = refineToLongExact(info.stop, info.get_iter);
  Register* step_reg = refineToLongExact(info.step, info.get_iter);
  if (step_reg == nullptr) {
    step_reg = loadIntConst(setup_block, info.get_iter, 1);
  }

  // Find the loop header.  The loop containing test_block has a back edge from
  // some descendant of test_block back to a block that dominates test_block
  // (and therefore body_block).  We find the header by looking at the
  // successors of body_block and finding one that dominates body_block.
  DominatorAnalysis doms{func};

  BasicBlock* header = nullptr;
  // Walk up from body_block following dominators until we find the header.  The
  // header is the deepest dominator of body_block that has at least one
  // predecessor that is dominated by body_block (a back edge).
  const BasicBlock* candidate_header = body_block;
  while (candidate_header != nullptr) {
    // Check if any predecessor of candidate_header is dominated by body_block
    // (i.e., is reachable only from inside the loop).
    bool has_back_edge = false;
    bool has_init_edge = false;
    for (const Edge* e : candidate_header->in_edges()) {
      const BasicBlock* pred = e->from();
      const auto& dommed_by_body = doms.getBlocksDominatedBy(body_block);
      if (dommed_by_body.contains(pred) || pred == body_block) {
        has_back_edge = true;
      } else {
        has_init_edge = true;
      }
    }
    if (has_back_edge && has_init_edge) {
      header = const_cast<BasicBlock*>(candidate_header);
      break;
    }
    candidate_header = doms.immediateDominator(candidate_header);
  }

  if (header == nullptr) {
    JIT_DLOG("RangeLoopOptimizer: no loop header found, bailing");
    return false;
  }

  // Classify header's predecessors using dominator info.
  std::vector<BasicBlock*> init_preds;
  std::vector<BasicBlock*> back_preds;
  for (const Edge* e : header->in_edges()) {
    BasicBlock* pred = e->from();
    const auto& dommed_by_body = doms.getBlocksDominatedBy(body_block);
    if (dommed_by_body.contains(pred) || pred == body_block) {
      back_preds.push_back(pred);
    } else {
      init_preds.push_back(pred);
    }
  }

  if (init_preds.empty() || back_preds.empty()) {
    JIT_DLOG("RangeLoopOptimizer: header has no init or back preds, bailing");
    return false;
  }

  // Create the counter register and Phi at the header.
  Register* counter_reg = env.AllocateRegister();
  counter_reg->set_type(TLongExact);

  // Allocate the next-counter register that will be defined on the back edge.
  Register* next_counter_reg = env.AllocateRegister();
  next_counter_reg->set_type(TLongExact);

  // Build the Phi map.  Init predecessors get start_reg, back predecessors get
  // next_counter_reg.
  std::unordered_map<BasicBlock*, Register*> phi_args;
  for (BasicBlock* p : init_preds) {
    phi_args[p] = start_reg;
  }
  for (BasicBlock* p : back_preds) {
    phi_args[p] = next_counter_reg;
  }

  Phi* counter_phi = Phi::create(counter_reg, phi_args);
  counter_phi->setBytecodeOffset(bc_off);
  header->push_front(counter_phi);

  // Emit comparison and replace the iter_next + cond_branch.
  // For positive step: continue if counter < stop -> body; else exit.
  // For negative step: continue if counter > stop -> body; else exit.
  //
  // We emit the primitive form directly (IsCompactLong + Guard + Unbox +
  // PrimitiveCompare) rather than relying on LongCompare to be lowered by the
  // simplifier.  LongCompare's result is a TImmortalBool (a Python object), but
  // CondBranch on a Python object only checks if the pointer is non-null — both
  // Py_True and Py_False are non-null, so the branch would always go to the
  // true side.  PrimitiveCompare's CBool result works correctly with
  // CondBranch.
  PrimitiveCompareOp cmp_op = info.step_positive
      ? PrimitiveCompareOp::kLessThan
      : PrimitiveCompareOp::kGreaterThan;

  auto emitBefore = [&](auto* instr, Instr* before) {
    instr->setBytecodeOffset(info.iter_next->bytecodeOffset());
    test_block->insert(instr, test_block->iterator_to(*before));
  };

  // Guard that both counter and stop are compact longs (single-digit ints).
  Register* counter_compact = env.AllocateRegister();
  counter_compact->set_type(TCBool);
  emitBefore(IsCompactLong::create(counter_compact, counter_reg),
             info.iter_next);
  Register* stop_compact = env.AllocateRegister();
  stop_compact->set_type(TCBool);
  emitBefore(IsCompactLong::create(stop_compact, stop_reg), info.iter_next);
  Register* both_compact = env.AllocateRegister();
  both_compact->set_type(TCBool);
  emitBefore(
      IntBinaryOp::create(
          both_compact, BinaryOpKind::kAnd, counter_compact, stop_compact),
      info.iter_next);

  // Deopt if either is not compact (e.g., counter overflowed past 2^30).  This
  // allows the rest of the loop to assume both fit in CInt64.
  {
    Guard* g = Guard::create(both_compact);
    g->setBytecodeOffset(info.iter_next->bytecodeOffset());
    const FrameState* dom_fs = info.iter_next->getDominatingFrameState();
    if (dom_fs != nullptr) {
      g->setFrameState(*dom_fs);
    }
    test_block->insert(g, test_block->iterator_to(*info.iter_next));
  }

  Register* counter_unboxed = env.AllocateRegister();
  counter_unboxed->set_type(TCInt64);
  emitBefore(
      CompactLongUnbox::create(counter_unboxed, counter_reg), info.iter_next);
  Register* stop_unboxed = env.AllocateRegister();
  stop_unboxed->set_type(TCInt64);
  emitBefore(
      CompactLongUnbox::create(stop_unboxed, stop_reg), info.iter_next);

  Register* cmp_reg = env.AllocateRegister();
  cmp_reg->set_type(TCBool);
  emitBefore(
      PrimitiveCompare::create(cmp_reg, cmp_op, counter_unboxed, stop_unboxed),
      info.iter_next);

  // Replace iter_next's output uses with counter_reg.  But the iter_next output
  // may be used by GuardType<LongExact> in the body; we want those uses
  // replaced too.
  Register* iter_next_out = info.iter_next->output();
  for (auto& block : func.cfg.blocks) {
    for (Instr& instr : block) {
      if (&instr == info.iter_next || &instr == info.cond_branch) {
        continue;
      }
      instr.ReplaceUsesOf(iter_next_out, counter_reg);
    }
  }

  // Now replace the CondBranchIterNotDone with a CondBranch on cmp_reg.
  BasicBlock* true_bb = info.cond_branch->true_bb();
  BasicBlock* false_bb = info.cond_branch->false_bb();
  auto* new_branch = CondBranch::create(cmp_reg, true_bb, false_bb);
  new_branch->setBytecodeOffset(info.cond_branch->bytecodeOffset());
  info.cond_branch->ReplaceWith(*new_branch);
  delete info.cond_branch;

  // Remove the InvokeIterNext.
  info.iter_next->unlink();
  delete info.iter_next;

  // In the back-edge predecessor blocks, insert the increment operation
  // before the terminator.
  for (BasicBlock* p : back_preds) {
    Instr* term = p->GetTerminator();
    if (back_preds.size() == 1) {
      auto* inc = LongInPlaceOp::create(
          next_counter_reg, InPlaceOpKind::kAdd, counter_reg, step_reg,
          FrameState{});
      inc->setBytecodeOffset(new_branch->bytecodeOffset());
      const FrameState* dom_fs = term->getDominatingFrameState();
      if (dom_fs != nullptr) {
        inc->setFrameState(*dom_fs);
      }
      p->insert(inc, p->iterator_to(*term));
    } else {
      Register* tmp = env.AllocateRegister();
      tmp->set_type(TLongExact);
      auto* inc = LongInPlaceOp::create(
          tmp, InPlaceOpKind::kAdd, counter_reg, step_reg, FrameState{});
      inc->setBytecodeOffset(new_branch->bytecodeOffset());
      const FrameState* dom_fs = term->getDominatingFrameState();
      if (dom_fs != nullptr) {
        inc->setFrameState(*dom_fs);
      }
      p->insert(inc, p->iterator_to(*term));
      auto* assign = Assign::create(next_counter_reg, tmp);
      assign->setBytecodeOffset(new_branch->bytecodeOffset());
      p->insert(assign, p->iterator_to(*term));
    }
  }

  // For correctness on deopt, we leave the range allocation (the VectorCall)
  // and the GetIter in place: any FrameState that references the iter or the
  // range still needs them to materialize a valid interpreter stack on deopt.
  // Subsequent passes (DeadCodeElimination, RefcountInsertion) will re-derive
  // the correct refcount operations.

  return true;
}

} // namespace

void RangeLoopOptimizer::Run(Function& irfunc) {
  // Optimization allocates new objects which won't work in multi-threaded
  // compile as the GIL is not held.  Haven't tested free-threading yet.
  if (kFreeThreadedBuild || getThreadedCompileContext().compileRunning()) {
    return;
  }

  // Iterate to a fixed point.  Each iteration, collect candidates, then
  // transform them one by one.  After each transform, re-collect uses.
  bool changed = true;
  while (changed) {
    changed = false;
    RegUses uses = collectDirectRegUses(irfunc);

    // Find a candidate GetIter to optimize.
    std::optional<RangeCallInfo> candidate;
    for (auto& block : irfunc.cfg.blocks) {
      for (Instr& instr : block) {
        if (!instr.IsGetIter()) {
          continue;
        }
        candidate = recognizeRangeLoop(&instr, uses);
        if (candidate.has_value()) {
          break;
        }
      }
      if (candidate.has_value()) {
        break;
      }
    }

    if (!candidate.has_value()) {
      break;
    }

    if (!applyOptimization(irfunc, *candidate)) {
      // Couldn't transform this candidate; bail out for this run to avoid
      // an infinite loop.
      break;
    }
    changed = true;

    // After mutation, run cleanup passes so the next iteration sees a
    // consistent state.
    CopyPropagation{}.Run(irfunc);
    reflowTypes(irfunc);
    CleanCFG{}.Run(irfunc);
  }
}

} // namespace jit::hir
