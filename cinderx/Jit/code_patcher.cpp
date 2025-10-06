// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cinderx/Jit/code_patcher.h"

#include "cinderx/Common/log.h"
#include "cinderx/Common/util.h"

#include <array>
#include <cstring>

namespace jit {

namespace {

static_assert(
    sizeof(CodePatcher) == 24,
    "CodePatcher should be kept small as there could be many per function");

// 5-byte nop - https://www.felixcloutier.com/x86/nop
//
// Asmjit supports multi-byte nops but for whatever reason we can't get it to
// emit the 5-byte version.
constexpr auto kJmpNopBytes =
    std::to_array<uint8_t>({0x0f, 0x1f, 0x44, 0x00, 0x00});

// Compute an x86-64 jump displacement operand.
uint32_t jumpDisplacement(uintptr_t from, uintptr_t to) {
  auto disp = to - (from + kJmpNopBytes.size());
  JIT_CHECK(
      fitsInt32(disp),
      "Can't encode jump from {:#x} to {:#x} as relative",
      from,
      to);
  return static_cast<uint32_t>(disp);
}

// Given the starting address and displacement operand of a jump instruction,
// resolve it to a target address.
uintptr_t resolveDisplacement(uintptr_t from, uint32_t displacement) {
  return from + displacement + kJmpNopBytes.size();
}

} // namespace

void CodePatcher::link(uintptr_t patchpoint, std::span<const uint8_t> data) {
  JIT_CHECK(!isLinked(), "Trying to re-link a patcher");

  patchpoint_ = reinterpret_cast<uint8_t*>(patchpoint);

  JIT_CHECK(
      data.size() <= data_.size(),
      "Trying to link a patch point with {} bytes of data but only {} are "
      "supported",
      data.size(),
      data_.size());

  std::memcpy(data_.data(), data.data(), data.size());
  data_len_ = data.size();

  onLink();
}

void CodePatcher::patch() {
  JIT_CHECK(isLinked(), "Trying to patch a patcher that isn't linked");
  JIT_DLOG("Patching DeoptPatchPoint at {}", static_cast<void*>(patchpoint_));

  swap();

  is_patched_ = true;
  onPatch();
}

void CodePatcher::unpatch() {
  JIT_CHECK(isLinked(), "Trying to unpatch a patcher that isn't linked");
  JIT_DLOG("Unpatching DeoptPatchPoint at {}", static_cast<void*>(patchpoint_));

  swap();

  is_patched_ = false;
  onUnpatch();
}

bool CodePatcher::isLinked() const {
  return patchpoint_ != nullptr;
}

bool CodePatcher::isPatched() const {
  return is_patched_;
}

uint8_t* CodePatcher::patchpoint() const {
  return patchpoint_;
}

std::span<const uint8_t> CodePatcher::storedBytes() const {
  return std::span{data_.data(), data_len_};
}

void CodePatcher::swap() {
  decltype(data_) temp;
  std::memcpy(temp.data(), patchpoint_, data_len_);
  std::memcpy(patchpoint_, data_.data(), data_len_);
  std::memcpy(data_.data(), temp.data(), data_len_);
}

JumpPatcher::JumpPatcher() {
  // Initializes to a nop.
  std::memcpy(data_.data(), kJmpNopBytes.data(), kJmpNopBytes.size());
  data_len_ = kJmpNopBytes.size();
}

void JumpPatcher::linkJump(uintptr_t patchpoint, uintptr_t jump_target) {
  auto disp = jumpDisplacement(patchpoint, jump_target);

  // 32 bit relative jump - https://www.felixcloutier.com/x86/jmp
  std::array<uint8_t, kJmpNopBytes.size()> buf{};
  buf[0] = 0xe9;
  std::memcpy(buf.data() + 1, &disp, sizeof(uint32_t));

  link(patchpoint, buf);
}

uint8_t* JumpPatcher::jumpTarget() const {
  JIT_CHECK(
      isLinked(), "Can't compute jump target before JumpPatcher is linked");

  std::span<const uint8_t> bytes = storedBytes();
  JIT_CHECK(
      bytes.size() == 5,
      "Must have linked a 5-byte 'jmp $DISP' instruction into a JumpPatcher");

  uint32_t disp = 0;
  std::memcpy(&disp, bytes.data() + 1, bytes.size() - 1);

  return reinterpret_cast<uint8_t*>(
      resolveDisplacement(reinterpret_cast<uintptr_t>(patchpoint_), disp));
}

} // namespace jit
