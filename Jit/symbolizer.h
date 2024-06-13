// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "cinderx/Jit/mmap_file.h"

#include <elf.h>
#include <fcntl.h>
#include <link.h> // for ElfW
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace jit {

class Symbolizer {
 public:
  Symbolizer(const char* exe_path = "/proc/self/exe");

  bool isInitialized() const {
    return file_.isOpen();
  }

  ~Symbolizer() {
    deinit();
  }

  // Return a string view whose lifetime is tied to the Symbolizer lifetime on
  // success. On failure, return std::nullopt.
  std::optional<std::string_view> symbolize(const void* func);

  std::optional<std::string_view> cache(
      const void* func,
      std::optional<std::string> name);

 private:
  void deinit();

  MmapFile file_;
  const ElfW(Shdr) * symtab_{nullptr};
  const ElfW(Shdr) * strtab_{nullptr};
  // This cache is useful for performance and also critical for correctness.
  // Some of the symbols (for example, to shared objects) do not return owned
  // pointers. We must keep an object in this map for the string_view to point
  // to.
  std::unordered_map<const void*, std::optional<std::string>> cache_;
};

std::optional<std::string> demangle(const std::string& mangled_name);

} // namespace jit
