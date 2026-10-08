#pragma once

#include <filesystem>
#include <string>

namespace tie::test {

std::filesystem::path models_dir();
std::filesystem::path fixtures_dir();

// A fresh path under the system temp directory for files a test writes.
std::filesystem::path temp_path(const std::string& name);

// True if `path` exists; otherwise records a doctest MESSAGE saying how to create it.
bool have(const std::filesystem::path& path);

}  // namespace tie::test

// Skips the rest of the current test case when optional test data is missing.
#define TIE_REQUIRE_DATA(path) \
  if (!::tie::test::have(path)) return
