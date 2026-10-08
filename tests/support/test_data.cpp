#include "support/test_data.h"

#include <doctest/doctest.h>

namespace tie::test {

std::filesystem::path models_dir() { return std::filesystem::path(TIE_SOURCE_DIR) / "models"; }

std::filesystem::path fixtures_dir() { return models_dir() / "fixtures"; }

std::filesystem::path temp_path(const std::string& name) {
  const auto dir = std::filesystem::temp_directory_path() / "tie_tests";
  std::filesystem::create_directories(dir);
  const auto path = dir / name;
  std::filesystem::remove(path);
  return path;
}

bool have(const std::filesystem::path& path) {
  if (std::filesystem::exists(path)) return true;
  MESSAGE("skipped: " << path.string() << " is missing; run tools/setup_test_data.sh");
  return false;
}

}  // namespace tie::test
