#include "model/mapped_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "core/error.h"

namespace tie {

MappedFile::MappedFile(const std::filesystem::path& path) : path_(path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) fail<LoadError>("{}: cannot open: {}", path.string(), std::strerror(errno));

  struct stat st {};
  if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    ::close(fd);
    fail<LoadError>("{}: not a regular file", path.string());
  }
  size_ = static_cast<size_t>(st.st_size);
  if (size_ == 0) {
    ::close(fd);
    fail<LoadError>("{}: file is empty", path.string());
  }

  void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  const int mmap_errno = errno;
  ::close(fd);
  if (p == MAP_FAILED) fail<LoadError>("{}: mmap failed: {}", path.string(), std::strerror(mmap_errno));
  data_ = static_cast<const uint8_t*>(p);
}

MappedFile::~MappedFile() {
  if (data_ != nullptr) ::munmap(const_cast<uint8_t*>(data_), size_);
}

}  // namespace tie
