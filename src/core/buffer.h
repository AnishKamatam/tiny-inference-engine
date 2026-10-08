#pragma once

#include <cstddef>

namespace tie {

// Page size on Apple Silicon. Page-aligned host memory can later be wrapped by
// Metal without copying, so every buffer tie allocates uses this alignment.
inline constexpr size_t kHostAlignment = 16384;

// A block of memory a backend computes on. Apple Silicon has unified memory, so
// every buffer is host-visible and `data()` is always valid; GPU backends
// subclass this to attach their native handle.
class Buffer {
 public:
  virtual ~Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  void* data() const { return data_; }
  size_t size() const { return size_; }

 protected:
  Buffer(void* data, size_t size) : data_(data), size_(size) {}

 private:
  void* data_;
  size_t size_;
};

// Owns page-aligned host memory.
class HostBuffer final : public Buffer {
 public:
  explicit HostBuffer(size_t bytes);
  ~HostBuffer() override;
};

// Borrows memory owned elsewhere (for example an mmapped weight file). The
// owner must outlive the buffer. Weight memory is never written through it.
class ExternalBuffer final : public Buffer {
 public:
  ExternalBuffer(const void* data, size_t bytes) : Buffer(const_cast<void*>(data), bytes) {}
};

}  // namespace tie
