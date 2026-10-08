#include "core/buffer.h"

#include <cstdlib>
#include <new>

namespace tie {

namespace {

void* allocate_aligned(size_t bytes) {
  // aligned_alloc requires the size to be a multiple of the alignment.
  const size_t rounded = (bytes + kHostAlignment - 1) / kHostAlignment * kHostAlignment;
  void* p = std::aligned_alloc(kHostAlignment, rounded == 0 ? kHostAlignment : rounded);
  if (p == nullptr) throw std::bad_alloc();
  return p;
}

}  // namespace

HostBuffer::HostBuffer(size_t bytes) : Buffer(allocate_aligned(bytes), bytes) {}

HostBuffer::~HostBuffer() { std::free(data()); }

}  // namespace tie
