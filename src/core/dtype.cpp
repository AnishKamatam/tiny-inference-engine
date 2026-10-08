#include "core/dtype.h"

#include "core/error.h"

namespace tie {

size_t bytes_for(DType t, int64_t n) {
  const DTypeTraits tr = traits(t);
  if (n < 0 || n % tr.block_elems != 0) {
    fail<InvalidArgument>("{} elements of {} do not fill whole blocks of {}", n, tr.name, tr.block_elems);
  }
  return static_cast<size_t>(n / tr.block_elems * tr.block_bytes);
}

}  // namespace tie
