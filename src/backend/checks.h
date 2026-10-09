#pragma once

#include "core/error.h"
#include "core/tensor.h"

namespace tie {

// Argument checks shared by every backend, so CPU and Metal reject bad input identically.
inline void expect_dtype(const Tensor& t, DType dtype, const char* what) {
  if (t.dtype() != dtype) fail<InvalidArgument>("{} must be {}, got {}", what, name(dtype), name(t.dtype()));
}

inline void expect_shape(const Tensor& t, const Shape& shape, const char* what) {
  if (!(t.shape() == shape)) fail<InvalidArgument>("{} must be {}, got {}", what, shape.str(), t.shape().str());
}

}  // namespace tie
