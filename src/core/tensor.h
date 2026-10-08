#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>

#include "core/buffer.h"
#include "core/dtype.h"

namespace tie {

class Shape {
 public:
  static constexpr int kMaxDims = 4;

  Shape() = default;
  Shape(std::initializer_list<int64_t> dims);
  static Shape from(std::span<const int64_t> dims);

  int ndim() const { return ndim_; }
  int64_t operator[](int i) const { return dims_[static_cast<size_t>(i)]; }
  int64_t back() const { return dims_[static_cast<size_t>(ndim_ - 1)]; }
  int64_t numel() const;
  std::string str() const;
  bool operator==(const Shape& other) const;

 private:
  std::array<int64_t, kMaxDims> dims_{};
  int ndim_ = 0;
};

// A non-owning view of contiguous, row-major data inside a Buffer. "Rows" are
// all leading dimensions flattened; "cols" is the last dimension.
class Tensor {
 public:
  Tensor() = default;
  Tensor(Buffer* buffer, size_t offset, DType dtype, Shape shape);

  DType dtype() const { return dtype_; }
  const Shape& shape() const { return shape_; }
  int64_t dim(int i) const { return shape_[i]; }
  int64_t numel() const { return shape_.numel(); }
  size_t nbytes() const { return bytes_for(dtype_, numel()); }
  int64_t cols() const { return shape_.back(); }
  int64_t rows() const { return numel() / cols(); }
  size_t row_bytes() const { return bytes_for(dtype_, cols()); }

  Buffer* buffer() const { return buffer_; }
  size_t offset() const { return offset_; }
  bool empty() const { return buffer_ == nullptr; }

  void* raw() const { return static_cast<char*>(buffer_->data()) + offset_; }
  template <typename T>
  T* data() const {
    return static_cast<T*>(raw());
  }

  // View of rows [begin, begin + count) as a 2-D tensor [count, cols].
  Tensor slice_rows(int64_t begin, int64_t count) const;
  // Same bytes, new shape; the element count must not change.
  Tensor reshape(Shape shape) const;

 private:
  Buffer* buffer_ = nullptr;
  size_t offset_ = 0;
  DType dtype_ = DType::F32;
  Shape shape_;
};

}  // namespace tie
