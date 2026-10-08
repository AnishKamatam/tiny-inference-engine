#include "core/tensor.h"

#include "core/error.h"

namespace tie {

Shape::Shape(std::initializer_list<int64_t> dims) : Shape(from(std::span<const int64_t>(dims.begin(), dims.size()))) {}

Shape Shape::from(std::span<const int64_t> dims) {
  if (dims.empty() || dims.size() > kMaxDims) {
    fail<InvalidArgument>("shape must have 1 to {} dims, got {}", kMaxDims, dims.size());
  }
  Shape s;
  for (int64_t d : dims) {
    if (d < 0) fail<InvalidArgument>("negative dimension {}", d);
    s.dims_[static_cast<size_t>(s.ndim_++)] = d;
  }
  return s;
}

int64_t Shape::numel() const {
  if (ndim_ == 0) return 0;
  int64_t n = 1;
  for (int i = 0; i < ndim_; ++i) n *= (*this)[i];
  return n;
}

std::string Shape::str() const {
  std::string out = "[";
  for (int i = 0; i < ndim_; ++i) {
    if (i > 0) out += ", ";
    out += std::to_string((*this)[i]);
  }
  return out + "]";
}

bool Shape::operator==(const Shape& other) const {
  if (ndim_ != other.ndim_) return false;
  for (int i = 0; i < ndim_; ++i) {
    if ((*this)[i] != other[i]) return false;
  }
  return true;
}

Tensor::Tensor(Buffer* buffer, size_t offset, DType dtype, Shape shape)
    : buffer_(buffer), offset_(offset), dtype_(dtype), shape_(shape) {
  if (buffer_ == nullptr) fail<InvalidArgument>("tensor {} has no buffer", shape_.str());
  const size_t end = offset_ + nbytes();
  if (end > buffer_->size()) {
    fail<InvalidArgument>("tensor {} {} at offset {} ends at byte {}, past its {}-byte buffer", shape_.str(),
                          name(dtype_), offset_, end, buffer_->size());
  }
}

Tensor Tensor::slice_rows(int64_t begin, int64_t count) const {
  if (begin < 0 || count < 0 || begin + count > rows()) {
    fail<InvalidArgument>("rows [{}, {}) out of range for tensor {}", begin, begin + count, shape_.str());
  }
  return Tensor(buffer_, offset_ + static_cast<size_t>(begin) * row_bytes(), dtype_, Shape{count, cols()});
}

Tensor Tensor::reshape(Shape shape) const {
  if (shape.numel() != numel()) {
    fail<InvalidArgument>("cannot reshape {} to {}", shape_.str(), shape.str());
  }
  return Tensor(buffer_, offset_, dtype_, shape);
}

}  // namespace tie
