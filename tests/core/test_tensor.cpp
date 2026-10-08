#include <doctest/doctest.h>

#include <cstdint>

#include "core/buffer.h"
#include "core/error.h"
#include "core/tensor.h"

using namespace tie;

TEST_CASE("Shape reports dims, element count and a readable string") {
  Shape s{2, 3, 4};
  CHECK(s.ndim() == 3);
  CHECK(s.numel() == 24);
  CHECK(s.back() == 4);
  CHECK(s.str() == "[2, 3, 4]");
  CHECK((Shape{2, 3} == Shape{2, 3}));
  CHECK_FALSE((Shape{2, 3} == Shape{3, 2}));
  CHECK_THROWS_AS(Shape({2, -1}), InvalidArgument);
}

TEST_CASE("Tensor views rows and reshapes without copying") {
  HostBuffer buf(6 * sizeof(float));
  Tensor t(&buf, 0, DType::F32, Shape{2, 3});
  float* p = t.data<float>();
  for (int i = 0; i < 6; ++i) p[i] = static_cast<float>(i);

  CHECK(t.rows() == 2);
  CHECK(t.cols() == 3);
  CHECK(t.nbytes() == 24);
  CHECK(t.row_bytes() == 12);

  Tensor r = t.slice_rows(1, 1);
  CHECK((r.shape() == Shape{1, 3}));
  CHECK(r.data<float>()[0] == 3.0f);

  Tensor flat = t.reshape(Shape{6});
  CHECK(flat.data<float>() == p);
  CHECK_THROWS_AS(t.reshape(Shape{4}), InvalidArgument);
  CHECK_THROWS_AS(t.slice_rows(1, 2), InvalidArgument);
}

TEST_CASE("Tensor rejects views that overrun their buffer") {
  HostBuffer buf(16);
  CHECK_THROWS_AS(Tensor(&buf, 8, DType::F32, Shape{4}), InvalidArgument);
}

TEST_CASE("HostBuffer is page aligned so GPU backends can wrap it without copying") {
  HostBuffer buf(100);
  CHECK(reinterpret_cast<uintptr_t>(buf.data()) % kHostAlignment == 0);
  CHECK(buf.size() == 100);
}
