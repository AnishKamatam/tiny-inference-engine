#include <doctest/doctest.h>

#include "core/dtype.h"
#include "core/error.h"
#include "core/half.h"

using namespace tie;

TEST_CASE("bytes_for covers plain and block-quantized types") {
  CHECK(bytes_for(DType::F32, 10) == 40);
  CHECK(bytes_for(DType::BF16, 10) == 20);
  CHECK(bytes_for(DType::I32, 3) == 12);
  CHECK(bytes_for(DType::Q8_0, 64) == 68);
  CHECK_THROWS_AS(bytes_for(DType::Q8_0, 33), InvalidArgument);
  CHECK(name(DType::Q8_0) == "Q8_0");
}

TEST_CASE("half conversions") {
  for (float f : {0.0f, 1.0f, -2.5f, 65504.0f, 1e-3f}) {
    CHECK(f16_to_f32(f32_to_f16(f)) == doctest::Approx(f).epsilon(1e-3));
  }
  CHECK(f32_to_f16(1.5f) == 0x3e00);
  CHECK(bf16_to_f32(0x3fc0) == 1.5f);
  CHECK(f32_to_bf16(1.5f) == 0x3fc0);
  // Exactly-halfway values round to the even mantissa.
  CHECK(f32_to_bf16(1.00390625f) == 0x3f80);  // 1 + 2^-8  -> 1.0
  CHECK(f32_to_bf16(1.01171875f) == 0x3f82);  // 1 + 3*2^-8 -> 1 + 2^-6
}
