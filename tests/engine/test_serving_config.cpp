#include <doctest/doctest.h>

#include "core/error.h"
#include "engine/serving_config.h"

using namespace tie;

TEST_CASE("ServingConfig must be positive with no more sequences than step tokens") {
  CHECK_NOTHROW(validate(ServingConfig{}));
  CHECK_NOTHROW(validate(ServingConfig{8, 8}));
  CHECK_THROWS_AS(validate(ServingConfig{0, 1}), InvalidArgument);
  CHECK_THROWS_AS(validate(ServingConfig{8, 0}), InvalidArgument);
  // Every running sequence must be able to decode in the same step.
  CHECK_THROWS_WITH_AS(validate(ServingConfig{8, 9}), doctest::Contains("max_seqs"), InvalidArgument);
}

TEST_CASE("model_limits gives one logit row per sequence") {
  const ModelLimits l = model_limits(ServingConfig{256, 16}, 40);
  CHECK(l.max_tokens_per_step == 256);
  CHECK(l.max_seqs == 16);
  CHECK(l.max_logit_rows == 16);
  CHECK(l.max_blocks_per_seq == 40);
}
