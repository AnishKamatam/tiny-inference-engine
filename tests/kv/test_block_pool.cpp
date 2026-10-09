#include <doctest/doctest.h>

#include "core/error.h"
#include "kv/block_pool.h"

using namespace tie;

TEST_CASE("BlockPool hands out blocks lowest first and recycles them") {
  BlockPool pool(3);
  CHECK(pool.num_blocks() == 3);
  CHECK(pool.allocate() == 0);
  CHECK(pool.allocate() == 1);
  CHECK(pool.num_free() == 1);
  pool.release(0);
  CHECK(pool.num_free() == 2);
  CHECK(pool.allocate() == 0);
}

TEST_CASE("BlockPool reference counts keep shared blocks alive") {
  BlockPool pool(2);
  const int32_t b = pool.allocate();
  pool.retain(b);
  CHECK(pool.ref_count(b) == 2);
  pool.release(b);
  CHECK(pool.num_free() == 1);
  pool.release(b);
  CHECK(pool.num_free() == 2);
}

TEST_CASE("BlockPool rejects misuse") {
  BlockPool pool(1);
  CHECK_THROWS_AS(pool.release(0), InvalidArgument);  // never allocated
  CHECK_THROWS_AS(pool.retain(0), InvalidArgument);
  CHECK_THROWS_AS(pool.release(5), InvalidArgument);  // out of range
  pool.allocate();
  CHECK_THROWS_AS(pool.allocate(), CapacityError);
  CHECK_THROWS_AS(BlockPool(0), InvalidArgument);
}
