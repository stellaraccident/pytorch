#include <ATen/pyre/dispatch/StridedCopyPlan.h>
#include <gtest/gtest.h>

using namespace at::pyre;

// Contiguous → Tier 0
TEST(StridedCopyPlanTest, ContiguousTensor) {
  auto plan = planCopy({4, 8}, {8, 1}, {8, 1}, 0, 0, 4);
  EXPECT_EQ(plan.tier, CopyPlan::kSingleCopy);
  ASSERT_EQ(plan.chunks.size(), 1u);
  EXPECT_EQ(plan.chunks[0].length, 4 * 8 * 4);
  EXPECT_EQ(plan.chunks[0].src_offset, 0);
  EXPECT_EQ(plan.chunks[0].dst_offset, 0);
}

// Sliced (every other row) → Tier 1
TEST(StridedCopyPlanTest, SlicedRows) {
  // shape=[4,8], src_strides=[16,1], dst_strides=[8,1]
  auto plan = planCopy({4, 8}, {16, 1}, {8, 1}, 0, 0, 4);
  EXPECT_EQ(plan.tier, CopyPlan::kDecomposed);
  ASSERT_EQ(plan.chunks.size(), 4u);
  for (auto& c : plan.chunks) EXPECT_EQ(c.length, 8 * 4);
  // Check offsets: src advances by 16*4=64 bytes per row, dst by 8*4=32.
  EXPECT_EQ(plan.chunks[0].src_offset, 0);
  EXPECT_EQ(plan.chunks[1].src_offset, 16 * 4);
  EXPECT_EQ(plan.chunks[0].dst_offset, 0);
  EXPECT_EQ(plan.chunks[1].dst_offset, 8 * 4);
}

// Narrowed (first K cols of [M,N]) → Tier 1
TEST(StridedCopyPlanTest, NarrowedColumns) {
  // shape=[4,3], src_strides=[8,1], dst_strides=[3,1]
  auto plan = planCopy({4, 3}, {8, 1}, {3, 1}, 0, 0, 4);
  EXPECT_EQ(plan.tier, CopyPlan::kDecomposed);
  ASSERT_EQ(plan.chunks.size(), 4u);
  for (auto& c : plan.chunks) EXPECT_EQ(c.length, 3 * 4);
}

// Transpose → Tier 2
TEST(StridedCopyPlanTest, Transpose2D) {
  // shape=[4,8], src_strides=[1,4], dst_strides=[8,1]
  auto plan = planCopy({4, 8}, {1, 4}, {8, 1}, 0, 0, 4);
  EXPECT_EQ(plan.tier, CopyPlan::kCompiledKernel);
  // Should have coalesced dims for kernel parametrization.
  EXPECT_FALSE(plan.dims.empty());
}

// 3D: contiguous inner dims coalesce → Tier 0
TEST(StridedCopyPlanTest, Coalescing3D) {
  auto plan = planCopy({2, 3, 4}, {12, 4, 1}, {12, 4, 1}, 0, 0, 4);
  EXPECT_EQ(plan.tier, CopyPlan::kSingleCopy);
  ASSERT_EQ(plan.chunks.size(), 1u);
  EXPECT_EQ(plan.chunks[0].length, 2 * 3 * 4 * 4);
}

// 3D with sliced outer dim → Tier 1
TEST(StridedCopyPlanTest, SlicedOuter3D) {
  // shape=[2,3,4], src_strides=[24,4,1], dst_strides=[12,4,1]
  // Inner 3×4=12 elements coalesce. 2 chunks of 12 elements.
  auto plan = planCopy({2, 3, 4}, {24, 4, 1}, {12, 4, 1}, 0, 0, 4);
  EXPECT_EQ(plan.tier, CopyPlan::kDecomposed);
  ASSERT_EQ(plan.chunks.size(), 2u);
  EXPECT_EQ(plan.chunks[0].length, 12 * 4);
  EXPECT_EQ(plan.chunks[1].length, 12 * 4);
}

// Storage offsets
TEST(StridedCopyPlanTest, StorageOffset) {
  auto plan = planCopy({4, 8}, {8, 1}, {8, 1}, 5, 0, 4);
  EXPECT_EQ(plan.tier, CopyPlan::kSingleCopy);
  ASSERT_EQ(plan.chunks.size(), 1u);
  EXPECT_EQ(plan.chunks[0].src_offset, 5 * 4);
  EXPECT_EQ(plan.chunks[0].dst_offset, 0);
}

// Scalar (0-dim)
TEST(StridedCopyPlanTest, ScalarTensor) {
  auto plan = planCopy({}, {}, {}, 0, 0, 4);
  EXPECT_EQ(plan.tier, CopyPlan::kSingleCopy);
  ASSERT_EQ(plan.chunks.size(), 1u);
  EXPECT_EQ(plan.chunks[0].length, 4);
}

// Broadcast dim (stride=0 in source)
TEST(StridedCopyPlanTest, BroadcastDim) {
  // shape=[4,8], src_strides=[0,1], dst_strides=[8,1]
  auto plan = planCopy({4, 8}, {0, 1}, {8, 1}, 0, 0, 4);
  EXPECT_EQ(plan.tier, CopyPlan::kDecomposed);
  ASSERT_EQ(plan.chunks.size(), 4u);
  for (auto& c : plan.chunks) {
    EXPECT_EQ(c.src_offset, 0);
    EXPECT_EQ(c.length, 8 * 4);
  }
}

// Verify chunk offsets precisely for sliced rows.
TEST(StridedCopyPlanTest, SlicedRowsOffsets) {
  auto plan = planCopy({4, 8}, {16, 1}, {8, 1}, 0, 0, 4);
  ASSERT_EQ(plan.chunks.size(), 4u);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(plan.chunks[i].src_offset, i * 16 * 4);
    EXPECT_EQ(plan.chunks[i].dst_offset, i * 8 * 4);
  }
}

// Verify narrowed column offsets.
TEST(StridedCopyPlanTest, NarrowedColumnsOffsets) {
  auto plan = planCopy({4, 3}, {8, 1}, {3, 1}, 0, 0, 4);
  ASSERT_EQ(plan.chunks.size(), 4u);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(plan.chunks[i].src_offset, i * 8 * 4);
    EXPECT_EQ(plan.chunks[i].dst_offset, i * 3 * 4);
  }
}

// Large number of chunks exceeds threshold → Tier 2.
TEST(StridedCopyPlanTest, TooManyChunksFallsToTier2) {
  // shape=[256,4], src_strides=[8,1], dst_strides=[4,1]
  // 256 outer chunks > kMaxCopyChunks=128 → Tier 2.
  auto plan = planCopy({256, 4}, {8, 1}, {4, 1}, 0, 0, 4);
  EXPECT_EQ(plan.tier, CopyPlan::kCompiledKernel);
}

// Storage offset with decomposed plan.
TEST(StridedCopyPlanTest, StorageOffsetDecomposed) {
  auto plan = planCopy({4, 8}, {16, 1}, {8, 1}, 3, 7, 4);
  EXPECT_EQ(plan.tier, CopyPlan::kDecomposed);
  ASSERT_EQ(plan.chunks.size(), 4u);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(plan.chunks[i].src_offset, (3 + i * 16) * 4);
    EXPECT_EQ(plan.chunks[i].dst_offset, (7 + i * 8) * 4);
  }
}

// 1D contiguous.
TEST(StridedCopyPlanTest, Contiguous1D) {
  auto plan = planCopy({100}, {1}, {1}, 0, 0, 8);
  EXPECT_EQ(plan.tier, CopyPlan::kSingleCopy);
  ASSERT_EQ(plan.chunks.size(), 1u);
  EXPECT_EQ(plan.chunks[0].length, 100 * 8);
}

// 1D strided → Tier 2.
TEST(StridedCopyPlanTest, Strided1D) {
  // Every other element: stride=2 → not contiguous inner.
  auto plan = planCopy({50}, {2}, {1}, 0, 0, 4);
  EXPECT_EQ(plan.tier, CopyPlan::kCompiledKernel);
}

// Single size-1 dim is contiguous.
TEST(StridedCopyPlanTest, SingleElement) {
  auto plan = planCopy({1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 0, 0, 4);
  EXPECT_EQ(plan.tier, CopyPlan::kSingleCopy);
  ASSERT_EQ(plan.chunks.size(), 1u);
  EXPECT_EQ(plan.chunks[0].length, 4);
}

// Tier 2 plans carry base offsets.
TEST(StridedCopyPlanTest, Tier2BaseOffsets) {
  // Transpose with storage offsets → Tier 2.
  auto plan = planCopy({4, 8}, {1, 4}, {8, 1}, 10, 20, 4);
  EXPECT_EQ(plan.tier, CopyPlan::kCompiledKernel);
  EXPECT_EQ(plan.src_base_offset, 10);
  EXPECT_EQ(plan.dst_base_offset, 20);
  EXPECT_FALSE(plan.dims.empty());
}

// Tier 2 from too many chunks also carries offsets.
TEST(StridedCopyPlanTest, Tier2TooManyChunksBaseOffsets) {
  auto plan = planCopy({256, 4}, {8, 1}, {4, 1}, 5, 7, 4);
  EXPECT_EQ(plan.tier, CopyPlan::kCompiledKernel);
  EXPECT_EQ(plan.src_base_offset, 5);
  EXPECT_EQ(plan.dst_base_offset, 7);
}

// Different element sizes.
TEST(StridedCopyPlanTest, ElementSize2) {
  auto plan = planCopy({4, 8}, {8, 1}, {8, 1}, 0, 0, 2);
  EXPECT_EQ(plan.tier, CopyPlan::kSingleCopy);
  EXPECT_EQ(plan.chunks[0].length, 4 * 8 * 2);
}
