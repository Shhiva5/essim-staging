/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "test_ssim.h"

#include <gtest/gtest.h>

#include <array>
#include <thread>
#include <vector>

TEST(ssimTest, constants) {
  // constant values shouldn't be changed
  ASSERT_EQ(SSIM_OK, 0);
  ASSERT_EQ(SSIM_ERR_NULL_PTR, 1);
  ASSERT_EQ(SSIM_ERR_BAD_PARAM, 2);
  ASSERT_EQ(SSIM_ERR_BAD_SIZE, 3);
  ASSERT_EQ(SSIM_ERR_UNSUPPORTED, 4);
  ASSERT_EQ(SSIM_ERR_ALLOC, 5);
  ASSERT_EQ(SSIM_ERR_FAILED, 6);

  ASSERT_EQ(SSIM_DATA_8BIT, 1);
  ASSERT_EQ(SSIM_DATA_16BIT, 2);

} // TEST(ssimTest, constants)

namespace {
enum { MAX_NUM_THREADS = 8 };

struct Dim {
  uint32_t width;
  uint32_t height;
};

const std::array<Dim, 10> dims{{{16, 16},
                                {18, 18},
                                {20, 20},
                                {60, 16},
                                {62, 16},
                                {64, 16},
                                {66, 16},
                                {68, 16},
                                {320, 240},
                                {720, 480}}};

struct ctx_array_deallocator {
  void operator()(SSIM_CTX_ARRAY *const array) { ssim_free_ctx_array(array); }
};

template <typename data_t, typename ptr_t>
inline std::tuple<data_t *, ptrdiff_t>
AllocateAndFill(ptr_t &ptr, const uint32_t width, const uint32_t height,
                const data_t maxValue) {
  const ptrdiff_t stride = (sizeof(data_t) * width) & -LOG2_ALIGN;
  const size_t requiredSize = stride * height;
  ptr.reset((data_t *)ssim_alloc_aligned(requiredSize, LOG2_ALIGN));
  data_t *p = (data_t *)ptr.get();
  for (size_t s = 0; s < requiredSize / sizeof(data_t); ++s) {
    p[s] = std::rand() & maxValue;
  }

  return {p, stride};
}

// Compute eSSIM for 8bit data using multiple-threads.
eSSIMResult ssim_compute_threaded(
    uint32_t *const pSsimScore, uint32_t *const pEssimScore, const void *ref,
    const ptrdiff_t refStride, const void *cmp, const ptrdiff_t cmpStride,
    const uint32_t width, const uint32_t height, const uint32_t bitDepthMinus8,
    const uint32_t windowSize, const uint32_t windowStride, const uint32_t d2h,
    const eSSIMMode mode, const eSSIMFlags flags, const uint32_t numThreads) {
  std::unique_ptr<SSIM_CTX_ARRAY, ctx_array_deallocator> ctx_array(
      ssim_allocate_ctx_array(
          numThreads, width, height, bitDepthMinus8,
          bitDepthMinus8 > 0 ? SSIM_DATA_16BIT : SSIM_DATA_8BIT, windowSize,
          windowStride, 1, mode, SSIM_SPATIAL_POOLING_COEFF_OF_VARIANCE));
  if (!ctx_array) {
    return SSIM_ERR_ALLOC;
  }

  std::vector<std::thread> threads(numThreads);

  auto thread_proc = [&](const size_t t) -> eSSIMResult {
    SSIM_CTX *ctx = ssim_access_ctx(ctx_array.get(), t);
    if (!ctx) {
      return SSIM_ERR_FAILED;
    }

    ssim_reset_ctx(ctx);
    const uint32_t beginHeight = height * t / numThreads;
    const uint32_t endHeight = height * (t + 1) / numThreads;
    return ssim_compute_ctx(ctx, ref, refStride, cmp, cmpStride, beginHeight,
                            endHeight - beginHeight);
  };

  for (size_t t = 1; t < numThreads; ++t) {
    threads.push_back(std::thread(thread_proc, t));
  }

  eSSIMResult res = thread_proc(0);

  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  if (SSIM_OK == res) {
    res = ssim_aggregate_score(pSsimScore, pEssimScore, ctx_array.get());
  }

  return res;

} // eSSIMResult ssim_compute_threaded(

// Compute eSSIM for 16bit data using multiple-threads.

} // namespace

TEST(ssimTest, threading) {
  for (auto &dim : dims) {
    std::unique_ptr<void, ssim::aligned_memory_deleter_t> pRefAllocated,
        pCmpAllocated;
    const size_t width = dim.width;
    const size_t height = dim.height;
    const uint32_t bitDepthMinus8 = 0;
    // allocate frames
    auto maxValue = std::numeric_limits<uint8_t>::max();
    auto [pRef, refStride] =
        AllocateAndFill<uint8_t>(pRefAllocated, width, height, maxValue);

    auto [pCmp, cmpStride] =
        AllocateAndFill<uint8_t>(pCmpAllocated, width, height, maxValue);

    // test all available windows & stride combination
    for (uint32_t windowSize = 8; windowSize <= 16; windowSize *= 2) {
      for (uint32_t windowStride = 4; windowStride <= windowSize;
           windowStride *= 2) {
        // call the reference function
        uint32_t essimRef = 0;
        uint32_t ssimRef = 0;
        auto resRef = ssim_compute_8u(&ssimRef, &essimRef, pRef, refStride, pCmp,
                                      cmpStride, width, height, windowSize,
                                      windowStride, 1, SSIM_MODE_PERF_INT,
                                      SSIM_SPATIAL_POOLING_COEFF_OF_VARIANCE);

        // call and test threaded versions
        for (uint32_t numThreads = 1; numThreads < MAX_NUM_THREADS;
             ++numThreads) {
          uint32_t ssimTst = 0;
          uint32_t essimTst = 0;
          auto resTst = ssim_compute_threaded(
              &ssimTst, &essimTst, pRef, refStride, pCmp, cmpStride, width, height,
              bitDepthMinus8, windowSize, windowStride, 1, SSIM_MODE_PERF_INT,
              SSIM_SPATIAL_POOLING_COEFF_OF_VARIANCE, numThreads);
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          ASSERT_EQ(resRef, resTst);
          ASSERT_EQ(ssimRef, ssimTst)
              << "mean pooled ssim failed with " << numThreads << " threads. Window size is "
              << windowSize << ", window stride is " << windowStride
              << ". Frame size is " << width << "x" << height;
          ASSERT_EQ(essimRef, essimTst)
              << "essim failed with " << numThreads << " threads. Window size is "
              << windowSize << ", window stride is " << windowStride
              << ". Frame size is " << width << "x" << height;
        }
      }
    }
  }

} // TEST(ssimTest, threading)

TEST(ssimTest, threading_10bit) {
  for (auto &dim : dims) {
    std::unique_ptr<void, ssim::aligned_memory_deleter_t> pRefAllocated,
        pCmpAllocated;
    const size_t width = dim.width;
    const size_t height = dim.height;
    const uint32_t bitDepthMinus8 = 2;
    // allocate frames
    auto maxValue = std::numeric_limits<uint16_t>::max();
    auto [pRef, refStride] =
        AllocateAndFill<uint16_t>(pRefAllocated, width, height, maxValue);

    auto [pCmp, cmpStride] =
        AllocateAndFill<uint16_t>(pCmpAllocated, width, height, maxValue);

    // test all available windows & stride combination
    for (uint32_t windowSize = 8; windowSize <= 16; windowSize *= 2) {
      for (uint32_t windowStride = 4; windowStride <= windowSize;
           windowStride *= 2) {
        // call the reference function
        uint32_t ssimRef = 0;
        uint32_t essimRef = 0 ;
        auto resRef = ssim_compute_16u(
            &ssimRef, &essimRef, pRef, refStride, pCmp, cmpStride, width, height,
            bitDepthMinus8, windowSize, windowStride, 1, SSIM_MODE_PERF_INT,
            SSIM_SPATIAL_POOLING_COEFF_OF_VARIANCE);

        // call and test threaded versions
        for (uint32_t numThreads = 1; numThreads < MAX_NUM_THREADS;
             ++numThreads) {
          uint32_t ssimTst = 0;
          uint32_t essimTst = 0;
          auto resTst = ssim_compute_threaded(
              &ssimTst, &essimTst, pRef, refStride, pCmp, cmpStride, width, height,
              bitDepthMinus8, windowSize, windowStride, 1, SSIM_MODE_PERF_INT,
              SSIM_SPATIAL_POOLING_COEFF_OF_VARIANCE, numThreads);
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          ASSERT_EQ(resRef, resTst);
          ASSERT_EQ(ssimRef, ssimTst)
              << "mean pooled ssim failed with " << numThreads << " threads. Window size is "
              << windowSize << ", window stride is " << windowStride
              << ". Frame size is " << width << "x" << height;
          ASSERT_EQ(essimRef, essimTst)
              << "essim failed with " << numThreads << " threads. Window size is "
              << windowSize << ", window stride is " << windowStride
              << ". Frame size is " << width << "x" << height;
        }
      }
    }
  }

} // TEST(ssimTest, threading_10bit)
