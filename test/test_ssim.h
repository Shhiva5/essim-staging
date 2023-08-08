/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <essim/essim.h>
#include <essim/inc/cpu.h>
#include <essim/inc/internal.h>
#include <essim/inc/memory.h>
#include <essim/inc/xplatform.h>

#define NOT_TESTING_PATHS 0
#define BUG_FIX 1
#define PROFILING_PRINTS 0

namespace ssim {

inline bool CheckSIMD(const eCPUType requiredSIMD) {
  return (requiredSIMD == (GetCpuType() & requiredSIMD));
}

struct aligned_memory_deleter_t {
  void operator()(void *p) { ssim_free_aligned(p); }
};

} // namespace ssim
