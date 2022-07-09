/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if !defined(__SSIM_SSIM_INTERNAL_H)
#define __SSIM_SSIM_INTERNAL_H

#include <essim/inc/xplatform.h>
#include <essim/essim.h>

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

enum { LOG2_ALIGN = 6, ALIGN = 1 << LOG2_ALIGN };

#define SSIM_POOLING_MINKOWSKI_P 4

#pragma pack(push, 1)

typedef struct WINDOW_STATS {
  uint32_t ref_sum;
  uint32_t cmp_sum;
  uint64_t ref_sigma_sqd;
  uint64_t cmp_sigma_sqd;
  uint64_t sigma_both;

} WINDOW_STATS;

/* the temporal buffer has the following structure:
a row of 16u(32u) interleaving sum values for ref and cmp,
a row of 32u(64u) ref squared values,
a row of 32u(64u) cmp squared values,
a row of 32u(64u) sigma values, then again a line of 16u values etc.
the byte distance between rows is stride */

typedef struct SSIM_4X4_WINDOW_BUFFER {
  void* p;
  ptrdiff_t stride;

} SSIM_4X4_WINDOW_BUFFER;

typedef struct SSIM_4X4_WINDOW_ROW {
  /* pointers to row data */
  SSIM_4X4_WINDOW_BUFFER ptrs;

  /* vertical coordinate of the last processed row */
  uint32_t y;

} SSIM_4X4_WINDOW_ROW;

typedef struct SSIM_RES {
  /* integer function results */
  uint64_t ssim_sum;
  uint64_t ssim_mink_sum;

  /* float function results */
  float ssim_sum_f;
  double ssim_mink_sum_f;

  /* number of windows summed */
  size_t numWindows;
} SSIM_RES;

typedef struct SSIM_SRC {
  /* pointer to reference data */
  const void* ref;
  /* reference data stride */
  ptrdiff_t refStride;
  /* pointer to reconstructed data */
  const void* cmp;
  /* reconstructed data stride */
  ptrdiff_t cmpStride;
} SSIM_SRC;

/*
    precision function types
*/

#define LOAD_WINDOW_FORMAL_ARGS                         \
  WINDOW_STATS *const pWnd, const SSIM_SRC *const pSrc, \
      const uint32_t windowSize
#define LOAD_WINDOW_ACTUAL_ARGS pWnd, pSrc, windowSize

#define CALC_WINDOW_SSIM_FORMAL_ARGS                                      \
  WINDOW_STATS *const pWnd, const uint32_t windowSize, const uint32_t C1, \
      const uint32_t C2
#define CALC_WINDOW_SSIM_ACTUAL_ARGS pWnd, windowSize, C1, C2

typedef void (*load_window_proc_t)(LOAD_WINDOW_FORMAL_ARGS);
typedef int64_t (*calc_window_ssim_proc_t)(CALC_WINDOW_SSIM_FORMAL_ARGS);

/*
    performance function types
*/

#define LOAD_4x4_WINDOWS_FORMAL_ARGS                                    \
  const SSIM_4X4_WINDOW_BUFFER *const pBuf, const size_t num4x4Windows, \
      const SSIM_SRC *const pSrc
#define LOAD_4x4_WINDOWS_ACTUAL_ARGS pBuf, num4x4Windows, pSrc

#define SUM_WINDOWS_FORMAL_ARGS                            \
  SSIM_RES *const res, SSIM_4X4_WINDOW_BUFFER *const pBuf, \
      const size_t numWindows, const uint32_t windowSize,  \
      const uint32_t windowStride, const uint32_t bitDepthMinus8
#define SUM_WINDOWS_ACTUAL_ARGS \
  res, pBuf, numWindows, windowSize, windowStride, bitDepthMinus8

typedef void (*load_4x4_windows_proc_t)(LOAD_4x4_WINDOWS_FORMAL_ARGS);
typedef void (*sum_windows_proc_t)(SUM_WINDOWS_FORMAL_ARGS);

typedef struct SSIM_PARAMS {
  /* stream parameters */
  uint32_t width;
  uint32_t height;
  uint32_t bitDepthMinus8;
  eSSIMDataType dataType;

  /* SSIM parameters */
  uint32_t windowSize;
  uint32_t windowStride;
  eSSIMMode mode;
  eSSIMFlags flags;

  /* processing functions */
  load_window_proc_t load_window_proc;
  calc_window_ssim_proc_t calc_window_ssim_proc;
  load_4x4_windows_proc_t load_4x4_windows_proc;
  sum_windows_proc_t sum_windows_proc;
} SSIM_PARAMS;

struct SSIM_CTX {
  void* buffer;
  size_t bufferSize;
  ptrdiff_t bufferStride;

  SSIM_4X4_WINDOW_ROW* windowRows;
  size_t numWindowRows;

  SSIM_RES res;

  const SSIM_PARAMS* params;
};

struct SSIM_CTX_ARRAY {
  SSIM_CTX** ctx;
  size_t numCtx;

  SSIM_PARAMS params;

  uint32_t d2h;
};

#pragma pack(pop)

/*
    declare tool functions
*/

/* get the number of windows 1D */
uint32_t GetNum4x4Windows(
    const uint32_t value,
    const uint32_t windowSize,
    const uint32_t windowStride);
uint32_t GetNumWindows(
    const uint32_t value,
    const uint32_t windowSize,
    const uint32_t windowStride);

/* get the number of windows 2D */
uint32_t GetTotalWindows(
    const uint32_t width,
    const uint32_t height,
    const uint32_t windowSize,
    const uint32_t windowStride);

/* advance a pointer on a stride in bytes */
void* AdvancePointer(const void* p, const ptrdiff_t stride);

uint32_t get_ssim_int_constant(
    const uint32_t constIdx,
    const uint32_t bitDepthMinus8,
    const uint32_t windowSize);
float get_ssim_float_constant(
    const uint32_t constIdx,
    const uint32_t bitDepthMinus8);

void load_window_8u_c(LOAD_WINDOW_FORMAL_ARGS);
void load_window_16u_c(LOAD_WINDOW_FORMAL_ARGS);

int64_t calc_window_ssim_int_8u(CALC_WINDOW_SSIM_FORMAL_ARGS);
int64_t calc_window_ssim_int_16u(CALC_WINDOW_SSIM_FORMAL_ARGS);
float calc_window_ssim_float(
    WINDOW_STATS* const pWnd,
    const uint32_t windowSize,
    const float C1,
    const float C2);

eSSIMResult ssim_compute_prec(
    SSIM_CTX* const ctx,
    const void* ref,
    const ptrdiff_t refStride,
    const void* cmp,
    const ptrdiff_t cmpStride);

eSSIMResult ssim_compute_perf(
    SSIM_CTX* const ctx,
    const void* ref,
    const ptrdiff_t refStride,
    const void* cmp,
    const ptrdiff_t cmpStride,
    const uint32_t roiY,
    const uint32_t roiHeight);

/*
    declare optimized functions callers
*/

void load_4x4_windows_8u(LOAD_4x4_WINDOWS_FORMAL_ARGS);
void sum_windows_int_8u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x4_int_8u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_12x4_int_8u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_float_8u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x4_float_8u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_12x4_float_8u(SUM_WINDOWS_FORMAL_ARGS);

void load_4x4_windows_16u(LOAD_4x4_WINDOWS_FORMAL_ARGS);
void sum_windows_int_16u(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_float_16u(SUM_WINDOWS_FORMAL_ARGS);

/*
    declare reference functions
*/

void load_4x4_windows_8u_c(LOAD_4x4_WINDOWS_FORMAL_ARGS);
void sum_windows_int_8u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x4_int_8u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_12x4_int_8u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_float_8u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x4_float_8u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_12x4_float_8u_c(SUM_WINDOWS_FORMAL_ARGS);

void load_4x4_windows_16u_c(LOAD_4x4_WINDOWS_FORMAL_ARGS);
void sum_windows_int_16u_c(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_float_16u_c(SUM_WINDOWS_FORMAL_ARGS);

/*
    declare optimized functions
*/

#if defined(_X86) || defined(_X64)

void load_4x4_windows_8u_ssse3(LOAD_4x4_WINDOWS_FORMAL_ARGS);
void load_4x4_windows_8u_avx2(LOAD_4x4_WINDOWS_FORMAL_ARGS);

void load_4x4_windows_16u_ssse3(LOAD_4x4_WINDOWS_FORMAL_ARGS);
void load_4x4_windows_16u_avx2(LOAD_4x4_WINDOWS_FORMAL_ARGS);

void sum_windows_8x4_int_8u_sse41(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_12x4_int_8u_sse41(SUM_WINDOWS_FORMAL_ARGS);

void sum_windows_8x4_float_8u_ssse3(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_8x4_float_8u_avx2(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_12x4_float_8u_ssse3(SUM_WINDOWS_FORMAL_ARGS);
void sum_windows_12x4_float_8u_avx2(SUM_WINDOWS_FORMAL_ARGS);

#elif defined(_ARM) || defined(_ARM64)

void load_4x4_windows_8u_neon(LOAD_4x4_WINDOWS_FORMAL_ARGS);
void load_4x4_windows_16u_neon(LOAD_4x4_WINDOWS_FORMAL_ARGS);

void sum_windows_8x4_int_8u_neon(SUM_WINDOWS_FORMAL_ARGS);

void sum_windows_8x4_float_8u_neon(SUM_WINDOWS_FORMAL_ARGS);

#endif /* defined(_X86) || defined(_X64) */

#if defined(__cplusplus)
} // extern "C"
#endif /* defined(__cplusplus) */

#endif /* !defined(__SSIM_SSIM_INTERNAL_H) */
