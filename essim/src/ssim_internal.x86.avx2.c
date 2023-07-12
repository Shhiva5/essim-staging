/**
 * Copyright (c) 2021-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD3 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <essim/inc/internal.h>

#if (defined(_X86) || defined(_X64)) && defined(__AVX2__)

#if defined(_WINDOWS)
#include <intrin.h>
#elif defined(_LINUX) || defined(MAC_OS_X)
#include <x86intrin.h>
#endif /* defined(_WINDOWS) */

/* GCC bug fix */
#define _mm_loadu_si32(addr) _mm_cvtsi32_si128(*((uint32_t *)(addr)))

void load_4x4_windows_8u_avx2(LOAD_4x4_WINDOWS_FORMAL_ARGS) {
  enum { PREFETCH = 160, WIN_CHUNK = 8, WIN_SIZE = 4 };

  uint8_t *pDst = pBuf->p;
  const ptrdiff_t dstStride = pBuf->stride;
  SSIM_SRC src = *pSrc;
  const ptrdiff_t srcStep = sizeof(uint8_t) * WIN_SIZE;

  const __m256i zero = _mm256_setzero_si256();

  size_t i = 0;
  for (; i + WIN_CHUNK <= num4x4Windows; i += WIN_CHUNK) {
    __m256i sum = _mm256_setzero_si256();
    __m256i ref_sigma_sqd = _mm256_setzero_si256();
    __m256i cmp_sigma_sqd = _mm256_setzero_si256();
    __m256i sigma_both = _mm256_setzero_si256();

    for (uint32_t y = 0; y < WIN_SIZE; ++y) {
      __m256i r0, r1, r2, r3, r4, r5, r6;

      _mm_prefetch((const char *)src.ref + PREFETCH, _MM_HINT_NTA);
      r0 = _mm256_loadu_si256((const __m256i *)(src.ref));
      r1 = _mm256_loadu_si256((const __m256i *)(src.cmp));

      /* sum ref_sum & cmp_sum */
      r2 = _mm256_unpackhi_epi8(r0, zero);
      r3 = _mm256_unpackhi_epi8(r1, zero);
      r0 = _mm256_unpacklo_epi8(r0, zero);
      r1 = _mm256_unpacklo_epi8(r1, zero);
      r4 = _mm256_sad_epu8(r0, zero);
      r5 = _mm256_sad_epu8(r1, zero);
      r5 = _mm256_slli_epi64(r5, 32);
      r4 = _mm256_or_si256(r4, r5);
      r5 = _mm256_sad_epu8(r2, zero);
      r6 = _mm256_sad_epu8(r3, zero);
      r6 = _mm256_slli_epi64(r6, 32);
      r5 = _mm256_or_si256(r5, r6);
      r4 = _mm256_packs_epi32(r4, r5);
      sum = _mm256_add_epi16(sum, r4);

      _mm_prefetch((const char *)src.cmp + PREFETCH, _MM_HINT_NTA);
      /* sum ref_sigma_sqd & cmp_sigma_sqd & sigma_both */
      r4 = r0;
      r5 = r2;
      r0 = _mm256_madd_epi16(r0, r0);
      r4 = _mm256_madd_epi16(r4, r1);
      r1 = _mm256_madd_epi16(r1, r1);
      r2 = _mm256_madd_epi16(r2, r2);
      r5 = _mm256_madd_epi16(r5, r3);
      r3 = _mm256_madd_epi16(r3, r3);
      r0 = _mm256_hadd_epi32(r0, r2);
      r1 = _mm256_hadd_epi32(r1, r3);
      r4 = _mm256_hadd_epi32(r4, r5);
      ref_sigma_sqd = _mm256_add_epi32(ref_sigma_sqd, r0);
      cmp_sigma_sqd = _mm256_add_epi32(cmp_sigma_sqd, r1);
      sigma_both = _mm256_add_epi32(sigma_both, r4);

      src.ref = AdvancePointer(src.ref, src.refStride);
      src.cmp = AdvancePointer(src.cmp, src.cmpStride);
    }

    _mm256_store_si256((__m256i *)(pDst + 0 * dstStride), sum);
    _mm256_store_si256((__m256i *)(pDst + 1 * dstStride), ref_sigma_sqd);
    _mm256_store_si256((__m256i *)(pDst + 2 * dstStride), cmp_sigma_sqd);
    _mm256_store_si256((__m256i *)(pDst + 3 * dstStride), sigma_both);
    pDst += WIN_CHUNK * sizeof(uint32_t);

    /* advance source pointers */
    src.ref =
        AdvancePointer(src.ref, WIN_CHUNK * srcStep - WIN_SIZE * src.refStride);
    src.cmp =
        AdvancePointer(src.cmp, WIN_CHUNK * srcStep - WIN_SIZE * src.cmpStride);
  }
  _mm256_zeroupper();

  if (i < num4x4Windows) {
    SSIM_4X4_WINDOW_BUFFER buf = {pDst, dstStride};

    load_4x4_windows_8u_ssse3(&buf, num4x4Windows - i, &src);
  }

} /* void load_4x4_windows_8u_avx2(LOAD_4x4_WINDOWS_FORMAL_ARGS) */

static const uint8_t lower_dword_mask[16] = {255, 255, 255, 255, 0, 0, 0, 0,
                                             255, 255, 255, 255, 0, 0, 0, 0};

enum { SWAP_MIDDLE_DWORD = 0xd8 };

void load_4x4_windows_16u_avx2(LOAD_4x4_WINDOWS_FORMAL_ARGS) {
  enum { PREFETCH = 160, WIN_CHUNK = 4, WIN_SIZE = 4 };

  uint8_t *pDst = pBuf->p;
  const ptrdiff_t dstStride = pBuf->stride;
  SSIM_SRC src = *pSrc;
  const ptrdiff_t srcStep = sizeof(uint16_t) * WIN_SIZE;

  const __m256i mask = _mm256_broadcastsi128_si256(
      _mm_loadu_si128((const __m128i *)lower_dword_mask));
  const __m256i ones = _mm256_set1_epi16(1);

  size_t i = 0;
  for (; i + WIN_CHUNK <= num4x4Windows; i += WIN_CHUNK) {
    __m256i sum = _mm256_setzero_si256();
    __m256i ref_sigma_sqd = _mm256_setzero_si256();
    __m256i cmp_sigma_sqd = _mm256_setzero_si256();
    __m256i sigma_both = _mm256_setzero_si256();

    for (uint32_t y = 0; y < WIN_SIZE; ++y) {
      __m256i r0, r1, r2, r3, r4, r5;

      _mm_prefetch((const char *)src.ref + PREFETCH, _MM_HINT_NTA);
      r0 = _mm256_loadu_si256((const __m256i *)(src.ref));
      r2 = _mm256_loadu_si256((const __m256i *)(src.cmp));

      /* sum ref_sum & cmp_sum */
      r1 = _mm256_madd_epi16(r0, ones);
      r3 = _mm256_madd_epi16(r2, ones);
      r1 = _mm256_hadd_epi32(r1, r3);
      r1 = _mm256_shuffle_epi32(r1, SWAP_MIDDLE_DWORD);
      sum = _mm256_add_epi32(sum, r1);

      _mm_prefetch((const char *)src.cmp + PREFETCH, _MM_HINT_NTA);
      /* sum ref_sigma_sqd & cmp_sigma_sqd & sigma_both */
      r1 = r0;
      r0 = _mm256_madd_epi16(r0, r0);
      r1 = _mm256_madd_epi16(r1, r2);
      r2 = _mm256_madd_epi16(r2, r2);
      r3 = _mm256_srli_epi64(r0, 32);
      r0 = _mm256_and_si256(r0, mask);
      r0 = _mm256_add_epi64(r0, r3);
      r4 = _mm256_srli_epi64(r1, 32);
      r1 = _mm256_and_si256(r1, mask);
      r1 = _mm256_add_epi64(r1, r4);
      r5 = _mm256_srli_epi64(r2, 32);
      r2 = _mm256_and_si256(r2, mask);
      r2 = _mm256_add_epi64(r2, r5);
      ref_sigma_sqd = _mm256_add_epi64(ref_sigma_sqd, r0);
      cmp_sigma_sqd = _mm256_add_epi64(cmp_sigma_sqd, r2);
      sigma_both = _mm256_add_epi64(sigma_both, r1);

      src.ref = AdvancePointer(src.ref, src.refStride);
      src.cmp = AdvancePointer(src.cmp, src.cmpStride);
    }

    _mm256_store_si256((__m256i *)(pDst + 0 * dstStride), sum);
    _mm256_store_si256((__m256i *)(pDst + 1 * dstStride), ref_sigma_sqd);
    _mm256_store_si256((__m256i *)(pDst + 2 * dstStride), cmp_sigma_sqd);
    _mm256_store_si256((__m256i *)(pDst + 3 * dstStride), sigma_both);
    pDst = AdvancePointer(pDst, WIN_CHUNK * sizeof(uint64_t));

    /* advance source pointers */
    src.ref =
        AdvancePointer(src.ref, WIN_CHUNK * srcStep - WIN_SIZE * src.refStride);
    src.cmp =
        AdvancePointer(src.cmp, WIN_CHUNK * srcStep - WIN_SIZE * src.cmpStride);
  }
  _mm256_zeroupper();

  if (i < num4x4Windows) {
    SSIM_4X4_WINDOW_BUFFER buf = {pDst, dstStride};

    load_4x4_windows_16u_ssse3(&buf, num4x4Windows - i, &src);
  }
} /* void load_4x4_windows_16u_avx2(LOAD_4x4_WINDOWS_FORMAL_ARGS) */

static const int8_t sum_8x8_shuffle[16] = {0, 1, 4, 5, 2, 3, 6,  7,
                                           4, 5, 8, 9, 6, 7, 10, 11};
static const int8_t sum_12x12_shuffle[16] = {0, 1, 4, 5, 8,  9,  -1, -1,
                                             2, 3, 6, 7, 10, 11, -1, -1};

#define ASM_LOAD_8X8_WINDOW_16_FLOAT_VALUES_AVX2(value0, value1, idx)          \
  {                                                                            \
    __m128i _r;                                                                \
    __m256i _r0, _r1;                                                          \
    _r0 = _mm256_load_si256((const __m256i *)(pSrc + (idx)*srcStride));        \
    _r = _mm_loadu_si32(pSrc + (idx)*srcStride + 32);                          \
    _r0 = _mm256_add_epi16(                                                    \
        _r0,                                                                   \
        _mm256_load_si256((const __m256i *)(pSrcNext + (idx)*srcStride)));     \
    _r = _mm_add_epi16(_r, _mm_loadu_si32(pSrcNext + (idx)*srcStride + 32));   \
    _r1 = _mm256_permute2x128_si256(_r0, _mm256_castsi128_si256(_r), 0x21);    \
    _r1 = _mm256_alignr_epi8(_r1, _r0, 8);                                     \
    _r0 = _mm256_shuffle_epi8(_r0, c_sum_shuffle_pattern);                     \
    _r1 = _mm256_shuffle_epi8(_r1, c_sum_shuffle_pattern);                     \
    _r0 = _mm256_hadd_epi16(_r0, _r1);                                         \
    _r1 = _mm256_srli_epi32(_r0, 16);                                          \
    _r0 = _mm256_srli_epi32(_mm256_slli_epi32(_r0, 16), 16);                   \
    value0 = _mm256_cvtepi32_ps(_r0);                                          \
    value1 = _mm256_cvtepi32_ps(_r1);                                          \
  }

#define ASM_LOAD_8X8_WINDOW_8_FLOAT_VALUES_AVX2(value, idx)                    \
  {                                                                            \
    __m128i _r;                                                                \
    __m256i _r0, _r1;                                                          \
    _r0 = _mm256_load_si256((const __m256i *)(pSrc + (idx)*srcStride));        \
    _r = _mm_loadu_si32(pSrc + (idx)*srcStride + 32);                          \
    _r0 = _mm256_add_epi32(                                                    \
        _r0,                                                                   \
        _mm256_load_si256((const __m256i *)(pSrcNext + (idx)*srcStride)));     \
    _r = _mm_add_epi32(_r, _mm_loadu_si32(pSrcNext + (idx)*srcStride + 32));   \
    _r1 = _mm256_permute2x128_si256(_r0, _mm256_castsi128_si256(_r), 0x21);    \
    _r1 = _mm256_alignr_epi8(_r1, _r0, 8);                                     \
    _r0 = _mm256_shuffle_epi32(_r0, 0x94);                                     \
    _r1 = _mm256_shuffle_epi32(_r1, 0x94);                                     \
    _r0 = _mm256_hadd_epi32(_r0, _r1);                                         \
    value = _mm256_mul_ps(_mm256_cvtepi32_ps(_r0), invWindowSize_sqd);         \
  }

#define ASM_CALC_8_FLOAT_SSIM_AVX2()                                           \
  {                                                                            \
    __m256 a, b, c, d, ssim_val;                                               \
    /* STEP 2. adjust values */                                                \
    __m256 both_sum_mul =                                                      \
        _mm256_mul_ps(_mm256_mul_ps(ref_sum, cmp_sum), invWindowSize_qd);      \
    __m256 ref_sum_sqd =                                                       \
        _mm256_mul_ps(_mm256_mul_ps(ref_sum, ref_sum), invWindowSize_qd);      \
    __m256 cmp_sum_sqd =                                                       \
        _mm256_mul_ps(_mm256_mul_ps(cmp_sum, cmp_sum), invWindowSize_qd);      \
    ref_sigma_sqd = _mm256_sub_ps(ref_sigma_sqd, ref_sum_sqd);                 \
    cmp_sigma_sqd = _mm256_sub_ps(cmp_sigma_sqd, cmp_sum_sqd);                 \
    sigma_both = _mm256_sub_ps(sigma_both, both_sum_mul);                      \
    /* STEP 3. process numbers, do scale */                                    \
    a = _mm256_add_ps(_mm256_add_ps(both_sum_mul, both_sum_mul), C1);          \
    b = _mm256_add_ps(sigma_both, halfC2);                                     \
    c = _mm256_add_ps(_mm256_add_ps(ref_sum_sqd, cmp_sum_sqd), C1);            \
    d = _mm256_add_ps(_mm256_add_ps(ref_sigma_sqd, cmp_sigma_sqd), C2);        \
    ssim_val = _mm256_mul_ps(a, b);                                            \
    ssim_val = _mm256_add_ps(ssim_val, ssim_val);                              \
    ssim_val = _mm256_div_ps(ssim_val, _mm256_mul_ps(c, d));                   \
    ssim_sum = _mm256_add_ps(ssim_sum, ssim_val);                              \
    ssim_val = _mm256_mul_ps(ssim_val, ssim_val);                              \
    ssim_mink_sum = _mm256_add_ps(ssim_mink_sum, ssim_val);                    \
  }

void sum_windows_8x4_float_8u_avx2(SUM_WINDOWS_FORMAL_ARGS) {
  enum { WIN_CHUNK = 8, WIN_SIZE = 8 };

  const __m256i c_sum_shuffle_pattern = _mm256_broadcastsi128_si256(
      _mm_loadu_si128((const __m128i *)sum_8x8_shuffle));
  const __m256 invWindowSize_sqd =
      _mm256_set1_ps(1.0f / (float)(windowSize * windowSize));
  const __m256 invWindowSize_qd =
      _mm256_mul_ps(invWindowSize_sqd, invWindowSize_sqd);
  const float fC1 = get_ssim_float_constant(1, bitDepthMinus8);
  const float fC2 = get_ssim_float_constant(2, bitDepthMinus8);
  const __m256 C1 = _mm256_set1_ps(fC1);
  const __m256 C2 = _mm256_set1_ps(fC2);
  const __m256 halfC2 = _mm256_set1_ps(fC2 / 2.0f);

  __m256 ssim_mink_sum = _mm256_setzero_ps();
  __m256 ssim_sum = _mm256_setzero_ps();
  const uint8_t *pSrc = pBuf->p;
  const ptrdiff_t srcStride = pBuf->stride;

  size_t i = 0;
  for (; i + WIN_CHUNK <= numWindows; i += WIN_CHUNK) {
    const uint8_t *pSrcNext = pSrc + 4 * srcStride;

    __m256 ref_sum, cmp_sum;
    __m256 ref_sigma_sqd;
    __m256 cmp_sigma_sqd;
    __m256 sigma_both;

    ASM_LOAD_8X8_WINDOW_16_FLOAT_VALUES_AVX2(ref_sum, cmp_sum, 0);
    ASM_LOAD_8X8_WINDOW_8_FLOAT_VALUES_AVX2(ref_sigma_sqd, 1);
    ASM_LOAD_8X8_WINDOW_8_FLOAT_VALUES_AVX2(cmp_sigma_sqd, 2);
    ASM_LOAD_8X8_WINDOW_8_FLOAT_VALUES_AVX2(sigma_both, 3);
    pSrc += sizeof(uint32_t) * WIN_CHUNK;

    ASM_CALC_8_FLOAT_SSIM_AVX2();
  }

  ssim_sum = _mm256_hadd_ps(ssim_sum, ssim_mink_sum);
  ssim_sum = _mm256_hadd_ps(ssim_sum, ssim_sum);
  ssim_sum =
      _mm256_add_ps(ssim_sum, _mm256_permute2f128_ps(ssim_sum, ssim_sum, 1));

  res->ssim_sum_f += _mm256_cvtss_f32(ssim_sum);
  ssim_sum = _mm256_shuffle_ps(ssim_sum, ssim_sum, 0x39);
  res->ssim_mink_sum_f += _mm256_cvtss_f32(ssim_sum);
  res->numWindows += i;

  _mm256_zeroupper();

  if (i < numWindows) {
    SSIM_4X4_WINDOW_BUFFER buf = {(uint8_t *)pSrc, srcStride};
#if UPDATED_INTEGER_IMPLEMENTATION
    sum_windows_8x4_float_8u_ssse3(res, &buf, numWindows - i, windowSize,
                                   windowStride, bitDepthMinus8,NULL,0,0);
#elif !UPDATED_INTEGER_IMPLEMENTATION
    sum_windows_8x4_float_8u_ssse3(res, &buf, numWindows - i, windowSize,
                                   windowStride, bitDepthMinus8);
#endif
  }

} /* void sum_windows_8x4_float_8u_avx2(SUM_WINDOWS_FORMAL_ARGS) */

#define ASM_LOAD_12X12_WINDOW_16_FLOAT_VALUES_AVX2(value0, value1, idx)        \
  {                                                                            \
    __m256i _r0, _r1, _r2, _r3;                                                \
    _r0 = _mm256_load_si256((const __m256i *)(pSrc + (idx)*srcStride));        \
    _r3 = _mm256_loadu_si256((const __m256i *)(pSrc + (idx)*srcStride + 8));   \
    _r0 = _mm256_add_epi16(                                                    \
        _r0,                                                                   \
        _mm256_load_si256((const __m256i *)(pSrcNext + (idx)*srcStride)));     \
    _r3 = _mm256_add_epi16(                                                    \
        _r3, _mm256_loadu_si256(                                               \
                 (const __m256i *)(pSrcNext + (idx)*srcStride + 8)));          \
    _r0 = _mm256_add_epi16(                                                    \
        _r0,                                                                   \
        _mm256_load_si256((const __m256i *)(pSrcNext2 + (idx)*srcStride)));    \
    _r3 = _mm256_add_epi16(                                                    \
        _r3, _mm256_loadu_si256(                                               \
                 (const __m256i *)(pSrcNext2 + (idx)*srcStride + 8)));         \
    _r3 = _mm256_srli_si256(_r3, 8);                                           \
    _r1 = _mm256_srli_si256(_r0, 4);                                           \
    _r2 = _mm256_alignr_epi8(_r3, _r0, 8);                                     \
    _r3 = _mm256_alignr_epi8(_r3, _r0, 12);                                    \
    _r0 = _mm256_shuffle_epi8(_r0, c_sum_shuffle_pattern);                     \
    _r1 = _mm256_shuffle_epi8(_r1, c_sum_shuffle_pattern);                     \
    _r2 = _mm256_shuffle_epi8(_r2, c_sum_shuffle_pattern);                     \
    _r3 = _mm256_shuffle_epi8(_r3, c_sum_shuffle_pattern);                     \
    _r0 = _mm256_hadd_epi16(_r0, _r1);                                         \
    _r2 = _mm256_hadd_epi16(_r2, _r3);                                         \
    _r0 = _mm256_hadd_epi16(_r0, _r2);                                         \
    _r1 = _mm256_srli_epi32(_r0, 16);                                          \
    _r0 = _mm256_srli_epi32(_mm256_slli_epi32(_r0, 16), 16);                   \
    value0 = _mm256_cvtepi32_ps(_r0);                                          \
    value1 = _mm256_cvtepi32_ps(_r1);                                          \
  }

#define ASM_LOAD_12X12_WINDOW_8_FLOAT_VALUES_AVX2(value, idx)                  \
  {                                                                            \
    __m256i _r0, _r1, _r2, _r3;                                                \
    _r0 = _mm256_load_si256((const __m256i *)(pSrc + (idx)*srcStride));        \
    _r3 = _mm256_loadu_si256((const __m256i *)(pSrc + (idx)*srcStride + 8));   \
    _r0 = _mm256_add_epi32(                                                    \
        _r0,                                                                   \
        _mm256_load_si256((const __m256i *)(pSrcNext + (idx)*srcStride)));     \
    _r3 = _mm256_add_epi32(                                                    \
        _r3, _mm256_loadu_si256(                                               \
                 (const __m256i *)(pSrcNext + (idx)*srcStride + 8)));          \
    _r0 = _mm256_add_epi32(                                                    \
        _r0,                                                                   \
        _mm256_load_si256((const __m256i *)(pSrcNext2 + (idx)*srcStride)));    \
    _r3 = _mm256_add_epi32(                                                    \
        _r3, _mm256_loadu_si256(                                               \
                 (const __m256i *)(pSrcNext2 + (idx)*srcStride + 8)));         \
    _r3 = _mm256_srli_si256(_r3, 8);                                           \
    _r1 = _mm256_srli_si256(_r0, 4);                                           \
    _r2 = _mm256_alignr_epi8(_r3, _r0, 8);                                     \
    _r2 = _mm256_slli_si256(_r2, 4);                                           \
    _r3 = _mm256_alignr_epi8(_r3, _r0, 12);                                    \
    _r0 = _mm256_slli_si256(_r0, 4);                                           \
    _r0 = _mm256_hadd_epi32(_r0, _r1);                                         \
    _r2 = _mm256_hadd_epi32(_r2, _r3);                                         \
    _r0 = _mm256_hadd_epi32(_r0, _r2);                                         \
    value = _mm256_mul_ps(_mm256_cvtepi32_ps(_r0), invWindowSize_sqd);         \
  }

void sum_windows_12x4_float_8u_avx2(SUM_WINDOWS_FORMAL_ARGS) {
  enum { WIN_CHUNK = 8, WIN_SIZE = 12 };

  const __m256i c_sum_shuffle_pattern = _mm256_broadcastsi128_si256(
      _mm_loadu_si128((const __m128i *)sum_12x12_shuffle));
  const __m256 invWindowSize_sqd =
      _mm256_set1_ps(1.0f / (float)(windowSize * windowSize));
  const __m256 invWindowSize_qd =
      _mm256_mul_ps(invWindowSize_sqd, invWindowSize_sqd);
  const float fC1 = get_ssim_float_constant(1, bitDepthMinus8);
  const float fC2 = get_ssim_float_constant(2, bitDepthMinus8);
  const __m256 C1 = _mm256_set1_ps(fC1);
  const __m256 C2 = _mm256_set1_ps(fC2);
  const __m256 halfC2 = _mm256_set1_ps(fC2 / 2.0f);

  __m256 ssim_mink_sum = _mm256_setzero_ps();
  __m256 ssim_sum = _mm256_setzero_ps();
  const uint8_t *pSrc = pBuf->p;
  const ptrdiff_t srcStride = pBuf->stride;

  size_t i = 0;
  for (; i + WIN_CHUNK <= numWindows; i += WIN_CHUNK) {
    const uint8_t *pSrcNext = pSrc + 4 * srcStride;
    const uint8_t *pSrcNext2 = pSrc + 8 * srcStride;

    __m256 ref_sum, cmp_sum;
    __m256 ref_sigma_sqd;
    __m256 cmp_sigma_sqd;
    __m256 sigma_both;

    ASM_LOAD_12X12_WINDOW_16_FLOAT_VALUES_AVX2(ref_sum, cmp_sum, 0);
    ASM_LOAD_12X12_WINDOW_8_FLOAT_VALUES_AVX2(ref_sigma_sqd, 1);
    ASM_LOAD_12X12_WINDOW_8_FLOAT_VALUES_AVX2(cmp_sigma_sqd, 2);
    ASM_LOAD_12X12_WINDOW_8_FLOAT_VALUES_AVX2(sigma_both, 3);
    pSrc += sizeof(uint32_t) * WIN_CHUNK;

    ASM_CALC_8_FLOAT_SSIM_AVX2();
  }

  ssim_sum = _mm256_hadd_ps(ssim_sum, ssim_mink_sum);
  ssim_sum = _mm256_hadd_ps(ssim_sum, ssim_sum);
  ssim_sum =
      _mm256_add_ps(ssim_sum, _mm256_permute2f128_ps(ssim_sum, ssim_sum, 1));

  res->ssim_sum_f += _mm256_cvtss_f32(ssim_sum);
  ssim_sum = _mm256_shuffle_ps(ssim_sum, ssim_sum, 0x39);
  res->ssim_mink_sum_f +=
      _mm256_cvtss_f32(ssim_sum); // TODO replace with (1 - ssim) ** 4
  res->numWindows += i;

  _mm256_zeroupper();

  if (i < numWindows) {
    SSIM_4X4_WINDOW_BUFFER buf = {(uint8_t *)pSrc, srcStride};
#if UPDATED_INTEGER_IMPLEMENTATION
    sum_windows_12x4_float_8u_ssse3(res, &buf, numWindows - i, windowSize,
                                    windowStride, bitDepthMinus8, NULL, 0, 0);
#elif !UPDATED_INTEGER_IMPLEMENTATION
    sum_windows_12x4_float_8u_ssse3(res, &buf, numWindows - i, windowSize,
                                    windowStride, bitDepthMinus8);
#endif
  }

} /* void sum_windows_12x4_float_8u_avx2(SUM_WINDOWS_FORMAL_ARGS) */

#endif /* defined(_X86) || defined(_X64) */
