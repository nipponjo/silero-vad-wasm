#ifndef SILERO_VAD_SIMD_H
#define SILERO_VAD_SIMD_H

#include <math.h>

#if defined(SILERO_VAD_ENABLE_AVX2) && defined(__AVX2__)
#define SILERO_VAD_SIMD_HAS_AVX2 1
#else
#define SILERO_VAD_SIMD_HAS_AVX2 0
#endif

#if defined(SILERO_VAD_ENABLE_SSE) && !defined(_M_ARM64EC) && \
    (defined(__x86_64__) || defined(__SSE__) || defined(_M_X64) || \
     (defined(_M_IX86_FP) && (_M_IX86_FP >= 1)))
#define SILERO_VAD_SIMD_HAS_SSE 1
#else
#define SILERO_VAD_SIMD_HAS_SSE 0
#endif

#if defined(SILERO_VAD_ENABLE_WASM_SIMD) && defined(__wasm_simd128__)
#define SILERO_VAD_SIMD_HAS_WASM_SIMD 1
#else
#define SILERO_VAD_SIMD_HAS_WASM_SIMD 0
#endif

#if SILERO_VAD_SIMD_HAS_AVX2
#include <immintrin.h>
#elif defined(SILERO_VAD_ENABLE_NEON) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#elif SILERO_VAD_SIMD_HAS_WASM_SIMD
#include <wasm_simd128.h>
#elif SILERO_VAD_SIMD_HAS_SSE
#include <xmmintrin.h>
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#define SILERO_VAD_SIMD_BACKEND_SCALAR 0
#define SILERO_VAD_SIMD_BACKEND_SSE 1
#define SILERO_VAD_SIMD_BACKEND_AVX2 2
#define SILERO_VAD_SIMD_BACKEND_NEON 3
#define SILERO_VAD_SIMD_BACKEND_WASM_SIMD 4

#if SILERO_VAD_SIMD_HAS_AVX2
#define SILERO_VAD_SIMD_BACKEND SILERO_VAD_SIMD_BACKEND_AVX2
#define SILERO_VAD_SIMD_LANES 8
typedef __m256 silero_vad_simd_f32;
#elif defined(SILERO_VAD_ENABLE_NEON) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#define SILERO_VAD_SIMD_BACKEND SILERO_VAD_SIMD_BACKEND_NEON
#define SILERO_VAD_SIMD_LANES 4
typedef float32x4_t silero_vad_simd_f32;
#elif SILERO_VAD_SIMD_HAS_WASM_SIMD
#define SILERO_VAD_SIMD_BACKEND SILERO_VAD_SIMD_BACKEND_WASM_SIMD
#define SILERO_VAD_SIMD_LANES 4
typedef v128_t silero_vad_simd_f32;
#elif SILERO_VAD_SIMD_HAS_SSE
#define SILERO_VAD_SIMD_BACKEND SILERO_VAD_SIMD_BACKEND_SSE
#define SILERO_VAD_SIMD_LANES 4
typedef __m128 silero_vad_simd_f32;
#else
#define SILERO_VAD_SIMD_BACKEND SILERO_VAD_SIMD_BACKEND_SCALAR
#define SILERO_VAD_SIMD_LANES 1
typedef struct silero_vad_simd_f32_scalar_tag {
  float lane0;
} silero_vad_simd_f32;
#endif

/* Load one SIMD-width vector of floats from an unaligned address. */
static inline silero_vad_simd_f32 silero_vad_simd_loadu_f32(const float *src) {
#if SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_AVX2
  return _mm256_loadu_ps(src);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_NEON
  return vld1q_f32(src);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_WASM_SIMD
  return wasm_v128_load(src);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_SSE
  return _mm_loadu_ps(src);
#else
  silero_vad_simd_f32 v;
  v.lane0 = src[0];
  return v;
#endif
}

/* Store one SIMD-width vector of floats to an unaligned address. */
static inline void silero_vad_simd_storeu_f32(float *dst, silero_vad_simd_f32 v) {
#if SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_AVX2
  _mm256_storeu_ps(dst, v);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_NEON
  vst1q_f32(dst, v);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_WASM_SIMD
  wasm_v128_store(dst, v);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_SSE
  _mm_storeu_ps(dst, v);
#else
  dst[0] = v.lane0;
#endif
}

/* Broadcast one scalar float into all SIMD lanes. */
static inline silero_vad_simd_f32 silero_vad_simd_splat_f32(float x) {
#if SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_AVX2
  return _mm256_set1_ps(x);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_NEON
  return vdupq_n_f32(x);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_WASM_SIMD
  return wasm_f32x4_splat(x);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_SSE
  return _mm_set1_ps(x);
#else
  silero_vad_simd_f32 v;
  v.lane0 = x;
  return v;
#endif
}

/* Return an all-zero SIMD vector. */
static inline silero_vad_simd_f32 silero_vad_simd_zero_f32(void) {
  return silero_vad_simd_splat_f32(0.0f);
}

/* Add two SIMD float vectors lane-wise. */
static inline silero_vad_simd_f32 silero_vad_simd_add_f32(silero_vad_simd_f32 a,
                                                           silero_vad_simd_f32 b) {
#if SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_AVX2
  return _mm256_add_ps(a, b);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_NEON
  return vaddq_f32(a, b);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_WASM_SIMD
  return wasm_f32x4_add(a, b);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_SSE
  return _mm_add_ps(a, b);
#else
  silero_vad_simd_f32 v;
  v.lane0 = a.lane0 + b.lane0;
  return v;
#endif
}

/* Multiply two SIMD float vectors lane-wise. */
static inline silero_vad_simd_f32 silero_vad_simd_mul_f32(silero_vad_simd_f32 a,
                                                           silero_vad_simd_f32 b) {
#if SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_AVX2
  return _mm256_mul_ps(a, b);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_NEON
  return vmulq_f32(a, b);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_WASM_SIMD
  return wasm_f32x4_mul(a, b);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_SSE
  return _mm_mul_ps(a, b);
#else
  silero_vad_simd_f32 v;
  v.lane0 = a.lane0 * b.lane0;
  return v;
#endif
}

/* Compute acc + (a * b) lane-wise. */
static inline silero_vad_simd_f32 silero_vad_simd_madd_f32(silero_vad_simd_f32 acc,
                                                            silero_vad_simd_f32 a,
                                                            silero_vad_simd_f32 b) {
#if SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_NEON
  return vmlaq_f32(acc, a, b);
#else
  return silero_vad_simd_add_f32(acc, silero_vad_simd_mul_f32(a, b));
#endif
}

/* Compute the lane-wise maximum of two SIMD float vectors. */
static inline silero_vad_simd_f32 silero_vad_simd_max_f32(silero_vad_simd_f32 a,
                                                           silero_vad_simd_f32 b) {
#if SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_AVX2
  return _mm256_max_ps(a, b);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_NEON
  return vmaxq_f32(a, b);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_WASM_SIMD
  return wasm_f32x4_max(a, b);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_SSE
  return _mm_max_ps(a, b);
#else
  silero_vad_simd_f32 v;
  v.lane0 = (a.lane0 > b.lane0) ? a.lane0 : b.lane0;
  return v;
#endif
}

/* Compute the lane-wise square root of a SIMD float vector. */
static inline silero_vad_simd_f32 silero_vad_simd_sqrt_f32(silero_vad_simd_f32 a) {
#if SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_AVX2
  return _mm256_sqrt_ps(a);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_NEON
  return vsqrtq_f32(a);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_WASM_SIMD
  return wasm_f32x4_sqrt(a);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_SSE
  return _mm_sqrt_ps(a);
#else
  silero_vad_simd_f32 v;
  v.lane0 = sqrtf(a.lane0);
  return v;
#endif
}

/* Reduce all lanes to one scalar sum. */
static inline float silero_vad_simd_hsum_f32(silero_vad_simd_f32 v) {
#if SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_AVX2
  __m128 low = _mm256_castps256_ps128(v);
  __m128 high = _mm256_extractf128_ps(v, 1);
  __m128 sum = _mm_add_ps(low, high);
  __m128 shuf = _mm_movehl_ps(sum, sum);
  sum = _mm_add_ps(sum, shuf);
  shuf = _mm_shuffle_ps(sum, sum, 0x01);
  sum = _mm_add_ss(sum, shuf);
  return _mm_cvtss_f32(sum);
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_NEON
  #if defined(__aarch64__)
  return vaddvq_f32(v);
  #else
  float tmp[4];
  vst1q_f32(tmp, v);
  return tmp[0] + tmp[1] + tmp[2] + tmp[3];
  #endif
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_WASM_SIMD
  float tmp[4];
  wasm_v128_store(tmp, v);
  return tmp[0] + tmp[1] + tmp[2] + tmp[3];
#elif SILERO_VAD_SIMD_BACKEND == SILERO_VAD_SIMD_BACKEND_SSE
  __m128 shuf = _mm_movehl_ps(v, v);
  __m128 sum = _mm_add_ps(v, shuf);
  shuf = _mm_shuffle_ps(sum, sum, 0x01);
  sum = _mm_add_ss(sum, shuf);
  return _mm_cvtss_f32(sum);
#else
  return v.lane0;
#endif
}

#if defined(__cplusplus)
}
#endif

#endif
