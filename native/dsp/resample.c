#include "resample.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct yl_resample_ctx {
  int orig0, new0;   // original requested (pre-gcd)
  int orig,  new;    // reduced by gcd
  int gcd;

  int lowpass_filter_width;
  float rolloff;
  yl_resampling_method method;
  float beta;

  int width;
  int kernel_len;    // 2*width + orig
  float* kernel;     // shape [new][kernel_len], row-major: kernel[p*kernel_len + k]
};

/* --------- helpers --------- */

static int yl_gcd_int(int a, int b) {
  if (a < 0) a = -a;
  if (b < 0) b = -b;
  while (b != 0) {
    int t = a % b;
    a = b;
    b = t;
  }
  return a == 0 ? 1 : a;
}

static inline float yl_sincf(float x) {
  // sin(x)/x with safe handling near 0
  const float ax = fabsf(x);
  if (ax < 1e-8f) return 1.0f;
  return sinf(x) / x;
}

/* Approximation of modified Bessel function I0 for float (good enough for windowing).
   (Cephes-style approximation, common in DSP codebases.)
*/
static float yl_i0f(float x) {
  float ax = fabsf(x);
  float y;
  if (ax < 3.75f) {
    y = x / 3.75f;
    y *= y;
    return 1.0f + y * (3.5156229f + y * (3.0899424f + y * (1.2067492f +
           y * (0.2659732f + y * (0.0360768f + y * 0.0045813f)))));
  } else {
    y = 3.75f / ax;
    return (expf(ax) / sqrtf(ax)) *
           (0.39894228f + y * (0.01328592f + y * (0.00225319f +
           y * (-0.00157565f + y * (0.00916281f + y * (-0.02057706f +
           y * (0.02635537f + y * (-0.01647633f + y * 0.00392377f))))))));
  }
}

static void yl_build_kernel(yl_resample_ctx* r) {

  const int orig = r->orig;
  const int newf = r->new;

  const float rolloff = r->rolloff;
  const int lpw = r->lowpass_filter_width;

  float base_freq = (float)((orig < newf) ? orig : newf);
  base_freq *= rolloff; 

  // width = ceil(lowpass_filter_width * orig / base_freq) 
  r->width = (int)ceilf((float)lpw * (float)orig / base_freq);

  r->kernel_len = 2 * r->width + orig; // idx = arange(-width, width+orig) => length 2*width+orig 

  const int width = r->width;
  const int K = r->kernel_len;

  free(r->kernel);
  r->kernel = (float*)malloc((size_t)newf * (size_t)K * sizeof(float));
  if (!r->kernel) return;

  const float scale = base_freq / (float)orig; 
  const float lpw_f = (float)lpw;

  const float beta = (r->method == YL_RESAMPLE_SINC_KAISER)
                       ? ((r->beta > 0.0f) ? r->beta : 14.769656459379492f) // default 
                       : 0.0f;
  const float i0_beta = (r->method == YL_RESAMPLE_SINC_KAISER) ? yl_i0f(beta) : 1.0f;

  // idx[k] = (k - width)/orig   where k corresponds to -width .. width+orig-1
  // phase p uses offset = -p/new 
  for (int p = 0; p < newf; p++) {
    const float phase = -(float)p / (float)newf;

    float* row = &r->kernel[(size_t)p * (size_t)K];

    for (int k = 0; k < K; k++) {
      float idx = (float)(k - width) / (float)orig;
      float t = (phase + idx) * base_freq; // t *= base_freq 

      // clamp to [-lpw, lpw] 
      if (t > lpw_f) t = lpw_f;
      else if (t < -lpw_f) t = -lpw_f;

      // window evaluated at t (not a regular grid) 
      float w;
      if (r->method == YL_RESAMPLE_SINC_HANN) {
        // window = cos(t*pi/lowpass_filter_width/2)^2 
        float a = t * (float)M_PI / lpw_f * 0.5f;
        float c = cosf(a);
        w = c * c;
      } else {
        // kaiser: i0(beta*sqrt(1-(t/lpw)^2)) / i0(beta) 
        float u = t / lpw_f;
        float inside = 1.0f - u * u;
        if (inside < 0.0f) inside = 0.0f;
        w = yl_i0f(beta * sqrtf(inside)) / i0_beta;
      }

      // sinc: kernels = sin(pi*t)/(pi*t) with t scaled then multiplied by pi 
      float x = t * (float)M_PI;
      float s = yl_sincf(x);

      row[k] = s * w * scale; // kernels *= window * scale 
    }
  }
}

/* --------- public API --------- */

yl_resample_ctx* yl_resample_create(int orig_freq, int new_freq,
                                    int lowpass_filter_width,
                                    float rolloff,
                                    yl_resampling_method method,
                                    float beta)
{
  if (orig_freq <= 0 || new_freq <= 0) return NULL;
  if (lowpass_filter_width <= 0) return NULL;
  if (rolloff <= 0.0f) return NULL;
  if (method != YL_RESAMPLE_SINC_HANN && method != YL_RESAMPLE_SINC_KAISER) return NULL;

  yl_resample_ctx* r = (yl_resample_ctx*)calloc(1, sizeof(*r));
  if (!r) return NULL;

  r->orig0 = orig_freq;
  r->new0  = new_freq;

  r->gcd = yl_gcd_int(orig_freq, new_freq);

  r->orig = orig_freq / r->gcd;
  r->new  = new_freq  / r->gcd;

  r->lowpass_filter_width = lowpass_filter_width;
  r->rolloff = rolloff;
  r->method = method;
  r->beta = beta;

  yl_build_kernel(r);
  if (!r->kernel) { yl_resample_destroy(r); return NULL; }

  return r;
}

void yl_resample_destroy(yl_resample_ctx* r) {
  if (!r) return;
  free(r->kernel);
  free(r);
}

void yl_resample_reset(yl_resample_ctx* r) {
  (void)r;
  // stateless: nothing to reset in this implementation
}

size_t yl_resample_out_len(int orig_freq, int new_freq, size_t in_len) {
  // ceil(a/b) = (a + b - 1) / b  for positive integers
  // here a = new_freq * in_len
  const size_t a = (size_t)new_freq * in_len;
  const size_t b = (size_t)orig_freq;
  return (a + b - 1) / b;
}

size_t yl_resample_process(const yl_resample_ctx* r,
                           const float* in, size_t in_len,
                           float* out)
{
  if (!r || !in || !out) return 0;

  // target_length uses original (pre-gcd) ratio like torchaudio API does
  const size_t target_len = yl_resample_out_len(r->orig0, r->new0, in_len);

  const int orig = r->orig;
  const int newf = r->new;
  const int width = r->width;
  const int K = r->kernel_len;

  // Pad: (width, width + orig)
  const size_t pad_left = (size_t)width;
  const size_t pad_right = (size_t)(width + orig);

  const size_t xpad_len = in_len + pad_left + pad_right;
  float* xpad = (float*)malloc(xpad_len * sizeof(float));
  if (!xpad) return 0;

  memset(xpad, 0, pad_left * sizeof(float));
  memcpy(xpad + pad_left, in, in_len * sizeof(float));
  memset(xpad + pad_left + in_len, 0, pad_right * sizeof(float));

  // Convolution with stride=orig, producing blocks of 'new' outputs per stride step,
  // then flatten in (step-major, phase-minor) order.
  size_t out_written = 0;

  // step s corresponds to starting input index s*orig into the padded signal
  for (size_t s = 0; out_written < target_len; s++) {
    const size_t base = s * (size_t)orig;
    if (base + (size_t)K > xpad_len) break; // no more full kernel support

    const float* xseg = xpad + base;

    for (int p = 0; p < newf && out_written < target_len; p++) {
      const float* h = &r->kernel[(size_t)p * (size_t)K];

      double acc = 0.0;
      for (int k = 0; k < K; k++) {
        acc += (double)xseg[k] * (double)h[k];
      }
      out[out_written++] = (float)acc;
    }
  }

  free(xpad);
  return out_written;
}