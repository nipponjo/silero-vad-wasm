#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  YL_RESAMPLE_SINC_HANN   = 1,  // "sinc_interp_hann"
  YL_RESAMPLE_SINC_KAISER = 2,  // "sinc_interp_kaiser"
} yl_resampling_method;

typedef struct yl_resample_ctx yl_resample_ctx;

/* Create a torchaudio-style sinc resampler context.
   - orig_freq/new_freq are integers (required for torchaudio quality behavior).
   - lowpass_filter_width: default 6
   - rolloff: default 0.99
   - method: HANN or KAISER
   - beta: kaiser beta (if <=0, uses torchaudio default 14.769656459379492)
*/
yl_resample_ctx* yl_resample_create(
   int orig_freq, 
   int new_freq,
   int lowpass_filter_width,
   float rolloff,
   yl_resampling_method method,
   float beta);

void yl_resample_destroy(yl_resample_ctx* r);

/* Reset internal state (not required for this stateless implementation, but kept for API symmetry). */
void yl_resample_reset(yl_resample_ctx* r);

/* Compute output length exactly like torchaudio: ceil(new_freq * in_len / orig_freq). */
size_t yl_resample_out_len(int orig_freq, int new_freq, size_t in_len);

/* Resample a single 1D waveform.
   - in: float input of length in_len
   - out: float output buffer length >= yl_resample_out_len(orig_freq,new_freq,in_len)
   Returns number of samples written (target_len).
*/
size_t yl_resample_process(const yl_resample_ctx* r,
                           const float* in, 
                           size_t in_len,
                           float* out);

#ifdef __cplusplus
}
#endif