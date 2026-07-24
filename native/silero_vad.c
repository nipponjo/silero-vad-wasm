#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200112L
#endif

#include "silero_vad.h"
#include "silero_vad_simd.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__MINGW32__) || defined(__MINGW64__)
#define silero_vad_win_aligned_malloc __mingw_aligned_malloc
#define silero_vad_win_aligned_free __mingw_aligned_free
#else
#define silero_vad_win_aligned_malloc _aligned_malloc
#define silero_vad_win_aligned_free _aligned_free
#endif

static int silero_vad_has_null_weights(const SileroVadWeights *weights) {
  return weights == NULL ||
         weights->conv1_weight == NULL ||
         weights->conv1_bias == NULL ||
         weights->conv2_weight == NULL ||
         weights->conv2_bias == NULL ||
         weights->conv3_weight == NULL ||
         weights->conv3_bias == NULL ||
         weights->conv4_weight == NULL ||
         weights->conv4_bias == NULL ||
         weights->lstm_weight_ih == NULL ||
         weights->lstm_weight_hh == NULL ||
         weights->lstm_bias_ih == NULL ||
         weights->lstm_bias_hh == NULL ||
         weights->final_conv_weight == NULL ||
         weights->final_conv_bias == NULL;
}

static int silero_vad_reflect_index(int index, size_t length) {
  int last;

  if (length == 0 || length == 1) {
    return 0;
  }

  last = (int)length - 1;
  while (index < 0 || index > last) {
    if (index < 0) {
      index = -index;
    } else {
      index = 2 * last - index;
    }
  }

  return index;
}

static void silero_vad_reflect_pad_right(const float *input,
                                         size_t input_frames,
                                         size_t right_pad,
                                         float *output) {
  size_t i;

  for (i = 0; i < input_frames; ++i) {
    output[i] = input[i];
  }

  for (i = 0; i < right_pad; ++i) {
    output[input_frames + i] = input[silero_vad_reflect_index((int)(input_frames + i), input_frames)];
  }
}

static void *silero_vad_aligned_calloc(size_t count, size_t size) {
  const size_t alignment = 32;
  const size_t total = count * size;
  void *ptr;

  if (count == 0 || size == 0) {
    return NULL;
  }

#if defined(_WIN32)
  ptr = silero_vad_win_aligned_malloc(total, alignment);
  if (ptr != NULL) {
    memset(ptr, 0, total);
  }
  return ptr;
#else
  ptr = NULL;
  if (posix_memalign(&ptr, alignment, total) != 0) {
    return NULL;
  }
  memset(ptr, 0, total);
  return ptr;
#endif
}

static void silero_vad_aligned_free(void *ptr) {
  if (ptr == NULL) {
    return;
  }

#if defined(_WIN32)
  silero_vad_win_aligned_free(ptr);
#else
  free(ptr);
#endif
}

static float silero_vad_dot_f32(const float *a, const float *b, size_t count) {
  size_t i = 0;
  float sum = 0.0f;

#if SILERO_VAD_SIMD_BACKEND != SILERO_VAD_SIMD_BACKEND_SCALAR
  silero_vad_simd_f32 acc = silero_vad_simd_zero_f32();
  for (; i + SILERO_VAD_SIMD_LANES <= count; i += SILERO_VAD_SIMD_LANES) {
    const silero_vad_simd_f32 va = silero_vad_simd_loadu_f32(a + i);
    const silero_vad_simd_f32 vb = silero_vad_simd_loadu_f32(b + i);
    acc = silero_vad_simd_madd_f32(acc, va, vb);
  }
  sum = silero_vad_simd_hsum_f32(acc);
#endif

  for (; i < count; ++i) {
    sum += a[i] * b[i];
  }

  return sum;
}

static void silero_vad_pack_k3_weights(SileroVadConv1d *conv) {
  const size_t block_width = 8;
  const size_t padded_output_channels =
    ((conv->output_channels + block_width - 1) / block_width) * block_width;
  const size_t packed_count = padded_output_channels * conv->input_channels;
  size_t oc;
  size_t ic;

  if (conv->kernel_size != 3 || conv->padding != 1) {
    return;
  }

  conv->packed_weight_t0 = (float *)silero_vad_aligned_calloc(packed_count, sizeof(float));
  conv->packed_weight_t1 = (float *)silero_vad_aligned_calloc(packed_count, sizeof(float));
  conv->packed_weight_t2 = (float *)silero_vad_aligned_calloc(packed_count, sizeof(float));
  if (conv->packed_weight_t0 == NULL || conv->packed_weight_t1 == NULL || conv->packed_weight_t2 == NULL) {
    silero_vad_aligned_free(conv->packed_weight_t0);
    silero_vad_aligned_free(conv->packed_weight_t1);
    silero_vad_aligned_free(conv->packed_weight_t2);
    conv->packed_weight_t0 = NULL;
    conv->packed_weight_t1 = NULL;
    conv->packed_weight_t2 = NULL;
    return;
  }

  for (oc = 0; oc < padded_output_channels; ++oc) {
    const size_t block = oc / block_width;
    const size_t lane = oc % block_width;
    for (ic = 0; ic < conv->input_channels; ++ic) {
      const size_t dst = ((block * conv->input_channels) + ic) * block_width + lane;
      if (oc < conv->output_channels) {
        const size_t src = ((oc * conv->input_channels) + ic) * 3;
        conv->packed_weight_t0[dst] = conv->weight[src];
        conv->packed_weight_t1[dst] = conv->weight[src + 1];
        conv->packed_weight_t2[dst] = conv->weight[src + 2];
      } else {
        conv->packed_weight_t0[dst] = 0.0f;
        conv->packed_weight_t1[dst] = 0.0f;
        conv->packed_weight_t2[dst] = 0.0f;
      }
    }
  }
}

static void silero_vad_pack_lstm_weights(SileroVadLstmCell *cell) {
  const size_t rows = cell->hidden_size * 4;
  const size_t cols_ih = cell->input_size;
  const size_t cols_hh = cell->hidden_size;
  const size_t block_width = 8;
  const size_t blocks = rows / block_width;
  size_t block;
  size_t col;
  size_t lane;

  if (rows == 0 || (rows % block_width) != 0) {
    return;
  }

  cell->packed_weight_ih = (float *)silero_vad_aligned_calloc(rows * cols_ih, sizeof(float));
  cell->packed_weight_hh = (float *)silero_vad_aligned_calloc(rows * cols_hh, sizeof(float));
  if (cell->packed_weight_ih == NULL || cell->packed_weight_hh == NULL) {
    silero_vad_aligned_free(cell->packed_weight_ih);
    silero_vad_aligned_free(cell->packed_weight_hh);
    cell->packed_weight_ih = NULL;
    cell->packed_weight_hh = NULL;
    return;
  }

  for (block = 0; block < blocks; ++block) {
    const size_t row_base = block * block_width;
    for (col = 0; col < cols_ih; ++col) {
      const size_t dst_base = ((block * cols_ih) + col) * block_width;
      for (lane = 0; lane < block_width; ++lane) {
        cell->packed_weight_ih[dst_base + lane] =
          cell->weight_ih[((row_base + lane) * cols_ih) + col];
      }
    }
    for (col = 0; col < cols_hh; ++col) {
      const size_t dst_base = ((block * cols_hh) + col) * block_width;
      for (lane = 0; lane < block_width; ++lane) {
        cell->packed_weight_hh[dst_base + lane] =
          cell->weight_hh[((row_base + lane) * cols_hh) + col];
      }
    }
  }
}

static void silero_vad_relu(float *data, size_t count) {
  size_t i = 0;

#if SILERO_VAD_SIMD_BACKEND != SILERO_VAD_SIMD_BACKEND_SCALAR
  const silero_vad_simd_f32 zero = silero_vad_simd_zero_f32();
  for (; i + SILERO_VAD_SIMD_LANES <= count; i += SILERO_VAD_SIMD_LANES) {
    silero_vad_simd_f32 v = silero_vad_simd_loadu_f32(data + i);
    v = silero_vad_simd_max_f32(v, zero);
    silero_vad_simd_storeu_f32(data + i, v);
  }
#endif

  for (; i < count; ++i) {
    if (data[i] < 0.0f) {
      data[i] = 0.0f;
    }
  }
}

static float silero_vad_sigmoid(float x) {
  if (x >= 0.0f) {
    float z = expf(-x);
    return 1.0f / (1.0f + z);
  }

  {
    float z = expf(x);
    return z / (1.0f + z);
  }
}

static unsigned short silero_vad_bit_reverse_8(unsigned short x) {
  x = (unsigned short)(((x & 0x55u) << 1) | ((x >> 1) & 0x55u));
  x = (unsigned short)(((x & 0x33u) << 2) | ((x >> 2) & 0x33u));
  x = (unsigned short)(((x & 0x0Fu) << 4) | ((x >> 4) & 0x0Fu));
  return x;
}

static void silero_vad_init_fft_tables(SileroVadModel *model) {
  size_t i;

  for (i = 0; i < SILERO_VAD_STFT_FFT_SIZE; ++i) {
    const float angle = (float)((2.0 * 3.14159265358979323846 * (double)i) / (double)SILERO_VAD_STFT_FFT_SIZE);
    model->fft_twiddle_cos[i] = cosf(angle);
    model->fft_twiddle_sin[i] = -sinf(angle);
    model->fft_bitrev[i] = silero_vad_bit_reverse_8((unsigned short)i);
  }
}

static void silero_vad_fft256(float *real,
                              float *imag,
                              const float *twiddle_cos,
                              const float *twiddle_sin,
                              const unsigned short *bitrev) {
  float tmp_real[SILERO_VAD_STFT_FFT_SIZE];
  float tmp_imag[SILERO_VAD_STFT_FFT_SIZE];
  size_t i;

  for (i = 0; i < SILERO_VAD_STFT_FFT_SIZE; ++i) {
    const size_t j = bitrev[i];
    tmp_real[i] = real[j];
    tmp_imag[i] = imag[j];
  }

  memcpy(real, tmp_real, sizeof(tmp_real));
  memcpy(imag, tmp_imag, sizeof(tmp_imag));

  {
    size_t start;
    for (start = 0; start < SILERO_VAD_STFT_FFT_SIZE; start += 2) {
      const float r0 = real[start];
      const float i0 = imag[start];
      const float r1 = real[start + 1];
      const float i1 = imag[start + 1];
      real[start] = r0 + r1;
      imag[start] = i0 + i1;
      real[start + 1] = r0 - r1;
      imag[start + 1] = i0 - i1;
    }
  }

  {
    size_t start;
    for (start = 0; start < SILERO_VAD_STFT_FFT_SIZE; start += 4) {
      const float r0 = real[start];
      const float i0 = imag[start];
      const float r1 = real[start + 1];
      const float i1 = imag[start + 1];
      const float r2 = real[start + 2];
      const float i2 = imag[start + 2];
      const float r3 = real[start + 3];
      const float i3 = imag[start + 3];

      real[start] = r0 + r2;
      imag[start] = i0 + i2;
      real[start + 2] = r0 - r2;
      imag[start + 2] = i0 - i2;

      real[start + 1] = r1 + i3;
      imag[start + 1] = i1 - r3;
      real[start + 3] = r1 - i3;
      imag[start + 3] = i1 + r3;
    }
  }

  {
    size_t start;
    for (start = 0; start < SILERO_VAD_STFT_FFT_SIZE; start += 8) {
      size_t j;
      for (j = 0; j < 4; ++j) {
        const size_t even = start + j;
        const size_t odd = even + 4;
        const float wr = twiddle_cos[j * 32];
        const float wi = twiddle_sin[j * 32];
        const float orr = real[odd];
        const float oii = imag[odd];
        const float tr = (wr * orr) - (wi * oii);
        const float ti = (wr * oii) + (wi * orr);
        const float er = real[even];
        const float ei = imag[even];
        real[even] = er + tr;
        imag[even] = ei + ti;
        real[odd] = er - tr;
        imag[odd] = ei - ti;
      }
    }
  }

  {
    size_t start;
    for (start = 0; start < SILERO_VAD_STFT_FFT_SIZE; start += 16) {
      size_t j;
      for (j = 0; j < 8; ++j) {
        const size_t even = start + j;
        const size_t odd = even + 8;
        const float wr = twiddle_cos[j * 16];
        const float wi = twiddle_sin[j * 16];
        const float orr = real[odd];
        const float oii = imag[odd];
        const float tr = (wr * orr) - (wi * oii);
        const float ti = (wr * oii) + (wi * orr);
        const float er = real[even];
        const float ei = imag[even];
        real[even] = er + tr;
        imag[even] = ei + ti;
        real[odd] = er - tr;
        imag[odd] = ei - ti;
      }
    }
  }

  {
    size_t start;
    for (start = 0; start < SILERO_VAD_STFT_FFT_SIZE; start += 32) {
      size_t j;
      for (j = 0; j < 16; ++j) {
        const size_t even = start + j;
        const size_t odd = even + 16;
        const float wr = twiddle_cos[j * 8];
        const float wi = twiddle_sin[j * 8];
        const float orr = real[odd];
        const float oii = imag[odd];
        const float tr = (wr * orr) - (wi * oii);
        const float ti = (wr * oii) + (wi * orr);
        const float er = real[even];
        const float ei = imag[even];
        real[even] = er + tr;
        imag[even] = ei + ti;
        real[odd] = er - tr;
        imag[odd] = ei - ti;
      }
    }
  }

  {
    size_t start;
    for (start = 0; start < SILERO_VAD_STFT_FFT_SIZE; start += 64) {
      size_t j;
      for (j = 0; j < 32; ++j) {
        const size_t even = start + j;
        const size_t odd = even + 32;
        const float wr = twiddle_cos[j * 4];
        const float wi = twiddle_sin[j * 4];
        const float orr = real[odd];
        const float oii = imag[odd];
        const float tr = (wr * orr) - (wi * oii);
        const float ti = (wr * oii) + (wi * orr);
        const float er = real[even];
        const float ei = imag[even];
        real[even] = er + tr;
        imag[even] = ei + ti;
        real[odd] = er - tr;
        imag[odd] = ei - ti;
      }
    }
  }

  {
    size_t start;
    for (start = 0; start < SILERO_VAD_STFT_FFT_SIZE; start += 128) {
      size_t j;
      for (j = 0; j < 64; ++j) {
        const size_t even = start + j;
        const size_t odd = even + 64;
        const float wr = twiddle_cos[j * 2];
        const float wi = twiddle_sin[j * 2];
        const float orr = real[odd];
        const float oii = imag[odd];
        const float tr = (wr * orr) - (wi * oii);
        const float ti = (wr * oii) + (wi * orr);
        const float er = real[even];
        const float ei = imag[even];
        real[even] = er + tr;
        imag[even] = ei + ti;
        real[odd] = er - tr;
        imag[odd] = ei - ti;
      }
    }
  }

  {
    size_t j;
    for (j = 0; j < 128; ++j) {
      const size_t even = j;
      const size_t odd = j + 128;
      const float wr = twiddle_cos[j];
      const float wi = twiddle_sin[j];
      const float orr = real[odd];
      const float oii = imag[odd];
      const float tr = (wr * orr) - (wi * oii);
      const float ti = (wr * oii) + (wi * orr);
      const float er = real[even];
      const float ei = imag[even];
      real[even] = er + tr;
      imag[even] = ei + ti;
      real[odd] = er - tr;
      imag[odd] = ei - ti;
    }
  }
}

static void silero_vad_conv1_from_stft_frame(const SileroVadConv1d *conv,
                                             const float *prev_mag,
                                             const float *curr_mag,
                                             const float *next_mag,
                                             float *output_frame) {
  size_t ic;

#if SILERO_VAD_SIMD_BACKEND != SILERO_VAD_SIMD_BACKEND_SCALAR
  if (conv->packed_weight_t0 != NULL && conv->packed_weight_t1 != NULL && conv->packed_weight_t2 != NULL) {
    const size_t block_width = 8;
    const size_t block_count = conv->output_channels / block_width;
    const size_t simd_channels = block_count * block_width;
    const size_t simd_steps = block_width / SILERO_VAD_SIMD_LANES;
    size_t block;
    size_t oc;

    for (block = 0; block < block_count; ++block) {
      const size_t oc_base = block * block_width;
      const float *w0 = conv->packed_weight_t0 + (block * conv->input_channels * block_width);
      const float *w1 = conv->packed_weight_t1 + (block * conv->input_channels * block_width);
      const float *w2 = conv->packed_weight_t2 + (block * conv->input_channels * block_width);
      silero_vad_simd_f32 acc[8];
      size_t step;

      for (step = 0; step < simd_steps; ++step) {
        const size_t lane_offset = step * SILERO_VAD_SIMD_LANES;
        acc[step] = conv->bias != NULL
          ? silero_vad_simd_loadu_f32(conv->bias + oc_base + lane_offset)
          : silero_vad_simd_zero_f32();
      }

      for (ic = 0; ic < conv->input_channels; ++ic) {
        const size_t weight_offset = ic * block_width;
        if (prev_mag != NULL) {
          const silero_vad_simd_f32 v = silero_vad_simd_splat_f32(prev_mag[ic]);
          for (step = 0; step < simd_steps; ++step) {
            const size_t lane_offset = step * SILERO_VAD_SIMD_LANES;
            acc[step] = silero_vad_simd_madd_f32(
              acc[step],
              v,
              silero_vad_simd_loadu_f32(w0 + weight_offset + lane_offset));
          }
        }
        {
          const silero_vad_simd_f32 v = silero_vad_simd_splat_f32(curr_mag[ic]);
          for (step = 0; step < simd_steps; ++step) {
            const size_t lane_offset = step * SILERO_VAD_SIMD_LANES;
            acc[step] = silero_vad_simd_madd_f32(
              acc[step],
              v,
              silero_vad_simd_loadu_f32(w1 + weight_offset + lane_offset));
          }
        }
        if (next_mag != NULL) {
          const silero_vad_simd_f32 v = silero_vad_simd_splat_f32(next_mag[ic]);
          for (step = 0; step < simd_steps; ++step) {
            const size_t lane_offset = step * SILERO_VAD_SIMD_LANES;
            acc[step] = silero_vad_simd_madd_f32(
              acc[step],
              v,
              silero_vad_simd_loadu_f32(w2 + weight_offset + lane_offset));
          }
        }
      }

      for (step = 0; step < simd_steps; ++step) {
        silero_vad_simd_storeu_f32(output_frame + oc_base + (step * SILERO_VAD_SIMD_LANES), acc[step]);
      }
    }

    for (oc = simd_channels; oc < conv->output_channels; ++oc) {
      float acc = conv->bias != NULL ? conv->bias[oc] : 0.0f;
      for (ic = 0; ic < conv->input_channels; ++ic) {
        const float *weight = conv->weight + (((oc * conv->input_channels) + ic) * 3);
        if (prev_mag != NULL) {
          acc += prev_mag[ic] * weight[0];
        }
        acc += curr_mag[ic] * weight[1];
        if (next_mag != NULL) {
          acc += next_mag[ic] * weight[2];
        }
      }
      output_frame[oc] = acc;
    }
    return;
  }
#endif

  {
    size_t oc;
    for (oc = 0; oc < conv->output_channels; ++oc) {
      float acc = conv->bias != NULL ? conv->bias[oc] : 0.0f;
      for (ic = 0; ic < conv->input_channels; ++ic) {
        const float *weight = conv->weight + (((oc * conv->input_channels) + ic) * 3);
        if (prev_mag != NULL) {
          acc += prev_mag[ic] * weight[0];
        }
        acc += curr_mag[ic] * weight[1];
        if (next_mag != NULL) {
          acc += next_mag[ic] * weight[2];
        }
      }
      output_frame[oc] = acc;
    }
  }
}

static void silero_vad_stft_conv1_fused_forward(SileroVadModel *model,
                                                const float *input) {
  float mag_ring[3][SILERO_VAD_STFT_BINS];
  size_t frame;
  size_t i;

  if (model->stft_frames == 0) {
    return;
  }

  memset(mag_ring, 0, sizeof(mag_ring));

  for (frame = 0; frame < model->stft_frames; ++frame) {
    const float *frame_input = input + (frame * SILERO_VAD_STFT_HOP_SIZE);
    const size_t slot = frame % 3;
    float *curr_mag = mag_ring[slot];

    for (i = 0; i < SILERO_VAD_STFT_FFT_SIZE; ++i) {
      const float hann = 0.5f - (0.5f * model->fft_twiddle_cos[i]);
      model->fft_real[i] = frame_input[i] * hann;
      model->fft_imag[i] = 0.0f;
    }

    silero_vad_fft256(model->fft_real,
                      model->fft_imag,
                      model->fft_twiddle_cos,
                      model->fft_twiddle_sin,
                      model->fft_bitrev);

#if SILERO_VAD_SIMD_BACKEND != SILERO_VAD_SIMD_BACKEND_SCALAR
    {
      size_t bin = 0;
      for (; bin + SILERO_VAD_SIMD_LANES <= SILERO_VAD_STFT_BINS; bin += SILERO_VAD_SIMD_LANES) {
        const silero_vad_simd_f32 vr = silero_vad_simd_loadu_f32(model->fft_real + bin);
        const silero_vad_simd_f32 vi = silero_vad_simd_loadu_f32(model->fft_imag + bin);
        const silero_vad_simd_f32 mag2 =
          silero_vad_simd_madd_f32(silero_vad_simd_mul_f32(vr, vr), vi, vi);
        silero_vad_simd_storeu_f32(curr_mag + bin, silero_vad_simd_sqrt_f32(mag2));
      }
      for (; bin < SILERO_VAD_STFT_BINS; ++bin) {
        curr_mag[bin] =
          sqrtf((model->fft_real[bin] * model->fft_real[bin]) +
                (model->fft_imag[bin] * model->fft_imag[bin]));
      }
    }
#else
    for (i = 0; i < SILERO_VAD_STFT_BINS; ++i) {
      curr_mag[i] =
        sqrtf((model->fft_real[i] * model->fft_real[i]) +
              (model->fft_imag[i] * model->fft_imag[i]));
    }
#endif

    if (frame > 0) {
      const size_t out_frame = frame - 1;
      const float *prev_mag = (out_frame == 0) ? NULL : mag_ring[(frame + 1) % 3];
      const float *center_mag = mag_ring[(frame + 2) % 3];
      float conv1_frame[128];
      size_t oc;

      silero_vad_conv1_from_stft_frame(&model->conv1, prev_mag, center_mag, curr_mag, conv1_frame);
      for (oc = 0; oc < 128; ++oc) {
        model->conv1_out[(oc * model->stft_frames) + out_frame] = conv1_frame[oc] > 0.0f ? conv1_frame[oc] : 0.0f;
      }
    }
  }

  {
    const size_t out_frame = model->stft_frames - 1;
    const float *prev_mag = (out_frame == 0) ? NULL : mag_ring[(model->stft_frames + 1) % 3];
    const float *center_mag = mag_ring[(model->stft_frames - 1) % 3];
    float conv1_frame[128];
    size_t oc;

    silero_vad_conv1_from_stft_frame(&model->conv1, prev_mag, center_mag, NULL, conv1_frame);
    for (oc = 0; oc < 128; ++oc) {
      model->conv1_out[(oc * model->stft_frames) + out_frame] = conv1_frame[oc] > 0.0f ? conv1_frame[oc] : 0.0f;
    }
  }
}

static void silero_vad_zero_model(SileroVadModel *model) {
  memset(model, 0, sizeof(*model));
}

size_t silero_vad_conv1d_output_frames(size_t input_frames,
                                       size_t kernel_size,
                                       size_t stride,
                                       size_t padding) {
  size_t padded_frames;

  if (stride == 0) {
    return 0;
  }

  padded_frames = input_frames + (2 * padding);
  if (padded_frames < kernel_size) {
    return 0;
  }

  return 1 + ((padded_frames - kernel_size) / stride);
}

SileroVadStatus silero_vad_conv1d_init(SileroVadConv1d *conv,
                                       size_t input_channels,
                                       size_t output_channels,
                                       size_t kernel_size,
                                       size_t stride,
                                       size_t padding,
                                       size_t max_input_frames,
                                       const float *weight,
                                       const float *bias) {
  if (conv == NULL || weight == NULL || input_channels == 0 || output_channels == 0 ||
      kernel_size == 0 || stride == 0 || max_input_frames == 0) {
    return SILERO_VAD_STATUS_INVALID_ARGUMENT;
  }

  memset(conv, 0, sizeof(*conv));
  conv->input_channels = input_channels;
  conv->output_channels = output_channels;
  conv->kernel_size = kernel_size;
  conv->stride = stride;
  conv->padding = padding;
  conv->weight = weight;
  conv->bias = bias;
  conv->max_input_frames = max_input_frames;

  if (!(kernel_size == 3 && padding == 1)) {
    const size_t padded_frames = max_input_frames + (2 * padding);
    const size_t padded_values = input_channels * padded_frames;
    conv->padded_input = (float *)silero_vad_aligned_calloc(padded_values, sizeof(float));
    if (conv->padded_input == NULL) {
      return SILERO_VAD_STATUS_ALLOCATION_FAILED;
    }
    conv->padded_capacity = padded_values;
  }

  silero_vad_pack_k3_weights(conv);
  return SILERO_VAD_STATUS_OK;
}

void silero_vad_conv1d_reset(SileroVadConv1d *conv) {
  if (conv == NULL || conv->padded_input == NULL) {
    return;
  }

  memset(conv->padded_input, 0, conv->padded_capacity * sizeof(float));
}

void silero_vad_conv1d_free(SileroVadConv1d *conv) {
  if (conv == NULL) {
    return;
  }

  silero_vad_aligned_free(conv->padded_input);
  silero_vad_aligned_free(conv->packed_weight_t0);
  silero_vad_aligned_free(conv->packed_weight_t1);
  silero_vad_aligned_free(conv->packed_weight_t2);
  memset(conv, 0, sizeof(*conv));
}

SileroVadStatus silero_vad_conv1d_forward(SileroVadConv1d *conv,
                                          const float *input,
                                          size_t input_frames,
                                          float *output) {
  size_t padded_frames;
  size_t out_frames;
  size_t ic;
  size_t oc;
  size_t of;
  size_t k;

  if (conv == NULL || input == NULL || output == NULL) {
    return SILERO_VAD_STATUS_INVALID_ARGUMENT;
  }

  if (input_frames == 0 || input_frames > conv->max_input_frames) {
    return SILERO_VAD_STATUS_INVALID_SHAPE;
  }

  padded_frames = input_frames + (2 * conv->padding);
  out_frames = silero_vad_conv1d_output_frames(input_frames,
                                               conv->kernel_size,
                                               conv->stride,
                                               conv->padding);
  if (out_frames == 0) {
    return SILERO_VAD_STATUS_INVALID_SHAPE;
  }

  for (ic = 0; ic < conv->input_channels; ++ic) {
    float *dst = conv->padded_input + (ic * padded_frames);
    const float *src = input + (ic * input_frames);
    memset(dst, 0, padded_frames * sizeof(float));
    memcpy(dst + conv->padding, src, input_frames * sizeof(float));
  }

  for (oc = 0; oc < conv->output_channels; ++oc) {
    for (of = 0; of < out_frames; ++of) {
      float acc = conv->bias != NULL ? conv->bias[oc] : 0.0f;
      const size_t input_start = of * conv->stride;

      for (ic = 0; ic < conv->input_channels; ++ic) {
        const float *padded = conv->padded_input + (ic * padded_frames) + input_start;
        const float *weight = conv->weight + (((oc * conv->input_channels) + ic) * conv->kernel_size);

        for (k = 0; k < conv->kernel_size; ++k) {
          acc += padded[k] * weight[k];
        }
      }

      output[(oc * out_frames) + of] = acc;
    }
  }

  return SILERO_VAD_STATUS_OK;
}

static SileroVadStatus silero_vad_conv1d_k3_p1_forward(const SileroVadConv1d *conv,
                                                       const float *input,
                                                       size_t input_frames,
                                                       float *output) {
  size_t out_frames;
  size_t of;
  size_t ic;

  if (conv == NULL || input == NULL || output == NULL) {
    return SILERO_VAD_STATUS_INVALID_ARGUMENT;
  }

  if (conv->kernel_size != 3 || conv->padding != 1 || conv->stride == 0) {
    return SILERO_VAD_STATUS_INVALID_ARGUMENT;
  }

  if (input_frames == 0 || input_frames > conv->max_input_frames) {
    return SILERO_VAD_STATUS_INVALID_SHAPE;
  }

  out_frames = silero_vad_conv1d_output_frames(input_frames,
                                               conv->kernel_size,
                                               conv->stride,
                                               conv->padding);
  if (out_frames == 0) {
    return SILERO_VAD_STATUS_INVALID_SHAPE;
  }

#if SILERO_VAD_SIMD_BACKEND != SILERO_VAD_SIMD_BACKEND_SCALAR
  if (conv->packed_weight_t0 != NULL && conv->packed_weight_t1 != NULL && conv->packed_weight_t2 != NULL) {
    const size_t block_width = 8;
    const size_t block_count = conv->output_channels / block_width;
    const size_t simd_channels = block_count * block_width;
    const size_t simd_steps = block_width / SILERO_VAD_SIMD_LANES;
    size_t block;
    size_t oc;

    for (of = 0; of < out_frames; ++of) {
      const size_t center = of * conv->stride;
      for (block = 0; block < block_count; ++block) {
        const size_t oc_base = block * block_width;
        const float *w0 = conv->packed_weight_t0 + (block * conv->input_channels * block_width);
        const float *w1 = conv->packed_weight_t1 + (block * conv->input_channels * block_width);
        const float *w2 = conv->packed_weight_t2 + (block * conv->input_channels * block_width);
        silero_vad_simd_f32 acc[8];
        size_t step;

        for (step = 0; step < simd_steps; ++step) {
          const size_t lane_offset = step * SILERO_VAD_SIMD_LANES;
          acc[step] = conv->bias != NULL
            ? silero_vad_simd_loadu_f32(conv->bias + oc_base + lane_offset)
            : silero_vad_simd_zero_f32();
        }

        for (ic = 0; ic < conv->input_channels; ++ic) {
          const float *src = input + (ic * input_frames);
          const size_t weight_offset = ic * block_width;

          if (center > 0) {
            const silero_vad_simd_f32 v = silero_vad_simd_splat_f32(src[center - 1]);
            for (step = 0; step < simd_steps; ++step) {
              const size_t lane_offset = step * SILERO_VAD_SIMD_LANES;
              acc[step] = silero_vad_simd_madd_f32(
                acc[step],
                v,
                silero_vad_simd_loadu_f32(w0 + weight_offset + lane_offset));
            }
          }
          {
            const silero_vad_simd_f32 v = silero_vad_simd_splat_f32(src[center]);
            for (step = 0; step < simd_steps; ++step) {
              const size_t lane_offset = step * SILERO_VAD_SIMD_LANES;
              acc[step] = silero_vad_simd_madd_f32(
                acc[step],
                v,
                silero_vad_simd_loadu_f32(w1 + weight_offset + lane_offset));
            }
          }
          if ((center + 1) < input_frames) {
            const silero_vad_simd_f32 v = silero_vad_simd_splat_f32(src[center + 1]);
            for (step = 0; step < simd_steps; ++step) {
              const size_t lane_offset = step * SILERO_VAD_SIMD_LANES;
              acc[step] = silero_vad_simd_madd_f32(
                acc[step],
                v,
                silero_vad_simd_loadu_f32(w2 + weight_offset + lane_offset));
            }
          }
        }

        {
          float tmp[8];
          size_t lane;
          for (step = 0; step < simd_steps; ++step) {
            silero_vad_simd_storeu_f32(tmp + (step * SILERO_VAD_SIMD_LANES), acc[step]);
          }
          for (lane = 0; lane < block_width; ++lane) {
            output[((oc_base + lane) * out_frames) + of] = tmp[lane];
          }
        }
      }

      for (oc = simd_channels; oc < conv->output_channels; ++oc) {
        float acc = conv->bias != NULL ? conv->bias[oc] : 0.0f;
        for (ic = 0; ic < conv->input_channels; ++ic) {
          const float *src = input + (ic * input_frames);
          const float *weight = conv->weight + (((oc * conv->input_channels) + ic) * 3);

          if (center > 0) {
            acc += src[center - 1] * weight[0];
          }
          acc += src[center] * weight[1];
          if ((center + 1) < input_frames) {
            acc += src[center + 1] * weight[2];
          }
        }
        output[(oc * out_frames) + of] = acc;
      }
    }

    return SILERO_VAD_STATUS_OK;
  }
#endif

  {
    size_t oc;
    for (oc = 0; oc < conv->output_channels; ++oc) {
      for (of = 0; of < out_frames; ++of) {
        const size_t center = of * conv->stride;
        float acc = conv->bias != NULL ? conv->bias[oc] : 0.0f;
        for (ic = 0; ic < conv->input_channels; ++ic) {
          const float *src = input + (ic * input_frames);
          const float *weight = conv->weight + (((oc * conv->input_channels) + ic) * 3);

          if (center > 0) {
            acc += src[center - 1] * weight[0];
          }
          acc += src[center] * weight[1];
          if ((center + 1) < input_frames) {
            acc += src[center + 1] * weight[2];
          }
        }

        output[(oc * out_frames) + of] = acc;
      }
    }
  }

  return SILERO_VAD_STATUS_OK;
}

SileroVadStatus silero_vad_lstm_cell_init(SileroVadLstmCell *cell,
                                          size_t input_size,
                                          size_t hidden_size,
                                          const float *weight_ih,
                                          const float *weight_hh,
                                          const float *bias_ih,
                                          const float *bias_hh) {
  if (cell == NULL || weight_ih == NULL || weight_hh == NULL ||
      bias_ih == NULL || bias_hh == NULL || input_size == 0 || hidden_size == 0) {
    return SILERO_VAD_STATUS_INVALID_ARGUMENT;
  }

  memset(cell, 0, sizeof(*cell));
  cell->input_size = input_size;
  cell->hidden_size = hidden_size;
  cell->weight_ih = weight_ih;
  cell->weight_hh = weight_hh;
  cell->bias_ih = bias_ih;
  cell->bias_hh = bias_hh;

  cell->hidden_state = (float *)silero_vad_aligned_calloc(hidden_size, sizeof(float));
  cell->cell_state = (float *)silero_vad_aligned_calloc(hidden_size, sizeof(float));
  cell->gates = (float *)silero_vad_aligned_calloc(hidden_size * 4, sizeof(float));
  silero_vad_pack_lstm_weights(cell);

  if (cell->hidden_state == NULL || cell->cell_state == NULL || cell->gates == NULL) {
    silero_vad_lstm_cell_free(cell);
    return SILERO_VAD_STATUS_ALLOCATION_FAILED;
  }

  return SILERO_VAD_STATUS_OK;
}

void silero_vad_lstm_cell_reset(SileroVadLstmCell *cell) {
  if (cell == NULL) {
    return;
  }

  if (cell->hidden_state != NULL) {
    memset(cell->hidden_state, 0, cell->hidden_size * sizeof(float));
  }

  if (cell->cell_state != NULL) {
    memset(cell->cell_state, 0, cell->hidden_size * sizeof(float));
  }

  if (cell->gates != NULL) {
    memset(cell->gates, 0, (cell->hidden_size * 4) * sizeof(float));
  }
}

void silero_vad_lstm_cell_free(SileroVadLstmCell *cell) {
  if (cell == NULL) {
    return;
  }

  silero_vad_aligned_free(cell->hidden_state);
  silero_vad_aligned_free(cell->cell_state);
  silero_vad_aligned_free(cell->gates);
  silero_vad_aligned_free(cell->packed_weight_ih);
  silero_vad_aligned_free(cell->packed_weight_hh);
  memset(cell, 0, sizeof(*cell));
}

SileroVadStatus silero_vad_lstm_cell_forward(SileroVadLstmCell *cell,
                                             const float *input,
                                             float *hidden_out) {
  size_t h;
  const size_t hidden_size = cell != NULL ? cell->hidden_size : 0;

  if (cell == NULL || input == NULL || hidden_out == NULL) {
    return SILERO_VAD_STATUS_INVALID_ARGUMENT;
  }

#if SILERO_VAD_SIMD_BACKEND != SILERO_VAD_SIMD_BACKEND_SCALAR
  if (cell->packed_weight_ih != NULL && cell->packed_weight_hh != NULL && ((hidden_size * 4) % 8) == 0) {
    const size_t rows = hidden_size * 4;
    const size_t blocks = rows / 8;
    const size_t simd_steps = 8 / SILERO_VAD_SIMD_LANES;
    size_t block;

    for (block = 0; block < blocks; ++block) {
      const size_t row_base = block * 8;
      const float *packed_ih = cell->packed_weight_ih + (block * cell->input_size * 8);
      const float *packed_hh = cell->packed_weight_hh + (block * hidden_size * 8);
      silero_vad_simd_f32 gate[8];
      size_t step;
      size_t i;

      for (step = 0; step < simd_steps; ++step) {
        const size_t lane_offset = step * SILERO_VAD_SIMD_LANES;
        gate[step] = silero_vad_simd_add_f32(
          silero_vad_simd_loadu_f32(cell->bias_ih + row_base + lane_offset),
          silero_vad_simd_loadu_f32(cell->bias_hh + row_base + lane_offset));
      }

      for (i = 0; i < cell->input_size; ++i) {
        const silero_vad_simd_f32 v = silero_vad_simd_splat_f32(input[i]);
        for (step = 0; step < simd_steps; ++step) {
          const size_t lane_offset = step * SILERO_VAD_SIMD_LANES;
          gate[step] = silero_vad_simd_madd_f32(
            gate[step],
            v,
            silero_vad_simd_loadu_f32(packed_ih + (i * 8) + lane_offset));
        }
      }
      for (i = 0; i < hidden_size; ++i) {
        const silero_vad_simd_f32 v = silero_vad_simd_splat_f32(cell->hidden_state[i]);
        for (step = 0; step < simd_steps; ++step) {
          const size_t lane_offset = step * SILERO_VAD_SIMD_LANES;
          gate[step] = silero_vad_simd_madd_f32(
            gate[step],
            v,
            silero_vad_simd_loadu_f32(packed_hh + (i * 8) + lane_offset));
        }
      }

      for (step = 0; step < simd_steps; ++step) {
        silero_vad_simd_storeu_f32(cell->gates + row_base + (step * SILERO_VAD_SIMD_LANES), gate[step]);
      }
    }
  } else
#endif
  {
    for (h = 0; h < hidden_size * 4; ++h) {
      float gate = cell->bias_ih[h] + cell->bias_hh[h];
      const float *w_ih = cell->weight_ih + (h * cell->input_size);
      const float *w_hh = cell->weight_hh + (h * hidden_size);

      gate += silero_vad_dot_f32(w_ih, input, cell->input_size);
      gate += silero_vad_dot_f32(w_hh, cell->hidden_state, hidden_size);
      cell->gates[h] = gate;
    }
  }

  for (h = 0; h < hidden_size; ++h) {
    const float input_gate = silero_vad_sigmoid(cell->gates[h]);
    const float forget_gate = silero_vad_sigmoid(cell->gates[hidden_size + h]);
    const float cell_gate = tanhf(cell->gates[(2 * hidden_size) + h]);
    const float output_gate = silero_vad_sigmoid(cell->gates[(3 * hidden_size) + h]);
    const float new_cell = (forget_gate * cell->cell_state[h]) + (input_gate * cell_gate);
    const float new_hidden = output_gate * tanhf(new_cell);

    cell->cell_state[h] = new_cell;
    cell->hidden_state[h] = new_hidden;
    hidden_out[h] = new_hidden;
  }

  return SILERO_VAD_STATUS_OK;
}

static void silero_vad_model_free_buffers(SileroVadModel *model) {
  silero_vad_aligned_free(model->padded_audio);
  silero_vad_aligned_free(model->fft_real);
  silero_vad_aligned_free(model->fft_imag);
  silero_vad_aligned_free(model->stft_mag);
  silero_vad_aligned_free(model->conv1_out);
  silero_vad_aligned_free(model->conv2_out);
  silero_vad_aligned_free(model->conv3_out);
  silero_vad_aligned_free(model->conv4_out);
  silero_vad_aligned_free(model->lstm_out);
  silero_vad_aligned_free(model->final_out);
  silero_vad_aligned_free(model->fft_twiddle_cos);
  silero_vad_aligned_free(model->fft_twiddle_sin);
  silero_vad_aligned_free(model->fft_bitrev);

  model->padded_audio = NULL;
  model->fft_real = NULL;
  model->fft_imag = NULL;
  model->stft_mag = NULL;
  model->conv1_out = NULL;
  model->conv2_out = NULL;
  model->conv3_out = NULL;
  model->conv4_out = NULL;
  model->lstm_out = NULL;
  model->final_out = NULL;
  model->fft_twiddle_cos = NULL;
  model->fft_twiddle_sin = NULL;
  model->fft_bitrev = NULL;
}

static SileroVadStatus silero_vad_model_alloc_buffers(SileroVadModel *model) {
  const size_t padded_audio_samples = model->input_samples + SILERO_VAD_STFT_RIGHT_PAD;

  model->padded_audio = (float *)silero_vad_aligned_calloc(padded_audio_samples, sizeof(float));
  model->fft_real = (float *)silero_vad_aligned_calloc(SILERO_VAD_STFT_FFT_SIZE, sizeof(float));
  model->fft_imag = (float *)silero_vad_aligned_calloc(SILERO_VAD_STFT_FFT_SIZE, sizeof(float));
  model->stft_mag = (float *)silero_vad_aligned_calloc(SILERO_VAD_STFT_BINS * model->stft_frames, sizeof(float));
  model->conv1_out = (float *)silero_vad_aligned_calloc(128 * model->stft_frames, sizeof(float));
  model->conv2_out = (float *)silero_vad_aligned_calloc(64 * model->conv2_frames, sizeof(float));
  model->conv3_out = (float *)silero_vad_aligned_calloc(64 * model->conv3_frames, sizeof(float));
  model->conv4_out = (float *)silero_vad_aligned_calloc(128 * model->conv4_frames, sizeof(float));
  model->lstm_out = (float *)silero_vad_aligned_calloc(SILERO_VAD_HIDDEN_SIZE, sizeof(float));
  model->final_out = (float *)silero_vad_aligned_calloc(model->conv4_frames, sizeof(float));
  model->fft_twiddle_cos = (float *)silero_vad_aligned_calloc(SILERO_VAD_STFT_FFT_SIZE, sizeof(float));
  model->fft_twiddle_sin = (float *)silero_vad_aligned_calloc(SILERO_VAD_STFT_FFT_SIZE, sizeof(float));
  model->fft_bitrev = (unsigned short *)silero_vad_aligned_calloc(SILERO_VAD_STFT_FFT_SIZE, sizeof(unsigned short));

  if (model->padded_audio == NULL || model->fft_real == NULL || model->fft_imag == NULL || model->stft_mag == NULL ||
      model->conv1_out == NULL || model->conv2_out == NULL || model->conv3_out == NULL ||
      model->conv4_out == NULL || model->lstm_out == NULL || model->final_out == NULL ||
      model->fft_twiddle_cos == NULL || model->fft_twiddle_sin == NULL || model->fft_bitrev == NULL) {
    silero_vad_model_free_buffers(model);
    return SILERO_VAD_STATUS_ALLOCATION_FAILED;
  }

  silero_vad_init_fft_tables(model);
  return SILERO_VAD_STATUS_OK;
}

static size_t silero_vad_round_ratio(size_t numerator_a, size_t numerator_b, size_t denominator) {
  if (denominator == 0 || numerator_a == 0 || numerator_b == 0) {
    return 0;
  }

  return ((numerator_a * numerator_b) + (denominator / 2)) / denominator;
}

static SileroVadStatus silero_vad_model_ensure_resampled_capacity(SileroVadModel *model,
                                                                  size_t samples) {
  float *resampled_audio;

  if (model->resampled_audio_capacity >= samples) {
    return SILERO_VAD_STATUS_OK;
  }

  resampled_audio = (float *)silero_vad_aligned_calloc(samples, sizeof(float));
  if (resampled_audio == NULL) {
    return SILERO_VAD_STATUS_ALLOCATION_FAILED;
  }

  silero_vad_aligned_free(model->resampled_audio);
  model->resampled_audio = resampled_audio;
  model->resampled_audio_capacity = samples;
  return SILERO_VAD_STATUS_OK;
}

SileroVadStatus silero_vad_model_init(SileroVadModel *model,
                                      const SileroVadWeights *weights,
                                      size_t input_samples) {
  return silero_vad_model_init_with_sample_rate(model,
                                                weights,
                                                input_samples,
                                                SILERO_VAD_SAMPLE_RATE);
}

SileroVadStatus silero_vad_model_init_with_sample_rate(SileroVadModel *model,
                                                       const SileroVadWeights *weights,
                                                       size_t input_samples,
                                                       size_t sampling_rate) {
  SileroVadStatus status;

  if (model == NULL || input_samples == 0 || sampling_rate == 0 || silero_vad_has_null_weights(weights)) {
    return SILERO_VAD_STATUS_INVALID_ARGUMENT;
  }

  silero_vad_zero_model(model);
  model->input_samples = input_samples;
  model->sampling_rate = sampling_rate;
  model->source_window_samples = silero_vad_model_source_window_samples(sampling_rate);

  if (sampling_rate != SILERO_VAD_SAMPLE_RATE) {
    model->resampler = yl_resample_create((int)sampling_rate,
                                          SILERO_VAD_SAMPLE_RATE,
                                          6,
                                          0.99f,
                                          YL_RESAMPLE_SINC_HANN,
                                          0.0f);
    if (model->resampler == NULL) {
      silero_vad_model_free(model);
      return SILERO_VAD_STATUS_ALLOCATION_FAILED;
    }
  }

  model->stft_frames = silero_vad_conv1d_output_frames(input_samples + SILERO_VAD_STFT_RIGHT_PAD,
                                                       SILERO_VAD_STFT_FFT_SIZE,
                                                       SILERO_VAD_STFT_HOP_SIZE,
                                                       0);
  model->conv2_frames = silero_vad_conv1d_output_frames(model->stft_frames, 3, 2, 1);
  model->conv3_frames = silero_vad_conv1d_output_frames(model->conv2_frames, 3, 2, 1);
  model->conv4_frames = silero_vad_conv1d_output_frames(model->conv3_frames, 3, 1, 1);

  if (model->stft_frames == 0 || model->conv2_frames == 0 ||
      model->conv3_frames == 0 || model->conv4_frames != 1) {
    silero_vad_model_free(model);
    return SILERO_VAD_STATUS_INVALID_SHAPE;
  }

  status = silero_vad_conv1d_init(&model->conv1,
                                  129, 128, 3, 1, 1,
                                  model->stft_frames,
                                  weights->conv1_weight, weights->conv1_bias);
  if (status != SILERO_VAD_STATUS_OK) {
    silero_vad_model_free(model);
    return status;
  }

  status = silero_vad_conv1d_init(&model->conv2,
                                  128, 64, 3, 2, 1,
                                  model->stft_frames,
                                  weights->conv2_weight, weights->conv2_bias);
  if (status != SILERO_VAD_STATUS_OK) {
    silero_vad_model_free(model);
    return status;
  }

  status = silero_vad_conv1d_init(&model->conv3,
                                  64, 64, 3, 2, 1,
                                  model->conv2_frames,
                                  weights->conv3_weight, weights->conv3_bias);
  if (status != SILERO_VAD_STATUS_OK) {
    silero_vad_model_free(model);
    return status;
  }

  status = silero_vad_conv1d_init(&model->conv4,
                                  64, 128, 3, 1, 1,
                                  model->conv3_frames,
                                  weights->conv4_weight, weights->conv4_bias);
  if (status != SILERO_VAD_STATUS_OK) {
    silero_vad_model_free(model);
    return status;
  }

  status = silero_vad_conv1d_init(&model->final_conv,
                                  128, 1, 1, 1, 0,
                                  model->conv4_frames,
                                  weights->final_conv_weight, weights->final_conv_bias);
  if (status != SILERO_VAD_STATUS_OK) {
    silero_vad_model_free(model);
    return status;
  }

  status = silero_vad_lstm_cell_init(&model->lstm,
                                     128, 128,
                                     weights->lstm_weight_ih,
                                     weights->lstm_weight_hh,
                                     weights->lstm_bias_ih,
                                     weights->lstm_bias_hh);
  if (status != SILERO_VAD_STATUS_OK) {
    silero_vad_model_free(model);
    return status;
  }

  status = silero_vad_model_alloc_buffers(model);
  if (status != SILERO_VAD_STATUS_OK) {
    silero_vad_model_free(model);
    return status;
  }

  return SILERO_VAD_STATUS_OK;
}

SileroVadModel *silero_vad_model_create(const SileroVadWeights *weights,
                                        size_t input_samples) {
  return silero_vad_model_create_with_sample_rate(weights,
                                                 input_samples,
                                                 SILERO_VAD_SAMPLE_RATE);
}

SileroVadModel *silero_vad_model_create_with_sample_rate(const SileroVadWeights *weights,
                                                        size_t input_samples,
                                                        size_t sampling_rate) {
  SileroVadModel *model;
  SileroVadStatus status;

  model = (SileroVadModel *)calloc(1, sizeof(*model));
  if (model == NULL) {
    return NULL;
  }

  status = silero_vad_model_init_with_sample_rate(model, weights, input_samples, sampling_rate);
  if (status != SILERO_VAD_STATUS_OK) {
    free(model);
    return NULL;
  }

  return model;
}

void silero_vad_model_reset(SileroVadModel *model) {
  if (model == NULL) {
    return;
  }

  silero_vad_lstm_cell_reset(&model->lstm);
  memset(model->stream_context, 0, sizeof(model->stream_context));
  if (model->resampler != NULL) {
    yl_resample_reset(model->resampler);
  }
}

void silero_vad_model_free(SileroVadModel *model) {
  if (model == NULL) {
    return;
  }

  silero_vad_conv1d_free(&model->conv1);
  silero_vad_conv1d_free(&model->conv2);
  silero_vad_conv1d_free(&model->conv3);
  silero_vad_conv1d_free(&model->conv4);
  silero_vad_conv1d_free(&model->final_conv);
  silero_vad_lstm_cell_free(&model->lstm);
  silero_vad_model_free_buffers(model);
  yl_resample_destroy(model->resampler);
  silero_vad_aligned_free(model->resampled_audio);

  model->input_samples = 0;
  model->sampling_rate = 0;
  model->source_window_samples = 0;
  model->stft_frames = 0;
  model->conv2_frames = 0;
  model->conv3_frames = 0;
  model->conv4_frames = 0;
  model->resampler = NULL;
  model->resampled_audio = NULL;
  model->resampled_audio_capacity = 0;
}

void silero_vad_model_destroy(SileroVadModel *model) {
  if (model == NULL) {
    return;
  }

  silero_vad_model_free(model);
  free(model);
}

size_t silero_vad_model_audio_prob_count(size_t audio_samples) {
  return silero_vad_model_audio_prob_count_for_sample_rate(audio_samples,
                                                           SILERO_VAD_SAMPLE_RATE);
}

size_t silero_vad_model_audio_prob_count_for_sample_rate(size_t audio_samples,
                                                        size_t sampling_rate) {
  size_t resampled_samples;

  if (audio_samples == 0 || sampling_rate == 0) {
    return 0;
  }

  if (sampling_rate == SILERO_VAD_SAMPLE_RATE) {
    resampled_samples = audio_samples;
  } else {
    resampled_samples = yl_resample_out_len((int)sampling_rate,
                                            SILERO_VAD_SAMPLE_RATE,
                                            audio_samples);
  }

  return (resampled_samples + SILERO_VAD_WINDOW_SAMPLES - 1) / SILERO_VAD_WINDOW_SAMPLES;
}

size_t silero_vad_model_source_window_samples(size_t sampling_rate) {
  if (sampling_rate == 0) {
    return 0;
  }

  return silero_vad_round_ratio(sampling_rate,
                                SILERO_VAD_WINDOW_SAMPLES,
                                SILERO_VAD_SAMPLE_RATE);
}

size_t silero_vad_model_get_source_window_samples(const SileroVadModel *model) {
  if (model == NULL) {
    return 0;
  }

  return model->source_window_samples;
}

SileroVadStatus silero_vad_model_forward(SileroVadModel *model,
                                         const float *input,
                                         float *speech_probability) {
  SileroVadStatus status;

  if (model == NULL || input == NULL || speech_probability == NULL) {
    return SILERO_VAD_STATUS_INVALID_ARGUMENT;
  }

  silero_vad_reflect_pad_right(input,
                               model->input_samples,
                               SILERO_VAD_STFT_RIGHT_PAD,
                               model->padded_audio);

  silero_vad_stft_conv1_fused_forward(model, model->padded_audio);

  status = silero_vad_conv1d_k3_p1_forward(&model->conv2,
                                           model->conv1_out,
                                           model->stft_frames,
                                           model->conv2_out);
  if (status != SILERO_VAD_STATUS_OK) {
    return status;
  }
  silero_vad_relu(model->conv2_out, 64 * model->conv2_frames);

  status = silero_vad_conv1d_k3_p1_forward(&model->conv3,
                                           model->conv2_out,
                                           model->conv2_frames,
                                           model->conv3_out);
  if (status != SILERO_VAD_STATUS_OK) {
    return status;
  }
  silero_vad_relu(model->conv3_out, 64 * model->conv3_frames);

  status = silero_vad_conv1d_k3_p1_forward(&model->conv4,
                                           model->conv3_out,
                                           model->conv3_frames,
                                           model->conv4_out);
  if (status != SILERO_VAD_STATUS_OK) {
    return status;
  }
  silero_vad_relu(model->conv4_out, 128 * model->conv4_frames);

  status = silero_vad_lstm_cell_forward(&model->lstm, model->conv4_out, model->lstm_out);
  if (status != SILERO_VAD_STATUS_OK) {
    return status;
  }
  silero_vad_relu(model->lstm_out, SILERO_VAD_HIDDEN_SIZE);

  model->final_out[0] = (model->final_conv.bias != NULL ? model->final_conv.bias[0] : 0.0f) +
                        silero_vad_dot_f32(model->final_conv.weight,
                                           model->lstm_out,
                                           SILERO_VAD_HIDDEN_SIZE);

  *speech_probability = silero_vad_sigmoid(model->final_out[0]);
  return SILERO_VAD_STATUS_OK;
}

SileroVadStatus silero_vad_model_forward_source_chunk(SileroVadModel *model,
                                                      const float *input,
                                                      size_t input_samples,
                                                      float *speech_probability) {
  float chunk[SILERO_VAD_INPUT_SAMPLES];
  const float *fresh_samples = input;
  float fresh_resampled[SILERO_VAD_WINDOW_SAMPLES];
  SileroVadStatus status;

  if (model == NULL || input == NULL || speech_probability == NULL) {
    return SILERO_VAD_STATUS_INVALID_ARGUMENT;
  }

  if (model->input_samples != SILERO_VAD_INPUT_SAMPLES ||
      model->sampling_rate == 0 ||
      model->source_window_samples == 0 ||
      input_samples != model->source_window_samples) {
    return SILERO_VAD_STATUS_INVALID_SHAPE;
  }

  if (model->sampling_rate != SILERO_VAD_SAMPLE_RATE) {
    size_t resampled_samples = yl_resample_out_len((int)model->sampling_rate,
                                                   SILERO_VAD_SAMPLE_RATE,
                                                   input_samples);
    size_t copy_samples;
    size_t written;

    if (model->resampler == NULL) {
      return SILERO_VAD_STATUS_INVALID_SHAPE;
    }

    status = silero_vad_model_ensure_resampled_capacity(model, resampled_samples);
    if (status != SILERO_VAD_STATUS_OK) {
      return status;
    }

    written = yl_resample_process(model->resampler,
                                  input,
                                  input_samples,
                                  model->resampled_audio);
    if (written == 0) {
      return SILERO_VAD_STATUS_ALLOCATION_FAILED;
    }

    memset(fresh_resampled, 0, sizeof(fresh_resampled));
    copy_samples = written < SILERO_VAD_WINDOW_SAMPLES ? written : SILERO_VAD_WINDOW_SAMPLES;
    memcpy(fresh_resampled, model->resampled_audio, copy_samples * sizeof(float));
    fresh_samples = fresh_resampled;
  }

  memset(chunk, 0, sizeof(chunk));
  memcpy(chunk, model->stream_context, SILERO_VAD_CONTEXT_SAMPLES * sizeof(float));
  memcpy(chunk + SILERO_VAD_CONTEXT_SAMPLES,
         fresh_samples,
         SILERO_VAD_WINDOW_SAMPLES * sizeof(float));

  status = silero_vad_model_forward(model, chunk, speech_probability);
  if (status != SILERO_VAD_STATUS_OK) {
    return status;
  }

  memcpy(model->stream_context,
         chunk + SILERO_VAD_WINDOW_SAMPLES,
         SILERO_VAD_CONTEXT_SAMPLES * sizeof(float));
  return SILERO_VAD_STATUS_OK;
}

SileroVadStatus silero_vad_model_forward_audio(SileroVadModel *model,
                                               const float *audio,
                                               size_t audio_samples,
                                               float *speech_probabilities,
                                               size_t speech_probabilities_capacity,
                                               size_t *speech_probabilities_written) {
  const size_t window_size = SILERO_VAD_WINDOW_SAMPLES;
  const size_t context_size = SILERO_VAD_CONTEXT_SAMPLES;
  const size_t chunk_size = SILERO_VAD_INPUT_SAMPLES;
  const float *model_audio = audio;
  size_t model_audio_samples = audio_samples;
  size_t required_probs;
  float context[64] = {0};
  float chunk[576];
  size_t start;
  size_t out_index = 0;

  if (model == NULL || audio == NULL || speech_probabilities == NULL || speech_probabilities_written == NULL) {
    return SILERO_VAD_STATUS_INVALID_ARGUMENT;
  }

  if (model->input_samples != chunk_size) {
    return SILERO_VAD_STATUS_INVALID_SHAPE;
  }

  if (model->sampling_rate == 0) {
    return SILERO_VAD_STATUS_INVALID_SHAPE;
  }

  required_probs = silero_vad_model_audio_prob_count_for_sample_rate(audio_samples,
                                                                     model->sampling_rate);
  if (speech_probabilities_capacity < required_probs) {
    return SILERO_VAD_STATUS_INVALID_SHAPE;
  }

  silero_vad_model_reset(model);
  *speech_probabilities_written = 0;

  if (audio_samples == 0) {
    return SILERO_VAD_STATUS_OK;
  }

  if (model->sampling_rate != SILERO_VAD_SAMPLE_RATE) {
    size_t resampled_samples = yl_resample_out_len((int)model->sampling_rate,
                                                   SILERO_VAD_SAMPLE_RATE,
                                                   audio_samples);
    size_t written;

    if (model->resampler == NULL) {
      return SILERO_VAD_STATUS_INVALID_SHAPE;
    }

    if (model->resampled_audio_capacity < resampled_samples) {
      SileroVadStatus status = silero_vad_model_ensure_resampled_capacity(model, resampled_samples);
      if (status != SILERO_VAD_STATUS_OK) {
        return status;
      }
    }

    written = yl_resample_process(model->resampler,
                                  audio,
                                  audio_samples,
                                  model->resampled_audio);
    if (written != resampled_samples) {
      return SILERO_VAD_STATUS_ALLOCATION_FAILED;
    }

    model_audio = model->resampled_audio;
    model_audio_samples = resampled_samples;
  }

  for (start = 0; start < model_audio_samples; start += window_size) {
    size_t chunk_samples = model_audio_samples - start;
    float prob = 0.0f;
    SileroVadStatus status;

    if (chunk_samples > window_size) {
      chunk_samples = window_size;
    }

    memset(chunk, 0, sizeof(chunk));
    memcpy(chunk, context, context_size * sizeof(float));
    memcpy(chunk + context_size, model_audio + start, chunk_samples * sizeof(float));

    status = silero_vad_model_forward(model, chunk, &prob);
    if (status != SILERO_VAD_STATUS_OK) {
      return status;
    }

    speech_probabilities[out_index++] = prob;
    memcpy(context, chunk + (chunk_size - context_size), context_size * sizeof(float));
  }

  *speech_probabilities_written = out_index;
  return SILERO_VAD_STATUS_OK;
}
