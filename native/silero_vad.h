#ifndef SILERO_VAD_H
#define SILERO_VAD_H

#include <stddef.h>

#include "dsp/resample.h"

/*
 * Standalone C port of the 16 kHz Silero VAD model.
 *
 * This API exposes both low-level reusable building blocks (Conv1d, LSTMCell)
 * and a full-model interface for chunked or full-audio inference.
 *
 * Current scope:
 * - model topology: fixed 16 kHz Silero VAD path
 * - full-audio input can be resampled to 16 kHz when initialized with a
 *   non-16 kHz sampling_rate
 * - streaming chunk shape: typically 576 samples
 *   (64 samples of left context + 512 new samples)
 *
 * Weight layout:
 * - weights are supplied through SileroVadWeights
 * - tensor shapes and ordering are expected to match the exporter scripts in
 *   this repository
 * - for parity with the Torch hub JIT model, use the JIT weight exporter
 */

#if defined(_WIN32)
#  if defined(SILERO_VAD_BUILD_DLL)
#    define SILERO_VAD_API __declspec(dllexport)
#  elif defined(SILERO_VAD_USE_DLL)
#    define SILERO_VAD_API __declspec(dllimport)
#  else
#    define SILERO_VAD_API
#  endif
#else
#  define SILERO_VAD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
  SILERO_VAD_SAMPLE_RATE = 16000,
  SILERO_VAD_STFT_FFT_SIZE = 256,
  SILERO_VAD_STFT_HOP_SIZE = 128,
  SILERO_VAD_STFT_RIGHT_PAD = 64,
  SILERO_VAD_CONTEXT_SAMPLES = 64,
  SILERO_VAD_WINDOW_SAMPLES = 512,
  SILERO_VAD_INPUT_SAMPLES = 576,
  SILERO_VAD_STFT_BINS = 129,
  SILERO_VAD_HIDDEN_SIZE = 128
};

typedef enum SileroVadStatus {
  SILERO_VAD_STATUS_OK = 0,
  SILERO_VAD_STATUS_INVALID_ARGUMENT = 1,
  SILERO_VAD_STATUS_ALLOCATION_FAILED = 2,
  SILERO_VAD_STATUS_INVALID_SHAPE = 3
} SileroVadStatus;

typedef struct SileroVadWeights {
  // const float *stft_weight;

  const float *conv1_weight;
  const float *conv1_bias;
  const float *conv2_weight;
  const float *conv2_bias;
  const float *conv3_weight;
  const float *conv3_bias;
  const float *conv4_weight;
  const float *conv4_bias;

  const float *lstm_weight_ih;
  const float *lstm_weight_hh;
  const float *lstm_bias_ih;
  const float *lstm_bias_hh;

  const float *final_conv_weight;
  const float *final_conv_bias;
} SileroVadWeights;

typedef struct SileroVadConv1d {
  size_t input_channels;
  size_t output_channels;
  size_t kernel_size;
  size_t stride;
  size_t padding;

  const float *weight;
  const float *bias;
  float *packed_weight_t0;
  float *packed_weight_t1;
  float *packed_weight_t2;

  float *padded_input;
  size_t padded_capacity;
  size_t max_input_frames;
} SileroVadConv1d;

typedef struct SileroVadLstmCell {
  size_t input_size;
  size_t hidden_size;

  const float *weight_ih;
  const float *weight_hh;
  const float *bias_ih;
  const float *bias_hh;

  float *packed_weight_ih;
  float *packed_weight_hh;
  float *hidden_state;
  float *cell_state;
  float *gates;
} SileroVadLstmCell;

/**
 * @brief Complete state and workspace for a Silero VAD model instance.
 *
 * Initialize this structure with silero_vad_model_init() or
 * silero_vad_model_init_with_sample_rate(). Its layer contexts, scratch
 * buffers, recurrent state, resampler, and streaming context are managed by
 * the model API and should not be modified directly after initialization.
 *
 * Weight arrays referenced by the layer contexts remain externally owned and
 * must outlive this structure. All other dynamically allocated resources are
 * model-owned and are released by silero_vad_model_free().
 */
typedef struct SileroVadModel {
  size_t input_samples; /**< Number of 16 kHz samples consumed by one direct inference step. */
  size_t sampling_rate; /**< Configured source-audio sample rate in Hz. */
  size_t source_window_samples; /**< Source samples corresponding to one 512-sample VAD window. */
  size_t stft_frames; /**< Number of STFT frames produced for one model input. */
  size_t conv2_frames; /**< Number of temporal frames produced by the second convolution. */
  size_t conv3_frames; /**< Number of temporal frames produced by the third convolution. */
  size_t conv4_frames; /**< Number of temporal frames produced by the fourth convolution. */

  SileroVadConv1d conv1; /**< First encoder convolution and its internal workspace. */
  SileroVadConv1d conv2; /**< Second encoder convolution and its internal workspace. */
  SileroVadConv1d conv3; /**< Third encoder convolution and its internal workspace. */
  SileroVadConv1d conv4; /**< Fourth encoder convolution and its internal workspace. */
  SileroVadConv1d final_conv; /**< Final projection from the LSTM output to one logit. */
  SileroVadLstmCell lstm; /**< Recurrent cell, including hidden and cell state. */

  float *padded_audio; /**< Model-owned input buffer including right padding. */
  float *fft_real; /**< Model-owned real-component workspace for the 256-point FFT. */
  float *fft_imag; /**< Model-owned imaginary-component workspace for the 256-point FFT. */
  float *stft_mag; /**< Model-owned STFT magnitude workspace. */
  float *conv1_out; /**< Model-owned output buffer for the first convolution. */
  float *conv2_out; /**< Model-owned output buffer for the second convolution. */
  float *conv3_out; /**< Model-owned output buffer for the third convolution. */
  float *conv4_out; /**< Model-owned output buffer for the fourth convolution. */
  float *lstm_out; /**< Model-owned output buffer for the current LSTM step. */
  float *final_out; /**< Model-owned buffer for the final projection logits. */
  float *fft_twiddle_cos; /**< Model-owned cosine twiddle-factor table for the FFT. */
  float *fft_twiddle_sin; /**< Model-owned sine twiddle-factor table for the FFT. */
  unsigned short *fft_bitrev; /**< Model-owned bit-reversal permutation table for the FFT. */
  yl_resample_ctx *resampler; /**< Source-to-16 kHz resampler, or NULL for 16 kHz source audio. */
  float *resampled_audio; /**< Model-owned scratch buffer for resampled source audio. */
  size_t resampled_audio_capacity; /**< Capacity of resampled_audio in float samples. */
  float stream_context[SILERO_VAD_CONTEXT_SAMPLES]; /**< Left context retained across source chunks. */
} SileroVadModel;

/* Computes the output frame count for a 1D convolution with symmetric padding. */
SILERO_VAD_API size_t silero_vad_conv1d_output_frames(size_t input_frames,
                                                      size_t kernel_size,
                                                      size_t stride,
                                                      size_t padding);

/*
 * Initializes a reusable Conv1d context.
 *
 * This is primarily exposed for low-level use. The full model API below is the
 * intended entry point for normal inference.
 */
SILERO_VAD_API SileroVadStatus silero_vad_conv1d_init(SileroVadConv1d *conv,
                                                      size_t input_channels,
                                                      size_t output_channels,
                                                      size_t kernel_size,
                                                      size_t stride,
                                                      size_t padding,
                                                      size_t max_input_frames,
                                                      const float *weight,
                                                      const float *bias);

/* Clears any reusable scratch buffers owned by the convolution context. */
SILERO_VAD_API void silero_vad_conv1d_reset(SileroVadConv1d *conv);
/* Releases memory owned by the convolution context. */
SILERO_VAD_API void silero_vad_conv1d_free(SileroVadConv1d *conv);

/* Runs a single Conv1d forward pass using the initialized context. */
SILERO_VAD_API SileroVadStatus silero_vad_conv1d_forward(SileroVadConv1d *conv,
                                                         const float *input,
                                                         size_t input_frames,
                                                         float *output);

/* Initializes a reusable LSTMCell context with persistent hidden/cell state. */
SILERO_VAD_API SileroVadStatus silero_vad_lstm_cell_init(SileroVadLstmCell *cell,
                                                         size_t input_size,
                                                         size_t hidden_size,
                                                         const float *weight_ih,
                                                         const float *weight_hh,
                                                         const float *bias_ih,
                                                         const float *bias_hh);

/* Resets the hidden state, cell state, and internal gate scratch buffers. */
SILERO_VAD_API void silero_vad_lstm_cell_reset(SileroVadLstmCell *cell);
/* Releases memory owned by the LSTM context. */
SILERO_VAD_API void silero_vad_lstm_cell_free(SileroVadLstmCell *cell);

/* Runs one LSTMCell step and writes the new hidden state to hidden_out. */
SILERO_VAD_API SileroVadStatus silero_vad_lstm_cell_forward(SileroVadLstmCell *cell,
                                                            const float *input,
                                                            float *hidden_out);

/**
 * @brief Initializes a caller-allocated Silero VAD model for 16 kHz input.
 *
 * The supplied weights must use the tensor layouts expected by
 * SileroVadWeights. For the standard streaming model, input_samples is
 * SILERO_VAD_INPUT_SAMPLES (576): 64 samples of left context followed by
 * 512 new samples.
 *
 * @param model Model storage to initialize.
 * @param weights Model weights. The pointed-to weight arrays must remain valid
 * for the lifetime of the model.
 * @param input_samples Number of float samples accepted by
 * silero_vad_model_forward().
 * @return SILERO_VAD_STATUS_OK on success, or an error status if an argument,
 * allocation, or model shape is invalid.
 * @note Release model-owned resources with silero_vad_model_free().
 */
SILERO_VAD_API SileroVadStatus silero_vad_model_init(SileroVadModel *model,
                                                     const SileroVadWeights *weights,
                                                     size_t input_samples);

/**
 * @brief Initializes a caller-allocated model for audio at a specified sample
 * rate.
 *
 * Input passed to silero_vad_model_forward_audio() or
 * silero_vad_model_forward_source_chunk() is resampled internally to 16 kHz
 * when sampling_rate differs from SILERO_VAD_SAMPLE_RATE. Direct calls to
 * silero_vad_model_forward() always operate on model-rate samples and do not
 * perform resampling.
 *
 * @param model Model storage to initialize.
 * @param weights Model weights. The pointed-to weight arrays must remain valid
 * for the lifetime of the model.
 * @param input_samples Number of 16 kHz float samples accepted by
 * silero_vad_model_forward(); normally SILERO_VAD_INPUT_SAMPLES.
 * @param sampling_rate Sample rate in Hz of source audio and source chunks.
 * @return SILERO_VAD_STATUS_OK on success, or an error status if an argument,
 * allocation, or model shape is invalid.
 * @note Release model-owned resources with silero_vad_model_free().
 */
SILERO_VAD_API SileroVadStatus silero_vad_model_init_with_sample_rate(SileroVadModel *model,
                                                                      const SileroVadWeights *weights,
                                                                      size_t input_samples,
                                                                      size_t sampling_rate);

/**
 * @brief Allocates and initializes a Silero VAD model for 16 kHz input.
 *
 * @param weights Model weights. The pointed-to weight arrays must remain valid
 * for the lifetime of the model.
 * @param input_samples Number of float samples accepted by
 * silero_vad_model_forward(); normally SILERO_VAD_INPUT_SAMPLES.
 * @return A newly allocated model, or NULL if validation or allocation fails.
 * @note Destroy the returned model with silero_vad_model_destroy().
 */
SILERO_VAD_API SileroVadModel *silero_vad_model_create(const SileroVadWeights *weights,
                                                       size_t input_samples);

/**
 * @brief Allocates and initializes a model for audio at a specified sample
 * rate.
 *
 * Source audio is resampled internally to 16 kHz when sampling_rate differs
 * from SILERO_VAD_SAMPLE_RATE.
 *
 * @param weights Model weights. The pointed-to weight arrays must remain valid
 * for the lifetime of the model.
 * @param input_samples Number of 16 kHz float samples accepted by
 * silero_vad_model_forward(); normally SILERO_VAD_INPUT_SAMPLES.
 * @param sampling_rate Sample rate in Hz of source audio and source chunks.
 * @return A newly allocated model, or NULL if validation or allocation fails.
 * @note Destroy the returned model with silero_vad_model_destroy().
 */
SILERO_VAD_API SileroVadModel *silero_vad_model_create_with_sample_rate(const SileroVadWeights *weights,
                                                                        size_t input_samples,
                                                                        size_t sampling_rate);

/**
 * @brief Resets the recurrent, streaming-context, and resampler state.
 *
 * Call this before starting an independent utterance or stream when using
 * stateful chunk inference. A NULL model is ignored.
 *
 * @param model Model to reset.
 */
SILERO_VAD_API void silero_vad_model_reset(SileroVadModel *model);

/**
 * @brief Releases resources owned by a caller-allocated model.
 *
 * This function does not free the SileroVadModel structure itself and does not
 * release the externally owned weight arrays. A NULL model is ignored.
 *
 * @param model Model whose internal resources are to be released.
 */
SILERO_VAD_API void silero_vad_model_free(SileroVadModel *model);

/**
 * @brief Releases and deallocates a dynamically created model.
 *
 * A NULL model is ignored.
 *
 * @param model Model returned by silero_vad_model_create() or
 * silero_vad_model_create_with_sample_rate().
 */
SILERO_VAD_API void silero_vad_model_destroy(SileroVadModel *model);

/**
 * @brief Runs one inference step on a model-rate chunk.
 *
 * The input must contain exactly the number of samples configured at
 * initialization. For the standard model this is 64 samples of caller-provided
 * left context followed by 512 new samples. This function updates the model's
 * recurrent state but does not resample input or manage left context.
 *
 * @param model Initialized model.
 * @param input Array of model->input_samples normalized float samples.
 * @param speech_probability Output location for a probability in the range
 * [0, 1].
 * @return SILERO_VAD_STATUS_OK on success, or an error status if an argument or
 * internal shape is invalid.
 */
SILERO_VAD_API SileroVadStatus silero_vad_model_forward(SileroVadModel *model,
                                                        const float *input,
                                                        float *speech_probability);

/**
 * @brief Runs one streaming inference step on a source-rate audio chunk.
 *
 * The function resamples the chunk when needed, prepends the model's internal
 * left context, performs inference, and saves context for the next call. The
 * model must have been initialized with SILERO_VAD_INPUT_SAMPLES.
 *
 * @param model Initialized model.
 * @param input Array of normalized float samples at the model's configured
 * source sample rate.
 * @param input_samples Number of input samples. This must equal
 * silero_vad_model_get_source_window_samples(model).
 * @param speech_probability Output location for a probability in the range
 * [0, 1].
 * @return SILERO_VAD_STATUS_OK on success, or an error status if an argument,
 * allocation, or chunk shape is invalid.
 */
SILERO_VAD_API SileroVadStatus silero_vad_model_forward_source_chunk(SileroVadModel *model,
                                                                     const float *input,
                                                                     size_t input_samples,
                                                                     float *speech_probability);

/**
 * @brief Calculates the number of probabilities produced for 16 kHz audio.
 *
 * Partial final windows count as one probability.
 *
 * @param audio_samples Number of input audio samples.
 * @return Required probability count, or 0 when audio_samples is 0.
 */
SILERO_VAD_API size_t silero_vad_model_audio_prob_count(size_t audio_samples);

/**
 * @brief Calculates the number of probabilities produced for source-rate
 * audio.
 *
 * The count is based on the resampled 16 kHz length, with a partial final
 * 512-sample window counting as one probability.
 *
 * @param audio_samples Number of source audio samples.
 * @param sampling_rate Source sample rate in Hz.
 * @return Required probability count, or 0 when audio_samples or sampling_rate
 * is 0.
 */
SILERO_VAD_API size_t silero_vad_model_audio_prob_count_for_sample_rate(size_t audio_samples,
                                                                        size_t sampling_rate);

/**
 * @brief Calculates the source-rate chunk size corresponding to one VAD
 * window.
 *
 * @param sampling_rate Source sample rate in Hz.
 * @return Rounded number of source samples corresponding to
 * SILERO_VAD_WINDOW_SAMPLES at 16 kHz, or 0 when sampling_rate is 0.
 */
SILERO_VAD_API size_t silero_vad_model_source_window_samples(size_t sampling_rate);

/**
 * @brief Returns the source-rate chunk size configured for a model.
 *
 * @param model Initialized model.
 * @return Number of source samples expected by
 * silero_vad_model_forward_source_chunk(), or 0 when model is NULL.
 */
SILERO_VAD_API size_t silero_vad_model_get_source_window_samples(const SileroVadModel *model);

/**
 * @brief Runs inference over a complete source-rate audio buffer.
 *
 * The model is reset at the start of the call. Audio is resampled to 16 kHz
 * when required, split into 512-sample windows, and processed using internal
 * 64-sample left context. A zero-padded partial final window produces one
 * probability.
 *
 * @param model Initialized model configured with SILERO_VAD_INPUT_SAMPLES.
 * @param audio Array of normalized float samples at the model's configured
 * source sample rate.
 * @param audio_samples Number of samples in audio.
 * @param speech_probabilities Output array receiving one probability per
 * window.
 * @param speech_probabilities_capacity Number of floats available in
 * speech_probabilities. It must be at least the value returned by
 * silero_vad_model_audio_prob_count_for_sample_rate().
 * @param speech_probabilities_written Output location receiving the number of
 * probabilities written.
 * @return SILERO_VAD_STATUS_OK on success, or an error status if an argument,
 * allocation, output capacity, or model shape is invalid.
 */
SILERO_VAD_API SileroVadStatus silero_vad_model_forward_audio(SileroVadModel *model,
                                                              const float *audio,
                                                              size_t audio_samples,
                                                              float *speech_probabilities,
                                                              size_t speech_probabilities_capacity,
                                                              size_t *speech_probabilities_written);

#ifdef __cplusplus
}
#endif

#endif
