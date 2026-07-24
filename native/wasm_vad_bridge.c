#include "wasm_vad_bridge.h"

#include <stdlib.h>
#include <string.h>

#include "silero_vad.h"
#include "silero_vad_weights.h"

struct WasmVad {
  SileroVadModel *model;
  float *pending;
  size_t pending_count;
  size_t source_window_samples;
};

WasmVad *wasm_vad_create(size_t sampling_rate) {
  WasmVad *vad;

  if (sampling_rate == 0) {
    return NULL;
  }

  vad = (WasmVad *)calloc(1, sizeof(*vad));
  if (vad == NULL) {
    return NULL;
  }

  vad->model = silero_vad_model_create_with_sample_rate(
      silero_vad_get_embedded_weights(),
      SILERO_VAD_INPUT_SAMPLES,
      sampling_rate);
  if (vad->model == NULL) {
    free(vad);
    return NULL;
  }

  vad->source_window_samples =
      silero_vad_model_get_source_window_samples(vad->model);
  vad->pending =
      (float *)calloc(vad->source_window_samples, sizeof(*vad->pending));
  if (vad->pending == NULL) {
    silero_vad_model_destroy(vad->model);
    free(vad);
    return NULL;
  }

  return vad;
}

void wasm_vad_destroy(WasmVad *vad) {
  if (vad == NULL) {
    return;
  }

  silero_vad_model_destroy(vad->model);
  free(vad->pending);
  free(vad);
}

void wasm_vad_reset(WasmVad *vad) {
  if (vad == NULL) {
    return;
  }

  silero_vad_model_reset(vad->model);
  vad->pending_count = 0;
}

size_t wasm_vad_source_window_samples(const WasmVad *vad) {
  return vad == NULL ? 0 : vad->source_window_samples;
}

int wasm_vad_push(WasmVad *vad,
                  const float *samples,
                  size_t sample_count,
                  float *probabilities,
                  size_t probabilities_capacity) {
  size_t required_probabilities;
  size_t input_offset = 0;
  size_t output_count = 0;

  if (vad == NULL || probabilities == NULL ||
      (samples == NULL && sample_count != 0)) {
    return -1;
  }

  required_probabilities =
      (vad->pending_count + sample_count) / vad->source_window_samples;
  if (probabilities_capacity < required_probabilities) {
    return -2;
  }

  while (input_offset < sample_count) {
    size_t available = sample_count - input_offset;
    size_t needed = vad->source_window_samples - vad->pending_count;
    size_t copied = available < needed ? available : needed;

    memcpy(vad->pending + vad->pending_count,
           samples + input_offset,
           copied * sizeof(*samples));
    vad->pending_count += copied;
    input_offset += copied;

    if (vad->pending_count == vad->source_window_samples) {
      float probability = 0.0f;
      SileroVadStatus status = silero_vad_model_forward_source_chunk(
          vad->model,
          vad->pending,
          vad->source_window_samples,
          &probability);
      if (status != SILERO_VAD_STATUS_OK) {
        return -100 - (int)status;
      }

      probabilities[output_count++] = probability;
      vad->pending_count = 0;
    }
  }

  return (int)output_count;
}
