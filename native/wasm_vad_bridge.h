#ifndef WASM_VAD_BRIDGE_H
#define WASM_VAD_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WasmVad WasmVad;

WasmVad *wasm_vad_create(size_t sampling_rate);
void wasm_vad_destroy(WasmVad *vad);
void wasm_vad_reset(WasmVad *vad);
size_t wasm_vad_source_window_samples(const WasmVad *vad);
int wasm_vad_push(WasmVad *vad,
                  const float *samples,
                  size_t sample_count,
                  float *probabilities,
                  size_t probabilities_capacity);

#ifdef __cplusplus
}
#endif

#endif
