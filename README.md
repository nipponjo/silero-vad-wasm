# Silero VAD WebAssembly demo

A small browser microphone demo for the [C port](https://github.com/nipponjo/silero-vad-c) of
[Silero VAD](https://github.com/snakers4/silero-vad). It compiles the native
implementation vendored from
[silero-vad-crs](https://github.com/nipponjo/silero-vad-crs) with Emscripten,
runs inference locally in WebAssembly, and plots the live speech probability.

## Run

The demo must be served over HTTP; opening `index.html` directly will not load the ES module and WASM file correctly.

Run with Python 3, Node.js, or another local static-file server:

```sh
python -m http.server 8080
```
or 
```sh
npx http-server . -p 8080
```

Then open <http://localhost:8080>. Microphone access is available on localhost.
Press **Start**, allow microphone access, and speak to see probability values
between 0 and 1.

## Preview
<img width="1944" height="1560" alt="626343858-a21e244d-2a57-4231-ad69-8164bfc39938" src="https://github.com/user-attachments/assets/ebec49d1-6ebe-4a1c-9591-78c1c1189133" />


## Build

### Requirements

- [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html)
- A browser with WebAssembly, Web Audio, AudioWorklet, and microphone support

Activate the Emscripten environment before building. 

On macOS or Linux:

```sh
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
```

On Windows PowerShell:

```powershell
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
.\emsdk install latest
.\emsdk activate latest
.\emsdk_env.ps1
```

### Compile

Run this command from the repository root:

```sh
emcc native/silero_vad.c \
  native/silero_vad_weights.c \
  native/dsp/resample.c \
  native/wasm_vad_bridge.c \
  -Inative \
  -O3 \
  -msimd128 \
  -DSILERO_VAD_ENABLE_WASM_SIMD=1 \
  -sWASM=1 \
  -sMODULARIZE=1 \
  -sEXPORT_ES6=1 \
  -sEXPORT_NAME=createSileroVadModule \
  -sENVIRONMENT=web \
  -sALLOW_MEMORY_GROWTH=1 \
  -sFILESYSTEM=0 \
  -sEXPORTED_RUNTIME_METHODS='["HEAPF32"]' \
  -sEXPORTED_FUNCTIONS='["_wasm_vad_create","_wasm_vad_destroy","_wasm_vad_reset","_wasm_vad_source_window_samples","_wasm_vad_push","_malloc","_free"]' \
  -o dist/silero-vad.js
```

On Windows PowerShell:

```powershell
emcc native/silero_vad.c `
  native/silero_vad_weights.c `
  native/dsp/resample.c `
  native/wasm_vad_bridge.c `
  -Inative `
  -O3 `
  -msimd128 `
  -DSILERO_VAD_ENABLE_WASM_SIMD=1 `
  -sWASM=1 `
  -sMODULARIZE=1 `
  -sEXPORT_ES6=1 `
  -sEXPORT_NAME=createSileroVadModule `
  -sENVIRONMENT=web `
  -sALLOW_MEMORY_GROWTH=1 `
  -sFILESYSTEM=0 `
  '-sEXPORTED_RUNTIME_METHODS=["HEAPF32"]' `
  '-sEXPORTED_FUNCTIONS=["_wasm_vad_create","_wasm_vad_destroy","_wasm_vad_reset","_wasm_vad_source_window_samples","_wasm_vad_push","_malloc","_free"]' `
  -o dist/silero-vad.js
```

This creates `dist/silero-vad.js` and `dist/silero-vad.wasm`.

To build without WebAssembly SIMD, remove `-msimd128` and
`-DSILERO_VAD_ENABLE_WASM_SIMD=1`. The scalar implementation is selected automatically.

## How it works

`mic-processor.js` copies mono microphone blocks from an AudioWorklet to the
main thread. `app.js` writes those `Float32Array` samples into WASM memory and
calls `wasm_vad_push()`. The C bridge buffers partial input until it has one
source-rate window and then invokes the native streaming VAD API. The C model
resamples non-16 kHz browser audio internally and returns one probability per
32 ms model window.

The model weights are compiled into the WASM binary from
`native/silero_vad_weights.c`; no model download or ONNX runtime is needed.
