import createSileroVadModule from "./dist/silero-vad.js";

const MAX_GRAPH_POINTS = 240;

const elements = {
  startButton: document.querySelector("#start-button"),
  stopButton: document.querySelector("#stop-button"),
  statusDot: document.querySelector("#status-dot"),
  statusText: document.querySelector("#status-text"),
  probability: document.querySelector("#probability-value"),
  sampleRate: document.querySelector("#sample-rate-value"),
  window: document.querySelector("#window-value"),
  canvas: document.querySelector("#probability-graph"),
  emptyState: document.querySelector("#empty-state"),
  error: document.querySelector("#error-message"),
};

let module;
let vad;
let audioContext;
let mediaStream;
let sourceNode;
let workletNode;
let sinkNode;
let probabilities = [];
let starting = false;

/**
 * Small JavaScript owner for the C VAD instance and its WASM-side buffers.
 *
 * Emscripten exports C pointers as numeric byte offsets into WebAssembly
 * memory. This wrapper keeps those pointers private, copies microphone samples
 * through `HEAPF32`, and converts native output back to a JavaScript array.
 *
 * The C bridge keeps incomplete audio windows between calls to `push()`.
 * Call `destroy()` when the stream ends to release both the model and the
 * temporary WASM allocations.
 */
class WasmVad {
  /**
   * Creates a model configured for the browser's actual input sample rate.
   *
   * The native model resamples to 16 kHz internally when necessary.
   *
   * @param {object} wasmModule Initialized Emscripten module.
   * @param {number} sampleRate Microphone sample rate in Hz.
   */
  constructor(wasmModule, sampleRate) {
    this.module = wasmModule;
    this.handle = wasmModule._wasm_vad_create(sampleRate);
    if (!this.handle) {
      throw new Error("Could not create the Silero VAD model.");
    }

    this.windowSamples =
      wasmModule._wasm_vad_source_window_samples(this.handle);
    this.inputPointer = 0;
    this.inputCapacity = 0;
    this.outputPointer = 0;
    this.outputCapacity = 0;
  }

  /**
   * Sends an arbitrary number of mono float samples to the streaming VAD.
   *
   * A short input may return no values because the native bridge is still
   * filling a window. A longer input can complete several windows and return
   * multiple probabilities.
   *
   * @param {Float32Array} samples Normalized microphone samples.
   * @returns {number[]} Speech probabilities in the range 0 to 1.
   */
  push(samples) {
    const outputCapacity =
      Math.floor((samples.length + this.windowSamples - 1) / this.windowSamples) + 1;
    this.ensureInputCapacity(samples.length);
    this.ensureOutputCapacity(outputCapacity);

    this.module.HEAPF32.set(samples, this.inputPointer >>> 2);
    const written = this.module._wasm_vad_push(
      this.handle,
      this.inputPointer,
      samples.length,
      this.outputPointer,
      outputCapacity,
    );

    if (written < 0) {
      throw new Error(`VAD inference failed with bridge error ${written}.`);
    }

    return Array.from(
      this.module.HEAPF32.subarray(
        this.outputPointer >>> 2,
        (this.outputPointer >>> 2) + written,
      ),
    );
  }

  /**
   * Grows the reusable native input allocation when the next block needs it.
   *
   * @param {number} sampleCount Required number of float samples.
   * @private
   */
  ensureInputCapacity(sampleCount) {
    if (sampleCount <= this.inputCapacity) {
      return;
    }

    const newPointer = this.module._malloc(sampleCount * Float32Array.BYTES_PER_ELEMENT);
    if (!newPointer) {
      throw new Error("Could not allocate WASM input buffer.");
    }

    if (this.inputPointer) {
      this.module._free(this.inputPointer);
    }

    this.inputPointer = newPointer;
    this.inputCapacity = sampleCount;
  }

  /**
   * Grows the reusable native output allocation when the next call needs it.
   *
   * @param {number} probabilityCount Required number of float values.
   * @private
   */
  ensureOutputCapacity(probabilityCount) {
    if (probabilityCount <= this.outputCapacity) {
      return;
    }

    const newPointer = this.module._malloc(probabilityCount * Float32Array.BYTES_PER_ELEMENT);
    if (!newPointer) {
      throw new Error("Could not allocate WASM output buffer.");
    }

    if (this.outputPointer) {
      this.module._free(this.outputPointer);
    }
    
    this.outputPointer = newPointer;
    this.outputCapacity = probabilityCount;
  }

  /**
   * Releases all C and WASM allocations owned by this wrapper.
   *
   * Calling this method more than once is safe.
   */
  destroy() {
    if (this.inputPointer) {
      this.module._free(this.inputPointer);
    }
    if (this.outputPointer) {
      this.module._free(this.outputPointer);
    }
    if (this.handle) {
      this.module._wasm_vad_destroy(this.handle);
    }
    this.handle = 0;
  }
}

function setStatus(text, state = "") {
  elements.statusText.textContent = text;
  elements.statusDot.className = `status-dot ${state}`.trim();
}

function showError(error) {
  elements.error.textContent = error instanceof Error ? error.message : String(error);
  elements.error.hidden = false;
  setStatus("Error", "error");
}

function clearError() {
  elements.error.hidden = true;
  elements.error.textContent = "";
}

function addProbability(probability) {
  probabilities.push(probability);
  if (probabilities.length > MAX_GRAPH_POINTS) {
    probabilities = probabilities.slice(-MAX_GRAPH_POINTS);
  }
  elements.probability.textContent = probability.toFixed(3);
  elements.emptyState.classList.add("hidden");
  drawGraph();
}

function drawGraph() {
  const canvas = elements.canvas;
  const bounds = canvas.getBoundingClientRect();
  const pixelRatio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(bounds.width * pixelRatio));
  const height = Math.max(1, Math.round(bounds.height * pixelRatio));

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const context = canvas.getContext("2d");
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, bounds.width, bounds.height);

  const padding = { top: 22, right: 18, bottom: 20, left: 42 };
  const graphWidth = bounds.width - padding.left - padding.right;
  const graphHeight = bounds.height - padding.top - padding.bottom;

  context.font = "11px ui-sans-serif, system-ui, sans-serif";
  context.fillStyle = "#737873";
  context.textAlign = "right";
  context.textBaseline = "middle";

  for (let value = 0; value <= 1; value += 0.25) {
    const y = padding.top + graphHeight * (1 - value);
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(padding.left + graphWidth, y);
    context.strokeStyle = value === 0.5 ? "#d15f4e" : "rgba(23, 32, 29, 0.10)";
    context.setLineDash(value === 0.5 ? [6, 5] : []);
    context.lineWidth = value === 0.5 ? 1.5 : 1;
    context.stroke();
    context.fillText(value.toFixed(2), padding.left - 8, y);
  }

  context.setLineDash([]);
  if (probabilities.length === 0) {
    return;
  }

  context.beginPath();
  probabilities.forEach((probability, index) => {
    const x =
      padding.left +
      ((MAX_GRAPH_POINTS - probabilities.length + index) /
        Math.max(1, MAX_GRAPH_POINTS - 1)) *
        graphWidth;
    const y = padding.top + (1 - probability) * graphHeight;
    if (index === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  });
  context.strokeStyle = "#20778a";
  context.lineWidth = 2.5;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.stroke();

  const latest = probabilities[probabilities.length - 1];
  const latestX =
    padding.left +
    ((MAX_GRAPH_POINTS - 1) / Math.max(1, MAX_GRAPH_POINTS - 1)) * graphWidth;
  const latestY = padding.top + (1 - latest) * graphHeight;
  context.beginPath();
  context.arc(latestX, latestY, 4, 0, Math.PI * 2);
  context.fillStyle = latest >= 0.5 ? "#d15f4e" : "#20778a";
  context.fill();
}

async function start() {
  if (starting || audioContext) {
    return;
  }

  starting = true;
  clearError();
  elements.startButton.disabled = true;
  setStatus("Requesting microphone");

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    audioContext = new AudioContext({ latencyHint: "interactive" });
    await audioContext.audioWorklet.addModule("./mic-processor.js");
    await audioContext.resume();

    vad = new WasmVad(module, Math.round(audioContext.sampleRate));
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioContext, "mic-processor", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      channelCount: 1,
    });
    sinkNode = audioContext.createGain();
    sinkNode.gain.value = 0;
    workletNode.port.onmessage = ({ data }) => {
      try {
        vad.push(data).forEach(addProbability);
      } catch (error) {
        showError(error);
        stop();
      }
    };
    sourceNode.connect(workletNode);
    workletNode.connect(sinkNode);
    sinkNode.connect(audioContext.destination);

    probabilities = [];
    elements.probability.textContent = "0.000";
    elements.sampleRate.textContent = `${Math.round(audioContext.sampleRate).toLocaleString()} Hz`;
    elements.window.textContent = `${vad.windowSamples.toLocaleString()} samples`;
    elements.stopButton.disabled = false;
    setStatus("Listening", "recording");
    drawGraph();
  } catch (error) {
    await stop();
    showError(error);
  } finally {
    starting = false;
    elements.startButton.disabled = Boolean(audioContext);
  }
}

async function stop() {
  elements.stopButton.disabled = true;

  if (workletNode) {
    workletNode.port.onmessage = null;
    workletNode.disconnect();
    workletNode = null;
  }
  if (sinkNode) {
    sinkNode.disconnect();
    sinkNode = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  if (audioContext) {
    const context = audioContext;
    audioContext = null;
    await context.close();
  }
  if (vad) {
    vad.destroy();
    vad = null;
  }

  elements.startButton.disabled = !module;
  setStatus(module ? "Ready" : "Loading model", module ? "ready" : "");
}

elements.startButton.addEventListener("click", start);
elements.stopButton.addEventListener("click", stop);
window.addEventListener("resize", drawGraph);
window.addEventListener("beforeunload", () => {
  if (vad) {
    vad.destroy();
  }
});

drawGraph();

try {
  module = await createSileroVadModule();
  elements.startButton.disabled = false;
  setStatus("Ready", "ready");
} catch (error) {
  showError(new Error(`Could not load WebAssembly: ${error.message}`));
}
