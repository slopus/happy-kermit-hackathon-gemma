# turn-1-mini: ONNX streaming graphs

One 160 ms step of turn-1-mini as an ONNX graph with fixed shapes. You feed 2,560 samples of 16 kHz mono audio and the
state from the previous step; you get 8 values of P(end of turn), one per 20 ms frame, and the new state. Nothing else
is needed at run time: the 4 kHz low-pass and the log-mel are inside the graph, and no Python logic is involved beyond
feeding the state back. Opset 17. Runs with onnxruntime on CPU; needs only `numpy` and `onnxruntime`, not PyTorch.

| File | Size | What |
|---|---:|---|
| `turn-1-mini.step.onnx` | 21.8 MB | float32. Raw audio in; low-pass and log-mel in the graph. **Use this one unless you need a smaller file** |
| `turn-1-mini.step.nolp.onnx` | 21.8 MB | float32, without the low-pass: only for audio that is already band-limited to 4 kHz |
| `turn-1-mini.step.int8.onnx` | 6.2 MB | The 8-bit weights of the 5.4 MB build (per-channel int8), turned back into float32 when the session loads. Small download, float32 arithmetic and speed |
| `turn-1-mini.step.int8dyn.onnx` | 7.0 MB | onnxruntime dynamic quantization of the float32 graph: 8-bit weights and integer matrix products at run time. Small and fast. Not benchmarked on eot-bench |

Checksums: `SHA256SUMS`. All measurements below: `results.json`, produced by `verify.py`.

## Interface

All four graphs have the same names. All tensors are float32. Shapes are fixed.

| Input | Shape | Meaning |
|---|---|---|
| `audio` | [1, 2560] | The next 160 ms of 16 kHz mono audio, in [-1, 1] |
| `state_audio` | [1, 304] ([1, 240] in the `nolp` graph) | Most recent input samples |
| `state_conv1` | [1, 80, 2] | Last two log-mel frames |
| `state_conv2` | [1, 256, 2] | Last two frames after the first convolution |
| `state_k`, `state_v` | [6, 4, 60, 64] each | Attention keys and values of the last 60 frames (1.2 s), per layer and head |
| `state_k_valid` | [60] | 1 where a cached frame exists, 0 before the start of the stream |
| `state_h6` | [60, 256] | Last-layer output of the last 60 frames |

| Output | Shape | Meaning |
|---|---|---|
| `probs` | [8] | P(end of turn) for the 8 frames of this step, oldest first; `probs[7]` is "now" |
| `new_state_*` | as the matching input | Feed each one back as `state_*` at the next step |

**State handling.** Every state tensor starts as zeros: that is the start of a stream. After each step, pass every
`new_state_<name>` back as `state_<name>`, unchanged. To start a new stream (a new call, or after a long gap), reset them
to zeros. One set of state tensors per audio stream; the session itself can be shared. Memory per stream: 0.8 MB.

**Timing.** The model answers once per 160 ms step, with 8 values. If you need an answer at a moment that is not a step
boundary, pad the audio you have with zeros to 2,560 samples and run the step on a *copy* of the state; keep the real
state for when the full chunk has arrived.

**The low-pass.** The model needs audio band-limited to 4 kHz. `turn-1-mini.step.onnx` does that itself, with the same
filter as the benchmarked PyTorch path (16 kHz to 8 kHz and back), run causally: at each step the most recent audio is
filtered again, with the audio that has not arrived yet counted as zero. No delay is added. Do not put another
low-pass in front. A generic FIR low-pass in front of the `nolp` graph is **not** equivalent: with a 65-tap FIR we
measured 3 to 7% of decisions changing (table below).

## Example

```python
import numpy as np
import onnxruntime as ort
import soundfile as sf

STEP = 2560                                                     # 160 ms at 16 kHz
STATES = ["audio", "conv1", "conv2", "k", "v", "k_valid", "h6"]

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1                                   # one thread is enough for a model this small
sess = ort.InferenceSession("turn-1-mini.step.onnx", opts, providers=["CPUExecutionProvider"])
outputs = ["probs"] + [f"new_state_{s}" for s in STATES]


def new_state():                                                # zeros = start of a stream
    return {i.name: np.zeros(i.shape, np.float32) for i in sess.get_inputs() if i.name != "audio"}


audio, sr = sf.read("call.wav", dtype="float32")                # 16 kHz mono
assert sr == 16000 and audio.ndim == 1
state = new_state()
threshold = 0.5                                                 # calibrate on your own audio: the scores are not calibrated

for start in range(0, len(audio) - STEP + 1, STEP):
    chunk = audio[start:start + STEP].reshape(1, STEP)
    probs, *new = sess.run(outputs, {"audio": chunk, **state})
    state = {f"state_{s}": v for s, v in zip(STATES, new)}      # feed the state back, unchanged
    t = (start + STEP) / sr
    if probs[-1] > threshold:
        print(f"{t:7.2f} s  P(end of turn) = {probs[-1]:.2f}  -> the user may be done")
```

`p99turn_proposed/onnx_stream.py` wraps the same loop in a small class (`Turn1MiniOnnx.push(chunk)` for audio of any length).

## Does it match the PyTorch model?

337 s of public English read speech, 16,864 frames, one continuous stream, onnxruntime 1.30 on CPU:

| Compared | Largest difference in P(end of turn) | Mean |
|---|---:|---:|
| `step.onnx` against the PyTorch stream (low-pass on) | 0.00004 | 0.000001 |
| `step.nolp.onnx` against the PyTorch stream (band-limited input) | 0.00002 | 0.0000008 |
| `step.nolp.onnx` against the PyTorch offline forward (first 60 s) | 0.00002 | 0.000001 |
| `step.int8.onnx` against the PyTorch model with the same 8-bit weights | 0.00004 | 0.000001 |

State: feeding the audio in chunks of 1,000 samples gives bit-identical output to one call; a reset gives bit-identical
output to a fresh session; dropping the state changes the scores (by up to 0.39), so the state is doing its job.

## The 8-bit graphs against float32

8,326 decision moments in public English audio that the model was not trained on (6,826 inside pauses of simulated
telephone calls, 1,500 at random points of read speech). Reference: `step.onnx`. "Changed decisions" = the share of
moments where the score falls on the other side of a threshold; thresholds 0.06 / 0.16 / 0.17 / 0.21 are the operating
points of the float32 model in its eot-bench run.

| Graph | Mean score difference | Largest | Changed decisions at 0.06 / 0.16 / 0.17 / 0.21 | at 0.5 |
|---|---:|---:|---|---:|
| `step.int8.onnx` | 0.0048 | 0.105 | 0.34% / 0.26% / 0.23% / 0.32% | 0.41% |
| `step.int8dyn.onnx` | 0.0022 | 0.097 | 0.24% / 0.23% / 0.19% / 0.25% | 0.20% |

On the read-speech part alone both reach 0.7% at some thresholds (1,500 moments, so 0.07% is one decision).
eot-bench: the 8-bit weights of `step.int8.onnx` are the ones behind the published 8-bit result (23.8% / 9.5% / 853 ms /
583 ms). **`step.int8dyn.onnx` has not been run on eot-bench**; only the numbers above exist for it.

How the low-pass in front changes decisions, same moments, reference = the benchmarked PyTorch path:

| In front of the model | Changed decisions (range over the five thresholds), calls | read speech |
|---|---|---|
| The low-pass inside `step.onnx` | 0.00 to 0.06% | 0.1 to 0.7% |
| A 65-tap FIR low-pass (3.8 kHz), then `step.nolp.onnx` | 2.9 to 5.1% | 1.4 to 6.6% |
| No low-pass at all, `step.nolp.onnx` | 1.2 to 2.0% | 1.3 to 8.2% |

## Speed: a relative comparison only

Wall-clock time of one 160 ms step, one thread, low-pass and log-mel included, 3,000 steps after warm-up, all on the
same Arm server core (Cortex-X925 in an NVIDIA DGX Spark). **This is a development machine. These are not on-device
numbers** and say nothing about a phone; they only compare the runtimes with each other.

| Runtime | Median | 95th percentile |
|---|---:|---:|
| PyTorch 2.14, float32 | 6.6 ms | 8.6 ms |
| onnxruntime 1.30, `step.onnx` (float32) | 4.5 ms | 6.3 ms |
| onnxruntime 1.30, `step.int8.onnx` (8-bit file, float32 arithmetic) | 4.5 ms | 6.3 ms |
| onnxruntime 1.30, `step.int8dyn.onnx` (integer arithmetic) | 1.0 ms | 1.3 ms |

## Core ML

Not done here: a Core ML model has to be converted **and measured** on macOS. What it still needs:

1. Convert from PyTorch, not from these ONNX files: current `coremltools` does not read ONNX. `export_step.py` contains
   the module to convert (`StreamStep`: fixed shapes, state in and out as plain tensors, only convolutions, matrix
   products, layer norms, softmax, GELU, log). Trace it and convert to an ML Program; the 7 state tensors can stay
   explicit inputs and outputs, or become Core ML states on iOS 18 / macOS 15.
2. Check the converted model against the PyTorch stream on macOS, on minutes of audio, as above. Core ML defaults to
   float16 arithmetic: measure it, and compare with float32 precision before trusting it. The attention mask uses
   -1e9, which does not fit in float16 and must be clamped (about -1e4) for a float16 build.
3. Redo the 8-bit step in Core ML's own tooling (`coremltools.optimize`) and measure it again: the numbers above are
   for onnxruntime's arithmetic and do not carry over.
4. Measure time per step, memory and energy on the phone itself, on the CPU and on the Neural Engine.

The same four points apply to an ExecuTorch export.

## Files

`export_step.py` builds the four graphs from the weights. `verify.py` produces `results.json`. `tests/` checks the graphs
and the loader. `p99turn_proposed/` is a copy of the Python package with the changes proposed alongside these graphs.
