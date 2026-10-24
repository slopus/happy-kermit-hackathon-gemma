---
license: apache-2.0
language: [en]
pipeline_tag: voice-activity-detection
tags: [turn-detection, end-of-turn, endpointing, turn-taking, voice-activity-detection, voice-agents, conversational-ai, real-time, streaming, on-device, edge, onnx, eot-bench]
model-index:
- name: turn-1-mini
  results:
  - task:
      type: audio-classification
      name: End-of-turn detection
    dataset:
      name: eot-bench (English)
      type: livekit/eot-bench-data
      config: en
      split: validation
      revision: ca9d98a
    metrics:
    - type: false_cutoff_rate_300ms
      name: False cut-offs at 300 ms (%)
      value: 22.6
    - type: false_cutoff_rate_600ms
      name: False cut-offs at 600 ms (%)
      value: 9.6
    - type: latency_at_5pct_false_cutoffs_ms
      name: Latency at 5% false cut-offs (ms)
      value: 849
    - type: latency_at_10pct_false_cutoffs_ms
      name: Latency at 10% false cut-offs (ms)
      value: 581
---
# turn-1-mini

**A 5.3M-parameter model that knows when you have finished talking.** Streaming, audio only, no transcript needed.
7 MB as an 8-bit ONNX file, 1.3 ms per 160 ms step on one laptop core, Apache-2.0.

Ahead of every downloadable model on LiveKit's public eot-bench (English), on point estimates:

| Model | False cut-offs @ 300 ms | Latency @ 5% cut-offs | Size |
|---|---:|---:|---:|
| **turn-1-mini** | **22.6%** | **849 ms** | 5.3M parameters |
| ultraVAD | 27.7% | 899 ms | |
| LiveKit Turn Detector v1-mini | 27.8% | 1,070 ms | |
| Smart Turn v3.2 | 35.2% | 1,051 ms | 8M parameters |

Lower is better. Full table, intervals and limits are below.

## Quick start

No PyTorch needed:

```bash
pip install "p99turn[onnx] @ https://huggingface.co/p99lab/turn-1-mini/resolve/main/dist/p99turn-1.2.3-py3-none-any.whl"
```
```python
from p99turn.onnx_stream import Turn1MiniOnnx

turn = Turn1MiniOnnx.from_pretrained()        # downloads the 7 MB graph
for chunk in microphone_chunks():             # 16 kHz mono float32, any chunk size
    probs = turn.push(chunk)                  # one P(end of turn) per finished 20 ms frame
    if len(probs) and probs[-1] > 0.5:        # calibrate this threshold on your own audio
        respond()
```

With PyTorch: install `p99turn[torch]` from the same link, then `from p99turn import Turn1Mini`.

Made by p99lab (p99lab.com), which today works on turn-taking that runs on the device.
Version 1.2.3, October 2026. Questions and issues: research@p99lab.com

## What it is

turn-1-mini is p99lab's own model: the architecture, the weights and the training are ours. It contains no
third-party weights and needs none at run time.

| | |
|---|---|
| Parameters | 5,262,337 (trunk 5.0M + head 0.26M) |
| Size | 21 MB (float32, `turn-1-mini.safetensors`) or 5.4 MB (8-bit, `turn-1-mini.int8.safetensors`, slightly lossy: see the evaluation section). ONNX streaming graphs of 21.8 MB and 7.0 MB are in `onnx/` |
| Input | 16 kHz mono audio, **low-passed at 4 kHz**, 80-bin log-mel, 10 ms hop, causal framing |
| Trunk | 2 causal convolutions (20 ms frames) + 6 transformer layers, width 256, 4 heads, causal attention limited to 1.2 s per layer, relative positions (ALiBi). No look-ahead |
| Head | MLP on layers 2, 4, 6 at the current frame + the mean of layer 6 from 1.2 s to 0.2 s back |
| Output | P(end of turn) per 20 ms frame; benchmarked read-out: any-time (the score can be read at any frame of a pause; no fixed score point) |
| Streaming | 160 ms steps with cached attention state; equals the offline forward to 1e-4 (tested) |

**The 4 kHz low-pass is required.** The model was trained on audio band-limited to 4 kHz. Feed it wideband
microphone audio without the low-pass and it is out of distribution. `p99turn.audio.bandlimit_4k` is the filter
behind the benchmark numbers. From version 1.2.0 the live stream (`model.stream()`) applies the same filter causally,
so live scores follow the benchmarked path; earlier versions used a different live filter.

```bash
pip install "p99turn[torch] @ https://huggingface.co/p99lab/turn-1-mini/resolve/main/dist/p99turn-1.2.3-py3-none-any.whl"
```
```python
from p99turn import Turn1Mini
model = Turn1Mini()                         # downloads p99lab/turn-1-mini; or set P99TURN_WEIGHTS=/local/dir
p = model.score(audio_so_far)             # as benchmarked: low-pass + run from a zero state
s = model.stream()                        # live: probs = s.push(chunk)  -> one value per finished 20 ms frame
```

## Training data

turn-1-mini was trained on read, conversational and telephone speech in English. No training data is redistributed
with this release. eot-bench data was used for evaluation only, never for training.

## Evaluation: eot-bench English

`livekit/eot-bench-data`, config `en`, split `validation`, revision `ca9d98a`, harness commit `9ee21b5`: 400 turns,
705 scored mid-turn pauses. One run, any-time read-out. Lower is better.

| | False cut-offs @ 300 ms | False cut-offs @ 600 ms | Latency @ 5% | Latency @ 10% | Score AUC (diagnostic) |
|---|---:|---:|---:|---:|---:|
| **turn-1-mini** (5.3M) | **22.6%** | **9.6%** | **849 ms** | **581 ms** | 0.970 |
| 95% interval (turn-level bootstrap, 400 resamples) | 18.8 to 27.8% | 6.3 to 12.1% | 678 to 1,012 ms | 462 to 672 ms | |
| ultraVAD (open weights) | 27.7% | 11.9% | 899 ms | 663 ms | |
| LiveKit Turn Detector v1-mini (downloadable) | 27.8% | 12.1% | 1,070 ms | 698 ms | 0.890 |
| Smart Turn v3.2 (open weights, 8M) | 35.2% | 14.8% | 1,051 ms | 739 ms | 0.845 |
| *For scale: LiveKit Turn Detector v1 (hosted, board leader)* | *9.9%* | *4.5%* | *543 ms* | *295 ms* | *0.969* |

What this supports, and what it does not:

- On these four measures turn-1-mini is **ahead of every model on the board whose weights can be downloaded**
  (ultraVAD, LiveKit v1-mini, Smart Turn v3.2, VAP), on point estimates. No paired interval against them was
  computed, and its own interval at 300 ms (18.8 to 27.8%) overlaps LiveKit v1-mini's (20.7 to 32.9%).
- Counted alone against the 13 rows of the board at that commit it would rank 4th / 4th / 4th / 5th.
- It is behind the large hosted models (LiveKit v1, JoinIn Baton), which run in a data centre.
- An 8-bit build of the same weights (5.4 MB, per-channel int8) scores 23.8% / 9.5% / 853 ms / 583 ms: about
  one point worse at 300 ms, unchanged on the other three measures. Load it with `Turn1Mini(variant="int8")`.
- English only. No other language was run for this checkpoint and no multilingual number is claimed.

Latency here is dead air chosen by the benchmark's policy sweep, not compute time. As for every model on the board,
the sweep picks threshold, action delay and timeout on the same 400 turns it scores.

## Evaluation discipline

eot-bench audio and labels were never used to train, select or threshold this model. The checkpoint and its read-out
were fixed beforehand on our own development data, and the checkpoint was scored on eot-bench English once. The
released package reproduces the numbers above.

## Compute per decision

About 0.57 GFLOP per second of audio; fixed memory. Streaming cost is constant per 160 ms step.

Measured on a laptop CPU (Apple M4 Pro, one core, PyTorch 2.14, float32; 3,000 streaming steps after warm-up): one
160 ms step, including the low-pass and the log-mel, takes **1.29 ms at the median and 1.45 ms at the 95th
percentile**, which is under 1% of one core. More threads do not help a model this small. The model loads in 17 ms.
On a server GPU (NVIDIA GB10) a full-prefix decision takes about 8 ms.

Not measured: an iPhone, a Core ML or ExecuTorch export, battery use, and end-to-end latency of a full voice
pipeline. This is the model's compute only.

## ONNX (no PyTorch needed)

`onnx/` holds one 160 ms streaming step of the model as an ONNX graph with fixed shapes: raw 16 kHz audio and the
previous state in, 8 frame probabilities and the new state out. The low-pass and the log-mel are inside the graph. It
runs with `onnxruntime` and `numpy` only. The float32 graph matches the PyTorch stream to 0.00004 on 337 s of audio. The
8-bit graph (7.0 MB) changes about 0.2% of decisions on our development audio and has not been run on eot-bench.
Interface, state handling and an example: `onnx/README.md`. A Core ML build is not available yet.

## TurnBench

turn-1-mini is also the model behind our entry on Sesame's TurnBench, submitted on 2026-10-05. Write-up:
https://github.com/P99Lab/turn-1-mini

## Intended use

On-device end-of-turn detection inside a voice SDK, as one signal in an endpointing policy (threshold + minimum
silence + timeout), where a server model is not available or as a first stage before one.

## Limits

- **English only measured.** Other languages are untested for this checkpoint.
- **4 kHz band only** (see the low-pass requirement above).
- **Scores are not calibrated.** The 300 ms operating point uses a threshold of 0.06. Calibrate thresholds on your
  own audio.
- Audio only: it does not read words. At its 300 ms operating point it cuts off 27 of 126 pauses inside numbers,
  letters and email addresses, 36 of 119 after a finished sentence with more to come, 54 of 214 after a content word
  mid-sentence, and 40 of 68 pauses longer than 1 s.
- No accent, age or gender breakdown; not evaluated with overlapping speech, far-field audio or agent echo.

## Responsible use

turn-1-mini outputs one number about turn-taking. It does not identify speakers, transcribe, or infer emotion,
health or identity. A false "end of turn" interrupts a person and a missed one makes them wait: keep a timeout
fallback and do not rely on it alone where either could cause harm. Performance for accents, speech impairments
and non-English speakers is unmeasured; test on your own users.

## How to credit

If you use this model or build on it, please credit "turn-1-mini by p99lab" with a link to
https://huggingface.co/p99lab/turn-1-mini. Redistributed copies and derived models must keep the `NOTICE` file
(Apache-2.0, section 4). Please do not name a derived model "turn-1-mini".

## Licence

Weights and code: Apache-2.0. Credits required by third-party licences: see `NOTICE`.

## Citation

```bibtex
@misc{p99lab2026turn1mini,
  title  = {turn-1-mini: a 5M-parameter streaming end-of-turn model},
  author = {p99lab},
  year   = {2026},
  url    = {https://p99lab.com}
}
```
Please also cite eot-bench (LiveKit) when reporting numbers.
