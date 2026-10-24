# Breeze TTS 2 local GPU benchmark

Upstream source: https://github.com/breezeblue-ai/breeze-tts at commit `6eba901d1845353802652c708a4740fdfc5baa75`. Checkpoint: `BreezeBlue/Breeze-TTS-2`.

## Install and run

From `kermit/voice/`:

```bash
export HF_HOME="$PWD/.hf" UV_CACHE_DIR="$PWD/.uv-cache" CUDA_VISIBLE_DEVICES=0
uv venv --python /usr/bin/python3.12 .venvs/breeze
git clone --depth 1 https://github.com/breezeblue-ai/breeze-tts.git bench/breeze/upstream
uv pip install --python .venvs/breeze/bin/python -r bench/breeze/upstream/requirements.txt
.venvs/breeze/bin/python bench/breeze/run.py
```

Upstream pins torch/torchaudio 2.9.1, qwen-tts 0.1.1 and transformers 4.57.3. No flash-attn build. Eager attention, bf16, eager streaming (`fast_all=False`), CFG 4 for voice direction. GPU 0 is the only visible device. Benchmark models are run sequentially to avoid GPU contention.

## API and timing

The script uses upstream `load_runtime`, `prepare_inputs`, and `FastBreezeStreamingRuntime.iter_audio_chunks`. It provides `ref_audio_path`, exact `ref_text`, `instruction` for each emotion, and the demo `text`. Reference comes from `../qwen3/ref_frog.wav` and `ref_frog.txt`.

One untimed warmup precedes the demo lines. TTFA is measured from before input preparation through arrival of the first CPU audio chunk, including reference audio loading/tokenization and text processing. `input_prepare_s` makes that overhead explicit. `gen_s` includes preparation and all chunks; file writing is outside timing. Per-chunk arrival and duration are retained. Load time excludes checkpoint download. Peak allocated/reserved GPU memory and maximum waveform amplitude are included. Eager streaming emits two codec frames per chunk. These are Linux CUDA proxy results, not Apple Silicon numbers.

## Results

All six lines completed. Median TTFA is **0.709 s**, range **0.667–0.741 s**. Audio-duration-weighted RTF is **3.239** (median per-line RTF 3.252). Peak allocated VRAM is **8.072 GiB**, reserved **8.191 GiB**. Local model load took **39.750 s**. Input preparation took **0.075–0.081 s** per line. Complete generation times range **9.929–21.494 s** for **3.04–6.64 s** of audio. All output is 24 kHz mono PCM16. Waveforms are finite, nonempty and do not hit PCM clipping limits; transcript accuracy was not evaluated.

Eager streaming makes the first audio available quickly but cannot sustain real-time playback on this GPU. The upstream `--fast-all` CUDA graph/compilation path was not tested; these results do not establish optimized-path performance. No additional downloads were made after the coordinator's disk-capacity stop instruction. The benchmark ran with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` after the initial download.

Upstream prints `Residual tail decode detected ... tail_frames=1 strategy=eager` on requests ending with an odd codec frame count. It flushes the last partial chunk and closes the request; all samples completed without an exception.

## License

The source is Apache 2.0. Model weights, derivatives and self-hosted outputs use the **BreezeBlue Research and Non-Commercial License v1.1**, updated September 1, 2026. Commercial use requires a separate written license. The model card explicitly says paid hosted API access does not grant commercial rights to local weights or local outputs. See the checkpoint's `LICENSE`; a copy of its model card is `HF_README.md`. This benchmark is evaluation use; deployment licensing needs to account for the intended use.