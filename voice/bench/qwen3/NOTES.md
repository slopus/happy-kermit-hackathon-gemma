# Qwen3-TTS local GPU benchmark

Test device: NVIDIA RTX 4000 Ada 20 GB, GPU 0 only. These Linux CUDA timings are a proxy, not Apple Silicon results. No inference calls use cloud APIs; Hugging Face is used only for checkpoint downloads.

## Install and run

From `kermit/voice/`:

```bash
export HF_HOME="$PWD/.hf" UV_CACHE_DIR="$PWD/.uv-cache" CUDA_VISIBLE_DEVICES=0
uv venv --python /usr/bin/python3.12 .venvs/qwen3
uv pip install --python .venvs/qwen3/bin/python qwen-tts
uv pip install --python .venvs/qwen3/bin/python torch==2.9.1 torchaudio==2.9.1
.venvs/qwen3/bin/python bench/qwen3/run.py
```

`qwen-tts==0.1.1`, `transformers==4.57.3`. Separate venv from Breeze. bf16, SDPA; no flash-attn build attempted. The initial unconstrained install selected torch 2.14.1, whose automatic native Triton dispatch failed because `/usr/include/python3.12/Python.h` is absent. Pinning torch and torchaudio 2.9.1 fixes inference without installing system packages. Missing SoX prints a warning but waveform generation works.

## API and measurement

`Qwen3TTSModel.from_pretrained(..., device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')` loads each model. VoiceDesign uses `generate_voice_design(text=..., language='English', instruct=voice_description + emotion_direction)`. Base uses `create_voice_clone_prompt(ref_audio=..., ref_text=...)`, reused with `generate_voice_clone(text=..., language='English', voice_clone_prompt=...)`.

The official package returns whole waveforms, not an audio iterator. Its source explicitly says `non_streaming_mode=False` only simulates streaming text, not true input or audio streaming. `ttfa_s` is therefore null; `gen_s` is the time until the complete waveform is available. CUDA is synchronized before/after each timed generation; each model gets a warmup. Peak VRAM is PyTorch's maximum allocated bytes, including model weights; reserved memory is also recorded. RTF = generation seconds / output seconds. Max generation length is 600 codec tokens and seed is 42.

Base does not accept natural-language emotion instructions; its demo lines clone the same neutral-cheerful reference. Only VoiceDesign gets per-emotion instructions. Base reference encoding is timed separately from synthesis. Neither model's generation timing includes model loading or WAV file writing.

The first benchmark run's model load field includes checkpoint retrieval if uncached (`load_s_including_download_if_cold`). The rerunnable script separates snapshot retrieval and local load time. The VoiceDesign model was already cached for the successful benchmark load.

## Samples

`ref_frog.wav` is the first baseline design, 9.84 s, with exact input transcript in `ref_frog.txt`. No existing character audio or named-character prompts were used. Two alternative descriptions are in `ref_variant_2.wav` and `ref_variant_3.wav`; their exact prompts and metrics are in `results.json`. The baseline is retained for consistent cross-model comparison; listening-based selection belongs to the coordinating agent.

The six demo lines are saved under `design/`, `base17/`, and `base06/`. Files are mono PCM16 at the model sample rate (24 kHz). Emotion labels correspond to excited, surprised, happy, thinking, sad, and embarrassed. No subjective voice-quality ranking or transcript accuracy claim is made.

Across the six demo lines, audio-duration-weighted RTF is 2.066 for VoiceDesign, 2.048 for Base 1.7B and 2.045 for Base 0.6B. Maximum allocated VRAM is respectively 4.064, 4.317 and 2.432 GiB. Cold load including downloads was 7.876 s (VoiceDesign, cached), 95.026 s (Base 1.7B, downloaded) and 31.381 s (Base 0.6B, downloaded). These last two are cold-start totals, not pure disk-to-GPU load measurements. Base reference encoding took 2.307 s and 0.058 s respectively; the first encode includes lazy library initialization.

The coordinating agent requested stopping all further downloads after the shared disk filled. The 0.6B checkpoint cache was deleted after its completed benchmark; its six audio samples and metrics remain. Rerunning that model requires downloading it again.

## Optional MLX CUDA check

```bash
export HF_HOME="$PWD/.hf" UV_CACHE_DIR="$PWD/.uv-cache" CUDA_VISIBLE_DEVICES=0
uv venv --python /usr/bin/python3.12 .venvs/mlx
uv pip install --python .venvs/mlx/bin/python 'mlx[cuda]' mlx-audio
.venvs/mlx/bin/python -c 'import mlx.core as mx; print(mx.__version__); print(mx.default_device()); print(mx.array([1,2])+1)'
```

**Install and GPU array operation succeeded**: mlx 0.32.4, `Device(gpu, 0)`, result `[2, 3]`. Logs are `mlx-install.log` and `mlx-smoke.log`. No MLX Qwen checkpoint was downloaded or tested because the coordinator stopped downloads for disk capacity. This confirms the Linux MLX CUDA runtime works, not that the mlx-audio Qwen inference path works or that it reproduces Mac performance.