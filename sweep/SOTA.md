# Gemma models available as of 2026-10-24 (checked on the HF API and model cards)

The newest google/gemma* repo on HF is from 2026-06-05, so nothing has been released since the 12B QAT drops.

| model | params | modalities in | context | license | GGUF | MLX |
|---|---|---|---|---|---|---|
| Gemma 4 E2B-it | 2.3B effective (5.1B with PLE embeddings), dense, 35 layers | text, image, audio, video | 128K | Apache 2.0 | unsloth, ggml-org, bartowski, lmstudio; Google QAT q4_0 | lmstudio 4/5/6/8-bit, mlx-community 4-bit |
| Gemma 4 E4B-it | 4.5B effective (8B with PLE), dense, 42 layers | text, image, audio, video | 128K | Apache 2.0 | same + Google QAT q4_0 (5.15 GB) | lmstudio 4/5/6/8-bit |
| Gemma 4 12B-it ("Unified", added 2026-05-23) | 11.95B, dense, 48 layers, no separate encoders | text, image, audio, video | 256K | Apache 2.0 | unsloth Q4_K_M 7.1 GB / Q8 12.7 GB, ggml-org (with MTP head), Google QAT q4_0 7.0 GB | lmstudio 4–8-bit, mlx-community QAT mxfp4 |
| Gemma 4 26B-A4B-it (MoE) | 25.2B total / 3.8B active, 128 experts (8 active + 1 shared), 30 layers | text, image, video (no audio) | 256K | Apache 2.0 | unsloth UD-Q4_K_M 17.0 GB, Google QAT q4_0 14.4 GB | lmstudio 4–8-bit + QAT 4-bit |
| Gemma 4 31B-it | 30.7B, dense, 60 layers | text, image, video (no audio) | 256K | Apache 2.0 | unsloth Q4_K_M 18.3 GB, Google QAT q4_0 17.7 GB | lmstudio 4/5/6/8-bit |
| Gemma 4 *-assistant (2026-04-23) | small MTP drafters for every size | – | – | Apache 2.0 | ggml-org `mtp-*.gguf`; llama.cpp arch `gemma4-assistant` | – |
| Gemma 3n E2B / E4B (2025-06) | 5B / 8B raw (2B / 4B effective), MatFormer | text, image, audio, video | 32K | Gemma license (gated) | ggml-org, bartowski, lmstudio | lmstudio |
| FunctionGemma 270M | 270M (Gemma 3 base) | text | – | Gemma license | – | – |

Other variants from Google: QAT checkpoints (q4_0 GGUF, w4a16, "mobile"); DiffusionGemma, EmbeddingGemma v2, ShieldGemma 2,
PaliGemma 2. None of them fits a persona/dialogue brain.

Community fine-tunes relevant to persona or dialogue (all have GGUFs on bartowski):
- Gryphe/Gemma-4-31B-StyleTune (Apache 2.0)
- ValiantLabs/gemma-4-12B-it-Esper4 (Apache 2.0)
- rpDungeon/Gemma-4-E4B-Luchador (role-play, Gemma license)
- several 31B RP or "Heretic" (abliterated) tunes

None of them is better suited than our own Kermit LoRA: they're tuned for general role-play, while ours is tuned for the
persona and the KML format.
