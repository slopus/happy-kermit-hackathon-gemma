# Turn-taking on the M5 Max: runbook addendum

This sits next to brain/MAC_RUNBOOK.md and voice/README.md. Everything here runs on CPU; it doesn't touch the Metal GPU, which BRAIN uses.

## 0. Copy
Copy all of `kermit/turn/`. You need at least:
- `endpoint.py`
- `setup_models.sh`
- `models/fusion.json`
- `models/smart-turn/mel80.npy`

The `bench/` and `data/` folders aren't needed on the Mac.

## 1. One-time setup (network, ~3 min)
```bash
cd kermit/voice && ./setup_mac.sh   # now also installs onnxruntime, tokenizers, livekit and runs ../turn/setup_models.sh
# or, if voice/.venv already exists:
VIRTUAL_ENV=voice/.venv uv pip install onnxruntime tokenizers livekit && kermit/turn/setup_models.sh
```
`setup_models.sh` downloads:
- turn-1-mini, 22 MB
- Smart Turn v3.2 CPU, 9 MB
- LiveKit turn-detector intl q8, 396 MB
- Nemotron-3.5 streaming 160 ms, 475 MB

## 2. Run
```bash
python run_kermit.py --camera 0 --tts-backend mlx                 # default: --endpoint eou, AEC on, speculative on
python run_kermit.py --camera 0 --tts-backend mlx --no-aec        # Kermit's speaker is in headphones / far away
python run_kermit.py --camera 0 --tts-backend mlx --endpoint vad  # FALLBACK 1: Silero 0.5 s silence (old behaviour)
python run_kermit.py --camera 0 --tts-backend mlx --ptt           # FALLBACK 2: push-to-talk (Enter / Enter)
```
At startup, stderr shows `listening (end-of-turn model)`. If it shows `endpoint eou unavailable (…); falling back to VAD`, a model or package is missing; rerun step 1.

## 3. Check (1 min)
Speak the hero line with pauses: "So, Kermit… [1 s] how are you feeling about, um… [1 s] AI these days?"
- stdout should show `turn_candidate` / `turn_cancel` in the pauses and **one** `turn_end` after "days".
- `turn_end.feats` shows why it fired.
- Talk over Kermit while he answers. You should see `bargein_detected`, then `stop` within ~0.3 s.
- If Kermit stops himself with nobody talking (false barge-in), turn the speaker down or move it away from the mic. As a last resort use `--bargein off`, which keeps the logging but never stops him.

## 4. Mic choice
Use a **handheld or close-talk mic** for whoever talks to Kermit. In ROOM SIM, the table mic at 1.3 m in 60 dBA babble (speech SNR around 0 dB) lost 7/17 turns. The handheld lost 0–2/17 even at 70 dBA.

## 5. Expected cost on the Mac
- turn-1-mini: ~1.3 ms per 160 ms
- Silero: <1 ms per 32 ms
- Smart Turn: ~10–20 ms, once per pause
- LiveKit text: ~20–50 ms per new partial
- Nemotron streaming: ~0.1–0.2 RTF on 4 threads
- TDT at the pause: ~50–150 ms, and it replaces the post-utterance transcription that VOICE used to do

Endpoint delay after the last word: median ~0.45 s, mean ~0.65 s at a 5% cut-in rate in replay. On a contended CPU it's more (ROOM SIM: 1.2–1.5 s).

## 6. Knobs (models/fusion.json → "policy")
`threshold` 0.92, `action_delay` 0.4 s, `timeout` 2.0 s, `spec_threshold` 0.7. If Kermit feels slow, try threshold 0.88 and action_delay 0.2. On eot-bench that's ~200 ms faster, but it cuts into ~2× as many pauses (6–9%).
