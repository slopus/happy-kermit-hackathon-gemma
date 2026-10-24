"""Qwen GPU benchmark. Run from voice/ with HF_HOME, UV_CACHE_DIR, CUDA_VISIBLE_DEVICES=0."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'bench/qwen3'
VOICE = 'A cheerful cartoon frog puppet character: a slightly nasal, reedy, high-pitched male tenor voice, warm and friendly, quick, bouncy and excitable delivery with a little rasp.'
REF_TEXT = "Hello there, friend! What a lovely day to make something together. I've got a head full of ideas and a little spark of excitement. Let's see what we can create!"
EMOTIONS = {'excited': 'Speak very excitedly, fast and bouncy.', 'surprised': 'Speak with wide-eyed surprise and rising pitch.', 'happy': 'Speak happily, warmly and playfully.', 'thinking': 'Speak thoughtfully with a curious hum and a short pause.', 'sad': 'Speak sadly and a little deflated, but gently reassuring.', 'embarrassed': 'Speak sheepishly and hesitantly, a little embarrassed.', 'neutral': 'Speak naturally and warmly.', 'annoyed': 'Speak with mild irritation.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modes', nargs='+', default=['design', 'base17', 'base06'])
    parser.add_argument('--reference-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    results_path = OUT / 'results.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else {'device': torch.cuda.get_device_name(0), 'dtype': 'bfloat16', 'attention': 'sdpa', 'streaming': False, 'ttfa_note': 'Official qwen-tts 0.1.1 returns whole waveforms. non_streaming_mode=False simulates text streaming; no audio iterator.', 'models': {}, 'lines': [], 'references': []}

    def save():
        results_path.write_text(json.dumps(results, indent=2))

    for mode in args.modes:
        name = {'design': '1.7B-VoiceDesign', 'base17': '1.7B-Base', 'base06': '0.6B-Base'}[mode]
        model_id = f'Qwen/Qwen3-TTS-12Hz-{name}'
        download_start = time.perf_counter()
        local_model = snapshot_download(model_id)
        download_s = time.perf_counter() - download_start
        start = time.perf_counter()
        model = Qwen3TTSModel.from_pretrained(local_model, device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
        torch.cuda.synchronize()
        load_s = time.perf_counter() - start
        results['models'][mode] = {'id': model_id, 'load_s': load_s, 'snapshot_download_s': download_s}
        print('LOADED', mode, load_s, flush=True)

        def generate(text, emotion='neutral', voice=VOICE):
            if mode == 'design':
                return model.generate_voice_design(text=text, language='English', instruct=voice + ' ' + EMOTIONS[emotion], max_new_tokens=600)
            return model.generate_voice_clone(text=text, language='English', voice_clone_prompt=clone_prompt, max_new_tokens=600)

        def timed(text, emotion='neutral', voice=VOICE):
            torch.manual_seed(42)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            wavs, sr = generate(text, emotion, voice)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            wav = np.asarray(wavs[0])
            duration = len(wav) / sr
            return wav, sr, {'gen_s': elapsed, 'ttfa_s': None, 'duration_s': duration, 'rtf': elapsed / duration, 'peak_vram_gib': torch.cuda.max_memory_allocated() / 2**30, 'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30, 'peak_amplitude': float(np.max(np.abs(wav))), 'model_load_s': load_s}

        if mode == 'design' and not (OUT / 'ref_frog.wav').exists():
            generate('Hello, friend! It is nice to see you.')
            wav, sr, metric = timed(REF_TEXT)
            sf.write(OUT / 'ref_frog.wav', wav, sr)
            (OUT / 'ref_frog.txt').write_text(REF_TEXT + '\n')
            results['references'].append({'file': 'ref_frog.wav', 'instruct': VOICE, **metric})
            save()
            print('REFERENCE_READY', OUT / 'ref_frog.wav', metric, flush=True)
            if args.reference_only:
                return
            for i, variant in enumerate([
                'A friendly male cartoon frog puppet, bright nasal tenor with a reedy buzz, gentle rasp, lively rhythm and playful warmth.',
                'A young adult male puppet voice, high light tenor, rounded vowels, a slightly croaky nasal resonance, warm, cheerful and quick.'
            ], 2):
                wav, sr, metric = timed(REF_TEXT, voice=variant)
                sf.write(OUT / f'ref_variant_{i}.wav', wav, sr)
                results['references'].append({'file': f'ref_variant_{i}.wav', 'instruct': variant, **metric})
                save()
        if mode != 'design':
            start = time.perf_counter()
            clone_prompt = model.create_voice_clone_prompt(ref_audio=str(OUT / 'ref_frog.wav'), ref_text=(OUT / 'ref_frog.txt').read_text().strip())
            torch.cuda.synchronize()
            results['models'][mode]['reference_encode_s'] = time.perf_counter() - start
        generate('Hello, friend! It is nice to see you.')
        for line in json.loads((ROOT / 'lines/demo_turns.json').read_text()):
            wav, sr, metric = timed(line['text'], line['emotion'])
            folder = OUT / mode
            folder.mkdir(exist_ok=True)
            sf.write(folder / f"{line['id']}.wav", wav, sr)
            row = {'mode': mode, 'id': line['id'], 'text': line['text'], 'emotion': line['emotion'], 'emotion_conditioned': mode == 'design', **metric}
            results['lines'] = [r for r in results['lines'] if (r['mode'], r['id']) != (mode, line['id'])] + [row]
            save()
            print(json.dumps(row), flush=True)
        del model
        if mode != 'design':
            del clone_prompt
        gc.collect()
        torch.cuda.empty_cache()
        save()


if __name__ == '__main__':
    main()