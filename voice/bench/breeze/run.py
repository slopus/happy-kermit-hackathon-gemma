"""Breeze TTS 2 voice-direction streaming benchmark. Run from voice/."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'bench/breeze'
sys.path.insert(0, str(OUT / 'upstream'))
sys.path.insert(0, str(ROOT / 'bench/qwen3'))
from run import EMOTIONS
from breeze_infer.runtime import load_runtime, set_all_seeds, update_generation_config_for_breeze
from breeze_infer.templates import get_template, prepare_inputs, select_template_name
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig


def main():
    torch.set_num_threads(4)
    ckpt = Path(snapshot_download('BreezeBlue/Breeze-TTS-2'))
    start = time.perf_counter()
    tokenizer, model, audio_tokenizer = load_runtime(ckpt, device='cuda:0', attn_implementation='eager')
    update_generation_config_for_breeze(model)
    runtime = FastBreezeStreamingRuntime(model, audio_tokenizer, FastStreamingConfig(max_new_tokens=600, max_seq_len=2048, fast_all=False, repetition_penalty=1.1), tokenizer=tokenizer)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - start
    results = {'model': 'BreezeBlue/Breeze-TTS-2', 'device': torch.cuda.get_device_name(0), 'dtype': 'bfloat16', 'attention': 'eager', 'fast_all': False, 'streaming': True, 'cfg_scale': 4.0, 'load_s': load_s, 'license': 'BreezeBlue Research and Non-Commercial License: weights, derivatives and self-hosted outputs non-commercial; source Apache 2.0.', 'lines': []}
    print('LOADED', load_s, flush=True)

    def generate(line):
        set_all_seeds(42)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        req = {'id': line['id'], 'text': line['text'], 'speaker': 'S0', 'instruction': EMOTIONS[line['emotion']], 'ref_audio_path': str(ROOT / 'bench/qwen3/ref_frog.wav'), 'ref_text': (ROOT / 'bench/qwen3/ref_frog.txt').read_text().strip()}
        inputs = prepare_inputs(tokenizer, audio_tokenizer, model, [req], get_template(select_template_name(req)), guidance_scale=4.0, guidance_scale_ref=None, guidance_scale_ins=None)
        torch.cuda.synchronize()
        prep_s = time.perf_counter() - start
        chunks = []
        arrivals = []
        for chunk in runtime.iter_audio_chunks(inputs, request_id=line['id'], seed=42):
            torch.cuda.synchronize()
            arrivals.append({'arrival_s': time.perf_counter() - start, 'duration_s': len(chunk.audio) / runtime.sample_rate})
            chunks.append(chunk.audio.copy())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        wav = np.concatenate(chunks)
        duration = len(wav) / runtime.sample_rate
        return wav, {'id': line['id'], 'text': line['text'], 'emotion': line['emotion'], 'model_load_s': load_s, 'input_prepare_s': prep_s, 'ttfa_s': arrivals[0]['arrival_s'], 'gen_s': elapsed, 'duration_s': duration, 'rtf': elapsed / duration, 'peak_vram_gib': torch.cuda.max_memory_allocated() / 2**30, 'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30, 'peak_amplitude': float(np.max(np.abs(wav))), 'chunks': arrivals}

    generate({'id': 'warmup', 'text': 'Hello, friend! It is nice to see you.', 'emotion': 'neutral'})
    for line in json.loads((ROOT / 'lines/demo_turns.json').read_text()):
        wav, metric = generate(line)
        sf.write(OUT / f"{line['id']}.wav", wav, runtime.sample_rate)
        results['lines'].append(metric)
        (OUT / 'results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps({k: v for k, v in metric.items() if k != 'chunks'}), flush=True)


if __name__ == '__main__':
    main()