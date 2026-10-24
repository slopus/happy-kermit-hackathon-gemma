"""Transcript-side scores on the eot-bench grid.

Step 1 (asr): stream each turn through Nemotron-3.5 streaming 0.6b 160 ms (sherpa-onnx int8, CPU) exactly as the
live endpointer would, and record the partial transcript after each 160 ms chunk -> text_nemo at each grid point.
Oracle text (dataset word timings, eot-bench's 0.5 s transcript lag) is kept as text_oracle for comparability.

Step 2 (models): score (previous assistant message, user text so far) with
  lk     LiveKit turn-detector v0.4.1-intl (Qwen2.5-0.5B distilled, int8 ONNX; P(<|im_end|>))
  namo   Namo-Turn-Detector-v1-English (DistilBERT int8 ONNX; P(end))

usage: scores_text.py DATASET asr [N_WORKERS] | scores_text.py DATASET models
"""
import os, re, sys, time, unicodedata
import numpy as np, pandas as pd
from common import ROOT, OUT, SR, load_rows, grid, at, run_sharded, load_out

NEMO = os.environ.get("NEMO_DIR", os.path.join(ROOT, "../../.context/gemma-bench/parakeet/models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11"))
_rec = {}


def nemo():
    if "r" not in _rec:
        import sherpa_onnx
        d = NEMO
        _rec["r"] = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.int8.onnx", joiner=f"{d}/joiner.int8.onnx",
            tokens=f"{d}/tokens.txt", num_threads=1, provider="cpu", decoding_method="greedy_search")
    return _rec["r"]


def stream_asr(x, chunk=0.16):
    rec = nemo(); s = rec.create_stream(); n = int(chunk * SR)
    T, TX, C = [], [], []
    for i in range(0, len(x), n):
        t0 = time.perf_counter()
        s.accept_waveform(SR, x[i:i + n])
        while rec.is_ready(s): rec.decode_stream(s)
        C.append(time.perf_counter() - t0)
        T.append(min(len(x), i + n) / SR); TX.append(rec.get_result(s).strip())
    return np.array(T), TX, C


def asr_row(args):
    r, g = args
    T, TX, C = stream_asr(r["audio"])
    out = g[["id", "span_index", "timestamp"]].copy()
    out["text_nemo"] = [at(T, TX, t, "") for t in g.timestamp]
    out["text_oracle"] = [" ".join(w for s, e, w in r["words"] if e <= t - 0.5 + 1e-6) for t in g.timestamp]
    final = TX[-1] if TX else ""
    return out, dict(id=r["id"], dur=len(r["audio"]) / SR, chunk_ms_med=1000 * float(np.median(C)),
                     rtf=sum(C) / (len(r["audio"]) / SR), final=final, ref=" ".join(w for _, _, w in r["words"]))


def norm(t):
    t = unicodedata.normalize("NFKC", (t or "").lower())
    t = "".join(ch for ch in t if not (unicodedata.category(ch).startswith("P") and ch not in "'-"))
    return re.sub(r"\s+", " ", t).strip()


class LiveKit:
    def __init__(self):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        d = os.path.join(ROOT, "models/livekit-ml")
        self.tok = AutoTokenizer.from_pretrained(d, truncation_side="left")
        o = ort.SessionOptions(); o.intra_op_num_threads = 4; o.log_severity_level = 3
        self.s = ort.InferenceSession(f"{d}/onnx/model_q8.onnx", o, providers=["CPUExecutionProvider"])
        self.cache = {}

    def prompt(self, messages, user_text):
        chat = [{"role": m["role"], "content": norm(m["content"])} for m in messages[-4:] if m.get("role") in ("user", "assistant") and norm(m.get("content"))]
        if norm(user_text): chat.append({"role": "user", "content": norm(user_text)})
        if not chat or chat[-1]["role"] != "user": return None
        text = self.tok.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
        cut = text.rfind("<|im_end|>")
        return text[:cut] if cut != -1 else text

    def __call__(self, messages, user_text):
        p = self.prompt(messages, user_text)
        if p is None: return 0.0
        if p not in self.cache:
            ids = self.tok(p, add_special_tokens=False, return_tensors="np", max_length=128, truncation=True)["input_ids"].astype(np.int64)
            self.cache[p] = float(self.s.run(None, {"input_ids": ids})[0][0, -1])
        return self.cache[p]


class Namo:
    def __init__(self):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        d = os.path.join(ROOT, "models/namo")
        self.tok = AutoTokenizer.from_pretrained(d)
        o = ort.SessionOptions(); o.intra_op_num_threads = 4; o.log_severity_level = 3
        self.s = ort.InferenceSession(f"{d}/model_quant.onnx", o, providers=["CPUExecutionProvider"])
        self.cache = {}

    def __call__(self, messages, user_text):
        t = (user_text or "").strip()
        if not t: return 0.0
        if t not in self.cache:
            e = self.tok(t, truncation=True, max_length=512, return_tensors="np")
            lg = self.s.run(None, {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)})[0][0]
            p = np.exp(lg - lg.max()); self.cache[t] = float(p[1] / p.sum())
        return self.cache[t]


if __name__ == "__main__":
    name, step = sys.argv[1], sys.argv[2]
    rows = load_rows(name); G = grid(rows)
    if step == "asr":
        jobs = [(r, G[G.id == r["id"]]) for r in rows]
        res, sfx = run_sharded(asr_row, jobs, "asr")
        pd.concat([o for o, _ in res]).to_parquet(f"{OUT}/{name}_asr{sfx}.parquet")
        pd.DataFrame([c for _, c in res]).to_parquet(f"{OUT}/{name}_asrfinal{sfx}.parquet")
    else:
        A = load_out(f"{name}_asr")
        msgs = {r["id"]: r["messages"] for r in rows}
        lk = LiveKit()
        out = A.copy(); t = time.time()
        for src in ("nemo", "oracle"):
            for mname, m in (("lk", lk),):
                t0 = time.perf_counter(); n0 = len(m.cache)
                out[f"{mname}_{src}"] = [m(msgs[i], tx) for i, tx in zip(A.id, A[f"text_{src}"])]
                dt = time.perf_counter() - t0; n = len(m.cache) - n0
                print(f"{mname}_{src}: {n} unique prompts, {1000 * dt / max(1, n):.1f} ms each", flush=True)
        out.to_parquet(f"{OUT}/{name}_text.parquet")
