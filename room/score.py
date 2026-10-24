"""Score one live-room run against the room's ground truth.

  .venv/bin/python score.py run/<name> [--json]

Everything is on one perf_counter clock (room engine, run_kermit stdout t_rx, truth). Definitions:
  reply to step i      first Kermit loudspeaker onset IN THE ROOM after step i's line ended and before the
                       next human line started (for the interjection beat: during the line, after word 3)
  beat landed          scripted: the reply text matches the expected turn text (word overlap ≥ 0.6)
                       live (Q&A) / no-reply steps: a reply happened (resp. none happened)
  missed turn          a reply was expected and none came
  false cut-in         Kermit's audio starts while a human is speaking (not the scripted interjection)
  spontaneous turn     a Kermit onset not attributable to any step (e.g. triggered by applause / noise)
  self-hearing         run_kermit 'heard' events whose words are mostly Kermit's own (echo); 'leaked' if
                       they were NOT flagged echo (they then reach BRAIN)
  latency              end of the human's last word (room clock) → first sample of Kermit's reply in the room
  tracking             perception speaker_id (gaze stream, 20 Hz) vs the truth speaker while a human talks;
                       facing (oracle/camera) vs truth for the active speaker
"""
import argparse, glob, json, os, re, sys
import numpy as np


def words(s):
    return re.sub(r"[^a-z' ]", " ", (s or "").lower()).split()


def overlap(a, b):
    A, B = words(a), words(b)
    if not A or not B: return 0.0
    from collections import Counter
    ca, cb = Counter(A), Counter(B)
    return sum((ca & cb).values()) / max(len(A), len(B))


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), int); d[:, 0] = range(len(r) + 1); d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1] / max(1, len(r))


def load(run):
    L = lambda f: [json.loads(l) for l in open(os.path.join(run, f)) if l.strip()]
    truth = L("truth.ndjson"); kerm = L("kermit.ndjson")
    meta = json.load(open(os.path.join(run, "meta.json")))
    script = {s["step"]: s for s in json.load(open(meta["script"]))["steps"]}
    return truth, kerm, meta, script


def score(run):
    truth, kerm, meta, script = load(run)
    ev = [e for e in truth if "event" in e]
    frames = [e for e in truth if "people" in e]
    utts = {}
    for e in ev:
        if e["event"] == "utt_on": utts[e["step"]] = dict(on=e["t"], who=e["who"], text=e["text"], words=e["words"])
        if e["event"] == "utt_off": utts[e["step"]].update(off=e["t"], cut=e["cut"])
    steps = sorted(utts)
    k_on = [e["t"] for e in ev if e["event"] == "kermit_audio_on"]
    k_off = [e["t"] for e in ev if e["event"] == "kermit_audio_off"]
    plays = [e for e in kerm if e.get("event") == "play"]
    done = {e["turn"]: e for e in kerm if e.get("event") == "done"}
    heard = [e for e in kerm if e.get("event") == "heard"]

    def turn_at(t_room):
        """run_kermit turn whose first sentence started playing just before this room onset."""
        c = [p for p in plays if p["index"] == 0 and -0.3 < t_room - p.get("t_audio", p["t_wall"]) < 0.4]
        if not c: c = [p for p in plays if -0.3 < t_room - p.get("t_audio", p["t_wall"]) < 0.4]
        return c[0]["turn"] if c else None

    def text_of(turn):
        d = done.get(turn)
        if d and os.path.exists(d.get("timeline", "")):
            return json.load(open(d["timeline"])).get("text", "")
        segs = [e for e in kerm if e.get("event") == "segment" and e.get("turn", "").startswith(turn or "-")]
        return " ".join(w["w"] for s in segs for w in s.get("words", []))

    used, rows, lat = set(), [], []
    for i, st in enumerate(steps):
        u = utts[st]; sc = script[st]; k = sc.get("kermit")
        nxt = utts[steps[i + 1]]["on"] if i + 1 < len(steps) else 1e18
        interject = bool(k) and k.get("beat") == "interject"
        w3 = u["words"][min(3, len(u["words"]) - 1)]["t0"]
        # human speech end = last word actually spoken
        t_end = u.get("off", u["on"])
        if interject:
            cand = [t for t in k_on if w3 <= t < t_end + 1.0 and t not in used]
        else:
            cand = [t for t in k_on if t_end - 0.05 <= t < nxt and t not in used]
        reply_t = cand[0] if cand else None
        if reply_t: used.add(reply_t)
        turn = turn_at(reply_t) if reply_t else None
        txt = text_of(turn) if turn else ""
        exp = (k or {}).get("turn") or (k or {}).get("fallback_turn") or {}
        row = dict(step=st, who=u["who"], beat=(k or {}).get("beat"), mode=(k or {}).get("mode"),
                   expected_reply=bool(k), reply=bool(reply_t), reply_text=txt[:120])
        if turn:
            stops = [e for e in kerm if e.get("event") == "stop" and e.get("turn") == turn]
            d = done.get(turn) or {}
            full = json.load(open(d["timeline"])).get("duration") if os.path.exists(d.get("timeline", "")) else None
            row["cut_short"] = bool(stops)
            if stops: row["cut_at_s"] = stops[0].get("at_s"); row["cut_index"] = stops[0].get("index")
            if full: row["full_s"] = round(full, 2)
        if reply_t is not None and not interject:
            row["latency_s"] = round(reply_t - t_end, 3); lat.append(reply_t - t_end)
            # decomposition: speech end → turn_end (endpointer) → heard (final STT) → play (BRAIN+TTS) → room
            te = [e for e in kerm if e.get("event") == "turn_end" and t_end - 1.0 < e["t_rx"] <= reply_t]
            hd = [e for e in kerm if e.get("event") == "heard" and not e.get("echo") and t_end - 1.0 < e["t_rx"] <= reply_t]
            pl = [p for p in plays if p["index"] == 0 and t_end < p["t_wall"] <= reply_t + 0.05]
            if te:
                row["endpoint_s"] = round(te[-1]["t_rx"] - t_end, 3); row["asr_behind_ms"] = te[-1].get("asr_behind_ms")
            if hd: row["stt_done_s"] = round(hd[-1]["t_rx"] - t_end, 3)
            if pl: row["play_s"] = round(pl[-1]["t_wall"] - t_end, 3)
        if interject and reply_t is not None:
            row["interject_after_turn_s"] = round(reply_t - w3, 3)
        if not k:
            row["landed"] = reply_t is None
        elif reply_t is None:
            row["landed"] = False; row["missed"] = True
        elif k.get("mode") == "scripted":
            row["text_match"] = round(overlap(txt, exp.get("text", "")), 2)
            row["landed"] = row["text_match"] >= 0.6 and not row.get("cut_short")
        else:
            row["landed"] = len(words(txt)) >= 3 and not row.get("cut_short")
        row["right_reply"] = bool(reply_t) and (row.get("text_match", 1.0) >= 0.6)   # landed, ignoring cut-offs
        # STT: what run_kermit heard for this line (heard events arriving after the line's onset, before the reply)
        hs = [h for h in heard if u["on"] < h["t_rx"] < (reply_t or nxt) + 0.5 and not h.get("echo")]
        row["heard"] = " | ".join(h["text"] for h in hs)[:160]
        row["wer"] = round(wer(u["text"], " ".join(h["text"] for h in hs)), 2) if hs else 1.0
        rows.append(row)

    # false cut-ins / spontaneous turns
    speaking = [(utts[s]["on"], utts[s].get("off", utts[s]["on"]), s) for s in steps]
    false_cut, spont = [], []
    for t in k_on:
        if t in used: continue
        inside = [s for a, b, s in speaking if a <= t <= b]
        (false_cut if inside else spont).append({"t": round(t - meta["t0"], 2), "step": inside[0] if inside else None})
    # replies that started while a human was still talking (counted in rows as replies but also cut-ins)
    for r in rows:
        pass

    # barge-ins (run_kermit stopped itself because it "heard a human" over its own speech)
    bi = [e for e in kerm if e.get("event") == "bargein"]
    def human_talking(t, pre=1.0):
        return any(a - 0.1 <= t and t - pre <= b for a, b, _ in speaking)
    self_bi = [round(e["t"] - meta["t0"], 2) for e in bi if not human_talking(e["t"])]
    human_bi = [round(e["t"] - meta["t0"], 2) for e in bi if human_talking(e["t"])]
    # self-hearing
    kwords_by_time = []
    for p in plays:
        d = done.get(p["turn"])
    k_text_all = " ".join(text_of(t) for t in done)
    self_heard = [h for h in heard if h.get("text") and overlap(h["text"], k_text_all) >= 0.5 and
                  len(set(words(h["text"])) & set(words(k_text_all))) / max(1, len(set(words(h["text"])))) >= 0.6]
    leaked = [h for h in self_heard if not h.get("echo")]
    # turns that BRAIN answered which were triggered by Kermit's own voice
    flagged = [h for h in heard if h.get("echo")]

    # tracking: speaker_id vs truth (map track ids to names by bearing at arrival)
    names_by_id = {}
    for e in kerm:
        if e.get("event") == "person_arrived" and "yaw_deg" in e:
            fr = min(frames, key=lambda f: abs(f["t"] - e["t"])) if frames else None
            if fr:
                best = min(fr["people"], key=lambda p: abs(p["yaw_deg"] - e["yaw_deg"]))
                names_by_id[e["id"]] = best["id"]
    gz = [e for e in kerm if e.get("event") == "gaze" and "speaker_id" in e]
    ft = np.array([f["t"] for f in frames])
    spk_ok = spk_n = fac_ok = fac_n = 0
    for g in gz:
        if not len(ft): break
        f = frames[int(np.clip(np.searchsorted(ft, g["t"]), 0, len(frames) - 1))]
        talking = [p for p in f["people"] if p["speaking"]]
        if len(talking) != 1: continue
        tp = talking[0]
        spk_n += 1; spk_ok += names_by_id.get(g["speaker_id"]) == tp["id"]
        if names_by_id.get(g.get("id")) == tp["id"] and "facing" in g:
            fac_n += 1; fac_ok += g["facing"] == tp["facing"]
    # turned_away events vs truth
    tw = [e for e in kerm if e.get("event") == "turned_away"]
    tw_true = [e for e in tw if names_by_id.get(e.get("id")) in ("kirill", "steve") and e.get("while_speaking")]

    L = [json.loads(l) for l in open(os.path.join(run, "levels.ndjson"))]
    def lvl_during(a, b, key):
        x = [10 ** (l[key] / 10) for l in L if a <= l["t"] <= b and l[key] > -150]
        return float(10 * np.log10(np.mean(x))) if x else None   # energy average
    snr = []   # A-weighted speech-to-noise at the mic over each line, from the component stems
    try:
        import soundfile as sf
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from acoustics import a_weight_db
        hw, fs = sf.read(os.path.join(run, "mic_human.wav"), dtype="float32"); nw, _ = sf.read(os.path.join(run, "mic_noise.wav"), dtype="float32")
        for s in steps:
            u = utts[s]; i0, i1 = int((u["on"] - meta["t0"]) * fs), int((u.get("off", u["on"]) - meta["t0"]) * fs)
            if i1 - i0 > fs // 4 and i1 <= len(hw): snr.append(a_weight_db(hw[i0:i1]) - a_weight_db(nw[i0:i1]))
    except Exception as e:
        print("snr:", e, file=sys.stderr)
    erl = [l["echo_dbfs"] for l in L if l["echo_dbfs"] > -100]
    eng = json.load(open(os.path.join(run, "engine.json"))) if os.path.exists(os.path.join(run, "engine.json")) else {}
    lat = np.array(lat)
    summary = dict(
        run=os.path.basename(run.rstrip("/")), noise=meta["noise"], venue_dba=meta["base_dba"],
        mic_noise_dba_median=round(float(np.median([l["noise_dba_spl"] for l in L])) +
                                   (20.0 if meta.get("input") == "B" and "levels_cal" not in meta else 0.0), 1),
        speech_snr_at_mic_db_median=round(float(np.median(snr)), 1) if snr else None,
        echo_to_speech_db=round(float(np.median(erl)) - float(np.median([l["human_dbfs"] for l in L if l["human_dbfs"] > -100])), 1) if erl else None,
        steps=len(rows), beats_landed=sum(r["landed"] for r in rows),
        right_reply_started=sum(r["right_reply"] for r in rows if r["expected_reply"]),
        replies_cut_short=sum(bool(r.get("cut_short")) for r in rows),
        self_bargeins=len(self_bi), bargeins_during_human=len(human_bi),
        expected_replies=sum(r["expected_reply"] for r in rows), missed_turns=sum(bool(r.get("missed")) for r in rows),
        false_cut_ins=len(false_cut), spontaneous_turns=len(spont),
        self_heard=len(self_heard), self_heard_flagged_echo=len(self_heard) - len(leaked), self_triggered_leaks=len(leaked),
        latency_mean_s=round(float(lat.mean()), 3) if len(lat) else None,
        latency_p95_s=round(float(np.percentile(lat, 95)), 3) if len(lat) else None,
        latency_max_s=round(float(lat.max()), 3) if len(lat) else None,
        endpoint_mean_s=round(float(np.mean([r["endpoint_s"] for r in rows if "endpoint_s" in r])), 3) if any("endpoint_s" in r for r in rows) else None,
        stt_done_mean_s=round(float(np.mean([r["stt_done_s"] for r in rows if "stt_done_s" in r])), 3) if any("stt_done_s" in r for r in rows) else None,
        play_mean_s=round(float(np.mean([r["play_s"] for r in rows if "play_s" in r])), 3) if any("play_s" in r for r in rows) else None,
        asr_behind_ms_median=float(np.median([r["asr_behind_ms"] for r in rows if r.get("asr_behind_ms") is not None])) if any(r.get("asr_behind_ms") is not None for r in rows) else None,
        stt_wer_mean=round(float(np.mean([r["wer"] for r in rows])), 2) if rows else None,
        speaker_id_acc=round(spk_ok / spk_n, 3) if spk_n else None, speaker_id_samples=spk_n,
        facing_acc=round(fac_ok / fac_n, 3) if fac_n else None,
        turned_away_events=len(tw), turned_away_presenter_while_speaking=len(tw_true),
        engine=dict((k, eng.get(k)) for k in ("overruns", "proc_ms_mean", "proc_ms_p99", "proc_ms_max", "spk_late_blocks", "duration_s")),
    )
    return summary, rows, dict(false_cut_ins=false_cut, spontaneous=spont, leaked=[h["text"] for h in leaked])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    s, rows, extra = score(a.run)
    json.dump(dict(summary=s, steps=rows, **extra), open(os.path.join(a.run, "score.json"), "w"), indent=1)
    if a.quiet: return
    for r in rows:
        print(f"{r['step']:>2} {r['who']:7s} {str(r['beat']):15s} {'OK ' if r['landed'] else 'XX '}{'CUT@%.2fs ' % r['cut_at_s'] if r.get('cut_short') else ''}"
              f"{'lat %.2fs' % r['latency_s'] if 'latency_s' in r else ('cut-in %.2fs' % r['interject_after_turn_s'] if 'interject_after_turn_s' in r else '          ')}"
              f"  wer {r['wer']:.2f}  heard: {r['heard'][:60]!r}  → {r['reply_text'][:50]!r}")
    print(json.dumps(s, indent=1))
    if extra["false_cut_ins"] or extra["spontaneous"]:
        print("false cut-ins:", extra["false_cut_ins"], "spontaneous:", extra["spontaneous"])


if __name__ == "__main__":
    main()
