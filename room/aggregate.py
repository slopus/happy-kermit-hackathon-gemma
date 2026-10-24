"""Markdown table over scored runs:  .venv/bin/python aggregate.py run/s1_* > results.md"""
import json, os, sys

COLS = [("venue", lambda s: f"{s['venue_dba']:.0f}"), ("input", lambda s: s["_input"]),
        ("speech SNR dB(A)", lambda s: s["speech_snr_at_mic_db_median"]), ("echo re speech dB", lambda s: s["echo_to_speech_db"]),
        ("beats landed", lambda s: f"{s['beats_landed']}/{s['steps']}"), ("right reply started", lambda s: f"{s['right_reply_started']}/{s['expected_replies']}"),
        ("missed", lambda s: s["missed_turns"]), ("false cut-ins", lambda s: s["false_cut_ins"]),
        ("spont.", lambda s: s["spontaneous_turns"]), ("self barge-in", lambda s: s["self_bargeins"]),
        ("cut short", lambda s: s["replies_cut_short"]), ("echo heard (flagged/leaked)", lambda s: f"{s['self_heard']} ({s['self_heard_flagged_echo']}/{s['self_triggered_leaks']})"),
        ("lat mean s", lambda s: s["latency_mean_s"]), ("lat p95 s", lambda s: s["latency_p95_s"]),
        ("→turn_end s", lambda s: s.get("endpoint_mean_s")), ("→STT s", lambda s: s.get("stt_done_mean_s")), ("→play s", lambda s: s.get("play_mean_s")),
        ("ASR behind ms", lambda s: s.get("asr_behind_ms_median")), ("STT WER", lambda s: s["stt_wer_mean"]), ("speaker-ID acc", lambda s: s["speaker_id_acc"]), ("facing acc", lambda s: s["facing_acc"])]

rows = []
for r in sys.argv[1:]:
    p = os.path.join(r, "score.json")
    if not os.path.exists(p): continue
    s = json.load(open(p))["summary"]; m = json.load(open(os.path.join(r, "meta.json")))
    s["_input"] = "A table" if m.get("input", "A") == "A" else "B handheld"
    rows.append((os.path.basename(r.rstrip("/")), s))
print("| run | " + " | ".join(c for c, _ in COLS) + " |")
print("|" + "---|" * (len(COLS) + 1))
for n, s in rows:
    print(f"| {n} | " + " | ".join(str(f(s)) for _, f in COLS) + " |")
