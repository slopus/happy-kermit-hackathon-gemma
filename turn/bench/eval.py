"""End-of-turn bench: merge the score tables, add fusion, report eot-bench operating points per method.

usage: eval.py DATASET [--train-on eotbench]   (fusion is 5-fold cross-validated by turn; --train-on X trains it
                                                 on dataset X and tests on DATASET)
Outputs out/{DATASET}_report.md, out/{DATASET}_merged.parquet
"""
import json, math, os, re, sys
import numpy as np, pandas as pd
from common import OUT, evaluate, load_out

FILLERS = {"um", "uh", "so", "and", "but", "or", "like", "the", "a", "to", "of", "about", "with", "because", "if",
           "that", "is", "i", "my", "you", "your", "for", "in", "on", "what", "how", "when", "then", "well", "kermit"}


def text_feats(t):
    t = (t or "").strip()
    w = re.sub(r"[^\w' ]", " ", t.lower()).split()
    return dict(n_words=min(len(w), 30), ends_q=float(t.endswith("?")), ends_p=float(t.endswith((".", "!"))),
                ends_comma=float(t.endswith((",", "...", "…"))), last_filler=float(bool(w) and w[-1] in FILLERS))


def merged(name):
    A = load_out(f"{name}_audio")
    T = load_out(f"{name}_text") if any(f.startswith(f"{name}_text") for f in os.listdir(OUT)) else load_out(f"{name}_asr")
    key = ["id", "span_index", "timestamp"]
    M = A.merge(T, on=key, how="left")
    for f in sorted(os.listdir(OUT)):
        if f.startswith(f"{name}_gemma_") and f.endswith(".parquet"):
            M = M.merge(pd.read_parquet(f"{OUT}/{f}"), on=key, how="left")
    tf = pd.DataFrame([text_feats(t) for t in M.text_nemo.fillna("")])
    return pd.concat([M.reset_index(drop=True), tf], axis=1)


FUSION_FEATS = ["silence_dur", "t1m", "smart", "peou", "peou_emit", "lk_nemo", "vad_p", "f0_slope", "en_slope",
                "n_words", "ends_q", "ends_p", "ends_comma", "last_filler"]


def fusion_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0)


def add_fusion(M, feats, col, train=None):
    feats = [f for f in feats if f in M.columns]
    y = (M.label == "eot").to_numpy()
    if train is not None:
        m = fusion_model().fit(train[feats].fillna(0), train.label == "eot")
        M[col] = m.predict_proba(M[feats].fillna(0))[:, 1]
        return M
    ids = M.id.unique(); rng = np.random.default_rng(0); rng.shuffle(ids)
    fold = {i: k % 5 for k, i in enumerate(ids)}; f = M.id.map(fold).to_numpy()
    p = np.zeros(len(M))
    for k in range(5):
        m = fusion_model().fit(M.loc[f != k, feats].fillna(0), y[f != k])
        p[f == k] = m.predict_proba(M.loc[f == k, feats].fillna(0))[:, 1]
    M[col] = p
    return M


def silence_rule(M, x, sil_col="silence_dur"):
    """Fixed silence threshold x (s): cut-in = hold span longer than x; latency = x on true ends (+ VAD lag
    when sil_col is the Silero-measured silence)."""
    sp = M.groupby(["id", "span_index", "label"]).agg(dur=("silence_dur", "max")).reset_index()
    fire = M[M[sil_col] >= x - 1e-9].groupby(["id", "span_index"]).silence_dur.min()
    sp["fire"] = sp.set_index(["id", "span_index"]).index.map(fire)
    h = sp[(sp.label == "hold") & (sp.dur >= 0.2) & (sp.dur <= 5)]
    e = sp[sp.label == "eot"]
    cut = float((h.fire.notna() & (h.fire < h.dur - 1e-9)).mean())
    lat = float(e.fire.fillna(e.dur).mean())
    turns_cut = h[h.fire.notna() & (h.fire < h.dur - 1e-9)].id.nunique() / M.id.nunique()
    return cut, lat, turns_cut


def main():
    name = sys.argv[1]
    train_on = sys.argv[sys.argv.index("--train-on") + 1] if "--train-on" in sys.argv else None
    M = merged(name)
    gem = [c for c in M.columns if c.startswith("gemma_")]
    T = merged(train_on) if train_on else None
    if T is not None:
        for g in gem:
            if g not in T.columns: T[g] = np.nan
    M = add_fusion(M, [f for f in FUSION_FEATS if f not in ("lk_nemo",)] , "fuse_audio", T)
    M = add_fusion(M, FUSION_FEATS, "fuse_audio_lk", T)
    if gem:
        M = add_fusion(M, FUSION_FEATS + gem, "fuse_all", T)
        M = add_fusion(M, [f for f in FUSION_FEATS if f != "lk_nemo"] + gem, "fuse_audio_gemma", T)
    M.to_parquet(f"{OUT}/{name}_merged.parquet")
    lines = [f"# End-of-turn bench: {name}  ({M.id.nunique()} turns, "
             f"{M[M.label == 'hold'].groupby(['id', 'span_index']).ngroups} hold spans, fusion {'trained on ' + train_on if train_on else '5-fold CV by turn'})", "",
             "eot-bench policy sweep (threshold × action delay 0.2–1.0 s × timeout 1–3.5 s). cut@X = share of mid-turn "
             "pauses Kermit would cut into at ≤X ms mean dead air; lat@Y = mean dead air after a true end at ≤Y cut-ins.", "",
             "| method | AUC | cut@300ms | cut@600ms | lat@2% | lat@5% | lat@10% |", "|---|---|---|---|---|---|---|"]
    methods = [("turn-1-mini (audio)", "t1m"), ("Smart Turn v3.2 (audio)", "smart"), ("parakeet EOU P(<EOU>)", "peou"),
               ("parakeet EOU emitted", "peou_emit"), ("LiveKit text, Nemotron partials", "lk_nemo"),
               ("LiveKit text, oracle words (0.5 s lag)", "lk_oracle")]
    methods += [(f"Gemma {g[6:]} judge, Nemotron partials", g) for g in gem]
    methods += [("fusion: audio models + prosody + text form", "fuse_audio"), ("fusion + LiveKit text", "fuse_audio_lk")]
    if gem: methods += [("fusion audio + Gemma", "fuse_audio_gemma"), ("fusion all", "fuse_all")]
    vad = None; res_all = {}
    for label, col in methods:
        if col not in M.columns or M[col].isna().all(): continue
        r, v, _ = evaluate(M, col); vad = vad or v; res_all[col] = r
        f = lambda x: "–" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x * 100:.1f}%" if x <= 1.0 and "cut" in k else f"{x * 1000:.0f} ms")
        cells = []
        for k in ("cut@300", "cut@600", "lat@2%", "lat@5%", "lat@10%"):
            x = r[k]; cells.append("–" if math.isnan(x) else (f"{x * 100:.1f}%" if k.startswith("cut") else f"{x * 1000:.0f} ms"))
        lines.append(f"| {label} | {r['auc']:.3f} | " + " | ".join(cells) + " |")
    cells = []
    for k in ("cut@300", "cut@600", "lat@2%", "lat@5%", "lat@10%"):
        x = vad[k]; cells.append("–" if math.isnan(x) else (f"{x * 100:.1f}%" if k.startswith("cut") else f"{x * 1000:.0f} ms"))
    lines.insert(6, "| silence only (ideal VAD) | – | " + " | ".join(cells) + " |")
    lines += ["", "## Fixed silence thresholds", "", "| rule | cut-ins (per pause) | turns with ≥1 cut-in | mean dead air |", "|---|---|---|---|"]
    for x in (0.3, 0.5, 0.8, 1.0):
        c, l, tc = silence_rule(M, x)
        lines.append(f"| ideal silence ≥ {x:.1f} s | {c * 100:.1f}% | {tc * 100:.1f}% | {l * 1000:.0f} ms |")
    for x in (0.3, 0.5, 0.8):
        c, l, tc = silence_rule(M, x, "vad_sil")
        lines.append(f"| Silero (p<0.5) silence ≥ {x:.1f} s | {c * 100:.1f}% | {tc * 100:.1f}% | {l * 1000:.0f} ms |")
    open(f"{OUT}/{name}_report.md", "w").write("\n".join(lines) + "\n")
    json.dump(res_all, open(f"{OUT}/{name}_results.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
