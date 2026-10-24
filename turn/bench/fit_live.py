"""Fit the live end-of-turn fusion (gradient-boosted trees on live features), export it as plain JSON trees for
endpoint.Fusion, choose the policy on grouped-CV predictions, and measure the speculative start.

Features (all computed live by endpoint.Endpointer): silence so far, turn-1-mini P, Smart Turn P (once, at 0.2 s
of silence), LiveKit text P on the Nemotron partial, text form (word count, ends with ? . , filler).
CV groups: eot-bench conversation / Kermit sentence (no sentence is in both train and test).

usage: fit_live.py [--budget 0.05]     (run after fit_fusion.py has built out/live_frame.parquet)
"""
import json, math, os, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from common import ROOT, OUT, evaluate

FEATS = ["silence_dur", "t1m", "smart1", "lk_nemo", "n_words", "ends_q", "ends_p", "ends_comma", "last_filler"]
KW = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0)


def cv_grouped(M, feats):
    g = M.grp.unique(); rng = np.random.default_rng(0); rng.shuffle(g)
    fold = M.grp.map({x: k % 5 for k, x in enumerate(g)}).to_numpy(); p = np.zeros(len(M)); y = (M.label == "eot").to_numpy()
    for k in range(5):
        m = HGB(**KW).fit(M.loc[fold != k, feats], y[fold != k]); p[fold == k] = m.predict_proba(M.loc[fold == k, feats])[:, 1]
    return p


def export(m):
    trees = []
    for (p,) in m._predictors:
        n = p.nodes
        trees.append({k: n[k].tolist() for k in ("value", "feature_idx", "num_threshold", "missing_go_to_left", "left", "right", "is_leaf")})
    return {"baseline": float(np.ravel(m._baseline_prediction)[0]), "trees": trees}


def predict_json(j, X):
    raw = np.full(len(X), j["baseline"])
    for t in j["trees"]:
        for r in range(len(X)):
            i = 0
            while not t["is_leaf"][i]:
                v = X[r, t["feature_idx"][i]]
                go_left = t["missing_go_to_left"][i] if np.isnan(v) else v <= t["num_threshold"][i]
                i = t["left"][i] if go_left else t["right"][i]
            raw[r] += t["value"][i]
    return 1 / (1 + np.exp(-raw))


def joint_policy(M, col, budget):
    key = ["threshold", "action_delay", "timeout"]; J = None
    for ds in ("eotbench", "kermit"):
        t = evaluate(M[M.ds == ds], col)[2]
        t = t[t.policy_type == "model"][key + ["cutoff_rate", "mean_latency"]].rename(columns={"cutoff_rate": f"cut_{ds}", "mean_latency": f"lat_{ds}"})
        J = t if J is None else J.merge(t, on=key)
    ok = J[(J.cut_eotbench <= budget + 1e-9) & (J.cut_kermit <= budget + 1e-9)].copy()
    ok["lat"] = (ok.lat_eotbench + ok.lat_kermit) / 2
    return ok.sort_values(["lat", "threshold"], ascending=[True, False]).iloc[0].to_dict()


def fire_times(M, col, thr, delay, timeout):
    """Per span: first grid time with silence ≥ delay and p ≥ thr (or the timeout), NaN if never in the span."""
    s = M[(M[col] >= thr) & (M.silence_dur >= delay - 1e-9) | (M.silence_dur >= timeout - 1e-9)]
    return s.groupby(["id", "span_index"]).silence_dur.min()


def speculative(M, col, pol, spec_thr, pipeline_s=0.65):
    """Speculative BRAIN start at p ≥ spec_thr (any silence ≥ 0.1 s); playback still waits for the commit.
    Saved on a true end = min(commit − candidate, pipeline); wasted = candidates on pauses that resume."""
    sp = M.groupby(["id", "span_index", "label", "ds"]).silence_dur.max().rename("dur").reset_index().set_index(["id", "span_index"])
    commit = fire_times(M, col, pol["threshold"], pol["action_delay"], pol["timeout"])
    cand = fire_times(M, col, spec_thr, 0.1, 99)
    sp["commit"], sp["cand"] = commit, cand
    sp = sp[(sp.label == "eot") | ((sp.dur >= 0.2) & (sp.dur <= 5))]
    out = {}
    for ds in ("eotbench", "kermit"):
        d = sp[sp.ds == ds]; e, h = d[d.label == "eot"], d[d.label == "hold"]
        ec = e.commit.fillna(e.dur); ek = e.cand.where(e.cand < ec, ec).fillna(ec)
        saved = np.minimum(ec - ek, pipeline_s)
        cut = h.commit.notna() & (h.commit < h.dur - 1e-9)
        waste = h.cand.notna() & (h.cand < h.dur - 1e-9) & ~cut
        out[ds] = dict(saved_ms_mean=round(1000 * float(saved.mean())), saved_ms_med=round(1000 * float(saved.median())),
                       eot_with_candidate=round(float((e.cand.notna() & (e.cand <= ec)).mean()), 3),
                       wasted_starts_per_pause=round(float(waste.mean()), 3),
                       wasted_starts_per_turn=round(float(waste.sum()) / d.index.get_level_values(0).nunique(), 3),
                       commit_lat_ms=round(1000 * float(ec.mean())))
    return out


if __name__ == "__main__":
    budget = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 0.05
    M = pd.read_parquet(f"{OUT}/live_frame.parquet")
    if "--text" in sys.argv and sys.argv[sys.argv.index("--text") + 1] == "tdt":
        # text features from Parakeet TDT at the pause (≥ 0.2 s of silence), as endpoint.py now computes them
        import eval as EV
        L = pd.concat([pd.read_parquet(f"{OUT}/{d}_live_text.parquet") for d in ("eotbench", "kermit")])
        M = M.drop(columns=[c for c in ("text_live", "lk_live") if c in M.columns]).merge(L, on=["id", "span_index", "timestamp"], how="left")
        tf = pd.DataFrame([EV.text_feats(t) for t in M.text_live.fillna("")])
        for c in tf.columns: M[c] = tf[c].to_numpy()
        M["lk_nemo"] = M.lk_live.fillna(M.lk_nemo)
        M["gbm_live"] = cv_grouped(M, FEATS)
        for ds in ("eotbench", "kermit"):
            r, _, _ = evaluate(M[M.ds == ds], "gbm_live"); print("tdt-text", ds, {k: round(v, 3) for k, v in r.items()})
    pols = {b: joint_policy(M, "gbm_live", b) for b in (0.02, 0.05, 0.10)}
    for b, p in pols.items(): print("budget", b, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()})
    pol = pols[budget]
    spec = {}
    for st in (0.3, 0.5, 0.7, 0.8, 0.9):
        if st >= pol["threshold"]: continue
        spec[st] = speculative(M, "gbm_live", pol, st); print("spec", st, spec[st])
    y = (M.label == "eot").to_numpy()
    m = HGB(**KW).fit(M[FEATS].to_numpy(), y)
    j = export(m)
    X = M[FEATS].to_numpy()[:2000]
    assert np.allclose(predict_json(j, X), m.predict_proba(X)[:, 1], atol=1e-6)
    j.update(kind="gbm", features=FEATS,
             policy={"threshold": pol["threshold"], "action_delay": pol["action_delay"], "timeout": pol["timeout"],
                     "spec_threshold": 0.5, "budget": budget},
             cv_policies={str(b): p for b, p in pols.items()}, speculative={str(k): v for k, v in spec.items()})
    j["text"] = "tdt" if "--text" in sys.argv else "stream"
    json.dump(j, open(os.path.join(ROOT, "models/fusion.json"), "w"), default=float)
    print("wrote models/fusion.json", j["policy"])
