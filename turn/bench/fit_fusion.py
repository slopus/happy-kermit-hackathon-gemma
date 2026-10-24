"""Fit the live fusion (logistic regression, portable: weights in models/fusion.json) and pick its policy.

Live features only (what endpoint.Endpointer computes): silence, turn-1-mini, Smart Turn once at 0.2 s of silence,
LiveKit text on Nemotron partials, text form. Cross-validated by turn on eot-bench + the Kermit demo set; then
refit on everything and choose (threshold, action_delay, timeout) for a cut-in budget.

usage: fit_fusion.py [--budget 0.05]
"""
import json, math, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from common import ROOT, OUT, evaluate
import eval as EV

LIVE = ["silence_dur", "sil_log", "t1m_l", "smart1_l", "lk_l", "n_words_l", "ends_q", "ends_p", "ends_comma", "last_filler"]


def logit(p): p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def live_frame(name):
    M = EV.merged(name)
    # Smart Turn is run once per pause in the endpointer, at the first evaluation with ≥ 0.2 s of silence
    first = M[M.silence_dur >= 0.2 - 1e-6].sort_values("silence_dur").groupby(["id", "span_index"]).smart.first()
    M["smart1"] = M.set_index(["id", "span_index"]).index.map(first).to_numpy()
    M.loc[M.silence_dur < 0.2 - 1e-6, "smart1"] = np.nan
    M["smart1"] = M.smart1.fillna(0.5)  # not run yet: uninformative
    M["sil_log"] = np.log(M.silence_dur + 0.1)
    M["t1m_l"], M["smart1_l"], M["lk_l"] = logit(M.t1m), logit(M.smart1), logit(M.lk_nemo.fillna(0))
    M["n_words_l"] = np.log1p(M.n_words)
    M["ds"] = name
    return M


def cv(M, feats, C=1.0):
    ids = M.id.unique(); rng = np.random.default_rng(0); rng.shuffle(ids)
    fold = M.id.map({i: k % 5 for k, i in enumerate(ids)}).to_numpy(); p = np.zeros(len(M))
    y = (M.label == "eot").to_numpy()
    for k in range(5):
        tr = fold != k
        mu, sd = M.loc[tr, feats].mean(), M.loc[tr, feats].std() + 1e-6
        m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit((M.loc[tr, feats] - mu) / sd, y[tr])
        p[~tr] = m.predict_proba((M.loc[~tr, feats] - mu) / sd)[:, 1]
    return p


def policy(M, col, budget):
    """Lowest mean latency over (threshold, action_delay, timeout) with cut-ins ≤ budget on both datasets."""
    best = None
    trades = {ds: evaluate(M[M.ds == ds], col)[2] for ds in M.ds.unique()}
    key = ["threshold", "action_delay", "timeout"]
    J = None
    for ds, t in trades.items():
        t = t[t.policy_type == "model"][key + ["cutoff_rate", "mean_latency"]].rename(columns={"cutoff_rate": f"cut_{ds}", "mean_latency": f"lat_{ds}"})
        J = t if J is None else J.merge(t, on=key)
    ok = J[(J[[c for c in J.columns if c.startswith("cut_")]] <= budget + 1e-9).all(axis=1)].copy()
    ok["lat"] = ok[[c for c in J.columns if c.startswith("lat_")]].mean(axis=1)
    return ok.sort_values(["lat", "threshold"], ascending=[True, False]).iloc[0].to_dict() if len(ok) else None


if __name__ == "__main__":
    budget = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 0.05
    M = pd.concat([live_frame("eotbench"), live_frame("kermit")], ignore_index=True)
    M["fuse_live"] = cv(M, LIVE)
    M["fuse_live_nolk"] = cv(M, [f for f in LIVE if f != "lk_l"])
    M["fuse_live_audio"] = cv(M, ["silence_dur", "sil_log", "t1m_l", "smart1_l"])
    rep = []
    for ds in ("eotbench", "kermit"):
        for col in ("t1m", "smart", "lk_nemo", "fuse_live_audio", "fuse_live_nolk", "fuse_live"):
            r, v, _ = evaluate(M[M.ds == ds], col)
            rep.append(dict(ds=ds, method=col, **{k: round(x, 3) for k, x in r.items()}))
        rep.append(dict(ds=ds, method="silence only", **{k: round(x, 3) for k, x in v.items()}))
    R = pd.DataFrame(rep); print(R.to_string(index=False))
    R.to_csv(f"{OUT}/fusion_cv.csv", index=False)
    pol = {b: policy(M, "fuse_live", b) for b in (0.02, 0.05, 0.10)}
    for b, p in pol.items(): print("budget", b, p)
    # refit on everything
    y = (M.label == "eot").to_numpy(); mu, sd = M[LIVE].mean(), M[LIVE].std() + 1e-6
    m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit((M[LIVE] - mu) / sd, y)
    p = pol[budget] or pol[0.10]
    spec = policy(M, "fuse_live", 0.25)  # speculative start: a looser point that fires earlier
    j = {"features": LIVE, "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
         "mean": mu.tolist(), "std": sd.tolist(),
         "policy": {"threshold": p["threshold"], "action_delay": p["action_delay"], "timeout": p["timeout"],
                    "spec_threshold": spec["threshold"] if spec else p["threshold"] * 0.6, "budget": budget},
         "cv": {f"{b}": pp for b, pp in pol.items()}}
    json.dump(j, open(os.path.join(ROOT, "models/fusion.json"), "w"), indent=1, default=float)
    M[["id", "span_index", "timestamp", "silence_dur", "label", "ds", "fuse_live", "text_nemo"]].to_parquet(f"{OUT}/fuse_live.parquet")
    print(json.dumps(j["policy"]), dict(zip(LIVE, np.round(m.coef_[0], 2))))
