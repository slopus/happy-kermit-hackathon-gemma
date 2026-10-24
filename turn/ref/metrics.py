from __future__ import annotations

import math

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = np.round(np.arange(0.0, 1.01, 0.01), 2).tolist()
DEFAULT_ACTION_DELAYS = np.round(np.arange(0.2, 1.01, 0.1), 2).tolist()
DEFAULT_TIMEOUTS = np.round(np.arange(1.0, 3.51, 0.5), 2).tolist()
DEFAULT_MIN_HOLD_SPAN_DURATION = 0.2
DEFAULT_MAX_HOLD_SPAN_DURATION = 5.0
# Backward-compatible name used by callers/tests before the joint-policy metric.
DEFAULT_MAX_SPAN_DURATION = DEFAULT_MAX_HOLD_SPAN_DURATION
CUTOFF_BUDGETS = (0.02, 0.05, 0.10)
FRONTIER_MAX_CUTOFF = 0.10
FRONTIER_BUDGET_STEP = 0.001
EPS = 1e-9


def load_predictions(predictions_path: str) -> pd.DataFrame:
    return pd.read_parquet(predictions_path)


def span_table_from_predictions(
    predictions_df: pd.DataFrame,
    *,
    min_hold_span_duration: float = DEFAULT_MIN_HOLD_SPAN_DURATION,
    max_hold_span_duration: float = DEFAULT_MAX_HOLD_SPAN_DURATION,
) -> pd.DataFrame:
    """Build the canonical span denominator from prediction rows.

    Hold/pause spans are filtered by duration. True EoT spans are kept and must
    later have an official score, otherwise the evaluation fails.
    """

    required = {"id", "span_index", "label", "silence_dur"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"prediction dataframe missing columns: {sorted(missing)}")
    if min_hold_span_duration < 0:
        raise ValueError("min_hold_span_duration must be non-negative")
    if max_hold_span_duration < min_hold_span_duration:
        raise ValueError("max_hold_span_duration must be >= min_hold_span_duration")

    labels = predictions_df[["id", "span_index", "label"]].drop_duplicates()
    if labels.duplicated(["id", "span_index"]).any():
        dupes = labels[labels.duplicated(["id", "span_index"], keep=False)]
        raise ValueError(f"conflicting labels for spans; examples:\n{dupes.head(10)}")

    durations = (
        predictions_df.groupby(["id", "span_index"], as_index=False, sort=False)["silence_dur"]
        .max()
        .rename(columns={"silence_dur": "span_dur"})
    )
    spans = labels.merge(durations, on=["id", "span_index"], how="inner", validate="one_to_one")
    hold_in_range = (
        (spans["span_dur"] >= min_hold_span_duration - EPS)
        & (spans["span_dur"] <= max_hold_span_duration + EPS)
    )
    spans = spans[(spans["label"] != "hold") | hold_in_range].copy()
    if spans.empty:
        raise ValueError("No spans remain after hold-duration filtering")
    spans["span_duration"] = spans["span_dur"]
    return spans.sort_values(["id", "span_index"], kind="stable").reset_index(drop=True)


def official_score_table(
    predictions_df: pd.DataFrame,
    spans: pd.DataFrame | None = None,
    *,
    score_point_s: float,
    min_hold_span_duration: float = DEFAULT_MIN_HOLD_SPAN_DURATION,
    max_hold_span_duration: float = DEFAULT_MAX_HOLD_SPAN_DURATION,
) -> pd.DataFrame:
    """Attach one official model score per span.

    The official score is the first prediction exactly at `score_point_s`. Missing
    scores are allowed only for hold spans that end before the score point.
    """

    if score_point_s < 0:
        raise ValueError("score_point_s must be non-negative")
    if spans is None:
        spans = span_table_from_predictions(
            predictions_df,
            min_hold_span_duration=min_hold_span_duration,
            max_hold_span_duration=max_hold_span_duration,
        )

    scored = (
        predictions_df[predictions_df["silence_dur"] >= score_point_s - EPS]
        .sort_values(["id", "span_index", "silence_dur", "timestamp"], kind="stable")
        .groupby(["id", "span_index"], as_index=False, sort=False)
        .first()[["id", "span_index", "p_eot", "silence_dur"]]
        .rename(columns={"silence_dur": "score_time"})
    )
    if scored.duplicated(["id", "span_index"]).any():
        raise ValueError("multiple official scores after grouping")

    out = spans.merge(scored, on=["id", "span_index"], how="left", validate="one_to_one")

    expected_scored = out["span_dur"] >= score_point_s - EPS
    has_score = out["p_eot"].notna()
    missing_expected = out[expected_scored & ~has_score]
    if not missing_expected.empty:
        raise ValueError(
            f"missing score at {score_point_s}s on {len(missing_expected)} spans that survive to the score point; "
            f"examples:\n{missing_expected.head(10)}",
        )

    wrong_time = out[has_score & ~np.isclose(out["score_time"], score_point_s, atol=EPS)]
    if not wrong_time.empty:
        raise ValueError(
            f"selected score time differs from requested {score_point_s}s; "
            f"examples:\n{wrong_time.head(10)}",
        )

    missing_eot = out[(out["label"] == "eot") & ~has_score]
    if not missing_eot.empty:
        raise ValueError(
            f"missing official score for true EoT spans; examples:\n{missing_eot.head(10)}",
        )

    out["score_point_s"] = float(score_point_s)
    out["score_mode"] = "score_point"
    return out


def max_score_table(
    predictions_df: pd.DataFrame,
    spans: pd.DataFrame | None = None,
    *,
    min_hold_span_duration: float = DEFAULT_MIN_HOLD_SPAN_DURATION,
    max_hold_span_duration: float = DEFAULT_MAX_HOLD_SPAN_DURATION,
) -> pd.DataFrame:
    """Attach one classification score per span using max p(eot).

    This is the scalar summary used for streaming adapters that produce scores
    over time instead of a single natural score point.
    """

    required = {"id", "span_index", "p_eot", "silence_dur"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"prediction dataframe missing columns: {sorted(missing)}")

    if spans is None:
        spans = span_table_from_predictions(
            predictions_df,
            min_hold_span_duration=min_hold_span_duration,
            max_hold_span_duration=max_hold_span_duration,
        )

    scoped = _prediction_rows_for_spans(predictions_df, spans)
    scored = (
        scoped[scoped["p_eot"].notna()]
        .sort_values(
            ["id", "span_index", "p_eot", "silence_dur", "timestamp"],
            ascending=[True, True, False, True, True],
            kind="stable",
        )
        .groupby(["id", "span_index"], as_index=False, sort=False)
        .first()[["id", "span_index", "p_eot", "silence_dur"]]
        .rename(columns={"silence_dur": "score_time"})
    )

    out = spans.merge(scored, on=["id", "span_index"], how="left", validate="one_to_one")
    missing_eot = out[(out["label"] == "eot") & out["p_eot"].isna()]
    if not missing_eot.empty:
        raise ValueError(
            f"missing max score for true EoT spans; examples:\n{missing_eot.head(10)}",
        )

    out["score_point_s"] = math.nan
    out["score_mode"] = "max"
    return out


def compute_classification_metrics(scored_spans: pd.DataFrame) -> dict:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for compute-metrics") from exc

    scored_only = scored_spans[scored_spans["p_eot"].notna()].copy()
    if scored_only.empty:
        raise ValueError("No scored spans available for classification metrics")
    y_true = (scored_only["label"] == "eot").astype(int).to_numpy()
    y_score = scored_only["p_eot"].to_numpy(dtype=float)

    auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else math.nan
    ap = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else math.nan
    return {
        "auc": auc,
        "ap": ap,
        "n_spans": int(len(scored_spans)),
        "n_hold_spans": int((scored_spans["label"] == "hold").sum()),
        "n_eot_spans": int((scored_spans["label"] == "eot").sum()),
        "n_scored_spans": int(len(scored_only)),
        "n_scored_hold_spans": int((scored_only["label"] == "hold").sum()),
        "n_scored_eot_spans": int((scored_only["label"] == "eot").sum()),
        "n_missing_scores": int(scored_spans["p_eot"].isna().sum()),
        "n_missing_hold_scores": int(((scored_spans["label"] == "hold") & scored_spans["p_eot"].isna()).sum()),
        "n_missing_eot_scores": int(((scored_spans["label"] == "eot") & scored_spans["p_eot"].isna()).sum()),
    }


def classification_curve(
    scored_spans: pd.DataFrame,
    *,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    thresholds_arr = _validate_grid(thresholds, name="thresholds")

    scored_only = scored_spans[scored_spans["p_eot"].notna()].copy()
    hold_scores = scored_only.loc[scored_only["label"] == "hold", "p_eot"].to_numpy(dtype=float)
    eot_scores = scored_only.loc[scored_only["label"] == "eot", "p_eot"].to_numpy(dtype=float)
    if hold_scores.size == 0 or eot_scores.size == 0:
        raise ValueError("need both hold and eot scored spans")

    return pd.DataFrame(
        [
            {
                "threshold": float(threshold),
                "cutoff_rate": float((hold_scores > threshold).mean()),
                "detect_rate": float((eot_scores > threshold).mean()),
            }
            for threshold in thresholds_arr
        ],
    )


def compute_joint_policy_sweep(
    scored_spans: pd.DataFrame,
    *,
    thresholds: list[float] | None = None,
    action_delays: list[float] | None = None,
    timeouts: list[float] | None = None,
    include_vad_baseline: bool = True,
    prediction_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Sweep threshold, action delay, and timeout from per-span fire times."""

    thresholds_arr, action_delay_arr, timeout_arr = _policy_grids(
        thresholds=thresholds,
        action_delays=action_delays,
        timeouts=timeouts,
    )
    model_action_delays = action_delay_arr
    if prediction_rows is None:
        score_point_s = _single_score_point(scored_spans)
        model_action_delays = action_delay_arr[action_delay_arr >= score_point_s - EPS]
        if model_action_delays.size == 0:
            raise ValueError(f"no action delays are >= score point {score_point_s}")

    rows = _policy_sweep_rows(
        scored_spans,
        thresholds_arr=thresholds_arr,
        action_delay_arr=model_action_delays,
        timeout_arr=timeout_arr,
        prediction_rows=prediction_rows,
    )
    if include_vad_baseline:
        vad_delays = np.sort(np.unique(np.concatenate([action_delay_arr, timeout_arr])))
        rows.extend(_vad_baseline_rows_from_scored_spans(scored_spans, delay_arr=vad_delays))

    return _tradeoff_frame(rows)


def compute_tradeoff(
    predictions_df: pd.DataFrame,
    *,
    score_point_s: float | None = None,
    thresholds: list[float] | None = None,
    action_delays: list[float] | None = None,
    timeouts: list[float] | None = None,
    delay_values: list[float] | None = None,
    min_hold_span_duration: float = DEFAULT_MIN_HOLD_SPAN_DURATION,
    max_hold_span_duration: float = DEFAULT_MAX_HOLD_SPAN_DURATION,
    max_span_duration: float | None = None,
    include_vad_baseline: bool = True,
) -> pd.DataFrame:
    if delay_values is not None:
        if action_delays is not None or timeouts is not None:
            raise ValueError("Use either delay_values or action_delays/timeouts, not both")
        action_delays = delay_values
        timeouts = delay_values
    if max_span_duration is not None:
        max_hold_span_duration = max_span_duration
    _validate_action_delay_floor(action_delays, min_hold_span_duration=min_hold_span_duration)
    _validate_action_delay_floor(timeouts, min_hold_span_duration=min_hold_span_duration)

    if score_point_s is not None:
        scored_spans = official_score_table(
            predictions_df,
            score_point_s=score_point_s,
            min_hold_span_duration=min_hold_span_duration,
            max_hold_span_duration=max_hold_span_duration,
        )
        return compute_joint_policy_sweep(
            scored_spans,
            thresholds=thresholds,
            action_delays=action_delays,
            timeouts=timeouts,
            include_vad_baseline=include_vad_baseline,
        )

    spans = span_table_from_predictions(
        predictions_df,
        min_hold_span_duration=min_hold_span_duration,
        max_hold_span_duration=max_hold_span_duration,
    )
    return compute_joint_policy_sweep(
        spans,
        thresholds=thresholds,
        action_delays=action_delays,
        timeouts=timeouts,
        include_vad_baseline=include_vad_baseline,
        prediction_rows=_prediction_rows_for_spans(predictions_df, spans),
    )


def compute_scalar_metrics(
    predictions_df: pd.DataFrame,
    *,
    score_point_s: float | None = None,
    min_hold_span_duration: float = DEFAULT_MIN_HOLD_SPAN_DURATION,
    max_hold_span_duration: float = DEFAULT_MAX_HOLD_SPAN_DURATION,
    max_span_duration: float | None = None,
) -> dict:
    if max_span_duration is not None:
        max_hold_span_duration = max_span_duration

    if score_point_s is None:
        scored_spans = max_score_table(
            predictions_df,
            min_hold_span_duration=min_hold_span_duration,
            max_hold_span_duration=max_hold_span_duration,
        )
    else:
        scored_spans = official_score_table(
            predictions_df,
            score_point_s=score_point_s,
            min_hold_span_duration=min_hold_span_duration,
            max_hold_span_duration=max_hold_span_duration,
        )
    return compute_classification_metrics(scored_spans)


def summarize_tradeoff(
    tradeoff_df: pd.DataFrame,
    *,
    cutoff_budgets: tuple[float, ...] = CUTOFF_BUDGETS,
    frontier_max_cutoff: float = FRONTIER_MAX_CUTOFF,
    frontier_budget_step: float = FRONTIER_BUDGET_STEP,
) -> dict:
    model_tradeoff = tradeoff_df[tradeoff_df["policy_type"] == "model"].copy()
    operating_points = {
        _format_budget_key(budget): _best_operating_point(model_tradeoff, budget)
        for budget in cutoff_budgets
    }

    budgets = np.round(np.arange(0.0, frontier_max_cutoff + frontier_budget_step, frontier_budget_step), 6)
    frontier_latencies = []
    for budget in budgets:
        point = _best_operating_point(model_tradeoff, float(budget))
        if point is not None:
            frontier_latencies.append(point["mean_latency"])

    mean_frontier_latency = float(np.mean(frontier_latencies)) if frontier_latencies else math.nan
    return {
        "operating_points": operating_points,
        "mean_frontier_latency_0_10": mean_frontier_latency,
    }


def get_pareto_operating_points(
    sweep: pd.DataFrame,
    budget_column: str,
    budget_values: list[float] | tuple[float, ...],
) -> pd.DataFrame:
    if budget_column == "mean_latency":
        optimize_column = "cutoff_rate"
        mode = "min_cutoff_under_latency"
    elif budget_column == "cutoff_rate":
        optimize_column = "mean_latency"
        mode = "min_latency_under_cutoff"
    else:
        raise ValueError("budget_column must be 'mean_latency' or 'cutoff_rate'")

    model_sweep = sweep[sweep["policy_type"] == "model"].copy() if "policy_type" in sweep.columns else sweep.copy()
    model_values = [None]
    if "model" in model_sweep.columns:
        model_values = list(model_sweep["model"].drop_duplicates())

    rows = []
    for model in model_values:
        mdf = model_sweep.copy() if model is None else model_sweep[model_sweep["model"] == model].copy()
        if mdf.empty:
            raise ValueError(f"no sweep rows for model={model!r}")
        for budget in budget_values:
            feasible = mdf[mdf[budget_column] <= float(budget) + EPS].copy()
            if feasible.empty:
                row = {
                    "budget_column": budget_column,
                    "budget_value": float(budget),
                    "optimize_column": optimize_column,
                    "mode": mode,
                    "feasible": False,
                    "value": math.nan,
                }
                if model is not None:
                    row["model"] = model
                rows.append(row)
                continue
            feasible = feasible.sort_values(
                [optimize_column, budget_column, "threshold", "action_delay", "timeout"],
                ascending=[True, True, False, True, True],
                kind="stable",
            )
            best = feasible.iloc[0].to_dict()
            best.update(
                {
                    "budget_column": budget_column,
                    "budget_value": float(budget),
                    "optimize_column": optimize_column,
                    "mode": mode,
                    "feasible": True,
                    "value": float(best[optimize_column]),
                },
            )
            rows.append(best)
    return pd.DataFrame(rows)


def compute_metrics_from_predictions(
    predictions_df: pd.DataFrame,
    *,
    score_point_s: float | None = None,
    thresholds: list[float] | None = None,
    action_delays: list[float] | None = None,
    timeouts: list[float] | None = None,
    delay_values: list[float] | None = None,
    min_hold_span_duration: float = DEFAULT_MIN_HOLD_SPAN_DURATION,
    max_hold_span_duration: float = DEFAULT_MAX_HOLD_SPAN_DURATION,
    max_span_duration: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    if delay_values is not None and (action_delays is not None or timeouts is not None):
        raise ValueError("Use either delay_values or action_delays/timeouts, not both")
    if max_span_duration is not None:
        max_hold_span_duration = max_span_duration

    sweep_action_delays = action_delays if action_delays is not None else delay_values
    sweep_timeouts = timeouts if timeouts is not None else delay_values
    _validate_action_delay_floor(sweep_action_delays, min_hold_span_duration=min_hold_span_duration)
    _validate_action_delay_floor(sweep_timeouts, min_hold_span_duration=min_hold_span_duration)

    if score_point_s is None:
        spans = span_table_from_predictions(
            predictions_df,
            min_hold_span_duration=min_hold_span_duration,
            max_hold_span_duration=max_hold_span_duration,
        )
        scored_spans = max_score_table(
            predictions_df,
            spans=spans,
        )
        tradeoff = compute_joint_policy_sweep(
            spans,
            thresholds=thresholds,
            action_delays=sweep_action_delays,
            timeouts=sweep_timeouts,
            include_vad_baseline=True,
            prediction_rows=_prediction_rows_for_spans(predictions_df, spans),
        )
        score_mode = "max"
    else:
        scored_spans = official_score_table(
            predictions_df,
            score_point_s=score_point_s,
            min_hold_span_duration=min_hold_span_duration,
            max_hold_span_duration=max_hold_span_duration,
        )
        tradeoff = compute_joint_policy_sweep(
            scored_spans,
            thresholds=thresholds,
            action_delays=sweep_action_delays,
            timeouts=sweep_timeouts,
            include_vad_baseline=True,
        )
        score_mode = "score_point"

    summary = compute_classification_metrics(scored_spans)
    summary.update(
        summarize_tradeoff(
            tradeoff,
            cutoff_budgets=CUTOFF_BUDGETS,
            frontier_max_cutoff=FRONTIER_MAX_CUTOFF,
        ),
    )
    summary["score_mode"] = score_mode
    summary["score_point_s"] = None if score_point_s is None else float(score_point_s)
    summary["min_hold_span_duration"] = float(min_hold_span_duration)
    summary["max_hold_span_duration"] = float(max_hold_span_duration)
    summary["sweep_action_delays"] = [float(v) for v in _grid_or_default(sweep_action_delays, DEFAULT_ACTION_DELAYS)]
    summary["sweep_timeouts"] = [float(v) for v in _grid_or_default(sweep_timeouts, DEFAULT_TIMEOUTS)]
    summary["sweep_delay_values"] = sorted(
        {float(v) for v in summary["sweep_action_delays"] + summary["sweep_timeouts"]},
    )
    return tradeoff, summary


def pareto_mask(tradeoff_df: pd.DataFrame) -> np.ndarray:
    ordered = tradeoff_df.sort_values(
        [
            "mean_latency",
            "cutoff_rate",
            "threshold",
            "action_delay",
            "timeout",
        ],
        ascending=[True, True, False, True, True],
        kind="stable",
    )
    mask = pd.Series(False, index=tradeoff_df.index, dtype=bool)
    best_cutoff = math.inf
    for row in ordered.itertuples():
        cutoff = float(row.cutoff_rate)
        if cutoff < best_cutoff - EPS:
            mask.loc[row.Index] = True
            best_cutoff = cutoff
    return mask.to_numpy(dtype=bool)


def pareto_frontier(tradeoff_df: pd.DataFrame) -> pd.DataFrame:
    model_df = tradeoff_df[tradeoff_df["policy_type"] == "model"].copy() if "policy_type" in tradeoff_df.columns else tradeoff_df.copy()
    if model_df.empty:
        raise ValueError("no model sweep rows")
    if "model" not in model_df.columns:
        out = model_df.copy()
        out["is_pareto"] = pareto_mask(out)
        return out[out["is_pareto"]].copy().reset_index(drop=True)

    parts = []
    for model, group in model_df.groupby("model", sort=False):
        group = group.copy()
        group["is_pareto"] = pareto_mask(group)
        parts.append(group[group["is_pareto"]].copy())
    return pd.concat(parts, ignore_index=True)


def _policy_grids(
    *,
    thresholds: list[float] | None,
    action_delays: list[float] | None,
    timeouts: list[float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        _validate_grid(DEFAULT_THRESHOLDS if thresholds is None else thresholds, name="thresholds"),
        _validate_grid(DEFAULT_ACTION_DELAYS if action_delays is None else action_delays, name="action_delays"),
        _validate_grid(DEFAULT_TIMEOUTS if timeouts is None else timeouts, name="timeouts"),
    )


def _tradeoff_frame(rows: list[dict]) -> pd.DataFrame:
    tradeoff = pd.DataFrame(rows).sort_values(
        [
            "policy_type",
            "cutoff_rate",
            "mean_latency",
            "timeout_rate",
            "timeout",
            "action_delay",
            "threshold",
        ],
        kind="stable",
    ).reset_index(drop=True)
    tradeoff["is_pareto"] = False
    model_mask = tradeoff["policy_type"] == "model"
    if model_mask.any():
        tradeoff.loc[model_mask, "is_pareto"] = pareto_mask(tradeoff.loc[model_mask])
    return tradeoff


def _prediction_rows_for_spans(predictions_df: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "span_index", "p_eot", "silence_dur"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"prediction dataframe missing columns: {sorted(missing)}")

    columns = ["id", "span_index", "p_eot", "silence_dur"]
    if "timestamp" in predictions_df.columns:
        columns.append("timestamp")
    scoped = predictions_df[columns].merge(
        spans[["id", "span_index"]],
        on=["id", "span_index"],
        how="inner",
        validate="many_to_one",
    )
    if scoped.empty:
        raise ValueError("No prediction rows remain after span filtering")
    if "timestamp" not in scoped.columns:
        scoped["timestamp"] = scoped["silence_dur"]
    return scoped


def _policy_sweep_rows(
    spans: pd.DataFrame,
    *,
    thresholds_arr: np.ndarray,
    action_delay_arr: np.ndarray,
    timeout_arr: np.ndarray,
    prediction_rows: pd.DataFrame | None = None,
) -> list[dict]:
    hold = spans[spans["label"] == "hold"].copy()
    eot = spans[spans["label"] == "eot"].copy()
    if hold.empty or eot.empty:
        raise ValueError("need both hold and eot spans")

    hold_dur = hold["span_dur"].to_numpy(dtype=float)
    score_mode = "max" if prediction_rows is not None else "score_point"
    score_point_s = math.nan if prediction_rows is not None else _single_score_point(spans)
    rows: list[dict] = []
    for threshold in thresholds_arr:
        if prediction_rows is None:
            hold_fire = _score_point_fire_times(hold, threshold=float(threshold))
            eot_fire = _score_point_fire_times(eot, threshold=float(threshold))
        else:
            crossing = _first_crossing_times(prediction_rows, threshold=float(threshold))
            hold_fire = _fire_times_from_crossing(hold, crossing)
            eot_fire = _fire_times_from_crossing(eot, crossing)
        hold_has_fire = np.isfinite(hold_fire)
        eot_has_fire = np.isfinite(eot_fire)
        for action_delay in action_delay_arr:
            hold_model_fire_time = np.maximum(action_delay, hold_fire)
            model_cutoff = hold_has_fire & (hold_dur > hold_model_fire_time + EPS)
            eot_model_fire_time = np.maximum(action_delay, eot_fire)
            for timeout in timeout_arr:
                if timeout + EPS < action_delay:
                    continue

                timeout_cutoff = hold_dur > timeout + EPS
                cutoff = timeout_cutoff | model_cutoff
                model_detect = eot_has_fire & (eot_model_fire_time <= timeout + EPS)
                eot_latency = np.where(model_detect, eot_model_fire_time, timeout)

                tp = int(model_detect.sum())
                fp = int(cutoff.sum())
                fn = int(len(eot) - tp)
                precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
                recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
                f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
                row = {
                    "policy_type": "model",
                    "threshold": float(threshold),
                    "action_delay": float(action_delay),
                    "timeout": float(timeout),
                    "score_point_s": score_point_s,
                    "score_mode": score_mode,
                    "cutoff_rate": float(cutoff.mean()),
                    "mean_latency": float(eot_latency.mean()),
                    "timeout_rate": float((~model_detect).mean()),
                    "detect_rate": float(model_detect.mean()),
                    "f1": f1,
                    "n_hold_spans": int(len(hold)),
                    "n_eot_spans": int(len(eot)),
                }
                if "model" in spans.columns:
                    row["model"] = _single_model(spans)
                rows.append(row)
    return rows


def _score_point_fire_times(spans: pd.DataFrame, *, threshold: float) -> np.ndarray:
    has_fire = spans["p_eot"].notna() & (spans["p_eot"] > threshold)
    score_time = spans["score_time"].fillna(np.inf).to_numpy(dtype=float)
    return np.where(has_fire, score_time, np.inf)


def _fire_times_from_crossing(spans: pd.DataFrame, crossing: pd.DataFrame) -> np.ndarray:
    merged = spans[["id", "span_index"]].merge(crossing, on=["id", "span_index"], how="left", validate="one_to_one")
    return pd.to_numeric(merged["fire_time"], errors="coerce").fillna(np.inf).to_numpy(dtype=float)


def _first_crossing_times(prediction_rows: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    positive = prediction_rows[prediction_rows["p_eot"] > threshold].copy()
    if positive.empty:
        return pd.DataFrame(columns=["id", "span_index", "fire_time"])
    return (
        positive.sort_values(["id", "span_index", "silence_dur", "timestamp"], kind="stable")
        .groupby(["id", "span_index"], as_index=False, sort=False)
        .first()[["id", "span_index", "silence_dur"]]
        .rename(columns={"silence_dur": "fire_time"})
    )


def _vad_baseline_rows_from_scored_spans(scored_spans: pd.DataFrame, *, delay_arr: np.ndarray) -> list[dict]:
    hold_durations = scored_spans.loc[scored_spans["label"] == "hold", "span_dur"].to_numpy(dtype=float)
    eot_count = int((scored_spans["label"] == "eot").sum())
    n_hold_spans = int(len(hold_durations))
    n_eot_spans = int(eot_count)

    rows: list[dict] = []
    for delay in delay_arr:
        hold_triggered = hold_durations > delay + EPS
        tp = n_eot_spans
        fp = int(hold_triggered.sum())
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = 1.0 if n_eot_spans else 0.0
        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
        rows.append(
            {
                "policy_type": "vad",
                "threshold": math.nan,
                "action_delay": float(delay),
                "timeout": float(delay),
                "score_point_s": math.nan,
                "score_mode": "vad",
                "cutoff_rate": float(hold_triggered.mean()) if n_hold_spans else 0.0,
                "mean_latency": float(delay) if n_eot_spans else math.nan,
                "timeout_rate": 0.0,
                "detect_rate": 1.0 if n_eot_spans else 0.0,
                "f1": f1,
                "n_hold_spans": int(n_hold_spans),
                "n_eot_spans": int(n_eot_spans),
            },
        )
    return rows


def _best_operating_point(tradeoff_df: pd.DataFrame, cutoff_budget: float) -> dict | None:
    feasible = tradeoff_df[tradeoff_df["cutoff_rate"] <= cutoff_budget + EPS].copy()
    if feasible.empty:
        return None

    best = feasible.sort_values(
        [
            "mean_latency",
            "cutoff_rate",
            "timeout_rate",
            "timeout",
            "action_delay",
            "threshold",
        ],
        kind="stable",
    ).iloc[0]
    score_point = best.get("score_point_s", math.nan)
    return {
        "cutoff_budget": float(cutoff_budget),
        "threshold": float(best["threshold"]),
        "cutoff_rate": float(best["cutoff_rate"]),
        "mean_latency": float(best["mean_latency"]),
        "timeout_rate": float(best["timeout_rate"]),
        "detect_rate": float(best["detect_rate"]),
        "action_delay": float(best["action_delay"]),
        "timeout": float(best["timeout"]),
        "score_point_s": None if pd.isna(score_point) else float(score_point),
        "score_mode": str(best.get("score_mode", "score_point")),
    }


def _format_budget_key(budget: float) -> str:
    return f"{int(round(budget * 100))}pct"


def _validate_grid(values: list[float] | tuple[float, ...], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError(f"{name} must be a non-empty 1D list")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.sort(np.unique(arr))


def _grid_or_default(values: list[float] | None, default: list[float]) -> list[float]:
    if values is None:
        return [float(v) for v in default]
    return [float(v) for v in _validate_grid(values, name="grid")]


def _validate_action_delay_floor(values: list[float] | None, *, min_hold_span_duration: float) -> None:
    if values is None:
        return
    arr = _validate_grid(values, name="action_delays")
    below = arr[arr < float(min_hold_span_duration) - EPS]
    if below.size:
        raise ValueError(
            "action delays/timeouts must be >= min_hold_span_duration "
            f"({min_hold_span_duration}); got {below.tolist()}",
        )


def _single_score_point(scored_spans: pd.DataFrame) -> float:
    values = scored_spans["score_point_s"].dropna().unique()
    if len(values) != 1:
        raise ValueError(f"expected exactly one score_point_s, got {values}")
    return float(values[0])


def _single_model(scored_spans: pd.DataFrame) -> str:
    values = scored_spans["model"].dropna().unique()
    if len(values) != 1:
        raise ValueError(f"expected exactly one model, got {values}")
    return str(values[0])


# Backward-compatible private aliases retained for older tests/imports.
def _span_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    spans = span_table_from_predictions(
        predictions_df,
        min_hold_span_duration=0.0,
        max_hold_span_duration=float("inf"),
    )
    return spans[["id", "span_index", "label", "span_duration"]].copy()


def _filter_predictions_by_span_duration(
    predictions_df: pd.DataFrame,
    *,
    max_span_duration: float,
) -> pd.DataFrame:
    spans = span_table_from_predictions(
        predictions_df,
        min_hold_span_duration=0.0,
        max_hold_span_duration=max_span_duration,
    )
    keep = spans[["id", "span_index"]]
    filtered = predictions_df.merge(keep, on=["id", "span_index"], how="inner")
    if filtered.empty:
        raise ValueError(f"No spans remain after filtering to span_duration <= {max_span_duration}")
    return filtered


def _pareto_mask(tradeoff_df: pd.DataFrame) -> np.ndarray:
    return pareto_mask(tradeoff_df)
