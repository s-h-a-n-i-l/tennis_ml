"""Utilities for constructing match state panels from point-level data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import (
    aces,
    aces_diff,
    aces_l60,
    ace_rate,
    ace_rate_l60,
    avg_srv_speed,
    avg_srv_speed_l60,
    best_of,
    bp_conv_rate,
    bp_defend_rate,
    df_diff,
    df_rate,
    df_rate_l60,
    double_faults,
    fs_in_pct,
    fs_in_pct_l60,
    fs_win_pct,
    fs_win_pct_l60,
    games_in_set_against,
    games_in_set_for,
    is_break_point,
    is_game_point_against,
    is_game_point_for,
    is_tiebreak,
    opp_aces,
    opp_aces_l60,
    opp_ace_rate,
    opp_ace_rate_l60,
    opp_avg_srv_speed,
    opp_avg_srv_speed_l60,
    opp_bp_conv_rate,
    opp_bp_defend_rate,
    opp_df_rate,
    opp_df_rate_l60,
    opp_double_faults,
    opp_fs_in_pct,
    opp_fs_in_pct_l60,
    opp_fs_win_pct,
    opp_fs_win_pct_l60,
    opp_ret_win_pct,
    opp_ret_win_pct_l60,
    opp_ss_in_pct,
    opp_ss_in_pct_l60,
    opp_ss_win_pct,
    opp_ss_win_pct_l60,
    point_idx,
    pts_in_game_against,
    pts_in_game_for,
    ret_win_pct,
    ret_win_pct_l60,
    server_is_persp,
    sets_against,
    sets_for,
    sets_needed_to_win,
    ss_in_pct,
    ss_in_pct_l60,
    ss_win_pct,
    ss_win_pct_l60,
    ttl_diff,
    y_match,
)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Safely divide two series, avoiding division by zero."""

    a = a.astype(float)
    b = b.astype(float)
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def _event_rolling_ratio(
    df: pd.DataFrame,
    num_cum_col: str,
    den_cum_col: str,
    event_cum_col: str,
    window: int,
) -> pd.Series:
    """Event-based rolling ratio over the last ``window`` events."""

    out = pd.Series(index=df.index, dtype=float)
    for mid, g in df.groupby("match_id", sort=False):
        events = g[event_cum_col].astype(float)
        ev_mask = events.diff().fillna(events) > 0
        if not ev_mask.any():
            out.loc[g.index] = 0.0
            continue
        ev_points = g.loc[ev_mask, [num_cum_col, den_cum_col]].copy()
        num_inc = ev_points[num_cum_col].diff().fillna(ev_points[num_cum_col])
        den_inc = ev_points[den_cum_col].diff().fillna(ev_points[den_cum_col])
        num_roll = num_inc.rolling(window, min_periods=1).sum()
        den_roll = den_inc.rolling(window, min_periods=1).sum()
        ratio = _safe_div(num_roll, den_roll)
        out.loc[g.index] = ratio.reindex(g.index).ffill().fillna(0.0).to_numpy()
    return out


def _event_rolling_sum(
    df: pd.DataFrame,
    cum_col: str,
    event_cum_col: str,
    window: int,
) -> pd.Series:
    """Event-based rolling sum over the last ``window`` events."""

    out = pd.Series(index=df.index, dtype=float)
    for mid, g in df.groupby("match_id", sort=False):
        events = g[event_cum_col].astype(float)
        ev_mask = events.diff().fillna(events) > 0
        if not ev_mask.any():
            out.loc[g.index] = 0.0
            continue
        ev_points = g.loc[ev_mask, [cum_col]].copy()
        increments = ev_points[cum_col].diff().fillna(ev_points[cum_col])
        rolling = increments.rolling(window, min_periods=1).sum()
        out.loc[g.index] = rolling.reindex(g.index).ffill().fillna(0.0).to_numpy()
    return out


def build_match_state_panel(
    input_data: pd.DataFrame,
    best_of_default: int = 5,
    ma_window: int = 60,
) -> pd.DataFrame:
    """Construct a point-level panel with state variables from both perspectives."""

    df = input_data.copy()

    needed = ["match_id", "SetNo", "GameNo", "PointNumber", "PointServer"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if "PointWinner" in df.columns:
        p1_won_point = (df["PointWinner"] == 1).astype(int)
        p2_won_point = (df["PointWinner"] == 2).astype(int)
    elif {"P1PointsWon", "P2PointsWon"}.issubset(df.columns):
        df = df.sort_values(["match_id", "SetNo", "GameNo", "PointNumber"]).copy()
        df["p1_cum_prev"] = df.groupby("match_id")["P1PointsWon"].shift(1).fillna(0)
        df["p2_cum_prev"] = df.groupby("match_id")["P2PointsWon"].shift(1).fillna(0)
        p1_won_point = (df["P1PointsWon"] > df["p1_cum_prev"]).astype(int)
        p2_won_point = (df["P2PointsWon"] > df["p2_cum_prev"]).astype(int)
    else:
        raise ValueError(
            "Need either PointWinner or (P1PointsWon, P2PointsWon) to derive point winners."
        )

    df = df.sort_values(["match_id", "SetNo", "GameNo", "PointNumber"]).copy()
    df["point_idx"] = point_idx.compute(df["match_id"])

    if "Time" in df.columns:
        df["_time_raw"] = pd.to_datetime(df["Time"])
        df["time"] = df.groupby("match_id")["_time_raw"].shift(1)
        df["elapsed_time"] = (
            df.groupby("match_id")["_time_raw"].transform(
                lambda s: (s - s.iloc[0]).dt.total_seconds()
            )
        )
        df["elapsed_time"] = df.groupby("match_id")["elapsed_time"].shift(1).fillna(0.0)
        df["elapsed_time"] = df["elapsed_time"].astype(float)
        df = df.drop(columns=["_time_raw"])
    else:
        df["time"] = pd.NaT
        df["elapsed_time"] = 0.0

    if "Speed_KMH" in df.columns:
        speed = df["Speed_KMH"].fillna(0).astype(float)
    else:
        speed = pd.Series(0.0, index=df.index)

    for side in (1, 2):
        srv_mask = (df["PointServer"] == side).astype(int)
        df[f"p{side}_sv_pts"] = srv_mask.groupby(df["match_id"]).cumsum().shift(1).fillna(0)
        df[f"p{side}_srv_speed_cum"] = (
            (speed * srv_mask).groupby(df["match_id"]).cumsum().shift(1).fillna(0)
        )
        df[f"p{side}_avg_srv_speed"] = _safe_div(
            df[f"p{side}_srv_speed_cum"], df[f"p{side}_sv_pts"]
        )

    df["ttl_p1"] = p1_won_point.groupby(df["match_id"]).cumsum().shift(1).fillna(0).astype(int)
    df["ttl_p2"] = p2_won_point.groupby(df["match_id"]).cumsum().shift(1).fillna(0).astype(int)

    srv_cols = {
        "P1FirstSrvIn",
        "P1FirstSrvWon",
        "P1SecondSrvIn",
        "P1SecondSrvWon",
        "P1DoubleFault",
        "P2FirstSrvIn",
        "P2FirstSrvWon",
        "P2SecondSrvIn",
        "P2SecondSrvWon",
        "P2DoubleFault",
    }
    have_srv = srv_cols.issubset(df.columns)

    for side in (1, 2):
        if have_srv:
            df[f"p{side}_fs_in_cum"] = (
                df[f"P{side}FirstSrvIn"].groupby(df["match_id"]).cumsum().shift(1).fillna(0)
            )
            df[f"p{side}_fs_won_cum"] = (
                df[f"P{side}FirstSrvWon"].groupby(df["match_id"]).cumsum().shift(1).fillna(0)
            )
            df[f"p{side}_ss_in_cum"] = (
                df[f"P{side}SecondSrvIn"].groupby(df["match_id"]).cumsum().shift(1).fillna(0)
            )
            df[f"p{side}_ss_won_cum"] = (
                df[f"P{side}SecondSrvWon"].groupby(df["match_id"]).cumsum().shift(1).fillna(0)
            )
            df[f"p{side}_df_cum"] = (
                df[f"P{side}DoubleFault"].groupby(df["match_id"]).cumsum().shift(1).fillna(0)
            )
        else:
            df[f"p{side}_fs_in_cum"] = 0.0
            df[f"p{side}_fs_won_cum"] = 0.0
            df[f"p{side}_ss_in_cum"] = 0.0
            df[f"p{side}_ss_won_cum"] = 0.0
            df[f"p{side}_df_cum"] = 0.0

        df[f"p{side}_fs_in_pct"] = _safe_div(df[f"p{side}_fs_in_cum"], df[f"p{side}_sv_pts"])
        df[f"p{side}_ss_att_cum"] = df[f"p{side}_ss_in_cum"] + df[f"p{side}_df_cum"]
        df[f"p{side}_ss_in_pct"] = _safe_div(
            df[f"p{side}_ss_in_cum"], df[f"p{side}_ss_att_cum"]
        )
        df[f"p{side}_fs_win_pct"] = _safe_div(
            df[f"p{side}_fs_won_cum"], df[f"p{side}_fs_in_cum"]
        )
        df[f"p{side}_ss_win_pct"] = _safe_div(
            df[f"p{side}_ss_won_cum"], df[f"p{side}_ss_in_cum"]
        )
        recv_mask = (df["PointServer"] != side).astype(int)
        won = (p1_won_point if side == 1 else p2_won_point) * recv_mask
        df[f"p{side}_ret_won_cum"] = (
            won.groupby(df["match_id"]).cumsum().shift(1).fillna(0)
        )
        df[f"p{side}_ret_pts_cum"] = (
            recv_mask.groupby(df["match_id"]).cumsum().shift(1).fillna(0)
        )
        df[f"p{side}_ret_win_pct"] = _safe_div(
            df[f"p{side}_ret_won_cum"], df[f"p{side}_ret_pts_cum"]
        )
        df[f"p{side}_double_faults"] = df[f"p{side}_df_cum"].astype(float)
        df[f"p{side}_df_rate"] = _safe_div(df[f"p{side}_df_cum"], df[f"p{side}_sv_pts"])

    if {"P1Ace", "P2Ace"}.issubset(df.columns):
        df["p1_aces_cum"] = df["P1Ace"].groupby(df["match_id"]).cumsum().shift(1).fillna(0)
        df["p2_aces_cum"] = df["P2Ace"].groupby(df["match_id"]).cumsum().shift(1).fillna(0)
    else:
        df["p1_aces_cum"] = 0.0
        df["p2_aces_cum"] = 0.0

    for side in (1, 2):
        df[f"p{side}_ace_rate"] = _safe_div(df[f"p{side}_aces_cum"], df[f"p{side}_sv_pts"])
        df[f"p{side}_avg_srv_speed_l60"] = _event_rolling_ratio(
            df, f"p{side}_srv_speed_cum", f"p{side}_sv_pts", f"p{side}_sv_pts", ma_window
        )
        df[f"p{side}_fs_in_pct_l60"] = _event_rolling_ratio(
            df, f"p{side}_fs_in_cum", f"p{side}_sv_pts", f"p{side}_sv_pts", ma_window
        )
        df[f"p{side}_ss_in_pct_l60"] = _event_rolling_ratio(
            df, f"p{side}_ss_in_cum", f"p{side}_ss_att_cum", f"p{side}_ss_att_cum", ma_window
        )
        df[f"p{side}_fs_win_pct_l60"] = _event_rolling_ratio(
            df, f"p{side}_fs_won_cum", f"p{side}_fs_in_cum", f"p{side}_fs_in_cum", ma_window
        )
        df[f"p{side}_ss_win_pct_l60"] = _event_rolling_ratio(
            df, f"p{side}_ss_won_cum", f"p{side}_ss_in_cum", f"p{side}_ss_in_cum", ma_window
        )
        df[f"p{side}_ret_win_pct_l60"] = _event_rolling_ratio(
            df, f"p{side}_ret_won_cum", f"p{side}_ret_pts_cum", f"p{side}_ret_pts_cum", ma_window
        )
        df[f"p{side}_aces_l60"] = _event_rolling_sum(
            df, f"p{side}_aces_cum", f"p{side}_sv_pts", ma_window
        )
        df[f"p{side}_ace_rate_l60"] = _event_rolling_ratio(
            df, f"p{side}_aces_cum", f"p{side}_sv_pts", f"p{side}_sv_pts", ma_window
        )
        df[f"p{side}_df_rate_l60"] = _event_rolling_ratio(
            df, f"p{side}_df_cum", f"p{side}_sv_pts", f"p{side}_sv_pts", ma_window
        )

    game_change = (
        (df["SetNo"].astype(str) + "-" + df["GameNo"].astype(str))
        .ne((df["SetNo"].astype(str) + "-" + df["GameNo"].astype(str)).shift(1))
    )
    df["game_key"] = game_change.groupby(df["match_id"]).cumsum()

    df["p1_pts_in_game"] = (
        p1_won_point.groupby([df["match_id"], df["game_key"]]).cumsum().shift(1).fillna(0).astype(int)
    )
    df["p2_pts_in_game"] = (
        p2_won_point.groupby([df["match_id"], df["game_key"]]).cumsum().shift(1).fillna(0).astype(int)
    )

    last_point_idx = df.groupby(["match_id", "SetNo", "GameNo"]).tail(1).index
    game_winner = pd.Series(
        index=last_point_idx,
        data=np.where(
            (p1_won_point + p2_won_point).loc[last_point_idx] == 1,
            np.where(p1_won_point.loc[last_point_idx] == 1, 1, 2),
            np.nan,
        ),
        dtype=float,
    )

    games_tbl = df.loc[last_point_idx, ["match_id", "SetNo", "GameNo"]].copy()
    games_tbl["p1_game_win"] = (game_winner == 1).astype(int)
    games_tbl["p2_game_win"] = (game_winner == 2).astype(int)
    games_tbl["p1_games_in_set_cum"] = (
        games_tbl.groupby(["match_id", "SetNo"])["p1_game_win"].cumsum().shift(1).fillna(0).astype(int)
    )
    games_tbl["p2_games_in_set_cum"] = (
        games_tbl.groupby(["match_id", "SetNo"])["p2_game_win"].cumsum().shift(1).fillna(0).astype(int)
    )

    df = df.merge(
        games_tbl[["match_id", "SetNo", "GameNo", "p1_games_in_set_cum", "p2_games_in_set_cum"]],
        on=["match_id", "SetNo", "GameNo"],
        how="left",
    )
    df = df.rename(
        columns={
            "p1_games_in_set_cum": "p1_games_before",
            "p2_games_in_set_cum": "p2_games_before",
        }
    )

    set_last_games = games_tbl.groupby(["match_id", "SetNo"]).tail(1).copy()
    set_last_games["set_winner"] = np.where(
        set_last_games["p1_games_in_set_cum"] > set_last_games["p2_games_in_set_cum"], 1, 2
    )
    set_last_games["p1_set_win"] = (set_last_games["set_winner"] == 1).astype(int)
    set_last_games["p2_set_win"] = (set_last_games["set_winner"] == 2).astype(int)
    set_last_games["p1_sets_before"] = (
        set_last_games.groupby("match_id")["p1_set_win"].cumsum().shift(1).fillna(0).astype(int)
    )
    set_last_games["p2_sets_before"] = (
        set_last_games.groupby("match_id")["p2_set_win"].cumsum().shift(1).fillna(0).astype(int)
    )

    df = (
        df.merge(
            set_last_games[["match_id", "SetNo", "p1_sets_before", "p2_sets_before"]],
            on=["match_id", "SetNo"],
            how="left",
        )
        .fillna({"p1_sets_before": 0, "p2_sets_before": 0})
    )

    df["is_tiebreak_game"] = (
        (df["p1_games_before"] == 6) & (df["p2_games_before"] == 6)
    ).astype(int)
    df["server_is_p1"] = (df["PointServer"] == 1).astype(int)

    p1_game_point = is_game_point_for.compute(
        df["p1_pts_in_game"], df["p2_pts_in_game"], df["is_tiebreak_game"]
    )
    p2_game_point = is_game_point_for.compute(
        df["p2_pts_in_game"], df["p1_pts_in_game"], df["is_tiebreak_game"]
    )
    df["p1_is_break_point"] = is_break_point.compute(df["server_is_p1"], p1_game_point)
    df["p2_is_break_point"] = is_break_point.compute(1 - df["server_is_p1"], p2_game_point)

    p1_bp_opp_cum = (
        df["p1_is_break_point"].astype(int).groupby(df["match_id"]).cumsum().shift(1).fillna(0)
    )
    p1_bp_conv_cum = (
        (
            df["p1_is_break_point"].astype(bool)
            & (p1_won_point == 1)
        )
        .astype(int)
        .groupby(df["match_id"])
        .cumsum()
        .shift(1)
        .fillna(0)
    )
    df["p1_bp_conv_rate"] = _safe_div(p1_bp_conv_cum, p1_bp_opp_cum)

    p2_bp_opp_cum = (
        df["p2_is_break_point"].astype(int).groupby(df["match_id"]).cumsum().shift(1).fillna(0)
    )
    p2_bp_conv_cum = (
        (
            df["p2_is_break_point"].astype(bool)
            & (p2_won_point == 1)
        )
        .astype(int)
        .groupby(df["match_id"])
        .cumsum()
        .shift(1)
        .fillna(0)
    )
    df["p2_bp_conv_rate"] = _safe_div(p2_bp_conv_cum, p2_bp_opp_cum)

    p1_bp_def_opp = (
        df["p2_is_break_point"].astype(int).groupby(df["match_id"]).cumsum().shift(1).fillna(0)
    )
    p1_bp_def_cum = (
        (
            df["p2_is_break_point"].astype(bool)
            & (p1_won_point == 1)
        )
        .astype(int)
        .groupby(df["match_id"])
        .cumsum()
        .shift(1)
        .fillna(0)
    )
    df["p1_bp_defend_rate"] = _safe_div(p1_bp_def_cum, p1_bp_def_opp)

    p2_bp_def_opp = (
        df["p1_is_break_point"].astype(int).groupby(df["match_id"]).cumsum().shift(1).fillna(0)
    )
    p2_bp_def_cum = (
        (
            df["p1_is_break_point"].astype(bool)
            & (p2_won_point == 1)
        )
        .astype(int)
        .groupby(df["match_id"])
        .cumsum()
        .shift(1)
        .fillna(0)
    )
    df["p2_bp_defend_rate"] = _safe_div(p2_bp_def_cum, p2_bp_def_opp)

    df["p1_double_faults"] = df["p1_double_faults"].astype(float)
    df["p2_double_faults"] = df["p2_double_faults"].astype(float)

    match_winners = (
        set_last_games.groupby("match_id").tail(1)[["match_id", "set_winner"]]
        .rename(columns={"set_winner": "match_winner"})
        .reset_index(drop=True)
    )

    base_cols = [
        "match_id",
        "SetNo",
        "GameNo",
        "PointNumber",
        "point_idx",
        "time",
        "elapsed_time",
        "server_is_p1",
        "p1_pts_in_game",
        "p2_pts_in_game",
        "p1_games_before",
        "p2_games_before",
        "p1_sets_before",
        "p2_sets_before",
        "is_tiebreak_game",
        "ttl_p1",
        "ttl_p2",
        "p1_aces_cum",
        "p2_aces_cum",
        "p1_df_cum",
        "p2_df_cum",
        "p1_avg_srv_speed",
        "p2_avg_srv_speed",
        "p1_fs_in_pct",
        "p2_fs_in_pct",
        "p1_ss_in_pct",
        "p2_ss_in_pct",
        "p1_fs_win_pct",
        "p2_fs_win_pct",
        "p1_ss_win_pct",
        "p2_ss_win_pct",
        "p1_ret_win_pct",
        "p2_ret_win_pct",
        "p1_double_faults",
        "p2_double_faults",
        "p1_df_rate",
        "p2_df_rate",
        "p1_ace_rate",
        "p2_ace_rate",
        "p1_avg_srv_speed_l60",
        "p2_avg_srv_speed_l60",
        "p1_fs_in_pct_l60",
        "p2_fs_in_pct_l60",
        "p1_ss_in_pct_l60",
        "p2_ss_in_pct_l60",
        "p1_fs_win_pct_l60",
        "p2_fs_win_pct_l60",
        "p1_ss_win_pct_l60",
        "p2_ss_win_pct_l60",
        "p1_ret_win_pct_l60",
        "p2_ret_win_pct_l60",
        "p1_aces_l60",
        "p2_aces_l60",
        "p1_df_rate_l60",
        "p2_df_rate_l60",
        "p1_ace_rate_l60",
        "p2_ace_rate_l60",
        "p1_bp_conv_rate",
        "p2_bp_conv_rate",
        "p1_bp_defend_rate",
        "p2_bp_defend_rate",
    ]

    panel = pd.concat(
        [
            df[base_cols].assign(perspective="P1"),
            df[base_cols].assign(perspective="P2"),
        ],
        ignore_index=True,
    )

    panel = panel.merge(match_winners, on="match_id", how="left")

    panel["server_is_persp"] = server_is_persp.compute(
        panel["perspective"], panel["server_is_p1"]
    )
    panel["pts_in_game_for"] = pts_in_game_for.compute(
        panel["perspective"], panel["p1_pts_in_game"], panel["p2_pts_in_game"]
    )
    panel["pts_in_game_against"] = pts_in_game_against.compute(
        panel["perspective"], panel["p1_pts_in_game"], panel["p2_pts_in_game"]
    )
    panel["games_in_set_for"] = games_in_set_for.compute(
        panel["perspective"], panel["p1_games_before"], panel["p2_games_before"]
    )
    panel["games_in_set_against"] = games_in_set_against.compute(
        panel["perspective"], panel["p1_games_before"], panel["p2_games_before"]
    )
    panel["sets_for"] = sets_for.compute(
        panel["perspective"], panel["p1_sets_before"], panel["p2_sets_before"]
    )
    panel["sets_against"] = sets_against.compute(
        panel["perspective"], panel["p1_sets_before"], panel["p2_sets_before"]
    )
    panel["is_tiebreak"] = is_tiebreak.compute(panel["is_tiebreak_game"])

    panel["ttl_diff"] = ttl_diff.compute(panel["ttl_p1"], panel["ttl_p2"])
    panel["aces_diff"] = aces_diff.compute(panel["p1_aces_cum"], panel["p2_aces_cum"])
    panel["df_diff"] = df_diff.compute(panel["p1_df_cum"], panel["p2_df_cum"])

    panel["best_of"] = best_of.compute(best_of_default, panel.index)
    panel["sets_needed_to_win"] = sets_needed_to_win.compute(panel["best_of"])

    panel["is_game_point_for"] = is_game_point_for.compute(
        panel["pts_in_game_for"], panel["pts_in_game_against"], panel["is_tiebreak"]
    )
    panel["is_game_point_against"] = is_game_point_against.compute(
        panel["pts_in_game_for"], panel["pts_in_game_against"], panel["is_tiebreak"]
    )
    panel["is_break_point"] = is_break_point.compute(
        panel["server_is_persp"], panel["is_game_point_for"]
    )

    panel["avg_srv_speed"] = avg_srv_speed.compute(
        panel["perspective"], panel["p1_avg_srv_speed"], panel["p2_avg_srv_speed"]
    )
    panel["fs_in_pct"] = fs_in_pct.compute(
        panel["perspective"], panel["p1_fs_in_pct"], panel["p2_fs_in_pct"]
    )
    panel["ss_in_pct"] = ss_in_pct.compute(
        panel["perspective"], panel["p1_ss_in_pct"], panel["p2_ss_in_pct"]
    )
    panel["fs_win_pct"] = fs_win_pct.compute(
        panel["perspective"], panel["p1_fs_win_pct"], panel["p2_fs_win_pct"]
    )
    panel["ss_win_pct"] = ss_win_pct.compute(
        panel["perspective"], panel["p1_ss_win_pct"], panel["p2_ss_win_pct"]
    )
    panel["ret_win_pct"] = ret_win_pct.compute(
        panel["perspective"], panel["p1_ret_win_pct"], panel["p2_ret_win_pct"]
    )
    panel["aces"] = aces.compute(
        panel["perspective"], panel["p1_aces_cum"], panel["p2_aces_cum"]
    )
    panel["double_faults"] = double_faults.compute(
        panel["perspective"], panel["p1_double_faults"], panel["p2_double_faults"]
    )
    panel["df_rate"] = df_rate.compute(
        panel["perspective"], panel["p1_df_rate"], panel["p2_df_rate"]
    )
    panel["ace_rate"] = ace_rate.compute(
        panel["perspective"], panel["p1_ace_rate"], panel["p2_ace_rate"]
    )
    panel["avg_srv_speed_l60"] = avg_srv_speed_l60.compute(
        panel["perspective"], panel["p1_avg_srv_speed_l60"], panel["p2_avg_srv_speed_l60"]
    )
    panel["fs_in_pct_l60"] = fs_in_pct_l60.compute(
        panel["perspective"], panel["p1_fs_in_pct_l60"], panel["p2_fs_in_pct_l60"]
    )
    panel["ss_in_pct_l60"] = ss_in_pct_l60.compute(
        panel["perspective"], panel["p1_ss_in_pct_l60"], panel["p2_ss_in_pct_l60"]
    )
    panel["fs_win_pct_l60"] = fs_win_pct_l60.compute(
        panel["perspective"], panel["p1_fs_win_pct_l60"], panel["p2_fs_win_pct_l60"]
    )
    panel["ss_win_pct_l60"] = ss_win_pct_l60.compute(
        panel["perspective"], panel["p1_ss_win_pct_l60"], panel["p2_ss_win_pct_l60"]
    )
    panel["ret_win_pct_l60"] = ret_win_pct_l60.compute(
        panel["perspective"], panel["p1_ret_win_pct_l60"], panel["p2_ret_win_pct_l60"]
    )
    panel["aces_l60"] = aces_l60.compute(
        panel["perspective"], panel["p1_aces_l60"], panel["p2_aces_l60"]
    )
    panel["df_rate_l60"] = df_rate_l60.compute(
        panel["perspective"], panel["p1_df_rate_l60"], panel["p2_df_rate_l60"]
    )
    panel["ace_rate_l60"] = ace_rate_l60.compute(
        panel["perspective"], panel["p1_ace_rate_l60"], panel["p2_ace_rate_l60"]
    )
    panel["bp_conv_rate"] = bp_conv_rate.compute(
        panel["perspective"], panel["p1_bp_conv_rate"], panel["p2_bp_conv_rate"]
    )
    panel["bp_defend_rate"] = bp_defend_rate.compute(
        panel["perspective"], panel["p1_bp_defend_rate"], panel["p2_bp_defend_rate"]
    )

    panel["opp_avg_srv_speed"] = opp_avg_srv_speed.compute(
        panel["perspective"], panel["p1_avg_srv_speed"], panel["p2_avg_srv_speed"]
    )
    panel["opp_fs_in_pct"] = opp_fs_in_pct.compute(
        panel["perspective"], panel["p1_fs_in_pct"], panel["p2_fs_in_pct"]
    )
    panel["opp_ss_in_pct"] = opp_ss_in_pct.compute(
        panel["perspective"], panel["p1_ss_in_pct"], panel["p2_ss_in_pct"]
    )
    panel["opp_fs_win_pct"] = opp_fs_win_pct.compute(
        panel["perspective"], panel["p1_fs_win_pct"], panel["p2_fs_win_pct"]
    )
    panel["opp_ss_win_pct"] = opp_ss_win_pct.compute(
        panel["perspective"], panel["p1_ss_win_pct"], panel["p2_ss_win_pct"]
    )
    panel["opp_ret_win_pct"] = opp_ret_win_pct.compute(
        panel["perspective"], panel["p1_ret_win_pct"], panel["p2_ret_win_pct"]
    )
    panel["opp_aces"] = opp_aces.compute(
        panel["perspective"], panel["p1_aces_cum"], panel["p2_aces_cum"]
    )
    panel["opp_double_faults"] = opp_double_faults.compute(
        panel["perspective"], panel["p1_double_faults"], panel["p2_double_faults"]
    )
    panel["opp_df_rate"] = opp_df_rate.compute(
        panel["perspective"], panel["p1_df_rate"], panel["p2_df_rate"]
    )
    panel["opp_ace_rate"] = opp_ace_rate.compute(
        panel["perspective"], panel["p1_ace_rate"], panel["p2_ace_rate"]
    )
    panel["opp_avg_srv_speed_l60"] = opp_avg_srv_speed_l60.compute(
        panel["perspective"], panel["p1_avg_srv_speed_l60"], panel["p2_avg_srv_speed_l60"]
    )
    panel["opp_fs_in_pct_l60"] = opp_fs_in_pct_l60.compute(
        panel["perspective"], panel["p1_fs_in_pct_l60"], panel["p2_fs_in_pct_l60"]
    )
    panel["opp_ss_in_pct_l60"] = opp_ss_in_pct_l60.compute(
        panel["perspective"], panel["p1_ss_in_pct_l60"], panel["p2_ss_in_pct_l60"]
    )
    panel["opp_fs_win_pct_l60"] = opp_fs_win_pct_l60.compute(
        panel["perspective"], panel["p1_fs_win_pct_l60"], panel["p2_fs_win_pct_l60"]
    )
    panel["opp_ss_win_pct_l60"] = opp_ss_win_pct_l60.compute(
        panel["perspective"], panel["p1_ss_win_pct_l60"], panel["p2_ss_win_pct_l60"]
    )
    panel["opp_ret_win_pct_l60"] = opp_ret_win_pct_l60.compute(
        panel["perspective"], panel["p1_ret_win_pct_l60"], panel["p2_ret_win_pct_l60"]
    )
    panel["opp_aces_l60"] = opp_aces_l60.compute(
        panel["perspective"], panel["p1_aces_l60"], panel["p2_aces_l60"]
    )
    panel["opp_df_rate_l60"] = opp_df_rate_l60.compute(
        panel["perspective"], panel["p1_df_rate_l60"], panel["p2_df_rate_l60"]
    )
    panel["opp_ace_rate_l60"] = opp_ace_rate_l60.compute(
        panel["perspective"], panel["p1_ace_rate_l60"], panel["p2_ace_rate_l60"]
    )
    panel["opp_bp_conv_rate"] = opp_bp_conv_rate.compute(
        panel["perspective"], panel["p1_bp_conv_rate"], panel["p2_bp_conv_rate"]
    )
    panel["opp_bp_defend_rate"] = opp_bp_defend_rate.compute(
        panel["perspective"], panel["p1_bp_defend_rate"], panel["p2_bp_defend_rate"]
    )

    panel["y_match"] = y_match.compute(panel["perspective"], panel["match_winner"])

    drop_cols = {
        "server_is_p1",
        "p1_pts_in_game",
        "p2_pts_in_game",
        "p1_games_before",
        "p2_games_before",
        "p1_sets_before",
        "p2_sets_before",
        "is_tiebreak_game",
        "ttl_p1",
        "ttl_p2",
        "p1_aces_cum",
        "p2_aces_cum",
        "p1_df_cum",
        "p2_df_cum",
        "p1_avg_srv_speed",
        "p2_avg_srv_speed",
        "p1_fs_in_pct",
        "p2_fs_in_pct",
        "p1_ss_in_pct",
        "p2_ss_in_pct",
        "p1_fs_win_pct",
        "p2_fs_win_pct",
        "p1_ss_win_pct",
        "p2_ss_win_pct",
        "p1_ret_win_pct",
        "p2_ret_win_pct",
        "p1_double_faults",
        "p2_double_faults",
        "p1_df_rate",
        "p2_df_rate",
        "p1_ace_rate",
        "p2_ace_rate",
        "p1_avg_srv_speed_l60",
        "p2_avg_srv_speed_l60",
        "p1_fs_in_pct_l60",
        "p2_fs_in_pct_l60",
        "p1_ss_in_pct_l60",
        "p2_ss_in_pct_l60",
        "p1_fs_win_pct_l60",
        "p2_fs_win_pct_l60",
        "p1_ss_win_pct_l60",
        "p2_ss_win_pct_l60",
        "p1_ret_win_pct_l60",
        "p2_ret_win_pct_l60",
        "p1_aces_l60",
        "p2_aces_l60",
        "p1_df_rate_l60",
        "p2_df_rate_l60",
        "p1_ace_rate_l60",
        "p2_ace_rate_l60",
        "p1_bp_conv_rate",
        "p2_bp_conv_rate",
        "p1_bp_defend_rate",
        "p2_bp_defend_rate",
        "match_winner",
    }
    panel = panel.drop(columns=[c for c in drop_cols if c in panel.columns])

    ordered = [
        "match_id",
        "SetNo",
        "GameNo",
        "PointNumber",
        "point_idx",
        "time",
        "elapsed_time",
        "perspective",
        "server_is_persp",
        "pts_in_game_for",
        "pts_in_game_against",
        "games_in_set_for",
        "games_in_set_against",
        "sets_for",
        "sets_against",
        "best_of",
        "sets_needed_to_win",
        "is_tiebreak",
        "is_game_point_for",
        "is_game_point_against",
        "is_break_point",
        "ttl_diff",
        "aces_diff",
        "df_diff",
        "avg_srv_speed",
        "fs_in_pct",
        "ss_in_pct",
        "fs_win_pct",
        "ss_win_pct",
        "ret_win_pct",
        "aces",
        "double_faults",
        "df_rate",
        "ace_rate",
        "avg_srv_speed_l60",
        "fs_in_pct_l60",
        "ss_in_pct_l60",
        "fs_win_pct_l60",
        "ss_win_pct_l60",
        "ret_win_pct_l60",
        "aces_l60",
        "df_rate_l60",
        "ace_rate_l60",
        "bp_conv_rate",
        "bp_defend_rate",
        "opp_avg_srv_speed",
        "opp_fs_in_pct",
        "opp_ss_in_pct",
        "opp_fs_win_pct",
        "opp_ss_win_pct",
        "opp_ret_win_pct",
        "opp_aces",
        "opp_double_faults",
        "opp_df_rate",
        "opp_ace_rate",
        "opp_avg_srv_speed_l60",
        "opp_fs_in_pct_l60",
        "opp_ss_in_pct_l60",
        "opp_fs_win_pct_l60",
        "opp_ss_win_pct_l60",
        "opp_ret_win_pct_l60",
        "opp_aces_l60",
        "opp_df_rate_l60",
        "opp_ace_rate_l60",
        "opp_bp_conv_rate",
        "opp_bp_defend_rate",
        "y_match",
    ]

    missing = [col for col in ordered if col not in panel.columns]
    if missing:
        raise KeyError(f"Missing expected columns in panel: {missing}")

    return panel[ordered].copy()
