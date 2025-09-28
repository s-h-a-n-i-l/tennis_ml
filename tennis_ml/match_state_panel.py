"""Utilities for constructing match state panels from point-level data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import (
    aces_diff,
    best_of,
    df_diff,
    games_in_set_against,
    games_in_set_for,
    is_break_point,
    is_game_point_against,
    is_game_point_for,
    is_tiebreak,
    point_idx,
    pts_in_game_against,
    pts_in_game_for,
    server_is_persp,
    sets_against,
    sets_for,
    sets_needed_to_win,
    ttl_diff,
    y_match,
)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Safely divide two series, avoiding division by zero."""

    a = a.astype(float)
    b = b.astype(float)
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def build_match_state_panel(
    input_data: pd.DataFrame, best_of_default: int = 5
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
    if have_srv:
        for side in (1, 2):
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
        df["p1_sv_pts"] = df["p1_fs_in_cum"] + df["p1_ss_in_cum"] + df["p1_df_cum"]
        df["p2_sv_pts"] = df["p2_fs_in_cum"] + df["p2_ss_in_cum"] + df["p2_df_cum"]
        df["p1_fsp"] = _safe_div(df["p1_fs_in_cum"], df["p1_sv_pts"])
        df["p2_fsp"] = _safe_div(df["p2_fs_in_cum"], df["p2_sv_pts"])
        df["p1_w1sp"] = _safe_div(df["p1_fs_won_cum"], df["p1_fs_in_cum"])
        df["p2_w1sp"] = _safe_div(df["p2_fs_won_cum"], df["p2_fs_in_cum"])
        df["p1_w2sp"] = _safe_div(df["p1_ss_won_cum"], df["p1_ss_in_cum"])
        df["p2_w2sp"] = _safe_div(df["p2_ss_won_cum"], df["p2_ss_in_cum"])
    else:
        df["p1_fsp"] = df["p2_fsp"] = 0.0
        df["p1_w1sp"] = df["p2_w1sp"] = 0.0
        df["p1_w2sp"] = df["p2_w2sp"] = 0.0
        df["p1_df_cum"] = df["p2_df_cum"] = 0.0

    if {"P1Ace", "P2Ace"}.issubset(df.columns):
        df["p1_aces_cum"] = df["P1Ace"].groupby(df["match_id"]).cumsum().shift(1).fillna(0)
        df["p2_aces_cum"] = df["P2Ace"].groupby(df["match_id"]).cumsum().shift(1).fillna(0)
    else:
        df["p1_aces_cum"] = df["p2_aces_cum"] = 0.0

    last_point_idx = df.groupby(["match_id", "SetNo", "GameNo"]).tail(1).index
    game_winner = pd.Series(
        index=last_point_idx,
        data=np.where(
            df.loc[last_point_idx, "p1_pts_in_game"]
            > df.loc[last_point_idx, "p2_pts_in_game"],
            1,
            2,
        ),
    )

    games_tbl = (
        df.assign(game_winner=game_winner)
        .groupby(["match_id", "SetNo", "GameNo"])
        .tail(1)
        .copy()
    )
    games_tbl["p1_games_in_set_cum"] = (
        (games_tbl["game_winner"] == 1)
        .groupby([games_tbl["match_id"], games_tbl["SetNo"]])
        .cumsum()
    )
    games_tbl["p2_games_in_set_cum"] = (
        (games_tbl["game_winner"] == 2)
        .groupby([games_tbl["match_id"], games_tbl["SetNo"]])
        .cumsum()
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
        set_last_games["p1_games_in_set_cum"] > set_last_games["p2_games_in_set_cum"],
        1,
        2,
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

    match_winners = (
        set_last_games.groupby("match_id").tail(1)[["match_id", "set_winner"]]
        .rename(columns={"set_winner": "match_winner"})
        .reset_index(drop=True)
    )

    df["is_tiebreak_game"] = (
        (df["p1_games_before"] == 6) & (df["p2_games_before"] == 6)
    ).astype(int)
    df["server_is_p1"] = (df["PointServer"] == 1).astype(int)

    base_cols = [
        "match_id",
        "SetNo",
        "GameNo",
        "PointNumber",
        "point_idx",
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
        "match_winner",
    }
    panel = panel.drop(columns=[c for c in drop_cols if c in panel.columns])

    ordered = [
        "match_id",
        "SetNo",
        "GameNo",
        "PointNumber",
        "point_idx",
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
        "y_match",
    ]

    missing = [col for col in ordered if col not in panel.columns]
    if missing:
        raise KeyError(f"Missing expected columns in panel: {missing}")

    return panel[ordered].copy()
