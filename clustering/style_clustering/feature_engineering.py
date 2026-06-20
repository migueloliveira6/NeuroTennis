from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .utils import normalize_player_name, safe_series_divide


SURFACE_NAMES = ["Hard", "Clay", "Grass"]


@dataclass(slots=True)
class FeatureFrameResult:
    """Container with the player-match and player-season feature tables."""

    player_match: pd.DataFrame
    player_season: pd.DataFrame


def _count_tiebreaks(score: Any) -> int:
    if score is None or pd.isna(score):
        return 0
    return len(re.findall(r"\((\d+)\)", str(score)))


def _parse_sets(score: Any) -> int:
    if score is None or pd.isna(score):
        return 0
    tokens = re.findall(r"\b\d+-\d+\b", str(score))
    return len(tokens)


def _to_float(value: Any) -> float:
    return float(pd.to_numeric(value, errors="coerce")) if pd.notna(pd.to_numeric(value, errors="coerce")) else float("nan")


def build_player_match_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert match-level ATP rows into one observation per player per match."""

    if matches.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for _, row in matches.iterrows():
        for side, prefix, opp_prefix, win_flag in (
            ("winner", "w_", "l_", 1),
            ("loser", "l_", "w_", 0),
        ):
            player_name = row.get(f"{side}_name")
            if pd.isna(player_name):
                continue
            opp_side = "loser" if side == "winner" else "winner"
            service_points = pd.to_numeric(row.get(f"{prefix}svpt"), errors="coerce")
            first_in = pd.to_numeric(row.get(f"{prefix}1stIn"), errors="coerce")
            first_won = pd.to_numeric(row.get(f"{prefix}1stWon"), errors="coerce")
            second_won = pd.to_numeric(row.get(f"{prefix}2ndWon"), errors="coerce")
            opp_service_points = pd.to_numeric(row.get(f"{opp_prefix}svpt"), errors="coerce")
            opp_first_won = pd.to_numeric(row.get(f"{opp_prefix}1stWon"), errors="coerce")
            opp_second_won = pd.to_numeric(row.get(f"{opp_prefix}2ndWon"), errors="coerce")
            opp_bp_faced = pd.to_numeric(row.get(f"{opp_prefix}bpFaced"), errors="coerce")
            opp_bp_saved = pd.to_numeric(row.get(f"{opp_prefix}bpSaved"), errors="coerce")
            service_games = pd.to_numeric(row.get(f"{prefix}SvGms"), errors="coerce")
            opp_service_games = pd.to_numeric(row.get(f"{opp_prefix}SvGms"), errors="coerce")
            return_points_total = opp_service_points
            return_points_won = opp_service_points - opp_first_won - opp_second_won
            break_points_opportunities = opp_bp_faced
            break_points_converted = opp_bp_faced - opp_bp_saved
            total_points_won = float(first_won or 0.0) + float(second_won or 0.0) + float(return_points_won or 0.0)
            total_points_played = float(service_points or 0.0) + float(return_points_total or 0.0)
            total_points_lost = total_points_played - total_points_won
            records.append(
                {
                    "match_id": f"{row.get('tourney_id')}_{row.get('match_num')}",
                    "player_id": row.get(f"{side}_id"),
                    "player_name": player_name,
                    "player_name_clean": normalize_player_name(player_name),
                    "opponent_id": row.get(f"{opp_side}_id"),
                    "opponent_name": row.get(f"{opp_side}_name"),
                    "surface": row.get("surface"),
                    "tourney_level": row.get("tourney_level"),
                    "tourney_date": row.get("tourney_date"),
                    "season": pd.to_datetime(row.get("tourney_date"), errors="coerce").year if not pd.isna(row.get("tourney_date")) else np.nan,
                    "round": row.get("round"),
                    "minutes": pd.to_numeric(row.get("minutes"), errors="coerce"),
                    "best_of": pd.to_numeric(row.get("best_of"), errors="coerce"),
                    "won": win_flag,
                    "sets_played": _parse_sets(row.get("score")),
                    "tiebreaks_played": _count_tiebreaks(row.get("score")),
                    "aces": pd.to_numeric(row.get(f"{prefix}ace"), errors="coerce"),
                    "double_faults": pd.to_numeric(row.get(f"{prefix}df"), errors="coerce"),
                    "service_points": service_points,
                    "first_serves_in": first_in,
                    "first_serve_points_won": first_won,
                    "second_serve_points_won": second_won,
                    "service_games": service_games,
                    "break_points_saved": pd.to_numeric(row.get(f"{prefix}bpSaved"), errors="coerce"),
                    "break_points_faced": pd.to_numeric(row.get(f"{prefix}bpFaced"), errors="coerce"),
                    "break_points_opportunities": break_points_opportunities,
                    "break_points_converted": break_points_converted,
                    "return_points_total": return_points_total,
                    "return_points_won": return_points_won,
                    "opponent_service_games": opp_service_games,
                    "player_rank": pd.to_numeric(row.get(f"{side}_rank"), errors="coerce"),
                    "player_rank_points": pd.to_numeric(row.get(f"{side}_rank_points"), errors="coerce"),
                    "opponent_rank": pd.to_numeric(row.get(f"{opp_side}_rank"), errors="coerce"),
                    "opponent_rank_points": pd.to_numeric(row.get(f"{opp_side}_rank_points"), errors="coerce"),
                    "total_points_won": total_points_won,
                    "total_points_played": total_points_played,
                    "total_points_lost": total_points_lost,
                    "surface_match": 1,
                    "hard_match": int(str(row.get("surface", "")).lower() == "hard"),
                    "clay_match": int(str(row.get("surface", "")).lower() == "clay"),
                    "grass_match": int(str(row.get("surface", "")).lower() == "grass"),
                }
            )

    player_match = pd.DataFrame.from_records(records)
    player_match["season"] = pd.to_datetime(player_match["tourney_date"], errors="coerce").dt.year
    return player_match


def _aggregate_surface_features(player_match: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for surface in SURFACE_NAMES:
        subset = player_match[player_match["surface"].astype("string").str.casefold() == surface.casefold()].copy()
        if subset.empty:
            continue
        grouped = subset.groupby(key_cols, dropna=False).agg(
            **{
                f"{surface.lower()}_matches": ("surface_match", "sum"),
                f"{surface.lower()}_wins": ("won", "sum"),
                f"{surface.lower()}_minutes": ("minutes", "sum"),
                f"{surface.lower()}_points_won": ("total_points_won", "sum"),
                f"{surface.lower()}_points_played": ("total_points_played", "sum"),
            }
        ).reset_index()
        rows.append(grouped)
    if not rows:
        return pd.DataFrame(columns=key_cols)
    surface_features = rows[0]
    for frame in rows[1:]:
        surface_features = surface_features.merge(frame, on=key_cols, how="outer")
    return surface_features


def build_player_season_table(player_match: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player-match records into player-season observations."""

    if player_match.empty:
        return pd.DataFrame()

    key_cols = ["player_id", "player_name", "player_name_clean", "season"]
    numeric_columns = [
        "won",
        "minutes",
        "sets_played",
        "tiebreaks_played",
        "aces",
        "double_faults",
        "service_points",
        "first_serves_in",
        "first_serve_points_won",
        "second_serve_points_won",
        "service_games",
        "break_points_saved",
        "break_points_faced",
        "break_points_opportunities",
        "break_points_converted",
        "return_points_total",
        "return_points_won",
        "opponent_service_games",
        "player_rank",
        "player_rank_points",
        "opponent_rank",
        "opponent_rank_points",
        "total_points_won",
        "total_points_played",
        "total_points_lost",
        "surface_match",
        "hard_match",
        "clay_match",
        "grass_match",
    ]
    available_numeric = [column for column in numeric_columns if column in player_match.columns]
    grouped = player_match.groupby(key_cols, dropna=False)[available_numeric].sum(min_count=1).reset_index()

    avg_minutes = player_match.groupby(key_cols, dropna=False)["minutes"].mean().reset_index(name="average_match_length")
    grouped = grouped.merge(avg_minutes, on=key_cols, how="left")
    avg_rank = player_match.groupby(key_cols, dropna=False)["player_rank"].mean().reset_index(name="average_player_rank")
    avg_opp_rank = player_match.groupby(key_cols, dropna=False)["opponent_rank"].mean().reset_index(name="average_opponent_rank")
    grouped = grouped.merge(avg_rank, on=key_cols, how="left").merge(avg_opp_rank, on=key_cols, how="left")

    surface_features = _aggregate_surface_features(player_match, key_cols)
    if not surface_features.empty:
        grouped = grouped.merge(surface_features, on=key_cols, how="left")

    grouped["matches"] = grouped["surface_match"]
    grouped["wins"] = grouped["won"]
    grouped["losses"] = grouped["matches"] - grouped["wins"]
    grouped["win_rate"] = safe_series_divide(grouped["wins"], grouped["matches"])
    grouped["ace_rate"] = safe_series_divide(grouped["aces"], grouped["service_points"])
    grouped["double_fault_rate"] = safe_series_divide(grouped["double_faults"], grouped["service_points"])
    grouped["first_serve_in_pct"] = safe_series_divide(grouped["first_serves_in"], grouped["service_points"])
    grouped["first_serve_win_pct"] = safe_series_divide(grouped["first_serve_points_won"], grouped["first_serves_in"])
    second_serve_opportunities = grouped["service_points"] - grouped["first_serves_in"]
    grouped["second_serve_win_pct"] = safe_series_divide(grouped["second_serve_points_won"], second_serve_opportunities)
    grouped["service_hold_rate"] = 1.0 - safe_series_divide(grouped["break_points_converted"], grouped["service_games"])
    grouped["return_points_won_pct"] = safe_series_divide(grouped["return_points_won"], grouped["return_points_total"])
    grouped["break_points_conversion_pct"] = safe_series_divide(grouped["break_points_converted"], grouped["break_points_opportunities"])
    grouped["return_games_won_pct"] = safe_series_divide(grouped["break_points_converted"], grouped["opponent_service_games"])
    grouped["tiebreak_frequency"] = safe_series_divide(grouped["tiebreaks_played"], grouped["matches"])
    grouped["dominance_ratio"] = safe_series_divide(grouped["total_points_won"], grouped["total_points_lost"])
    grouped["hard_win_rate"] = safe_series_divide(grouped.get("hard_wins", pd.Series(index=grouped.index, dtype=float)), grouped.get("hard_matches", pd.Series(index=grouped.index, dtype=float)))
    grouped["clay_win_rate"] = safe_series_divide(grouped.get("clay_wins", pd.Series(index=grouped.index, dtype=float)), grouped.get("clay_matches", pd.Series(index=grouped.index, dtype=float)))
    grouped["grass_win_rate"] = safe_series_divide(grouped.get("grass_wins", pd.Series(index=grouped.index, dtype=float)), grouped.get("grass_matches", pd.Series(index=grouped.index, dtype=float)))
    grouped["service_aggression_score"] = (
        grouped["ace_rate"].fillna(0)
        + grouped["first_serve_win_pct"].fillna(0)
        + 0.5 * grouped["second_serve_win_pct"].fillna(0)
        - grouped["double_fault_rate"].fillna(0)
    )
    grouped["return_aggression_score"] = (
        grouped["return_points_won_pct"].fillna(0)
        + grouped["break_points_conversion_pct"].fillna(0)
        + 0.5 * grouped["return_games_won_pct"].fillna(0)
    )
    grouped["average_match_length"] = grouped["average_match_length"].fillna(grouped["minutes"] / grouped["matches"].replace(0, np.nan))

    if {"hard_matches", "clay_matches", "grass_matches"}.issubset(grouped.columns):
        total_surface_matches = grouped[["hard_matches", "clay_matches", "grass_matches"]].sum(axis=1)
        shares = grouped[["hard_matches", "clay_matches", "grass_matches"]].div(total_surface_matches.replace(0, np.nan), axis=0)
        entropy = -(shares * np.log(shares.replace(0, np.nan))).sum(axis=1)
        grouped["surface_diversity"] = entropy.fillna(0.0)
        grouped["favorite_surface"] = shares.idxmax(axis=1).str.replace("_matches", "", regex=False).str.title()
    else:
        grouped["surface_diversity"] = 0.0
        grouped["favorite_surface"] = np.nan

    if {"winners", "unforced_errors"}.issubset(player_match.columns):
        charting = player_match.groupby(key_cols, dropna=False)[["winners", "unforced_errors"]].sum(min_count=1).reset_index()
        charting["winner_to_error_ratio"] = safe_series_divide(charting["winners"], charting["unforced_errors"])
        grouped = grouped.merge(charting[key_cols + ["winner_to_error_ratio"]], on=key_cols, how="left")
    else:
        grouped["winner_to_error_ratio"] = np.nan

    grouped["player_key"] = grouped["player_name_clean"].fillna(grouped["player_name"])
    return grouped.sort_values(["player_name", "season"]).reset_index(drop=True)


def build_feature_tables(matches: pd.DataFrame) -> FeatureFrameResult:
    """Build both the player-match and player-season feature tables."""

    player_match = build_player_match_table(matches)
    player_season = build_player_season_table(player_match)
    return FeatureFrameResult(player_match=player_match, player_season=player_season)
