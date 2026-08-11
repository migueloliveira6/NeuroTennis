from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import ensure_directory, normalize_player_name


MATCH_FILE_PATTERN = re.compile(r"atp_matches_(\d{4})\.csv$")


@dataclass(slots=True)
class MatchDataBundle:
    """Container for loaded raw ATP match data and optional auxiliary sources."""

    matches: pd.DataFrame
    charting: pd.DataFrame | None = None


def load_matches_from_sqlite(
    db_path: str | Path | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Load match data directly from the project's SQLite database."""

    database_path = Path(db_path) if db_path is not None else Path("datasets/tennis_data.db")
    conn = sqlite3.connect(database_path)
    try:
        matches = pd.read_sql_query("SELECT * FROM matches", conn, parse_dates=["tourney_date"])
    finally:
        conn.close()

    if "tourney_date" in matches.columns:
        matches["tourney_date"] = pd.to_datetime(matches["tourney_date"], errors="coerce")
    if start_year is not None:
        matches = matches[matches["tourney_date"].dt.year >= start_year]
    if end_year is not None:
        matches = matches[matches["tourney_date"].dt.year <= end_year]

    for column in ["winner_name", "loser_name", "winner_ioc", "loser_ioc", "tourney_level", "round", "surface", "tourney_name", "tournament_type"]:
        if column in matches.columns:
            matches[column] = matches[column].astype("string").str.strip()
    matches["source_file"] = "tennis_data.db"
    matches["player_count"] = 2
    return matches.reset_index(drop=True)


def get_latest_season_from_sqlite(db_path: str | Path | None = None) -> int:
    """Return the latest season available in the SQLite matches table."""

    database_path = Path(db_path) if db_path is not None else Path("datasets/tennis_data.db")
    conn = sqlite3.connect(database_path)
    try:
        result = pd.read_sql_query("SELECT MAX(strftime('%Y', tourney_date)) AS max_year FROM matches", conn)
    finally:
        conn.close()

    value = result.iloc[0, 0]
    if pd.isna(value):
        raise ValueError("Unable to determine the latest season from the SQLite database")
    return int(value)


def discover_match_files(data_dir: str | Path, include_qualifying: bool = True) -> list[Path]:
    """Discover ATP match CSVs from Jeff Sackmann's repository structure."""

    base_dir = Path(data_dir)
    files = sorted(base_dir.glob("atp_matches_*.csv"))
    if not include_qualifying:
        files = [path for path in files if "qual_chall" not in path.name and "doubles" not in path.name]
    return files


def _filter_years(paths: Iterable[Path], start_year: int | None, end_year: int | None) -> list[Path]:
    selected: list[Path] = []
    for path in paths:
        match = MATCH_FILE_PATTERN.search(path.name)
        if not match:
            continue
        year = int(match.group(1))
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        selected.append(path)
    return selected


def load_atp_matches(
    data_dir: str | Path,
    start_year: int | None = None,
    end_year: int | None = None,
    include_qualifying: bool = True,
) -> pd.DataFrame:
    """Load and concatenate yearly ATP match CSV files."""

    data_dir = Path(data_dir)
    files = discover_match_files(data_dir, include_qualifying=include_qualifying)
    files = _filter_years(files, start_year=start_year, end_year=end_year)
    if not files:
        raise FileNotFoundError(f"No ATP match CSV files found in {data_dir}")

    frames: list[pd.DataFrame] = []
    for file_path in files:
        frame = pd.read_csv(file_path, low_memory=False)
        frame["source_file"] = file_path.name
        frames.append(frame)

    matches = pd.concat(frames, ignore_index=True)
    if "tourney_date" in matches.columns:
        matches["tourney_date"] = pd.to_datetime(matches["tourney_date"], format="%Y%m%d", errors="coerce")
    if "surface" in matches.columns:
        matches["surface"] = matches["surface"].astype("string").str.strip().str.title()
    for column in ["winner_name", "loser_name", "winner_ioc", "loser_ioc", "tourney_level", "round"]:
        if column in matches.columns:
            matches[column] = matches[column].astype("string").str.strip()
    matches["player_count"] = 2
    return matches


def load_optional_charting_project(charting_path: str | Path | None) -> pd.DataFrame | None:
    """Load an optional Match Charting Project extract when the user supplies one."""

    if charting_path is None:
        return None
    path = Path(charting_path)
    if not path.exists():
        raise FileNotFoundError(f"Optional charting file not found: {path}")
    charting = pd.read_csv(path, low_memory=False)
    rename_map: dict[str, str] = {}
    for column in charting.columns:
        normalized = normalize_player_name(column) or column.lower()
        if normalized in {"player", "player_name"}:
            rename_map[column] = "player_name"
        elif normalized in {"match_id", "tourney_id"}:
            rename_map[column] = "match_id"
        elif normalized in {"winners", "winner_count"}:
            rename_map[column] = "winners"
        elif normalized in {"unforced_errors", "errors", "ue"}:
            rename_map[column] = "unforced_errors"
    if rename_map:
        charting = charting.rename(columns=rename_map)
    return charting


def merge_optional_charting(matches: pd.DataFrame, charting: pd.DataFrame | None) -> pd.DataFrame:
    """Merge optional Match Charting features when a compatible schema is present."""

    if charting is None or charting.empty:
        return matches

    merged = matches.copy()
    charting = charting.copy()
    if {"tourney_id", "match_num", "player_name"}.issubset(charting.columns) and {"tourney_id", "match_num"}.issubset(merged.columns):
        charting["player_name"] = charting["player_name"].astype("string").str.strip()
        merged["winner_name"] = merged["winner_name"].astype("string").str.strip()
        merged["loser_name"] = merged["loser_name"].astype("string").str.strip()
        winner_side = charting.merge(
            merged,
            how="left",
            left_on=["tourney_id", "match_num", "player_name"],
            right_on=["tourney_id", "match_num", "winner_name"],
            suffixes=("", "_winner"),
        )
        loser_side = charting.merge(
            merged,
            how="left",
            left_on=["tourney_id", "match_num", "player_name"],
            right_on=["tourney_id", "match_num", "loser_name"],
            suffixes=("", "_loser"),
        )
        winner_side["player_side"] = "winner"
        loser_side["player_side"] = "loser"
        combined = pd.concat([winner_side, loser_side], ignore_index=True, sort=False)
        return combined

    return merged


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Utility to save a dataframe to CSV with parent directory creation."""

    output_path = Path(path)
    ensure_directory(output_path.parent)
    df.to_csv(output_path, index=False)
    return output_path
