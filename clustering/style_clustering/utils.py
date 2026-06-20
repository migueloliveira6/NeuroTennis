from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def normalize_player_name(value: Any) -> str | None:
    """Normalize player names to improve joins and clustering aggregation."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text or None


def safe_series_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while protecting against zeros and missing values."""

    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    numerator = pd.to_numeric(numerator, errors="coerce")
    return numerator.divide(denominator)


def safe_scalar_divide(numerator: float | int | None, denominator: float | int | None) -> float:
    """Divide two scalars safely and return NaN for invalid denominators."""

    if numerator is None or denominator is None:
        return float("nan")
    try:
        if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0.0:
            return float("nan")
    except Exception:
        return float("nan")
    return float(numerator) / float(denominator)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    """Persist a dictionary as formatted JSON."""

    output_path = Path(path)
    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def top_n_dict(series: pd.Series, n: int = 5) -> dict[str, float]:
    """Return the top-N values from a series as an ordered dictionary."""

    cleaned = series.dropna().sort_values(ascending=False).head(n)
    return {str(index): float(value) for index, value in cleaned.items()}
