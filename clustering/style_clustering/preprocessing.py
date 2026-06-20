from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class PreprocessingResult:
    """Prepared matrix and metadata ready for clustering."""

    frame: pd.DataFrame
    feature_columns: list[str]
    scaler: StandardScaler
    imputer: SimpleImputer
    matrix: np.ndarray


def season_zscore(frame: pd.DataFrame, feature_columns: list[str], season_column: str = "season") -> pd.DataFrame:
    """Apply within-season z-scoring to reduce cross-season scale drift."""

    result = frame.copy()
    for column in feature_columns:
        if column not in result.columns:
            continue
        grouped = result.groupby(season_column)[column]
        means = grouped.transform("mean")
        stds = grouped.transform(lambda series: series.std(ddof=0))
        result[column] = (pd.to_numeric(result[column], errors="coerce") - means) / stds.replace(0, np.nan)
    result[feature_columns] = result[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result


def drop_high_missing_columns(frame: pd.DataFrame, feature_columns: list[str], threshold: float) -> list[str]:
    """Keep only feature columns whose missing ratio is below the requested threshold."""

    keep: list[str] = []
    for column in feature_columns:
        missing_ratio = frame[column].isna().mean()
        if missing_ratio <= threshold:
            keep.append(column)
    return keep


def drop_highly_correlated_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = 0.92,
) -> list[str]:
    """Remove redundant variables with pairwise correlation above the threshold."""

    if len(feature_columns) < 2:
        return feature_columns
    corr = frame[feature_columns].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return [column for column in feature_columns if column not in to_drop]


def prepare_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str],
    season_column: str = "season",
    use_season_zscore: bool = True,
    missing_threshold: float = 0.35,
    correlation_threshold: float = 0.92,
) -> PreprocessingResult:
    """Clean, normalize and scale the player-season feature matrix."""

    if frame.empty:
        raise ValueError("Cannot prepare features from an empty dataframe")

    valid_features = [column for column in feature_columns if column in frame.columns]
    cleaned = frame.copy()
    for column in valid_features:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    valid_features = drop_high_missing_columns(cleaned, valid_features, threshold=missing_threshold)
    cleaned = cleaned.dropna(subset=[season_column]).copy()

    if use_season_zscore:
        cleaned = season_zscore(cleaned, valid_features, season_column=season_column)

    valid_features = drop_highly_correlated_features(cleaned, valid_features, threshold=correlation_threshold)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    matrix_imputed = imputer.fit_transform(cleaned[valid_features])
    matrix_scaled = scaler.fit_transform(matrix_imputed)
    prepared = cleaned.copy()
    prepared[valid_features] = matrix_imputed
    return PreprocessingResult(frame=prepared, feature_columns=valid_features, scaler=scaler, imputer=imputer, matrix=matrix_scaled)
