from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ProjectConfig:
    """Global paths and toggles for the style clustering pipeline."""

    db_path: Path = Path("datasets/tennis_data.db")
    optional_charting_path: Path | None = None
    output_dir: Path = Path("outputs")
    include_qualifying: bool = True
    start_year: int | None = None
    end_year: int | None = None
    use_season_zscore: bool = True
    missing_threshold: float = 0.35
    correlation_threshold: float = 0.92
    random_state: int = 42


@dataclass(slots=True)
class ClusteringConfig:
    """Parameters used to search and compare clustering methods."""

    k_min: int = 3
    k_max: int = 10
    hdbscan_min_cluster_sizes: list[int] = field(default_factory=lambda: [5, 8, 12, 16])
    stability_iterations: int = 12
    stability_sample_fraction: float = 0.8
    random_state: int = 42
