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

    k_min: int = 5
    k_max: int = 14
    hdbscan_min_cluster_sizes: list[int] = field(default_factory=lambda: [8, 12, 16, 24])
    stability_iterations: int = 12
    stability_sample_fraction: float = 0.8
    preferred_k_min: int = 6
    k_diversity_bonus: float = 0.06
    low_k_penalty: float = 0.08
    cluster_merge_similarity_threshold: float = 0.92
    random_state: int = 42
