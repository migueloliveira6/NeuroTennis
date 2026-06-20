from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .clustering import (
    attach_cluster_labels,
    build_cluster_profile,
    choose_best_cluster_row,
    evaluate_hdbscan_family,
    evaluate_k_family,
    fit_hdbscan,
    fit_gmm,
    fit_hierarchical,
    fit_kmeans,
)
from .config import ClusteringConfig, ProjectConfig
from .data_loader import load_matches_from_sqlite, load_optional_charting_project, merge_optional_charting, save_dataframe
from .feature_engineering import build_feature_tables
from .preprocessing import prepare_feature_matrix
from .utils import ensure_directory, save_json
from .visualization import (
    embed_2d,
    plot_cluster_distribution,
    plot_cluster_radar,
    plot_correlation_heatmap,
    plot_feature_profiles,
    plot_histograms,
    plot_missing_values,
    plot_pca_embedding,
    plot_umap_embedding,
    plotly_embedding,
)


@dataclass(slots=True)
class PipelineOutputs:
    """Artifacts produced by the style clustering pipeline."""

    raw_matches: pd.DataFrame
    player_match: pd.DataFrame
    player_season: pd.DataFrame
    prepared_frame: pd.DataFrame
    feature_columns: list[str]
    cluster_frame: pd.DataFrame
    cluster_profile: pd.DataFrame
    metrics: pd.DataFrame
    best_method: str
    best_parameters: dict[str, Any]


def _feature_columns_for_analysis(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "ace_rate",
        "double_fault_rate",
        "first_serve_in_pct",
        "first_serve_win_pct",
        "second_serve_win_pct",
        "service_hold_rate",
        "return_points_won_pct",
        "break_points_conversion_pct",
        "return_games_won_pct",
        "tiebreak_frequency",
        "average_match_length",
        "dominance_ratio",
        "win_rate",
        "hard_win_rate",
        "clay_win_rate",
        "grass_win_rate",
        "service_aggression_score",
        "return_aggression_score",
        "winner_to_error_ratio",
        "surface_diversity",
        "average_player_rank",
        "average_opponent_rank",
    ]
    return [column for column in preferred if column in frame.columns]


def _generate_artifacts(
    prepared_frame: pd.DataFrame,
    feature_columns: list[str],
    cluster_frame: pd.DataFrame,
    matrix,
    output_dir: Path,
) -> None:
    try:
        figures_dir = ensure_directory(output_dir / "figures")
        if feature_columns:
            plot_missing_values(prepared_frame[feature_columns], figures_dir / "missing_values.png")
            plot_histograms(prepared_frame, feature_columns[:12], figures_dir / "histograms.png")
            plot_correlation_heatmap(prepared_frame, feature_columns, figures_dir / "correlation_heatmap.png")
            plot_feature_profiles(cluster_frame, feature_columns, path=figures_dir / "cluster_profiles.png")
        plot_cluster_distribution(cluster_frame, path=figures_dir / "cluster_distribution.png")
        plot_pca_embedding(matrix, cluster_frame["cluster"], cluster_frame, path=figures_dir / "pca_clusters.png")
        plot_umap_embedding(matrix, cluster_frame["cluster"], cluster_frame, path=figures_dir / "umap_clusters.png")
        umap_embedding = embed_2d(matrix, method="umap")
        cluster_plot = plotly_embedding(umap_embedding, cluster_frame, color_column="cluster", title="UMAP by cluster")
        cluster_plot.write_html(str(figures_dir / "umap_by_cluster.html"))
        if "favorite_surface" in cluster_frame.columns:
            surface_plot = plotly_embedding(umap_embedding, cluster_frame, color_column="favorite_surface", title="UMAP by favorite surface")
            surface_plot.write_html(str(figures_dir / "umap_by_surface.html"))
        if "average_player_rank" in cluster_frame.columns:
            rank_plot = plotly_embedding(umap_embedding, cluster_frame, color_column="average_player_rank", title="UMAP by average ranking")
            rank_plot.write_html(str(figures_dir / "umap_by_rank.html"))
        for cluster_id, subset in cluster_frame.groupby("cluster"):
            if subset.empty:
                continue
            radar_columns = [column for column in feature_columns[:8] if column in subset.columns]
            if radar_columns:
                plot_cluster_radar(subset[radar_columns].mean(numeric_only=True), radar_columns, figures_dir / f"cluster_{cluster_id}_radar.html")
    except Exception:
        pass


def _best_method_to_fit(method: str, parameter_value: int, matrix, random_state: int) -> Any:
    if method == "kmeans":
        fit = fit_kmeans(matrix, parameter_value, random_state=random_state)
    elif method == "gmm":
        fit = fit_gmm(matrix, parameter_value, random_state=random_state)
    elif method == "hierarchical":
        fit = fit_hierarchical(matrix, parameter_value)
    elif method == "hdbscan":
        fit = fit_hdbscan(matrix, parameter_value)
    else:
        raise ValueError(f"Unknown clustering method chosen: {method}")
    return fit


def run_pipeline(project: ProjectConfig, clustering: ClusteringConfig | None = None) -> PipelineOutputs:
    """Execute the full clustering pipeline end-to-end."""

    clustering = clustering or ClusteringConfig(random_state=project.random_state)
    ensure_directory(project.output_dir)

    matches = load_matches_from_sqlite(
        project.db_path,
        start_year=project.start_year,
        end_year=project.end_year,
    )
    charting = load_optional_charting_project(project.optional_charting_path)
    matches = merge_optional_charting(matches, charting)

    feature_tables = build_feature_tables(matches)
    player_match = feature_tables.player_match
    player_season = feature_tables.player_season
    if player_season.empty:
        raise ValueError("No player-season observations could be generated from the loaded matches")

    feature_columns = _feature_columns_for_analysis(player_season)
    preprocessing = prepare_feature_matrix(
        player_season,
        feature_columns,
        season_column="season",
        use_season_zscore=project.use_season_zscore,
        missing_threshold=project.missing_threshold,
        correlation_threshold=project.correlation_threshold,
    )

    matrix = preprocessing.matrix
    prepared_frame = preprocessing.frame
    feature_columns = preprocessing.feature_columns

    k_metrics = evaluate_k_family(
        matrix,
        k_min=clustering.k_min,
        k_max=clustering.k_max,
        random_state=clustering.random_state,
        stability_iterations=clustering.stability_iterations,
        stability_sample_fraction=clustering.stability_sample_fraction,
    )
    try:
        hdbscan_metrics = evaluate_hdbscan_family(
            matrix,
            min_cluster_sizes=clustering.hdbscan_min_cluster_sizes,
            random_state=clustering.random_state,
            stability_iterations=clustering.stability_iterations,
            stability_sample_fraction=clustering.stability_sample_fraction,
        )
    except ImportError:
        hdbscan_metrics = pd.DataFrame(columns=["method", "k", "silhouette", "davies_bouldin", "calinski_harabasz", "stability"])

    metrics = pd.concat([frame for frame in [k_metrics, hdbscan_metrics] if not frame.empty], ignore_index=True)
    if metrics.empty:
        raise ValueError("Could not evaluate any clustering configuration")
    best_row = choose_best_cluster_row(metrics)
    method = str(best_row["method"])
    parameter_value = int(best_row["k"])

    fit = _best_method_to_fit(method, parameter_value, matrix, clustering.random_state)
    labels = fit.labels
    cluster_frame = attach_cluster_labels(prepared_frame, labels)
    cluster_profile = build_cluster_profile(cluster_frame, feature_columns, label_column="cluster")
    cluster_frame = cluster_frame.merge(cluster_profile[["cluster", "style_label"]], on="cluster", how="left")

    outputs_dir = ensure_directory(project.output_dir)
    save_dataframe(cluster_frame, outputs_dir / "player_season_clusters.csv")
    save_dataframe(cluster_profile, outputs_dir / "cluster_profiles.csv")
    save_dataframe(metrics, outputs_dir / "clustering_metrics.csv")
    save_json(
        {
            "best_method": method,
            "best_parameters": fit.parameters,
            "metrics": metrics.to_dict(orient="records"),
            "cluster_profile": cluster_profile.to_dict(orient="records"),
        },
        outputs_dir / "cluster_report.json",
    )

    _generate_artifacts(prepared_frame, feature_columns, cluster_frame, matrix, outputs_dir)

    return PipelineOutputs(
        raw_matches=matches,
        player_match=player_match,
        player_season=player_season,
        prepared_frame=prepared_frame,
        feature_columns=feature_columns,
        cluster_frame=cluster_frame,
        cluster_profile=cluster_profile,
        metrics=metrics,
        best_method=method,
        best_parameters=fit.parameters,
    )
