from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture


@dataclass(slots=True)
class ClusteringFit:
    """Container for a fitted clustering method and its evaluation metrics."""

    method: str
    model: Any
    labels: np.ndarray
    metrics: dict[str, float]
    parameters: dict[str, Any]


def _safe_cluster_scores(matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    valid = labels != -1
    filtered_labels = labels[valid]
    filtered_matrix = matrix[valid]
    unique_labels = np.unique(filtered_labels)
    if filtered_matrix.shape[0] < 3 or unique_labels.size < 2:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"), "calinski_harabasz": float("nan")}
    return {
        "silhouette": float(silhouette_score(filtered_matrix, filtered_labels)),
        "davies_bouldin": float(davies_bouldin_score(filtered_matrix, filtered_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(filtered_matrix, filtered_labels)),
    }


def _refit_result(method: str, matrix: np.ndarray, parameters: dict[str, Any], random_state: int = 42) -> np.ndarray:
    if method == "kmeans":
        return KMeans(n_clusters=parameters["n_clusters"], n_init="auto", random_state=random_state).fit_predict(matrix)
    if method == "gmm":
        return GaussianMixture(n_components=parameters["n_components"], covariance_type="full", random_state=random_state).fit_predict(matrix)
    if method == "hierarchical":
        return AgglomerativeClustering(n_clusters=parameters["n_clusters"], linkage="ward").fit_predict(matrix)
    if method == "hdbscan":
        import hdbscan

        return hdbscan.HDBSCAN(min_cluster_size=parameters["min_cluster_size"]).fit_predict(matrix)
    raise ValueError(f"Unknown clustering method: {method}")


def _bootstrap_stability(
    matrix: np.ndarray,
    method: str,
    parameters: dict[str, Any],
    reference_labels: np.ndarray,
    iterations: int = 10,
    sample_fraction: float = 0.8,
    random_state: int = 42,
) -> float:
    rng = np.random.default_rng(random_state)
    scores: list[float] = []
    sample_size = max(3, int(round(matrix.shape[0] * sample_fraction)))
    for _ in range(iterations):
        sample_idx = rng.choice(matrix.shape[0], size=sample_size, replace=False)
        sample_matrix = matrix[sample_idx]
        sample_labels = _refit_result(method, sample_matrix, parameters, random_state=random_state)
        scores.append(float(adjusted_rand_score(reference_labels[sample_idx], sample_labels)))
    return float(np.mean(scores)) if scores else float("nan")


def fit_kmeans(matrix: np.ndarray, n_clusters: int, random_state: int = 42) -> ClusteringFit:
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = model.fit_predict(matrix)
    metrics = _safe_cluster_scores(matrix, labels)
    metrics["inertia"] = float(model.inertia_)
    return ClusteringFit(method="kmeans", model=model, labels=labels, metrics=metrics, parameters={"n_clusters": n_clusters})


def fit_gmm(matrix: np.ndarray, n_components: int, random_state: int = 42) -> ClusteringFit:
    model = GaussianMixture(n_components=n_components, covariance_type="full", random_state=random_state)
    labels = model.fit_predict(matrix)
    metrics = _safe_cluster_scores(matrix, labels)
    metrics["bic"] = float(model.bic(matrix))
    metrics["aic"] = float(model.aic(matrix))
    return ClusteringFit(method="gmm", model=model, labels=labels, metrics=metrics, parameters={"n_components": n_components})


def fit_hierarchical(matrix: np.ndarray, n_clusters: int) -> ClusteringFit:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(matrix)
    metrics = _safe_cluster_scores(matrix, labels)
    return ClusteringFit(method="hierarchical", model=model, labels=labels, metrics=metrics, parameters={"n_clusters": n_clusters})


def fit_hdbscan(matrix: np.ndarray, min_cluster_size: int) -> ClusteringFit:
    try:
        import hdbscan
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("hdbscan is required for density-based clustering") from exc

    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(matrix)
    metrics = _safe_cluster_scores(matrix, labels)
    metrics["noise_fraction"] = float((labels == -1).mean())
    return ClusteringFit(method="hdbscan", model=model, labels=labels, metrics=metrics, parameters={"min_cluster_size": min_cluster_size})


def evaluate_k_family(
    matrix: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int = 42,
    stability_iterations: int = 10,
    stability_sample_fraction: float = 0.8,
) -> pd.DataFrame:
    """Compare multiple cluster counts for KMeans, GMM and hierarchical clustering."""

    rows: list[dict[str, Any]] = []
    if matrix.shape[0] < 3:
        return pd.DataFrame(rows)
    upper_k = min(k_max, matrix.shape[0] - 1)
    for k in range(k_min, upper_k + 1):
        for result in [fit_kmeans(matrix, k, random_state=random_state), fit_gmm(matrix, k, random_state=random_state), fit_hierarchical(matrix, k)]:
            stability = _bootstrap_stability(
                matrix,
                result.method,
                result.parameters,
                result.labels,
                iterations=stability_iterations,
                sample_fraction=stability_sample_fraction,
                random_state=random_state,
            )
            rows.append({"method": result.method, "k": k, **result.metrics, "stability": stability})
    return pd.DataFrame(rows)


def evaluate_hdbscan_family(
    matrix: np.ndarray,
    min_cluster_sizes: list[int],
    random_state: int = 42,
    stability_iterations: int = 10,
    stability_sample_fraction: float = 0.8,
) -> pd.DataFrame:
    """Compare several min_cluster_size values for HDBSCAN."""

    rows: list[dict[str, Any]] = []
    if matrix.shape[0] < 3:
        return pd.DataFrame(rows)
    for size in min_cluster_sizes:
        result = fit_hdbscan(matrix, size)
        stability = _bootstrap_stability(
            matrix,
            result.method,
            result.parameters,
            result.labels,
            iterations=stability_iterations,
            sample_fraction=stability_sample_fraction,
            random_state=random_state,
        )
        rows.append({"method": result.method, "k": size, **result.metrics, "stability": stability})
    return pd.DataFrame(rows)


def choose_best_cluster_row(metrics: pd.DataFrame) -> pd.Series:
    """Select the strongest configuration using silhouette, stability and compactness."""

    if metrics.empty:
        raise ValueError("No clustering metrics available")
    scored = metrics.copy()
    scored["score"] = scored["silhouette"].fillna(-1.0) - scored["davies_bouldin"].fillna(999.0) / 10.0 + scored["stability"].fillna(0.0)
    best_idx = scored["score"].idxmax()
    return scored.loc[best_idx]


def attach_cluster_labels(frame: pd.DataFrame, labels: np.ndarray, label_column: str = "cluster") -> pd.DataFrame:
    """Return a copy of the frame with cluster labels attached."""

    enriched = frame.copy()
    enriched[label_column] = labels
    return enriched


def infer_style_label(centroid: pd.Series) -> str:
    """Assign an interpretable label to a cluster profile using heuristic rules."""

    scores = {
        "Serve Bot": float(centroid.get("ace_rate", 0) + centroid.get("service_hold_rate", 0) + centroid.get("first_serve_win_pct", 0) - centroid.get("return_points_won_pct", 0)),
        "Counterpuncher": float(centroid.get("return_points_won_pct", 0) + centroid.get("return_games_won_pct", 0) - centroid.get("ace_rate", 0)),
        "Clay Specialist": float(centroid.get("clay_win_rate", 0) + centroid.get("return_points_won_pct", 0)),
        "All-Court": float(centroid.get("hard_win_rate", 0) + centroid.get("clay_win_rate", 0) + centroid.get("grass_win_rate", 0)),
        "Defensive Player": float(centroid.get("tiebreak_frequency", 0) + centroid.get("average_match_length", 0) + centroid.get("return_points_won_pct", 0)),
        "Short-Rally Attacker": float(centroid.get("service_aggression_score", 0) + centroid.get("return_aggression_score", 0) + centroid.get("win_rate", 0)),
    }
    return max(scores, key=scores.get)


def build_cluster_profile(frame: pd.DataFrame, feature_columns: list[str], label_column: str = "cluster") -> pd.DataFrame:
    """Summarize cluster centroids, dominant features and representative players."""

    rows: list[dict[str, Any]] = []
    overall_means = frame[feature_columns].mean(numeric_only=True)
    for cluster_id, subset in frame.groupby(label_column):
        centroid = subset[feature_columns].mean(numeric_only=True)
        z_scores = centroid - overall_means
        top_positive = z_scores.sort_values(ascending=False).head(5)
        top_negative = z_scores.sort_values(ascending=True).head(5)
        if feature_columns:
            representative_players = subset.sort_values(feature_columns[0], ascending=False)[["player_name", "season"]].head(5).to_dict(orient="records")
        else:
            representative_players = subset[["player_name", "season"]].head(5).to_dict(orient="records")
        rows.append(
            {
                "cluster": int(cluster_id),
                "size": int(len(subset)),
                "top_positive_features": ", ".join(top_positive.index.tolist()),
                "top_negative_features": ", ".join(top_negative.index.tolist()),
                "mean_features": centroid.to_dict(),
                "representatives": representative_players,
                "style_label": infer_style_label(centroid),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
