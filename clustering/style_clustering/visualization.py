from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_directory

try:  # pragma: no cover - optional dependency
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional dependency
    px = None
    go = None


def _seaborn():
    import seaborn as sns

    return sns


def save_matplotlib_figure(fig: plt.Figure, path: str | Path | None) -> None:
    if path is None:
        return
    output_path = Path(path)
    ensure_directory(output_path.parent)
    fig.savefig(output_path, bbox_inches="tight", dpi=160)


def save_plotly_figure(fig, path: str | Path | None) -> None:
    if path is None:
        return
    output_path = Path(path)
    ensure_directory(output_path.parent)
    fig.write_html(str(output_path))


def plot_missing_values(frame: pd.DataFrame, path: str | Path | None = None) -> plt.Figure:
    sns = _seaborn()
    counts = frame.isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, max(4, len(counts) * 0.25)))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#2c7fb8")
    ax.set_xlabel("Missing ratio")
    ax.set_ylabel("Feature")
    ax.set_title("Missing Values by Feature")
    save_matplotlib_figure(fig, path)
    return fig


def plot_histograms(frame: pd.DataFrame, feature_columns: list[str], path: str | Path | None = None) -> plt.Figure:
    sns = _seaborn()
    if not feature_columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No features available", ha="center", va="center")
        ax.axis("off")
        return fig
    n_cols = 3
    n_rows = int(np.ceil(len(feature_columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, max(4, 3 * n_rows)))
    axes = np.atleast_1d(axes).ravel()
    for ax, column in zip(axes, feature_columns):
        sns.histplot(frame[column].dropna(), kde=True, ax=ax, color="#1f78b4")
        ax.set_title(column)
    for ax in axes[len(feature_columns):]:
        ax.axis("off")
    fig.tight_layout()
    save_matplotlib_figure(fig, path)
    return fig


def plot_correlation_heatmap(frame: pd.DataFrame, feature_columns: list[str], path: str | Path | None = None) -> plt.Figure:
    sns = _seaborn()
    corr = frame[feature_columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(10, len(feature_columns) * 0.55), max(8, len(feature_columns) * 0.45)))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    save_matplotlib_figure(fig, path)
    return fig


def embed_2d(matrix: np.ndarray, method: str = "umap") -> np.ndarray:
    if method == "umap":
        try:
            from umap import UMAP
        except Exception:
            method = "pca"
        else:
            return UMAP(n_components=2, random_state=42).fit_transform(matrix)
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=42).fit_transform(matrix)


def plot_embedding_scatter(
    embedding: np.ndarray,
    labels: pd.Series | np.ndarray,
    frame: pd.DataFrame,
    title: str = "2D Embedding",
    path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_frame = frame.copy()
    plot_frame["x"] = embedding[:, 0]
    plot_frame["y"] = embedding[:, 1]
    scatter = ax.scatter(plot_frame["x"], plot_frame["y"], c=labels, cmap="tab10", s=30, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.colorbar(scatter, ax=ax, label="cluster")
    fig.tight_layout()
    save_matplotlib_figure(fig, path)
    return fig


def plot_pca_embedding(matrix: np.ndarray, labels: pd.Series | np.ndarray, frame: pd.DataFrame, path: str | Path | None = None) -> plt.Figure:
    embedding = embed_2d(matrix, method="pca")
    return plot_embedding_scatter(embedding, labels, frame, title="PCA projection", path=path)


def plot_umap_embedding(matrix: np.ndarray, labels: pd.Series | np.ndarray, frame: pd.DataFrame, path: str | Path | None = None) -> plt.Figure:
    embedding = embed_2d(matrix, method="umap")
    return plot_embedding_scatter(embedding, labels, frame, title="UMAP projection", path=path)


def plot_cluster_distribution(frame: pd.DataFrame, label_column: str = "cluster", path: str | Path | None = None) -> plt.Figure:
    sns = _seaborn()
    counts = frame[label_column].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#4c78a8")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Observations")
    ax.set_title("Cluster Distribution")
    fig.tight_layout()
    save_matplotlib_figure(fig, path)
    return fig


def plot_cluster_radar(cluster_profile: pd.Series, feature_columns: list[str], path: str | Path | None = None):
    """Create a radar chart for a cluster profile using Plotly when available."""

    values = [float(cluster_profile.get(column, 0.0)) for column in feature_columns]
    if go is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
        angles = np.linspace(0, 2 * np.pi, len(feature_columns), endpoint=False).tolist()
        values_closed = values + values[:1]
        angles_closed = angles + angles[:1]
        ax.plot(angles_closed, values_closed, color="#1f77b4")
        ax.fill(angles_closed, values_closed, alpha=0.25)
        ax.set_xticks(angles)
        ax.set_xticklabels(feature_columns, fontsize=8)
        save_matplotlib_figure(fig, path)
        return fig

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=values + values[:1], theta=feature_columns + feature_columns[:1], fill="toself", name="cluster"))
    radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Cluster Radar Chart", showlegend=False)
    save_plotly_figure(radar, path)
    return radar


def plot_feature_profiles(frame: pd.DataFrame, feature_columns: list[str], label_column: str = "cluster", path: str | Path | None = None) -> plt.Figure:
    sns = _seaborn()
    profile = frame.groupby(label_column)[feature_columns].mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(12, len(feature_columns) * 0.65), max(4, len(profile) * 0.55)))
    sns.heatmap(profile, cmap="vlag", center=0, ax=ax)
    ax.set_title("Average Feature Profile by Cluster")
    fig.tight_layout()
    save_matplotlib_figure(fig, path)
    return fig


def plotly_embedding(
    embedding: np.ndarray,
    frame: pd.DataFrame,
    color_column: str = "cluster",
    title: str = "Embedding",
):
    if px is None:
        raise ImportError("plotly is not installed")
    plot_frame = frame.copy()
    plot_frame["x"] = embedding[:, 0]
    plot_frame["y"] = embedding[:, 1]
    return px.scatter(plot_frame, x="x", y="y", color=color_column, hover_data=["player_name", "season"], title=title)
