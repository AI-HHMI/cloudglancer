"""Core scatter plotting functionality for 3D point clouds."""

from typing import Optional, List, Dict, Union
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.graph_objects import Figure


def plot(
    points: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_map: Optional[Dict] = None,
    color_map: Optional[Union[List[str], float]] = None,
    size: float = 1.5,
    title: Optional[str] = None
) -> Figure:
    """
    Render an interactive 3D scatter plot using Plotly.

    Args:
        points (np.ndarray): Array of shape (n_points, 3) containing 3D coordinates.
        labels (np.ndarray, optional): Array of labels for color grouping. When provided
            with label_map, enables discrete color mapping. Without label_map, creates
            a continuous color scale.
        label_map (dict, optional): Maps label values to display names. When provided,
            enables discrete color mapping.
        color_map (list or float, optional): When label_map is provided, this should be
            a list of color strings for discrete coloring. Without label_map, this can be
            a float specifying the continuous color scale midpoint.
        size (float, optional): Size of the scatter plot markers. Default is 1.5.
        title (str, optional): Title of the plot.

    Returns:
        plotly.graph_objects.Figure: Interactive 3D scatter plot.

    Raises:
        ValueError: If points array is not of shape (n_points, 3).

    Examples:
        Basic scatter plot:
        >>> import numpy as np
        >>> points = np.random.randn(100, 3)
        >>> fig = plot(points, title="Random Points")
        >>> fig.show()

        Plot with categorical labels:
        >>> labels = np.random.choice([0, 1, 2], size=100)
        >>> label_map = {0: "Class A", 1: "Class B", 2: "Class C"}
        >>> color_map = ["red", "blue", "green"]
        >>> fig = plot(points, labels=labels, label_map=label_map, color_map=color_map)
        >>> fig.show()
    """
    if points.shape[1] != 3:
        raise ValueError("points must be of shape (n_points, 3)")

    # Create a DataFrame for easier plotting
    df = pd.DataFrame(points, columns=["x", "y", "z"])

    if labels is not None:
        df["label"] = labels
        if label_map:
            df["label"] = df["label"].map(label_map).fillna(df["label"])
            fig = px.scatter_3d(df, x="x", y="y", z="z", color="label",
                              color_discrete_sequence=color_map)
        else:
            fig = px.scatter_3d(df, x="x", y="y", z="z", color="label",
                              color_continuous_midpoint=color_map, range_color=[0, 1])
    else:
        fig = px.scatter_3d(df, x="x", y="y", z="z")

    fig.update_traces(marker=dict(size=size))

    # Fix: Apply title if provided
    if title:
        fig.update_layout(title=title)

    return fig


def combine_plots(figs: List[Figure], rows: int = 1, cols: int = 2) -> Figure:
    """
    Combine multiple 3D plots into a single figure with subplots.

    Args:
        figs (list): List of Plotly figures to combine.
        rows (int, optional): Number of rows in the subplot grid. Default is 1.
        cols (int, optional): Number of columns in the subplot grid. Default is 2.

    Returns:
        plotly.graph_objects.Figure: Combined figure with all plots arranged in a grid.

    Raises:
        ValueError: If the number of figures doesn't match rows * cols.

    Examples:
        Combine two plots side by side:
        >>> points1 = np.random.randn(100, 3)
        >>> points2 = np.random.randn(100, 3) + 5
        >>> fig1 = plot(points1, title="Dataset 1")
        >>> fig2 = plot(points2, title="Dataset 2")
        >>> combined = combine_plots([fig1, fig2], rows=1, cols=2)
        >>> combined.show()
    """
    from plotly.subplots import make_subplots

    # Fix: Validate figure count
    if len(figs) != rows * cols:
        raise ValueError(
            f"Number of figures ({len(figs)}) must equal rows * cols ({rows * cols})"
        )

    # Fix: Generate specs dynamically based on rows and cols
    specs = [[{"type": "scene"} for _ in range(cols)] for _ in range(rows)]

    combined_fig = make_subplots(
        rows=rows, cols=cols,
        specs=specs
    )

    for i, fig in enumerate(figs):
        for trace in fig.data:
            combined_fig.add_trace(trace, row=(i // cols) + 1, col=(i % cols) + 1)

    combined_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return combined_fig
