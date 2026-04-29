"""Core scatter plotting functionality for 3D point clouds."""

import math
from typing import Optional, List, Dict, Tuple, Union
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.graph_objects import Figure


def plot(
    points: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_map: Optional[Dict] = None,
    color_map: Optional[Union[List[str], Dict[str, str], float]] = None,
    batch_colors: Optional[List[str]] = None,
    size: float = 1.5,
    title: Optional[str] = None,
    aspectmode: Optional[str] = 'data'
) -> Figure:
    """
    Render an interactive 3D scatter plot using Plotly.

    Args:
        points (np.ndarray): Array of shape (n_points, 3) or (batch, n_points, 3)
            containing 3D coordinates. When a 3D array is provided, each batch element
            is rendered as a separate point cloud with a distinct color.
        labels (np.ndarray, optional): Array of labels for color grouping. When provided
            with label_map, enables discrete color mapping. Without label_map, creates
            a continuous color scale. Not supported for batched input.
        label_map (dict, optional): Maps label values to display names. When provided,
            enables discrete color mapping.
        color_map (list, dict, or float, optional): When label_map is provided, a
            {display_name: color} dict (preferred) or a list of color strings aligned
            with label_map values for discrete coloring. Without label_map, this can be
            a float specifying the continuous color scale midpoint.
        batch_colors (list, optional): List of color strings, one per batch element.
            Only used when points is 3D. Defaults to Plotly's qualitative palette.
        size (float, optional): Size of the scatter plot markers. Default is 1.5.
        title (str, optional): Title of the plot.
        aspectmode (str, optional): Aspect ratio mode for the 3D scene. Default is 'data' (Check plotly docs for available modes).

    Returns:
        plotly.graph_objects.Figure: Interactive 3D scatter plot.

    Raises:
        ValueError: If points array shape is invalid or incompatible options are used.

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

        Batched point clouds:
        >>> points = np.random.randn(3, 100, 3)
        >>> fig = plot(points, batch_colors=["red", "green", "blue"])
        >>> fig.show()
    """
    if points.ndim == 3:
        B, N, D = points.shape
        if D != 3:
            raise ValueError("points must be of shape (B, N, 3)")
        if labels is not None:
            raise ValueError("labels are not supported for batched point clouds")

        flat_points = points.reshape(-1, 3)
        batch_labels = np.repeat(np.arange(B), N).astype(str)

        if batch_colors is None:
            batch_colors = px.colors.qualitative.Plotly[:B]

        df = pd.DataFrame(flat_points, columns=["x", "y", "z"])
        df["batch"] = batch_labels
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="batch",
                            color_discrete_sequence=batch_colors)
    else:
        if points.shape[1] != 3:
            raise ValueError("points must be of shape (n_points, 3)")

        # Create a DataFrame for easier plotting
        df = pd.DataFrame(points, columns=["x", "y", "z"])

        if labels is not None:
            df["label"] = labels
            if label_map:
                df["label"] = df["label"].map(label_map).fillna(df["label"])
                category_orders = {"label": list(label_map.values())}
                if isinstance(color_map, dict):
                    discrete_map = color_map
                elif isinstance(color_map, list):
                    discrete_map = dict(zip(label_map.values(), color_map))
                else:
                    discrete_map = None
                fig = px.scatter_3d(
                    df, x="x", y="y", z="z", color="label",
                    color_discrete_map=discrete_map,
                    category_orders=category_orders,
                )
            else:
                fig = px.scatter_3d(df, x="x", y="y", z="z", color="label",
                                  color_continuous_midpoint=color_map, range_color=[0, 1])
        else:
            fig = px.scatter_3d(df, x="x", y="y", z="z")

    fig.update_traces(marker=dict(size=size))

    fig.update_layout(
        scene=dict(
            aspectmode=aspectmode
        )
    )

    if title:
        fig.update_layout(title=title)

    return fig


def combine_plots(figs: List[Figure], rows: int = 1, cols: int = 2, aspectmode: Optional[str] = 'data') -> Figure:
    """
    Combine multiple 3D plots into a single figure with subplots.

    Args:
        figs (list): List of Plotly figures to combine.
        rows (int, optional): Number of rows in the subplot grid. Default is 1.
        cols (int, optional): Number of columns in the subplot grid. Default is 2.
        aspectmode (str, optional): Aspect ratio mode for the 3D scene. Default is 'data' (Check plotly docs for available modes).

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

    scene_updates = {}
    for i in range(len(figs)):
        key = "scene" if i == 0 else f"scene{i + 1}"
        scene_updates[key] = dict(aspectmode=aspectmode)

    combined_fig.update_layout(**scene_updates,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return combined_fig


_DEFAULT_AXIS_STYLE = dict(
    backgroundcolor="white",
    gridcolor="lightgray",
    showbackground=True,
    showticklabels=False,
    title="",
    zerolinecolor="lightgray",
)


def beautify(
    fig: Figure,
    paper_bgcolor: str = "white",
    scene_bgcolor: str = "white",
    axis_style: Optional[Dict] = None,
) -> Figure:
    """
    Apply a clean, GIF-friendly style to every 3D scene in a figure.

    Hides axis tick labels and titles, gives axes a light gray grid, and
    paints both the figure paper and each 3D scene's background. Works on
    single-scene figures (from :func:`plot`) and multi-scene figures (from
    :func:`plot_grid` / :func:`combine_plots`), which use layout keys
    ``scene``, ``scene2``, ``scene3``, .... Mutates and returns ``fig``.

    Args:
        fig (plotly.graph_objects.Figure): Figure to style in place.
        paper_bgcolor (str, optional): Color of the area outside the 3D
            scenes. Default ``"white"``.
        scene_bgcolor (str, optional): Color of the area inside each 3D
            scene's box. Default ``"white"``.
        axis_style (dict, optional): Override the per-axis style applied
            to ``xaxis``/``yaxis``/``zaxis`` of every scene. When ``None``
            a clean default (light gray grid, hidden tick labels and
            titles) is used.

    Returns:
        plotly.graph_objects.Figure: The same ``fig``, restyled.

    Examples:
        >>> import numpy as np
        >>> import cloudglancer as cg
        >>> pts = np.random.randn(6, 200, 3)
        >>> fig = cg.beautify(cg.plot_grid(pts, colors="#1f77b4"))
        >>> cg.animate(fig, "grid.gif", n_frames=60)
    """
    style = dict(_DEFAULT_AXIS_STYLE if axis_style is None else axis_style)

    fig.update_layout(paper_bgcolor=paper_bgcolor)

    for key in fig.layout:
        if key == "scene" or (key.startswith("scene") and key[len("scene"):].isdigit()):
            fig.layout[key].update(
                bgcolor=scene_bgcolor,
                xaxis=style,
                yaxis=style,
                zaxis=style,
            )

    return fig


def _resolve_grid_shape(n: int, rows: Optional[int], cols: Optional[int]) -> Tuple[int, int]:
    if rows is None and cols is None:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
    elif rows is None:
        rows = int(math.ceil(n / cols))
    elif cols is None:
        cols = int(math.ceil(n / rows))
    if rows * cols < n:
        raise ValueError(f"rows*cols={rows * cols} < B={n}")
    return rows, cols


def plot_grid(
    points: np.ndarray,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    colors: Optional[Union[str, List[str]]] = None,
    size: float = 1.5,
    aspectmode: Optional[str] = 'data',
    showlegend: bool = False,
) -> Figure:
    """
    Render a batch of 3D point clouds as a grid of subplots, one cloud per cell.

    Args:
        points (np.ndarray): Array of shape (B, N, 3). Each batch element is
            rendered into its own subplot.
        rows (int, optional): Number of grid rows. If both ``rows`` and ``cols``
            are ``None``, a near-square grid is chosen automatically. If only
            one is given, the other is derived from ``B``.
        cols (int, optional): Number of grid columns. See ``rows``.
        colors (str or list of str, optional): Color for each subplot. Pass a
            single color string to use it for every cloud, or a list of length
            ``B`` to color each cloud individually. Defaults to Plotly's
            qualitative palette.
        size (float, optional): Marker size. Default is 1.5.
        aspectmode (str, optional): Aspect mode for each 3D scene. Default
            ``'data'``.
        showlegend (bool, optional): Whether to show the figure legend.
            Default ``False`` (legends are noisy in grids).

    Returns:
        plotly.graph_objects.Figure: A combined figure with the grid of
        subplots, ready for ``.show()`` or for passing to
        :func:`cloudglancer.animate`.

    Raises:
        ValueError: If ``points`` is not (B, N, 3), if ``colors`` is a list
            whose length does not match ``B``, or if ``rows*cols < B``.

    Examples:
        >>> import numpy as np
        >>> import cloudglancer as cg
        >>> pts = np.random.randn(6, 200, 3)
        >>> fig = cg.plot_grid(pts, colors="#1f77b4", size=1.5)
        >>> cg.animate(fig, "grid.gif", n_frames=60)
    """
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be of shape (B, N, 3); got {points.shape}")

    B = points.shape[0]
    rows, cols = _resolve_grid_shape(B, rows, cols)

    if colors is None:
        palette = px.colors.qualitative.Plotly
        cell_colors = [palette[i % len(palette)] for i in range(B)]
    elif isinstance(colors, str):
        cell_colors = [colors] * B
    else:
        if len(colors) != B:
            raise ValueError(
                f"colors must have length B={B}; got {len(colors)}"
            )
        cell_colors = list(colors)

    figs = []
    for i in range(rows * cols):
        if i < B:
            sub = plot(
                points[i:i + 1],
                batch_colors=[cell_colors[i]],
                size=size,
                aspectmode=aspectmode,
            )
        else:
            sub = plot(
                np.zeros((1, 1, 3), dtype=points.dtype),
                batch_colors=["rgba(0,0,0,0)"],
                size=size,
                aspectmode=aspectmode,
            )
        figs.append(sub)

    combined = combine_plots(figs, rows=rows, cols=cols, aspectmode=aspectmode)
    combined.update_layout(showlegend=showlegend)
    return combined
