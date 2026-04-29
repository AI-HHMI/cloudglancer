"""Unit tests for cloudglancer.scatter module."""

import numpy as np
import pytest
from cloudglancer import plot, combine_plots, plot_grid, beautify


def test_plot_basic():
    """Test basic plot creation with random points."""
    points = np.random.randn(100, 3)
    fig = plot(points)
    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter3d"


def test_plot_with_labels():
    """Test plot with continuous labels."""
    points = np.random.randn(100, 3)
    labels = np.random.rand(100)
    fig = plot(points, labels=labels)
    assert fig is not None
    assert len(fig.data) == 1


def test_plot_with_label_map():
    """Test plot with discrete color mapping."""
    points = np.random.randn(100, 3)
    labels = np.random.choice([0, 1, 2], size=100)
    label_map = {0: "Class A", 1: "Class B", 2: "Class C"}
    color_map = ["red", "blue", "green"]
    fig = plot(points, labels=labels, label_map=label_map, color_map=color_map)
    assert fig is not None
    assert len(fig.data) > 0


def test_plot_label_map_color_binding_is_order_independent():
    """Colors must bind to labels via label_map, not the point order
    (regression for issue #1)."""
    pts = np.random.randn(200, 3)
    labels_a = np.zeros(200, dtype=int); labels_a[:100] = 1
    labels_b = np.zeros(200, dtype=int); labels_b[100:] = 1
    label_map = {0: "unknown", 1: "known"}
    color_map = ["red", "steelblue"]

    fig_a = plot(pts, labels=labels_a, label_map=label_map, color_map=color_map)
    fig_b = plot(pts, labels=labels_b, label_map=label_map, color_map=color_map)

    def color_for(fig, name):
        trace = next(t for t in fig.data if t.name == name)
        return trace.marker.color

    assert color_for(fig_a, "unknown") == color_for(fig_b, "unknown") == "red"
    assert color_for(fig_a, "known") == color_for(fig_b, "known") == "steelblue"


def test_plot_label_map_accepts_color_dict():
    """color_map may be a {display_name: color} dict."""
    pts = np.random.randn(60, 3)
    labels = np.array([0] * 30 + [1] * 30)
    fig = plot(
        pts, labels=labels,
        label_map={0: "unknown", 1: "known"},
        color_map={"unknown": "red", "known": "steelblue"},
    )
    by_name = {t.name: t.marker.color for t in fig.data}
    assert by_name["unknown"] == "red"
    assert by_name["known"] == "steelblue"


def test_plot_invalid_shape():
    """Test that ValueError is raised for incorrect point shape."""
    points = np.random.randn(100, 2)  # Wrong shape
    with pytest.raises(ValueError, match="points must be of shape"):
        plot(points)


def test_plot_title():
    """Test that title is correctly applied to the figure."""
    points = np.random.randn(100, 3)
    title = "Test Plot Title"
    fig = plot(points, title=title)
    assert fig.layout.title.text == title


def test_plot_title_none():
    """Test that plot works without a title."""
    points = np.random.randn(100, 3)
    fig = plot(points, title=None)
    assert fig is not None


def test_combine_plots():
    """Test combining multiple plots."""
    points1 = np.random.randn(50, 3)
    points2 = np.random.randn(50, 3)
    fig1 = plot(points1)
    fig2 = plot(points2)
    combined = combine_plots([fig1, fig2], rows=1, cols=2)
    assert combined is not None
    assert len(combined.data) == 2


def test_combine_plots_grid():
    """Test combining plots in a 2x2 grid."""
    figs = [plot(np.random.randn(30, 3)) for _ in range(4)]
    combined = combine_plots(figs, rows=2, cols=2)
    assert combined is not None
    assert len(combined.data) == 4


def test_combine_plots_invalid_count():
    """Test that ValueError is raised when figure count doesn't match grid."""
    fig1 = plot(np.random.randn(50, 3))
    fig2 = plot(np.random.randn(50, 3))
    with pytest.raises(ValueError, match="Number of figures"):
        combine_plots([fig1, fig2], rows=2, cols=2)  # 2 figs, but 2x2 = 4 spaces


def test_plot_size_parameter():
    """Test that marker size parameter is applied."""
    points = np.random.randn(100, 3)
    size = 3.0
    fig = plot(points, size=size)
    assert fig.data[0].marker.size == size


def test_plot_batched_basic():
    """Test batched plot with default colors."""
    points = np.random.randn(3, 100, 3)
    fig = plot(points)
    assert fig is not None
    assert len(fig.data) == 3  # one trace per batch element


def test_plot_batched_custom_colors():
    """Test batched plot with explicit batch_colors."""
    points = np.random.randn(2, 50, 3)
    fig = plot(points, batch_colors=["red", "blue"])
    assert fig is not None
    assert len(fig.data) == 2


def test_plot_batched_invalid_shape():
    """Test that ValueError is raised for (B, N, 2) input."""
    points = np.random.randn(3, 100, 2)
    with pytest.raises(ValueError, match="points must be of shape"):
        plot(points)


def test_plot_batched_with_labels_raises():
    """Test that combining batch input with labels raises ValueError."""
    points = np.random.randn(3, 100, 3)
    labels = np.zeros(300)
    with pytest.raises(ValueError, match="labels are not supported"):
        plot(points, labels=labels)


def _scene_count(fig):
    return sum(
        1 for k in fig.layout
        if k == "scene" or (k.startswith("scene") and k[5:].isdigit())
    )


def test_plot_grid_auto_shape():
    """Auto grid for B=5 → 3 cols x 2 rows = 6 cells."""
    pts = np.random.randn(5, 30, 3)
    fig = plot_grid(pts)
    assert _scene_count(fig) == 6
    assert fig.layout.showlegend is False


def test_plot_grid_explicit_shape():
    pts = np.random.randn(4, 30, 3)
    fig = plot_grid(pts, rows=2, cols=2)
    assert _scene_count(fig) == 4
    assert len(fig.data) == 4


def test_plot_grid_single_color_string():
    pts = np.random.randn(3, 20, 3)
    fig = plot_grid(pts, colors="#1f77b4")
    cloud_traces = [t for t in fig.data if len(t.x) > 1]
    assert len(cloud_traces) == 3
    for t in cloud_traces:
        assert t.marker.color == "#1f77b4"


def test_plot_grid_color_list():
    pts = np.random.randn(3, 20, 3)
    fig = plot_grid(pts, colors=["red", "green", "blue"])
    cloud_traces = [t for t in fig.data if len(t.x) > 1]
    colors = [t.marker.color for t in cloud_traces]
    assert sorted(colors) == sorted(["red", "green", "blue"])


def test_plot_grid_invalid_shape():
    with pytest.raises(ValueError, match="points must be of shape"):
        plot_grid(np.random.randn(3, 20, 2))


def test_plot_grid_grid_too_small():
    pts = np.random.randn(5, 20, 3)
    with pytest.raises(ValueError, match="rows\\*cols"):
        plot_grid(pts, rows=2, cols=2)


def test_plot_grid_color_list_length_mismatch():
    pts = np.random.randn(3, 20, 3)
    with pytest.raises(ValueError, match="colors must have length"):
        plot_grid(pts, colors=["red", "green"])


def test_plot_grid_pads_when_grid_larger_than_b():
    """B=3 in a 2x2 grid → 4 scenes, with the last cell padded."""
    pts = np.random.randn(3, 20, 3)
    fig = plot_grid(pts, rows=2, cols=2)
    assert _scene_count(fig) == 4
    assert len(fig.data) == 4


def test_beautify_single_scene():
    pts = np.random.randn(50, 3)
    fig = beautify(plot(pts))
    assert fig.layout.paper_bgcolor == "white"
    assert fig.layout.scene.bgcolor == "white"
    assert fig.layout.scene.xaxis.showticklabels is False
    assert fig.layout.scene.yaxis.gridcolor == "lightgray"


def test_beautify_applies_to_every_grid_scene():
    """All scenes (not just `scene`) should get the styling."""
    pts = np.random.randn(4, 30, 3)
    fig = beautify(plot_grid(pts, rows=2, cols=2))
    scene_keys = [
        k for k in fig.layout
        if k == "scene" or (k.startswith("scene") and k[5:].isdigit())
    ]
    assert len(scene_keys) == 4
    for k in scene_keys:
        scene = fig.layout[k]
        assert scene.bgcolor == "white"
        assert scene.xaxis.showticklabels is False
        assert scene.yaxis.showticklabels is False
        assert scene.zaxis.showticklabels is False


def test_beautify_custom_colors():
    fig = beautify(plot(np.random.randn(20, 3)),
                   paper_bgcolor="black", scene_bgcolor="black")
    assert fig.layout.paper_bgcolor == "black"
    assert fig.layout.scene.bgcolor == "black"


def test_beautify_returns_same_figure():
    fig = plot(np.random.randn(20, 3))
    assert beautify(fig) is fig
