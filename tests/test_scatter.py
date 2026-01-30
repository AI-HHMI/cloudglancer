"""Unit tests for cloudglancer.scatter module."""

import numpy as np
import pytest
from cloudglancer import plot, combine_plots


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
