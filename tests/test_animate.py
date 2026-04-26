"""Unit tests for cloudglancer.animate module."""

import numpy as np
import pytest

from cloudglancer import plot, combine_plots, animate


def _is_gif(path) -> bool:
    with open(path, "rb") as f:
        head = f.read(6)
    return head in (b"GIF87a", b"GIF89a")


def test_animate_writes_file(tmp_path):
    points = np.random.randn(50, 3)
    fig = plot(points)
    out = tmp_path / "out.gif"
    result = animate(fig, str(out), n_frames=4, width=160, height=120, progress=False)
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 0
    assert _is_gif(out)


def test_animate_invalid_axis(tmp_path):
    fig = plot(np.random.randn(10, 3))
    with pytest.raises(ValueError, match="axis must be one of"):
        animate(fig, str(tmp_path / "x.gif"), axis="w", n_frames=2)


def test_animate_does_not_mutate_figure(tmp_path):
    fig = plot(np.random.randn(20, 3))
    fig.update_layout(scene=dict(camera=dict(eye=dict(x=2.0, y=2.0, z=2.0))))
    before = (
        fig.layout.scene.camera.eye.x,
        fig.layout.scene.camera.eye.y,
        fig.layout.scene.camera.eye.z,
    )
    animate(fig, str(tmp_path / "x.gif"), n_frames=3, width=120, height=120, progress=False)
    after = (
        fig.layout.scene.camera.eye.x,
        fig.layout.scene.camera.eye.y,
        fig.layout.scene.camera.eye.z,
    )
    assert before == after


def test_animate_combined_figure(tmp_path):
    f1 = plot(np.random.randn(20, 3))
    f2 = plot(np.random.randn(20, 3))
    combined = combine_plots([f1, f2], rows=1, cols=2)
    out = tmp_path / "combined.gif"
    animate(combined, str(out), axis="y", n_frames=3, width=200, height=120, progress=False)
    assert out.exists() and _is_gif(out)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_animate_each_axis(tmp_path, axis):
    fig = plot(np.random.randn(20, 3))
    out = tmp_path / f"{axis}.gif"
    animate(fig, str(out), axis=axis, n_frames=3, width=120, height=120, progress=False)
    assert _is_gif(out)
