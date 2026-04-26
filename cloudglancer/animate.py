"""Render a rotating GIF animation of a 3D Plotly figure."""

from io import BytesIO
from typing import Tuple
import copy

import numpy as np
from plotly.graph_objects import Figure
from tqdm import tqdm


_AXIS_VECTORS = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def _rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' rotation matrix for rotation by `theta` around unit `axis`."""
    x, y, z = axis
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _scene_keys(fig: Figure) -> list:
    """Return all scene-typed layout keys on `fig` (e.g. 'scene', 'scene2', ...)."""
    keys = []
    for key in fig.layout:
        if key == "scene" or (key.startswith("scene") and key[5:].isdigit()):
            keys.append(key)
    return keys


def _eye_to_vec(eye) -> np.ndarray:
    if eye is None or eye.x is None:
        return np.array([1.25, 1.25, 1.25])
    return np.array([eye.x, eye.y, eye.z])


def animate(
    fig: Figure,
    output_path: str,
    axis: str = "z",
    n_frames: int = 60,
    duration: int = 50,
    width: int = 800,
    height: int = 600,
    loop: int = 0,
    progress: bool = True,
) -> str:
    """
    Render a rotating GIF of a 3D Plotly figure.

    The camera orbits the scene around the chosen axis for one full revolution
    over `n_frames` frames, producing a turntable-style animation. The input
    figure is not mutated.

    Args:
        fig: A Plotly Figure (e.g. the output of :func:`cloudglancer.plot` or
            :func:`cloudglancer.combine_plots`).
        output_path: Path where the GIF is written.
        axis: Rotation axis, one of ``'x'``, ``'y'``, ``'z'``. Defaults to ``'z'``.
        n_frames: Number of frames in one full revolution. Defaults to 60.
        duration: Per-frame display time in milliseconds. Defaults to 50.
        width: Frame width in pixels. Defaults to 800.
        height: Frame height in pixels. Defaults to 600.
        loop: Number of times the GIF should loop (0 = infinite). Defaults to 0.
        progress: Show a tqdm progress bar while rendering frames. Defaults to True.

    Returns:
        The ``output_path`` it wrote to.

    Raises:
        ValueError: If `axis` is not one of ``'x'``, ``'y'``, ``'z'``.
        ImportError: If ``kaleido`` or ``Pillow`` is not installed.

    Examples:
        >>> import numpy as np
        >>> import cloudglancer as cg
        >>> pts = np.random.randn(500, 3)
        >>> fig = cg.plot(pts, size=2.0)
        >>> cg.animate(fig, "rotation.gif", axis="z", n_frames=60)
        'rotation.gif'
    """
    if axis not in _AXIS_VECTORS:
        raise ValueError(f"axis must be one of 'x', 'y', 'z' (got {axis!r})")

    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "animate() requires Pillow. Install it with `pip install Pillow`."
        ) from e

    axis_vec = _AXIS_VECTORS[axis]
    up_vec = {"x": dict(x=1, y=0, z=0),
              "y": dict(x=0, y=1, z=0),
              "z": dict(x=0, y=0, z=1)}[axis]

    scene_keys = _scene_keys(fig)
    if not scene_keys:
        raise ValueError("figure has no 3D scene to animate")

    original_cameras = {k: copy.deepcopy(fig.layout[k].camera) for k in scene_keys}
    initial_eyes = {k: _eye_to_vec(fig.layout[k].camera.eye) for k in scene_keys}

    frames = []
    try:
        for i in tqdm(range(n_frames), desc="Rendering frames",
                      unit="frame", disable=not progress):
            theta = 2.0 * np.pi * i / n_frames
            R = _rotation_matrix(axis_vec, theta)

            scene_updates = {}
            for k in scene_keys:
                eye = R @ initial_eyes[k]
                scene_updates[k] = dict(
                    camera=dict(
                        eye=dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])),
                        up=up_vec,
                    )
                )
            fig.update_layout(**scene_updates)

            try:
                png = fig.to_image(format="png", width=width, height=height)
            except ValueError as e:
                raise ImportError(
                    "animate() requires kaleido for PNG export. "
                    "Install it with `pip install 'kaleido<1.0'`."
                ) from e

            frames.append(Image.open(BytesIO(png)).convert("P", palette=Image.ADAPTIVE))
    finally:
        for k, cam in original_cameras.items():
            fig.update_layout(**{k: dict(camera=cam)})

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=True,
        disposal=2,
    )

    return output_path
