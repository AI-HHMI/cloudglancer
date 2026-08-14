"""Render a rotating GIF animation of a 3D Plotly figure."""

import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from typing import Optional, Tuple

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


def _eye_to_vec(eye: dict) -> np.ndarray:
    if not eye or eye.get("x") is None:
        return np.array([1.25, 1.25, 1.25])
    return np.array([eye["x"], eye["y"], eye["z"]], dtype=float)


_WORKER_FIG_DICT = None


def _init_worker(fig_dict: dict) -> None:
    global _WORKER_FIG_DICT
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning,
                            message=r".*[Kk]aleido.*")
    _WORKER_FIG_DICT = fig_dict


def _render_frame(args: Tuple[dict, int, int]) -> bytes:
    cameras, width, height = args
    import plotly.io as pio
    layout = _WORKER_FIG_DICT.setdefault("layout", {})
    for key, cam in cameras.items():
        layout.setdefault(key, {})["camera"] = cam
    return pio.to_image(_WORKER_FIG_DICT, format="png",
                        width=width, height=height, validate=False)


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
    n_workers: Optional[int] = None,
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
        n_workers: Number of parallel render processes. Defaults to
            ``min(16, cpu_count, n_frames)``. Raise it on machines with many
            cores for large frame counts. On non-Linux platforms worker
            processes are spawned, so scripts calling ``animate()`` there must
            be guarded by ``if __name__ == "__main__":``.

    Returns:
        The ``output_path`` it wrote to.

    Raises:
        ValueError: If `axis` is not one of ``'x'``, ``'y'``, ``'z'``, or
            `n_workers` is less than 1.
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

    try:
        import kaleido  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "animate() requires kaleido for PNG export. "
            "Install it with `pip install 'kaleido<1.0'`."
        ) from e

    axis_vec = _AXIS_VECTORS[axis]
    up_vec = {"x": dict(x=1, y=0, z=0),
              "y": dict(x=0, y=1, z=0),
              "z": dict(x=0, y=0, z=1)}[axis]

    scene_keys = _scene_keys(fig)
    if not scene_keys:
        raise ValueError("figure has no 3D scene to animate")

    fig_dict = fig.to_dict()
    layout_dict = fig_dict.setdefault("layout", {})

    orig_cams = {}
    initial_eyes = {}
    for k in scene_keys:
        cam = dict((layout_dict.get(k) or {}).get("camera") or {})
        orig_cams[k] = cam
        initial_eyes[k] = _eye_to_vec(cam.get("eye"))

    tasks = []
    for i in range(n_frames):
        theta = 2.0 * np.pi * i / n_frames
        R = _rotation_matrix(axis_vec, theta)
        cameras = {}
        for k in scene_keys:
            eye = R @ initial_eyes[k]
            cameras[k] = {
                **orig_cams[k],
                "eye": dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])),
                "up": up_vec,
            }
        tasks.append((cameras, width, height))

    if n_workers is None:
        n_workers = min(16, os.cpu_count() or 2)
    elif n_workers < 1:
        raise ValueError(f"n_workers must be >= 1 (got {n_workers})")
    n_workers = min(n_workers, n_frames)

    # fork keeps unguarded user scripts working and skips per-worker re-imports;
    # spawn elsewhere (fork is unsafe on macOS, unavailable on Windows).
    start_method = "fork" if sys.platform.startswith("linux") else "spawn"

    pngs: list = [None] * n_frames
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=multiprocessing.get_context(start_method),
        initializer=_init_worker,
        initargs=(fig_dict,),
    ) as executor:
        iterator = executor.map(_render_frame, tasks, chunksize=1)
        for i, png in enumerate(
            tqdm(iterator, total=n_frames, desc="Rendering frames",
                 unit="frame", disable=not progress)
        ):
            pngs[i] = png

    ref_palette = Image.open(BytesIO(pngs[0])).convert("P", palette=Image.ADAPTIVE)

    frames = [ref_palette]
    for png in pngs[1:]:
        rgb = Image.open(BytesIO(png)).convert("RGB")
        frames.append(rgb.quantize(palette=ref_palette, dither=Image.NONE))

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        disposal=2,
    )

    return output_path
