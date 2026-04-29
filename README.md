[![PyPI version](https://img.shields.io/pypi/v/cloudglancer.svg)](https://pypi.org/project/cloudglancer/)


# cloudglancer

Simple interactive visualization of 3D point clouds using Plotly.

## Features

- Interactive 3D scatter plots with pan, zoom, and rotation
- Support for categorical and continuous color mapping
- Combine multiple plots into subplot grids
- Easy-to-use API with sensible defaults
- Type hints for better IDE support
- Plot batched point clouds (B, N, 3)
- Render a (B, N, 3) batch as a grid of subplots (one cloud per cell)
- Export a rotating turntable GIF of any figure

## Installation

```bash
pip install cloudglancer
```

## Quick Start

```python
import numpy as np
import cloudglancer as cg

# Generate random 3D points
points = np.random.randn(500, 3)

# Create and display the plot
cg.plot(points, title="My Point Cloud", size=2.0).show()
```

Export a rotating GIF of the same figure:

```python
fig = cg.plot(points, size=2.0)
cg.animate(fig, "rotation.gif", axis="z", n_frames=60)
```

Render a batch of point clouds as a grid of subplots (one cloud per cell):

```python
batch = np.random.randn(6, 500, 3)  # (B, N, 3)

# Auto near-square grid, single color for every cloud
fig = cg.plot_grid(batch, colors="#1f77b4", size=1.5)

# Or explicit grid + per-cell colors
fig = cg.plot_grid(batch, rows=2, cols=3,
                   colors=["red", "green", "blue", "orange", "purple", "teal"])

# Combine with animate() for a rotating GIF of the whole grid
cg.animate(fig, "grid.gif", n_frames=60, width=1600, height=1200)
```

Apply a clean, GIF-friendly style (white backgrounds, hidden tick labels,
light gray axis grid) to every 3D scene in a figure — works for both
single plots and grids:

```python
fig = cg.beautify(cg.plot_grid(batch, colors="#1f77b4"))
cg.animate(fig, "grid.gif", n_frames=60)
```

More examples are in the `examples` folder.

### Development Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/yourusername/cloudglancer.git
cd cloudglancer
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Install with Development Dependencies

```bash
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.9
- plotly >= 5.0.0
- pandas >= 2.0.0
- numpy >= 1.24.0

### Building the Package

```bash
pip install build
python -m build
```

This will create both wheel and source distributions in the `dist/` directory.

### Deploy to PyPi
```bash
twine upload --verbose dist/*
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
