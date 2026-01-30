# cloudglancer

Simple interactive visualization of 3D point clouds using Plotly.

## Features

- Interactive 3D scatter plots with pan, zoom, and rotation
- Support for categorical and continuous color mapping
- Combine multiple plots into subplot grids
- Easy-to-use API with sensible defaults
- Type hints for better IDE support

## Installation

### From PyPI (once published)

```bash
pip install cloudglancer
```

### Development Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/yourusername/cloudglancer.git
cd cloudglancer
python -m venv cloudglancer
source cloudglancer/bin/activate
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

## Quick Start

### Basic 3D Scatter Plot

```python
import numpy as np
import cloudglancer

# Generate random 3D points
points = np.random.randn(500, 3)

# Create and display the plot
fig = cloudglancer.plot(points, title="My Point Cloud", size=2.0)
fig.show()
```

### Plot with Categorical Labels

```python
import numpy as np
import cloudglancer

# Generate three clusters
cluster1 = np.random.randn(150, 3)
cluster2 = np.random.randn(150, 3) + np.array([5, 0, 0])
cluster3 = np.random.randn(150, 3) + np.array([0, 5, 5])

points = np.vstack([cluster1, cluster2, cluster3])
labels = np.array([0] * 150 + [1] * 150 + [2] * 150)

# Define label mapping and colors
label_map = {0: "Cluster A", 1: "Cluster B", 2: "Cluster C"}
color_map = ["red", "blue", "green"]

fig = cloudglancer.plot(
    points,
    labels=labels,
    label_map=label_map,
    color_map=color_map,
    title="Three Clusters"
)
fig.show()
```

### Combine Multiple Plots

```python
import numpy as np
import cloudglancer

# Create multiple point clouds
points1 = np.random.randn(200, 3)
points2 = np.random.randn(200, 3) * 2

# Create individual plots
fig1 = cloudglancer.plot(points1, title="Dataset 1")
fig2 = cloudglancer.plot(points2, title="Dataset 2")

# Combine into a single figure
combined = cloudglancer.combine_plots([fig1, fig2], rows=1, cols=2)
combined.show()
```

## API Reference

### `cloudglancer.plot()`

Create an interactive 3D scatter plot.

**Parameters:**

- `points` (np.ndarray): Array of shape (n_points, 3) containing 3D coordinates.
- `labels` (np.ndarray, optional): Array of labels for color grouping.
- `label_map` (dict, optional): Maps label values to display names. Enables discrete color mapping.
- `color_map` (list or float, optional):
  - With `label_map`: List of color strings for discrete coloring
  - Without `label_map`: Float specifying the continuous color scale midpoint
- `size` (float, optional): Size of scatter plot markers. Default is 1.5.
- `title` (str, optional): Title of the plot.

**Returns:**

- `plotly.graph_objects.Figure`: Interactive 3D scatter plot.

**Raises:**

- `ValueError`: If points array is not of shape (n_points, 3).

### `cloudglancer.combine_plots()`

Combine multiple 3D plots into a single figure with subplots.

**Parameters:**

- `figs` (list): List of Plotly figures to combine.
- `rows` (int, optional): Number of rows in the subplot grid. Default is 1.
- `cols` (int, optional): Number of columns in the subplot grid. Default is 2.

**Returns:**

- `plotly.graph_objects.Figure`: Combined figure with all plots arranged in a grid.

**Raises:**

- `ValueError`: If the number of figures doesn't match rows * cols.

## Examples

See the [examples/](examples/) directory for more detailed usage examples.

Run the examples:

```bash
python examples/basic_usage.py
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black cloudglancer tests examples
ruff check cloudglancer tests examples
```

### Building the Package

```bash
pip install build
python -m build
```

This will create both wheel and source distributions in the `dist/` directory.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
