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
