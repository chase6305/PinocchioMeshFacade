# PinocchioMeshFacade
PinocchioMeshFacade is a Python library for extracting and processing robot model meshes using the Pinocchio library. It provides a simple API to extract, process, and export collision and visual meshes from URDF files. The tool automatically manages joint transforms, mesh transforms, and caching, and supports batch export and mesh merging.

## Installation

Recommended installation with pip:

```bash
pip install .
```

Or for development mode:

```bash
pip install -e .
```

## Usage Example

```python
from pinocchio_mesh_facade import PinocchioMeshFacade

# Example: Load URDF and process mesh
facade = PinocchioMeshFacade("your_robot.urdf")
# Further usage of facade methods...
```

For detailed API, please refer to the source code and documentation.
