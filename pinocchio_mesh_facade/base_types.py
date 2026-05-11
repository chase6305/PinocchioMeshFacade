"""Base type definitions module.

Contains core data structures such as MeshResource and MimicJointRule.
"""
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class MeshResource:
    """Data structure describing a single mesh resource."""
    link_name: str
    geometry_type: str
    mesh_path: str
    geometry_name: Optional[str] = None
    transform: Optional[np.ndarray] = None
    mesh_scale: Optional[np.ndarray] = None
    mesh: Optional[Any] = None


@dataclass(frozen=True)
class MimicJointRule:
    """Data structure describing a mimic joint rule."""
    child_joint: str
    parent_joint: str
    multiplier: float = 1.0
    offset: float = 0.0
