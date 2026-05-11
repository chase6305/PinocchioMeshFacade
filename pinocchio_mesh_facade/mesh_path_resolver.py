"""MeshPathResolver module.

Responsible for resolving mesh paths in URDF to absolute file system paths.
"""
import os
from typing import List, Optional


class MeshPathResolver:
    """Resolves mesh paths in URDF to absolute file system paths."""

    def __init__(self,
                 urdf_path: str,
                 package_dirs: Optional[List[str]] = None):
        """Initialize the MeshPathResolver.

        :param urdf_path: Path to the URDF file
        :param package_dirs: List of package directories for mesh lookup
        """
        self.urdf_path = os.fspath(urdf_path)
        self.package_dirs = [os.fspath(p) for p in (package_dirs or [])]

    def resolve(self, mesh_path: Optional[str]) -> Optional[str]:
        """Resolve a mesh path (possibly relative or package://) to an absolute file system path.

        :param mesh_path: Mesh path from URDF
        :return: Absolute file path if found, else original or None
        """
        if not mesh_path:
            return None
        mesh_path = os.fspath(mesh_path)
        candidate_paths = []
        if os.path.isabs(mesh_path):
            candidate_paths.append(mesh_path)
        else:
            if mesh_path.startswith("package://"):
                relative_path = mesh_path[len("package://"):]
                candidate_paths.extend(
                    os.path.join(pd, relative_path) for pd in self.package_dirs)
            else:
                candidate_paths.extend(
                    os.path.join(pd, mesh_path) for pd in self.package_dirs)
                candidate_paths.append(
                    os.path.join(os.path.dirname(self.urdf_path), mesh_path))
        for candidate in candidate_paths:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        return os.path.abspath(mesh_path) if os.path.isabs(
            mesh_path) else mesh_path
