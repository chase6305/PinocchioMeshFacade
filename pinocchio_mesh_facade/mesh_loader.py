"""MeshLoader module.

Responsible for loading mesh files and managing mesh/trimesh object caches.
"""
import os
from typing import Dict

import trimesh

from pinocchio_mesh_facade.logger import get_logger
from pinocchio_mesh_facade.mesh import Mesh
from pinocchio_mesh_facade.mesh_path_resolver import MeshPathResolver


class MeshLoader:
    """Loads mesh files and manages mesh/trimesh object caches for efficient reuse."""

    def __init__(self, path_resolver: MeshPathResolver):
        """Initialize the MeshLoader with a MeshPathResolver."""
        self._path_resolver = path_resolver
        self.logger = get_logger(self.__class__.__name__)
        self._mesh_cache: Dict[str, Mesh] = {}
        self._trimesh_cache: Dict[str, trimesh.Trimesh] = {}

    def load_mesh(self, mesh_path: str) -> Mesh:
        """Load a Mesh object from the given mesh file path, using cache if available."""
        resolved = self._path_resolver.resolve(mesh_path)
        if not resolved or not os.path.exists(resolved):
            raise FileNotFoundError(f"Mesh file does not exist: {mesh_path}")
        if resolved not in self._mesh_cache:
            self._mesh_cache[resolved] = Mesh.init_from_file(resolved)
        return self._mesh_cache[resolved]

    def load_trimesh(self, mesh_path: str) -> trimesh.Trimesh:
        """Load a trimesh.Trimesh object from the given mesh file path, using cache if available.

        :param mesh_path: Path to the mesh file
        :return: trimesh.Trimesh object
        """
        resolved = self._path_resolver.resolve(mesh_path)
        if not resolved or not os.path.exists(resolved):
            raise FileNotFoundError(f"Mesh file does not exist: {mesh_path}")
        if resolved not in self._trimesh_cache:
            loaded = trimesh.load(resolved, process=False)
            if isinstance(loaded, trimesh.Scene):
                geoms = [
                    g.copy()
                    for g in loaded.geometry.values()
                    if isinstance(g, trimesh.Trimesh)
                ]
                if not geoms:
                    raise RuntimeError(
                        f"No triangle mesh found in mesh file: {resolved}")
                loaded = trimesh.util.concatenate(geoms)
            elif not isinstance(loaded, trimesh.Trimesh):
                raise RuntimeError(f"Unsupported mesh type: {type(loaded)}")
            self._trimesh_cache[resolved] = loaded
        return self._trimesh_cache[resolved]
