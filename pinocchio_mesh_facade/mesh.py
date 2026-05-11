"""Mesh abstract base class module.

Defines the Mesh class and its related methods.
"""
from abc import ABC
from typing import Optional

import numpy as np
import trimesh

from pinocchio_mesh_facade.logger import get_logger


class Mesh(ABC):
    """Abstract Mesh class defining basic properties and methods for meshes."""

    def __init__(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
        pose: Optional[np.ndarray] = None,
        vertex_normals: Optional[np.ndarray] = None,
        triangle_normals: Optional[np.ndarray] = None,
    ):
        """Initialize a Mesh object.

        :param vertices: Array of vertices.
        :param triangles: Array of triangle faces.
        :param pose: Pose (optional).
        :param vertex_normals: Vertex normals (optional).
        :param triangle_normals: Face normals (optional).
        """
        super().__init__()
        self.logger = get_logger(self.__class__.__name__)
        self.vertices = vertices
        self.triangles = triangles
        self.pose = pose
        self.vertex_normals = vertex_normals
        self.triangle_normals = triangle_normals

        if self.vertex_normals is None or self.triangle_normals is None:
            try:
                mesh = trimesh.Trimesh(vertices=self.vertices,
                                       faces=self.triangles,
                                       process=False)
                if self.vertex_normals is None:
                    self.vertex_normals = mesh.vertex_normals
                if self.triangle_normals is None:
                    self.triangle_normals = mesh.face_normals
            except ImportError:
                pass

    @staticmethod
    def init_from_file(file_name: str, pose: Optional[np.ndarray] = None):
        """Initialize a Mesh object from a file.

        :param file_name: Path to the mesh file.
        :param pose: Pose (optional).
        :return: Mesh object.
        """
        mesh = trimesh.load_mesh(file_name, process=False)
        return Mesh(
            vertices=mesh.vertices,
            triangles=mesh.faces,
            pose=pose,
            vertex_normals=getattr(mesh, "vertex_normals", None),
            triangle_normals=getattr(mesh, "face_normals", None),
        )
