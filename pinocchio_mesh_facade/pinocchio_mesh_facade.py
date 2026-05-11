
"""Main module for PinocchioMeshFacade, providing the PinocchioMeshFacade class and.

encapsulating all mesh and kinematics-related functionality.
"""
import os
from typing import Dict, Iterable, List, Optional

import numpy as np

from pinocchio_mesh_facade.base_types import MeshResource
from pinocchio_mesh_facade.kinematics_manager import KinematicsManager
from pinocchio_mesh_facade.link_tree_graph import LinkTreeGraph
from pinocchio_mesh_facade.mesh import Mesh
from pinocchio_mesh_facade.mesh_loader import MeshLoader
from pinocchio_mesh_facade.mesh_path_resolver import MeshPathResolver
from pinocchio_mesh_facade.mesh_processor import MeshProcessor
from pinocchio_mesh_facade.mesh_resource_registry import MeshResourceRegistry
from pinocchio_mesh_facade.mimic_joint_resolver import MimicJointResolver
from pinocchio_mesh_facade.pinocchio_robot_model import PinocchioRobotModel


class PinocchioMeshFacade:
    """PinocchioMeshFacade main class, encapsulates all mesh and kinematics related functions."""

    def __init__(self,
                 urdf_path: str,
                 package_dirs: Optional[List[str]] = None,
                 verbose: bool = True):
        """Initialize the PinocchioMeshFacade main class, encapsulating all mesh and kinematics.

        Related functions.
        """
        self.verbose = verbose
        urdf_path = os.fspath(urdf_path)
        package_dirs = [os.fspath(p) for p in (package_dirs or [])]
        self.path_resolver = MeshPathResolver(urdf_path, package_dirs)
        self.model = PinocchioRobotModel(urdf_path, package_dirs)
        self.mimic = MimicJointResolver(urdf_path, self.model, verbose)
        self.kinematics = KinematicsManager(self.model, self.mimic, verbose)
        self.tree = LinkTreeGraph(self.model)
        self.registry = MeshResourceRegistry(self.model, self.kinematics,
                                             self.path_resolver)
        self.loader = MeshLoader(self.path_resolver)
        self.processor = MeshProcessor(self.registry, self.loader, self.tree,
                                       self.kinematics, self.model)
        self.kinematics.add_update_listener(self.registry)

        self.warmup_all_meshes(include_visual=False,
                               include_collision=True,
                               unique=True)

    def warmup_all_meshes(self,
                          include_visual: bool = True,
                          include_collision: bool = True,
                          unique: bool = True):
        """Warm up all mesh files into cache to avoid I/O delay on first access.

        :param include_visual: Include visual meshes
        :param include_collision: Include collision meshes
        :param unique: Only unique paths
        """
        """Warm up all mesh files into cache to avoid I/O delay on first access."""
        # Get all link names
        all_links = list(self.tree.get_all_link_names())
        # Get all mesh paths
        mesh_paths = self.registry.get_mesh_paths_for_links(
            all_links,
            include_visual=include_visual,
            include_collision=include_collision,
            unique=unique)
        # Batch load into cache
        for path in mesh_paths:
            try:
                self.loader.load_trimesh(path)
            except Exception as e:
                if self.verbose:
                    print(f"[warmup] Failed to load {path}: {e}")
        if self.verbose:
            print(f"[warmup] Warmed up {len(mesh_paths)} mesh files")

    def update_configuration(self, q: Iterable[float]) -> np.ndarray:
        """Update the robot configuration vector and perform forward kinematics.

        :param q: New configuration vector
        :return: Updated configuration vector
        """
        return self.kinematics.update_configuration(q)

    def update_joint_angle(
        self,
        joint_angles: Dict[str, float],
        base_configuration: Optional[Iterable[float]] = None
    ) -> np.ndarray:
        """Update specific joint angles (single-DoF only) and optionally the base configuration.

        :param joint_angles: Mapping from joint names to angles
        :param base_configuration: Optional base configuration (length 7)
        :return: Updated configuration vector
        """
        return self.kinematics.update_joint_angle(joint_angles, base_configuration)

    def get_current_configuration(self) -> np.ndarray:
        """Get the current robot configuration vector.

        :return: Current configuration vector
        """
        return self.kinematics.current_configuration

    def get_all_collision_oobbs(self) -> List[dict]:
        """Get the OOBB (oriented bounding box) for the collision mesh of all links.

        Returns a list of dicts, each containing:
            'link_name': str,
            'oobb': trimesh.primitives.Box,
            'transform': np.ndarray
        """
        resources = self.registry.get_resources_for_links(
            self.tree.get_all_link_names(),
            include_visual=False,
            include_collision=True)
        result = []
        for res in resources:
            mesh = self.loader.load_trimesh(res.mesh_path).copy()
            if res.mesh_scale is not None and not np.allclose(
                    res.mesh_scale, 1.0):
                s = np.eye(4)
                s[0, 0], s[1, 1], s[
                    2,
                    2] = res.mesh_scale[0], res.mesh_scale[1], res.mesh_scale[2]
                mesh.apply_transform(s)
            if res.transform is not None:
                mesh.apply_transform(res.transform)
            oobb = mesh.bounding_box_oriented
            result.append({
                'link_name': res.link_name,
                'oobb': oobb,
                'transform': res.transform
            })
        return result

    def get_mesh_paths_by_link(
        self,
        link_name: str,
        other_link_name: Optional[str] = None,
        include_self: bool = True,
        include_visual: bool = True,
        include_collision: bool = True,
        unique: bool = True,
    ) -> List[str]:
        """Get mesh file paths for a specific link and its children.

        :param link_name: Main link name
        :param other_link_name: Optional other link name
        :param include_self: Whether to include the main link
        :param include_visual: Include visual meshes
        :param include_collision: Include collision meshes
        :param unique: Only unique paths
        :return: List of file paths
        """
        links = self.tree.get_child_links_or_path(link_name, other_link_name, include_self)
        if not links:
            return []
        return self.registry.get_mesh_paths_for_links(
            links, include_visual, include_collision, unique
        )

    def get_mesh_paths_by_joints(
        self,
        joint_names: List[str],
        include_self: bool = True,
        include_visual: bool = True,
        include_collision: bool = True,
        unique: bool = True,
    ) -> List[List[str]]:
        """Get mesh file paths for a list of joints.

        :param joint_names: List of joint names
        :param include_self: Whether to include the main link
        :param include_visual: Include visual meshes
        :param include_collision: Include collision meshes
        :param unique: Only unique paths
        :return: List of file path lists
        """
        result = []
        for jn in joint_names:
            ln = self.tree.get_link_name_from_joint(jn)
            if ln is None:
                result.append([])
                continue
            links = [ln] if include_self else []
            result.append(
                self.registry.get_mesh_paths_for_links(
                    links, include_visual, include_collision, unique
                )
            )
        return result

    def get_meshes_by_link(
        self,
        link_name: str,
        other_link_name: Optional[str] = None,
        include_self: bool = True,
        include_visual: bool = True,
        include_collision: bool = True,
        unique: bool = True,
        merge: bool = False,
        reference_link: Optional[str] = None,
    ) -> List[MeshResource]:
        """Get mesh resources for a specific link and its children.

        :param link_name: Main link name
        :param other_link_name: Optional other link name
        :param include_self: Whether to include the main link
        :param include_visual: Include visual meshes
        :param include_collision: Include collision meshes
        :param unique: Only unique resources
        :param merge: Whether to merge meshes
        :param reference_link: Reference link for transforms
        :return: List of MeshResource
        """
        if merge:
            merged = self.processor.get_merged_trimesh(
                link_name, other_link_name, include_self, include_visual,
                include_collision, unique)
            if reference_link is not None:
                pass
            mesh_obj = Mesh(vertices=np.asarray(merged.vertices, dtype=float),
                            triangles=np.asarray(merged.faces, dtype=np.int64))
            return [
                MeshResource(
                    link_name=link_name,
                    geometry_type="merged",
                    mesh_path=f"merged://{link_name}",
                    geometry_name=f"merged_{link_name}",
                    transform=np.eye(4),
                    mesh_scale=np.ones(3),
                    mesh=mesh_obj,
                )
            ]
        return self.processor.get_meshes_with_cache(
            link_name,
            other_link_name,
            include_self,
            include_visual,
            include_collision,
            unique,
            reference_link,
        )

    def get_meshes_by_joints(
        self,
        joint_names: List[str],
        include_self: bool = True,
        include_visual: bool = True,
        include_collision: bool = True,
        unique: bool = True,
    ) -> List[List[MeshResource]]:
        """Get mesh resources for a list of joints.

        :param joint_names: List of joint names
        :param include_self: Whether to include the main link
        :param include_visual: Include visual meshes
        :param include_collision: Include collision meshes
        :param unique: Only unique resources
        :return: List of MeshResource lists
        """
        result = []
        for jn in joint_names:
            ln = self.tree.get_link_name_from_joint(jn)
            if ln is None:
                result.append([])
                continue
            links = [ln] if include_self else []
            resources = self.registry.get_resources_for_links(
                links,
                include_visual=include_visual,
                include_collision=include_collision,
                unique=unique,
            )
            for i, res in enumerate(resources):
                resources[i] = MeshResource(
                    link_name=res.link_name,
                    geometry_type=res.geometry_type,
                    mesh_path=res.mesh_path,
                    geometry_name=res.geometry_name,
                    transform=res.transform,
                    mesh_scale=res.mesh_scale,
                    mesh=self.loader.load_mesh(res.mesh_path),
                )
            result.append(resources)
        return result

    def export_merged_mesh_by_link(self, link_name: str, output_path: str, **kwargs):
        """Export a merged mesh for the specified link to a file."""
        return self.processor.export_merged_mesh(link_name, output_path, **kwargs)

    def export_validation_scene_by_link(self, link_name: str, output_path: str, **kwargs):
        """Export a validation scene for the specified link to a file."""
        return self.processor.export_validation_scene(link_name, output_path, **kwargs)

    def get_merged_trimesh_by_link(self, link_name: str, **kwargs):
        """Get a merged trimesh.Trimesh object for the specified link."""
        return self.processor.get_merged_trimesh(link_name, **kwargs)

    def create_validation_scene_by_link(self, link_name: str, **kwargs):
        """Create a validation scene for the specified link."""
        return self.processor.create_validation_scene(link_name, **kwargs)
