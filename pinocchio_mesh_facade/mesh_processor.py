"""MeshProcessor module, responsible for mesh processing, merging, and exporting operations."""
import os
from typing import Iterable, List, Optional

import numpy as np
import trimesh

from pinocchio_mesh_facade.base_types import MeshResource
from pinocchio_mesh_facade.kinematics_manager import KinematicsManager
from pinocchio_mesh_facade.link_tree_graph import LinkTreeGraph
from pinocchio_mesh_facade.mesh_loader import MeshLoader
from pinocchio_mesh_facade.mesh_resource_registry import MeshResourceRegistry
from pinocchio_mesh_facade.pinocchio_robot_model import PinocchioRobotModel

from .logger import get_logger


class MeshProcessor:
    """Mesh processor, responsible for mesh processing, merging, and exporting operations."""

    def __init__(
        self,
        registry: MeshResourceRegistry,
        loader: MeshLoader,
        tree: LinkTreeGraph,
        kinematics: KinematicsManager,
        model: PinocchioRobotModel,
    ):
        """Initialize the MeshProcessor.

        :param registry: MeshResourceRegistry instance
        :param loader: MeshLoader instance
        :param tree: LinkTreeGraph instance
        :param kinematics: KinematicsManager instance
        :param model: PinocchioRobotModel instance
        """
        self._registry = registry
        self._loader = loader
        self._tree = tree
        self._kinematics = kinematics
        self._model = model
        self.logger = get_logger(self.__class__.__name__)

    def get_meshes_with_cache(
        self,
        link_name: str,
        other_link_name: Optional[str] = None,
        include_self: bool = True,
        include_visual: bool = True,
        include_collision: bool = True,
        unique: bool = True,
        reference_link: Optional[str] = None,
    ) -> List[MeshResource]:
        """Get mesh resources for the specified links, loading meshes into cache.

        :param link_name: Main link name
        :param other_link_name: Optional other link name
        :param include_self: Whether to include the main link
        :param include_visual: Include visual meshes
        :param include_collision: Include collision meshes
        :param unique: Only unique resources
        :param reference_link: Reference link for transforms
        :return: List of MeshResource
        """
        links = self._tree.get_child_links_or_path(link_name, other_link_name,
                               include_self)
        if not links:
            return []
        ref_frame_id = None
        if reference_link is not None:
            ref_frame_id = self._model.get_frame_id(reference_link)
            if ref_frame_id is None:
                raise ValueError(f"reference_link '{reference_link}' not found")
        resources = self._registry.get_resources_for_links(
            links,
            include_visual=include_visual,
            include_collision=include_collision,
            unique=unique,
            reference_frame_id=ref_frame_id,
        )
        result = []
        for res in resources:
            loaded_mesh = self._loader.load_mesh(res.mesh_path)
            result.append(
                MeshResource(
                    link_name=res.link_name,
                    geometry_type=res.geometry_type,
                    mesh_path=res.mesh_path,
                    geometry_name=res.geometry_name,
                    transform=res.transform,
                    mesh_scale=res.mesh_scale,
                    mesh=loaded_mesh,
                ))
        return result

    def get_merged_trimesh(
        self,
        link_name: str,
        other_link_name: Optional[str] = None,
        include_self: bool = True,
        include_visual: bool = True,
        include_collision: bool = True,
        unique: bool = True,
    ) -> trimesh.Trimesh:
        """Get a merged trimesh.Trimesh object for the specified links.

        :param link_name: Main link name
        :param other_link_name: Optional other link name
        :param include_self: Whether to include the main link
        :param include_visual: Include visual meshes
        :param include_collision: Include collision meshes
        :param unique: Only unique resources
        :return: Merged trimesh.Trimesh object
        """
        resources = self._registry.get_resources_for_links(
            self._tree.get_child_links_or_path(link_name, other_link_name,
                                               include_self),
            include_visual=include_visual,
            include_collision=include_collision,
            unique=unique,
        )
        if not resources:
            raise ValueError(f"No mesh resource found for link '{link_name}'")
        trimesh_list = []
        for res in resources:
            # Prefer using MeshLoader's cache to avoid repeated disk loading
            tm = self._loader.load_trimesh(res.mesh_path).copy()
            scale = res.mesh_scale
            if scale is not None and not np.allclose(scale, 1.0):
                s = np.eye(4)
                s[0, 0], s[1, 1], s[2, 2] = scale[0], scale[1], scale[2]
                tm.apply_transform(s)
            if res.transform is not None:
                tm.apply_transform(res.transform)
            trimesh_list.append(tm)
        return trimesh.util.concatenate(trimesh_list)

    def export_merged_mesh(
        self,
        link_name: str,
        output_path: str,
        other_link_name: Optional[str] = None,
        include_self: bool = True,
        include_visual: bool = True,
        include_collision: bool = True,
        unique: bool = True,
        file_type: Optional[str] = None,
        reference_link: Optional[str] = None,
    ) -> str:
        """Export a merged mesh for the specified links to a file.

        :param link_name: Main link name
        :param output_path: Output file path
        :param other_link_name: Optional other link name
        :param include_self: Whether to include the main link
        :param include_visual: Include visual meshes
        :param include_collision: Include collision meshes
        :param unique: Only unique resources
        :param file_type: Output file type
        :param reference_link: Reference link for transforms
        :return: Output file path
        """
        output_path = os.fspath(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if reference_link is not None:
            resources = self.get_meshes_with_cache(
                link_name,
                other_link_name,
                include_self,
                include_visual,
                include_collision,
                unique,
                reference_link=reference_link,
            )
            trimesh_list = []
            for res in resources:
                tm = self._loader.load_trimesh(res.mesh_path).copy()
                scale = res.mesh_scale
                if scale is not None and not np.allclose(scale, 1.0):
                    s = np.eye(4)
                    s[0, 0], s[1, 1], s[2, 2] = scale[0], scale[1], scale[2]
                    tm.apply_transform(s)
                if res.transform is not None:
                    tm.apply_transform(res.transform)
                trimesh_list.append(tm)
            if not trimesh_list:
                raise ValueError("No mesh resource found to merge")
            merged = trimesh.util.concatenate(trimesh_list)
        else:
            merged = self.get_merged_trimesh(link_name, other_link_name,
                                             include_self, include_visual,
                                             include_collision, unique)
        export_kwargs = {}
        if file_type is not None:
            export_kwargs["file_type"] = file_type
        merged.export(output_path, **export_kwargs)
        return os.path.abspath(output_path)

    def create_validation_scene(
        self,
        link_name: str,
        other_link_name: Optional[str] = None,
        include_self: bool = True,
        include_visual: bool = True,
        include_collision: bool = True,
        unique: bool = True,
        merged_translation: Iterable[float] = (0.5, 0.0, 0.0),
    ) -> trimesh.Scene:
        """Create a validation scene with original and merged meshes for visual inspection.

        :param link_name: Main link name
        :param other_link_name: Optional other link name
        :param include_self: Whether to include the main link
        :param include_visual: Include visual meshes
        :param include_collision: Include collision meshes
        :param unique: Only unique resources
        :param merged_translation: Offset for merged mesh
        :return: trimesh.Scene object
        """
        resources = self._registry.get_resources_for_links(
            self._tree.get_child_links_or_path(link_name, other_link_name,
                                               include_self),
            include_visual=include_visual,
            include_collision=include_collision,
            unique=unique,
        )
        if not resources:
            raise ValueError(f"No mesh resource found for link '{link_name}'")
        scene = trimesh.Scene()
        original_color = np.array([70, 130, 180, 180], dtype=np.uint8)
        merged_color = np.array([220, 20, 60, 220], dtype=np.uint8)
        for idx, res in enumerate(resources):
            tm = self._loader.load_trimesh(res.mesh_path).copy()
            scale = res.mesh_scale
            if scale is not None and not np.allclose(scale, 1.0):
                s = np.eye(4)
                s[0, 0], s[1, 1], s[2, 2] = scale[0], scale[1], scale[2]
                tm.apply_transform(s)
            if res.transform is not None:
                tm.apply_transform(res.transform)
            tm.visual.face_colors = original_color
            scene.add_geometry(
                tm,
                node_name=f"original_{idx}_{res.geometry_name or res.link_name}"
            )
        merged = self.get_merged_trimesh(link_name, other_link_name,
                                         include_self, include_visual,
                                         include_collision, unique)
        merged.visual.face_colors = merged_color
        offset = np.eye(4)
        offset[:3, 3] = np.asarray(merged_translation, dtype=float).reshape(3)
        merged.apply_transform(offset)
        scene.add_geometry(merged, node_name="merged_mesh")
        return scene

    def export_validation_scene(
        self,
        link_name: str,
        output_path: str,
        other_link_name: Optional[str] = None,
        include_self: bool = True,
        include_visual: bool = True,
        include_collision: bool = True,
        unique: bool = True,
        merged_translation: Iterable[float] = (0.5, 0.0, 0.0),
    ) -> str:
        """Export a validation scene to a file.

        :param link_name: Main link name
        :param output_path: Output file path
        :param other_link_name: Optional other link name
        :param include_self: Whether to include the main link
        :param include_visual: Include visual meshes
        :param include_collision: Include collision meshes
        :param unique: Only unique resources
        :param merged_translation: Offset for merged mesh
        :return: Output file path
        """
        output_path = os.fspath(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        scene = self.create_validation_scene(
            link_name,
            other_link_name,
            include_self,
            include_visual,
            include_collision,
            unique,
            merged_translation,
        )
        scene.export(output_path)
        return os.path.abspath(output_path)
