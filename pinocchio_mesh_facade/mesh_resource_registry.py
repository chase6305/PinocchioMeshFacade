"""MeshResourceRegistry module, responsible for managing mesh resource registration and mapping."""
from typing import Dict, Iterable, List, Optional

import numpy as np

from pinocchio_mesh_facade.base_types import MeshResource
from pinocchio_mesh_facade.kinematics_manager import KinematicsManager
from pinocchio_mesh_facade.mesh_path_resolver import MeshPathResolver
from pinocchio_mesh_facade.pinocchio_robot_model import PinocchioRobotModel

from .logger import get_logger


class MeshResourceRegistry:
    """Mesh resource registry, responsible for managing mesh resource registration and mapping."""

    def __init__(
        self,
        model: PinocchioRobotModel,
        kinematics: KinematicsManager,
        path_resolver: MeshPathResolver,
    ):
        """Initialize the MeshResourceRegistry."""
        self._model = model
        self._kinematics = kinematics
        self._path_resolver = path_resolver
        self.logger = get_logger(self.__class__.__name__)
        self._link_to_visual_resources: Optional[Dict[str, List[MeshResource]]] = None
        self._link_to_collision_resources: Optional[Dict[str, List[MeshResource]]] = None
        self._link_to_visual_paths: Optional[Dict[str, List[str]]] = None
        self._link_to_collision_paths: Optional[Dict[str, List[str]]] = None
        self.build()

    def on_kinematics_updated(self):
        """Clear all cached resources and paths when kinematics are updated."""
        # Clear all cached resources and paths when kinematics are updated
        self._link_to_visual_resources = None
        self._link_to_collision_resources = None
        self._link_to_visual_paths = None
        self._link_to_collision_paths = None

    def build(self):
        """Build resource and path mappings for visual and collision meshes."""
        # Build resource and path mappings for visual and collision meshes
        self._link_to_visual_resources = {}
        self._link_to_collision_resources = {}
        self._link_to_visual_paths = {}
        self._link_to_collision_paths = {}
        for geom_id, geom in enumerate(self._model.visual_model.geometryObjects):
            link_name = self._model.model.frames[geom.parentFrame].name
            mesh_path = self._get_mesh_file_path(geom)
            if mesh_path:
                resource = self._build_resource(link_name, "visual", geom, mesh_path, geom_id)
                self._link_to_visual_resources.setdefault(link_name, []).append(resource)
                self._link_to_visual_paths.setdefault(link_name, []).append(resource.mesh_path)
        for geom_id, geom in enumerate(self._model.collision_model.geometryObjects):
            link_name = self._model.model.frames[geom.parentFrame].name
            mesh_path = self._get_mesh_file_path(geom)
            if mesh_path:
                resource = self._build_resource(link_name, "collision", geom, mesh_path, geom_id)
                self._link_to_collision_resources.setdefault(link_name, []).append(resource)
                self._link_to_collision_paths.setdefault(link_name, []).append(resource.mesh_path)

    def _get_mesh_file_path(self, geom) -> Optional[str]:
        # Get the mesh file path from geometry object
        mesh_path = None
        if hasattr(geom, "meshPath") and geom.meshPath:
            mesh_path = geom.meshPath
        elif hasattr(geom, "filename") and geom.filename:
            mesh_path = geom.filename
        return self._path_resolver.resolve(mesh_path)

    @staticmethod
    def _se3_to_matrix(placement) -> np.ndarray:
        # Convert SE3 placement to 4x4 transformation matrix
        if hasattr(placement, "homogeneous"):
            return np.array(placement.homogeneous, dtype=float)
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(placement.rotation, dtype=float)
        transform[:3, 3] = np.asarray(placement.translation, dtype=float)
        return transform

    def _get_world_transform(self, geom_id: int, geometry_type: str,
                             geom) -> np.ndarray:
        # Get the world transformation matrix for the geometry
        if geometry_type == "visual":
            return self._se3_to_matrix(self._model.visual_data.oMg[geom_id])
        elif geometry_type == "collision":
            return self._se3_to_matrix(self._model.collision_data.oMg[geom_id])
        else:
            frame = self._model.data.oMf[geom.parentFrame]
            world = frame * geom.placement
            return self._se3_to_matrix(world)

    @staticmethod
    def _get_mesh_scale(geom) -> np.ndarray:
        # Get the mesh scale as a 3-element array
        mesh_scale = getattr(geom, "meshScale", None)
        if mesh_scale is None:
            return np.ones(3, dtype=float)
        mesh_scale = np.asarray(mesh_scale, dtype=float).reshape(-1)
        if mesh_scale.size == 1:
            return np.repeat(mesh_scale[0], 3)
        if mesh_scale.size >= 3:
            return mesh_scale[:3]
        return np.ones(3, dtype=float)

    def _build_resource(self, link_name, geometry_type, geom, mesh_path,
                        geom_id) -> MeshResource:
        # Build a MeshResource object for a given geometry
        return MeshResource(
            link_name=link_name,
            geometry_type=geometry_type,
            mesh_path=mesh_path,
            geometry_name=getattr(geom, "name", None),
            transform=self._get_world_transform(geom_id, geometry_type, geom),
            mesh_scale=self._get_mesh_scale(geom),
        )

    def get_resources_for_links(
        self,
        link_names: Iterable[str],
        include_visual: bool = True,
        include_collision: bool = True,
        unique: bool = True,
        reference_frame_id: Optional[int] = None,
    ) -> List[MeshResource]:
        """Get mesh resources for the specified links, optionally filtered and transformed.

        :param link_names: Iterable of link names
        :param include_visual: Include visual meshes
        :param include_collision: Include collision meshes
        :param unique: Only unique resources
        :param reference_frame_id: Reference frame id for transforms
        :return: List of MeshResource
        """
        # Get mesh resources for the specified links, optionally filtered and transformed
        if self._link_to_visual_resources is None or self._link_to_collision_resources is None:
            self.build()
        resources = []
        seen = set()
        for ln in link_names:
            for geom_type, storage in [
                ("visual", self._link_to_visual_resources),
                ("collision", self._link_to_collision_resources)
            ]:
                if (geom_type == "visual" and
                        not include_visual) or (geom_type == "collision" and
                                                not include_collision):
                    continue
                for res in storage.get(ln, []):
                    key = (res.link_name, res.geometry_type, res.geometry_name,
                           res.mesh_path)
                    if unique and key in seen:
                        continue
                    new_res = self._copy_resource(res)
                    if reference_frame_id is not None:
                        ref_pose = self._model.data.oMf[reference_frame_id]
                        inv = self._se3_to_matrix(ref_pose.inverse())
                        if new_res.transform is not None:
                            new_res = MeshResource(
                                link_name=new_res.link_name,
                                geometry_type=new_res.geometry_type,
                                mesh_path=new_res.mesh_path,
                                geometry_name=new_res.geometry_name,
                                transform=inv @ new_res.transform,
                                mesh_scale=new_res.mesh_scale,
                                mesh=new_res.mesh,
                            )
                    resources.append(new_res)
                    seen.add(key)
        return resources

    def get_mesh_paths_for_links(
        self,
        link_names: Iterable[str],
        include_visual: bool = True,
        include_collision: bool = True,
        unique: bool = True,
    ) -> List[str]:
        """Get mesh file paths for the specified links.

        :param link_names: Iterable of link names
        :param include_visual: Include visual meshes
        :param include_collision: Include collision meshes
        :param unique: Only unique paths
        :return: List of file paths
        """
        if self._link_to_visual_paths is None or self._link_to_collision_paths is None:
            self.build()
        # Get mesh file paths for the specified links
        paths = []
        seen = set()

        # Determine which link names have resources based on the filters
        valid_link_names = set()
        if include_visual and self._link_to_visual_paths is not None:
            valid_link_names.update(self._link_to_visual_paths.keys())
        if include_collision and self._link_to_collision_paths is not None:
            valid_link_names.update(self._link_to_collision_paths.keys())

        for ln in link_names:
            if ln not in valid_link_names:
                continue
            for storage, enabled in [
                (self._link_to_visual_paths, include_visual),
                (self._link_to_collision_paths, include_collision),
            ]:
                if not enabled or storage is None:
                    continue
                for p in storage.get(ln, []):
                    if unique and p in seen:
                        continue
                    paths.append(p)
                    seen.add(p)
        return paths

    @staticmethod
    def _copy_resource(res: MeshResource) -> MeshResource:
        # Create a copy of a MeshResource object
        return MeshResource(
            link_name=res.link_name,
            geometry_type=res.geometry_type,
            mesh_path=res.mesh_path,
            geometry_name=res.geometry_name,
            transform=None if res.transform is None else np.array(res.transform,
                                                                  dtype=float),
            mesh_scale=None if res.mesh_scale is None else np.array(
                res.mesh_scale, dtype=float),
            mesh=res.mesh,
        )
