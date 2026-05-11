"""Robot link tree structure utility module.

Provides LinkTreeGraph class for traversing and querying robot link trees.
"""
from collections import deque
from typing import List, Optional

import pinocchio as pin

from pinocchio_mesh_facade.logger import get_logger
from pinocchio_mesh_facade.pinocchio_robot_model import PinocchioRobotModel


class LinkTreeGraph:
    """Utility for traversing and querying robot link tree structures."""

    def __init__(self, model: PinocchioRobotModel):
        """Initialize the LinkTreeGraph with a PinocchioRobotModel."""
        self._model = model
        self.logger = get_logger(self.__class__.__name__)

    def get_all_link_names(self) -> List[str]:
        """Return all link names in the model."""
        return [
            frame.name
            for frame in self._model.model.frames
            if hasattr(frame, 'type') and
            getattr(frame, 'type', None) == pin.FrameType.BODY
        ]

    def get_link_name_from_joint(self, joint_name: str) -> Optional[str]:
        """Return the link name associated with a given joint name, or None if not found."""
        try:
            joint_id = self._model.get_joint_id(joint_name)
            if joint_id > 0:
                for frame in self._model.model.frames:
                    if frame.parentJoint == joint_id and frame.type == pin.FrameType.BODY:
                        return frame.name
        except Exception:
            pass
        return None

    def get_frame_id(self, link_name: str) -> Optional[int]:
        """Return the frame id for a given link name, or None if not found."""
        try:
            return self._model.get_frame_id(link_name)
        except Exception:
            return None

    def get_all_child_links_ordered(self, root_link_name: str) -> List[str]:
        """Return all child link names (BFS order) starting from the given root link name."""
        child_links = [root_link_name]
        seen = {root_link_name}
        queue = deque([root_link_name])
        while queue:
            current = queue.popleft()
            try:
                current_id = self._model.get_frame_id(current)
            except Exception:
                continue
            for frame in self._model.model.frames:
                if frame.parentFrame == current_id and frame.name != current:
                    if frame.name not in seen:
                        seen.add(frame.name)
                        child_links.append(frame.name)
                        queue.append(frame.name)
        return child_links

    def get_path_between_links_ordered(self, link_a: str,
                                       link_b: str) -> List[str]:
        """Return the path (as link names) between two links in the tree.

        Returns an empty list if not connected.
        """
        id_a = self.get_frame_id(link_a)
        id_b = self.get_frame_id(link_b)
        if id_a is None or id_b is None:
            return []

        def path_to_root(frame_id):
            path = []
            visited = set()
            current = frame_id
            while current is not None and current not in visited:
                visited.add(current)
                path.append(current)
                parent = self._model.model.frames[current].parentFrame
                if parent == current:
                    break
                current = parent
            return path

        path_a = path_to_root(id_a)
        path_b = path_to_root(id_b)
        path_a_set = set(path_a)
        lca = None
        for fid in path_b:
            if fid in path_a_set:
                lca = fid
                break
        if lca is None:
            return []
        up_a = path_a[:path_a.index(lca) + 1]
        down_b = list(reversed(path_b[:path_b.index(lca)]))
        path_ids = up_a + down_b
        return [self._model.model.frames[fid].name for fid in path_ids]

    def get_child_links_or_path(self,
                                link_name: str,
                                other_link_name: Optional[str] = None,
                                include_self: bool = True) -> List[str]:
        """Return child links or the path between two links.

        Optionally includes the root link itself.
        """
        if self.get_frame_id(link_name) is None:
            return []
        if other_link_name:
            links = self.get_path_between_links_ordered(link_name,
                                                        other_link_name)
            if not links:
                return []
            return links
        links = self.get_all_child_links_ordered(link_name)
        if not include_self:
            links = [n for n in links if n != link_name]
        return links
