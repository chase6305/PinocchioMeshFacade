"""MimicJointResolver module.

Parses and applies mimic joint rules from URDF.
"""
import xml.etree.ElementTree as ET
from typing import Dict, Optional

import numpy as np

from pinocchio_mesh_facade.base_types import MimicJointRule
from pinocchio_mesh_facade.pinocchio_robot_model import PinocchioRobotModel


class MimicJointResolver:
    """Parse mimic joint rules from URDF and complete configuration vectors."""

    def __init__(self,
                 urdf_path: str,
                 model: PinocchioRobotModel,
                 verbose: bool = False):
        """Initialize the MimicJointResolver.

        :param urdf_path: Path to the URDF file
        :param model: PinocchioRobotModel instance
        :param verbose: Enable verbose logging
        """
        self._rules: Dict[str, MimicJointRule] = {}
        self._model = model
        self._verbose = verbose
        self._parse_urdf(urdf_path)

    def _parse_urdf(self, urdf_path: str):
        try:
            root = ET.parse(urdf_path).getroot()
        except Exception as e:
            if self._verbose:
                print(
                    f"Warning: Failed to parse URDF mimic relationships - {str(e)}"
                )
            return
        for joint_elem in root.findall("joint"):
            child_joint = joint_elem.get("name")
            mimic_elem = joint_elem.find("mimic")
            if not child_joint or mimic_elem is None:
                continue
            parent_joint = mimic_elem.get("joint")
            if not parent_joint:
                continue
            child_info = self._get_joint_q_info(child_joint)
            parent_info = self._get_joint_q_info(parent_joint)
            if child_info is None or parent_info is None:
                if self._verbose:
                    print(
                        f"Warning: mimic joint '{child_joint}' or '{parent_joint}' "
                        "is not a single-DoF joint, skipped"
                    )
                continue
            multiplier = float(mimic_elem.get("multiplier", "1.0"))
            offset = float(mimic_elem.get("offset", "0.0"))
            self._rules[child_joint] = MimicJointRule(
                child_joint=child_joint,
                parent_joint=parent_joint,
                multiplier=multiplier,
                offset=offset,
            )
        if self._verbose and self._rules:
            print(
                f"Automatically established {len(self._rules)} mimic joint relationships"
            )

    def _get_joint_q_info(self, joint_name: str) -> Optional[tuple]:
        try:
            joint_id = self._model.get_joint_id(joint_name)
        except Exception:
            return None
        if joint_id <= 0 or self._model.model.nqs[joint_id] != 1:
            return None
        return joint_id, self._model.model.idx_qs[joint_id]

    @property
    def rules(self) -> Dict[str, MimicJointRule]:
        """Return a copy of the mimic joint rules dictionary."""
        return dict(self._rules)

    def apply(self, q: np.ndarray) -> np.ndarray:
        """Apply mimic joint rules to a configuration vector.

        :param q: Input configuration vector
        :return: Configuration vector with mimic rules applied
        """
        if not self._rules:
            return np.asarray(q, dtype=float).reshape(-1).copy()
        updated_q = np.asarray(q, dtype=float).reshape(-1).copy()
        resolved = set()
        visiting = set()

        def _resolve(child: str):
            if child in resolved:
                return
            if child in visiting:
                raise ValueError(f"Detected mimic circular dependency: {child}")
            rule = self._rules[child]
            visiting.add(child)
            if rule.parent_joint in self._rules:
                _resolve(rule.parent_joint)
            child_info = self._get_joint_q_info(rule.child_joint)
            parent_info = self._get_joint_q_info(rule.parent_joint)
            if child_info is None or parent_info is None:
                visiting.remove(child)
                resolved.add(child)
                return
            _, child_idx = child_info
            _, parent_idx = parent_info
            updated_q[child_idx] = rule.multiplier * updated_q[
                parent_idx] + rule.offset
            visiting.remove(child)
            resolved.add(child)

        for child in self._rules:
            _resolve(child)
        return updated_q
