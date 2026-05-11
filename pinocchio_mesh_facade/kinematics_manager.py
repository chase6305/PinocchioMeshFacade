"""Kinematics manager module.

Provides KinematicsManager class for managing robot kinematics,
configuration updates, and joint angle modifications.
"""
from typing import Dict, Iterable, Optional

import numpy as np

from pinocchio_mesh_facade.mimic_joint_resolver import MimicJointResolver
from pinocchio_mesh_facade.pinocchio_robot_model import PinocchioRobotModel

from .logger import get_logger


class KinematicsManager:
    """Manages robot kinematics, configuration updates, and joint angle modifications.

    Handles forward kinematics and notifies listeners on updates.
    """

    def __init__(
        self,
        model: PinocchioRobotModel,
        mimic: MimicJointResolver,
        verbose: bool = False,
    ):
        """Initialize the KinematicsManager.

        :param model: PinocchioRobotModel instance
        :param mimic: MimicJointResolver instance
        :param verbose: Enable verbose logging
        """
        self._model = model
        self._mimic = mimic
        self._verbose = verbose
        self.logger = get_logger(self.__class__.__name__)
        self._current_q = mimic.apply(model.neutral_configuration().copy())
        self._forward_all(self._current_q)
        self._listeners = []

    def add_update_listener(self, listener):
        """Add a listener to be notified when kinematics are updated.

        :param listener: Listener object with on_kinematics_updated method
        """
        self._listeners.append(listener)

    def _forward_all(self, q: np.ndarray):
        """Perform forward kinematics and update all frame and geometry placements.

        :param q: Configuration vector
        """
        import pinocchio as pin
        pin.forwardKinematics(self._model.model, self._model.data, q)
        pin.updateFramePlacements(self._model.model, self._model.data)
        pin.updateGeometryPlacements(self._model.model, self._model.data,
                                     self._model.collision_model,
                                     self._model.collision_data)
        pin.updateGeometryPlacements(self._model.model, self._model.data,
                                     self._model.visual_model,
                                     self._model.visual_data)

    def _notify_listeners(self):
        """Notify all registered listeners that kinematics have been updated."""
        for listener in self._listeners:
            listener.on_kinematics_updated()

    @property
    def current_configuration(self) -> np.ndarray:
        """Get the current robot configuration vector.

        :return: Current configuration vector
        """
        return self._current_q.copy()

    def update_configuration(self, q: Iterable[float]) -> np.ndarray:
        """Update the robot configuration vector and perform forward kinematics.

        Notifies listeners after update.
        :param q: New configuration vector
        :return: Updated configuration vector
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != self._model.nq:
            raise ValueError(
                f"Configuration vector length mismatch: got {q.shape[0]}, "
                f"expected {self._model.nq}"
            )
        self._current_q = self._mimic.apply(q.copy())
        self._forward_all(self._current_q)
        self._notify_listeners()
        if self._verbose:
            self.logger.debug(f"Current configuration: {self._current_q}")
        return self.current_configuration

    def update_joint_angle(
        self,
        joint_angles: Dict[str, float],
        base_configuration: Optional[Iterable[float]] = None,
    ) -> np.ndarray:
        """Update specific joint angles (single-DoF only) and optionally the base configuration.

        Notifies listeners after update.
        :param joint_angles: Mapping from joint names to angles
        :param base_configuration: Optional base configuration (length 7)
        :return: Updated configuration vector
        """
        updated_q = self._current_q.copy()
        if base_configuration is not None:
            base_configuration = np.asarray(base_configuration,
                                            dtype=float).reshape(-1)
            if base_configuration.shape[0] != 7:
                raise ValueError("base_configuration length must be 7")
            if self._model.nq < 7:
                raise ValueError(
                    "Current model does not contain a free-flyer base configuration"
                )
            updated_q[:7] = base_configuration
        for joint_name, joint_angle in joint_angles.items():
            if joint_name in self._mimic.rules:
                rule = self._mimic.rules[joint_name]
                raise ValueError(
                    f"Joint '{joint_name}' is a mimic joint, controlled by '{rule.parent_joint}'"
                )
            joint_id = self._model.get_joint_id(joint_name)
            if joint_id <= 0:
                raise ValueError(f"Joint '{joint_name}' not found")
            joint_nq = self._model.model.nqs[joint_id]
            if joint_nq != 1:
                raise ValueError(
                    f"Joint '{joint_name}' is not a single-DoF joint (nq={joint_nq}), "
                    "please use update_configuration()"
                )
            joint_idx = self._model.model.idx_qs[joint_id]
            updated_q[joint_idx] = float(joint_angle)
        self._current_q = self._mimic.apply(updated_q)
        self._forward_all(self._current_q)
        self._notify_listeners()
        if self._verbose:
            self.logger.info(f"Updated {len(joint_angles)} joint angles")
        return self.current_configuration
