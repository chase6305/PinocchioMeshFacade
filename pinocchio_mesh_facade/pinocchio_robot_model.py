
"""PinocchioRobotModel module, encapsulates Pinocchio model and related data structures."""
from typing import List, Optional

import numpy as np
import pinocchio as pin


class PinocchioRobotModel:
    """Encapsulates Pinocchio model, collision/visual geometry models, and their data."""

    def __init__(
        self,
        urdf_path: str,
        package_dirs: Optional[List[str]] = None,
        root_joint: Optional[pin.JointModel] = None,
    ):
        """Initialize the PinocchioRobotModel with URDF和可选的package目录与root joint."""
        if root_joint is None:
            root_joint = pin.JointModelFreeFlyer()
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, package_dirs, root_joint
        )
        self.data = self.model.createData()
        self.collision_data = self.collision_model.createData()
        self.visual_data = self.visual_model.createData()

    @property
    def nq(self) -> int:
        """Return the dimension of the configuration vector (q)."""
        return self.model.nq

    @property
    def njoints(self) -> int:
        """Return the number of joints in the model."""
        return self.model.njoints

    def get_joint_id(self, name: str) -> int:
        """Get the joint id by joint name."""
        return self.model.getJointId(name)

    def get_frame_id(self, name: str) -> int:
        """Get the frame id by frame name."""
        return self.model.getFrameId(name)

    def get_joint_names(self) -> List[str]:
        """Return a list of all joint names in the model."""
        return list(self.model.names)

    def neutral_configuration(self) -> np.ndarray:
        """Return the neutral (zero) configuration for the robot model."""
        return pin.neutral(self.model)
