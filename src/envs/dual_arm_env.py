from dataclasses import dataclass
import mujoco
import numpy as np

from src.envs.base_env import BaseEnv, BaseEnvConfig
from src.components import Robot


@dataclass
class DualArmEnvConfig(BaseEnvConfig):
    left_arm_body_name: str
    """Name of the left arm robot body in MJCF."""

    right_arm_body_name: str
    """Name of the right arm robot body in MJCF."""

    left_gripper_body_name: str
    """Name of the left gripper robot body in MJCF."""

    right_gripper_body_name: str
    """Name of the right gripper robot body in MJCF."""

    left_gripper_act_name: str
    """Name of the left gripper actuator in MJCF."""

    right_gripper_act_name: str
    """Name of the right gripper actuator in MJCF."""

    left_tcp_site_name: str
    """Name of the left arm TCP site in MJCF."""

    right_tcp_site_name: str
    """Name of the right arm TCP site in MJCF."""

    left_arm_num_dofs: int = 7
    """Number of degrees of freedom for the left robot arm."""

    right_arm_num_dofs: int = 7
    """Number of degrees of freedom for the right robot arm."""


class DualArmEnv(BaseEnv):
    """Environment class for dual robot arms."""

    def __init__(self, config: DualArmEnvConfig):
        super().__init__(config)

        self.left_arm = Robot(
            self.model,
            self.data,
            config.left_arm_body_name,
            config.left_gripper_body_name,
            config.left_gripper_act_name,
            config.left_tcp_site_name,
            config.left_arm_num_dofs,
        )
        self.right_arm = Robot(
            self.model,
            self.data,
            config.right_arm_body_name,
            config.right_gripper_body_name,
            config.right_gripper_act_name,
            config.right_tcp_site_name,
            config.right_arm_num_dofs,
        )

        # Use left arm home position for both arms
        self.home_qpos: np.ndarray = np.zeros(len(self.left_arm.joint_ids))
        for i in range(self.model.nkey):
            key_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i)
            if "home" in key_name:
                self.home_qpos = self.model.key_qpos[i][self.left_arm.joint_ids].copy()
                break

    def reset(self):
        """Reset the environment to an initial state."""
        mujoco.mj_resetData(self._model, self._data)
        self.data.qpos[self.left_arm.joint_ids] = self.home_qpos
        self.data.qpos[self.right_arm.joint_ids] = self.home_qpos
        self.data.ctrl[self.left_arm.ctrl_ids] = self.home_qpos
        self.data.ctrl[self.right_arm.ctrl_ids] = self.home_qpos
        self.forward()
