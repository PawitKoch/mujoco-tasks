from dataclasses import dataclass
import mujoco

from src.envs.base_env import BaseEnv, BaseEnvConfig
from src.components import Robot


@dataclass
class SingleArmEnvConfig(BaseEnvConfig):
    arm_body_name: str
    """Name of the robot body in MJCF."""

    gripper_body_name: str
    """Name of the gripper body in MJCF."""

    gripper_act_name: str
    """Name of the gripper actuator in MJCF."""

    tcp_site_name: str
    """Name of the end-effector TCP site in MJCF."""
    
    arm_num_dofs: int
    """Number of degrees of freedom for the robot arm."""


class SingleArmEnv(BaseEnv):
    """Environment class for a single robot arm."""

    def __init__(self, config: SingleArmEnvConfig):
        super().__init__(config)
        self.arm = Robot(
            model=self._model,
            data=self._data,
            arm_body_name=config.arm_body_name,
            gripper_body_name=config.gripper_body_name,
            gripper_act_name=config.gripper_act_name,
            tcp_site_name=config.tcp_site_name,
            arm_num_dofs=config.arm_num_dofs,
        )

    def reset(self):
        """Reset the environment to an initial state."""
        # If a keyframe named 'home' exists, use it; else, just reset data
        key_id = self._model.key("home").id
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self._model, self._data, key_id)
        else:
            mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
