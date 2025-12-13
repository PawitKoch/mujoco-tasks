from src.envs.base_env import BaseEnv, BaseEnvConfig
from dataclasses import dataclass


@dataclass
class DualArmEnvConfig(BaseEnvConfig):
    left_arm_name: str
    """Name of the left arm robot body in MJCF."""

    right_arm_name: str
    """Name of the right arm robot body in MJCF."""

    left_gripper_name: str
    """Name of the left gripper actuator in MJCF."""

    right_gripper_name: str
    """Name of the right gripper actuator in MJCF."""


class DualArmEnv(BaseEnv):
    """Environment class for dual robotic arms."""

    def __init__(self, config: DualArmEnvConfig):
        super().__init__(config)

    def reset(self):
        """Reset the environment to an initial state and return the initial observation."""
        pass

    def step(self, action):
        """Apply action, step and return (observation, done, info)."""
        pass

    def render(self):
        """Render the environment."""
        pass
