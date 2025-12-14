from abc import ABC, abstractmethod
import numpy as np
import mujoco

from src.components import Robot
from src.utils.ik import LevenbergMarquardtIKSolver


class BasePlanner(ABC):
    """Abstract base class for planners."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, arm: Robot):
        self.model = model
        self.data = data
        self.arm = arm
        self.ik_solver = LevenbergMarquardtIKSolver(model, data, arm)

    @abstractmethod
    def plan(self, start_q: np.ndarray, target_pose_6d: np.ndarray) -> list[np.ndarray] | None:
        """Plan a path from start_q to target_pose_6d. Returns a list of joint waypoints or None on failure."""
        pass