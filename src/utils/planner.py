from abc import ABC, abstractmethod
import numpy as np
from loguru import logger
import mujoco

from src.components import Robot
from src.utils.ik import LevenbergMarquardtIKSolver


class BasePlanner(ABC):
    """Abstract base class for planners."""

    def __init__(
        self, model: mujoco.MjModel, data: mujoco.MjData, arm: Robot
    ):
        self.model = model
        self.data = data
        self.arm = arm
        self.ik_solver = LevenbergMarquardtIKSolver(model, data, arm)

    @abstractmethod
    def plan(self, start_q: np.ndarray, target_pose_6d: np.ndarray) -> list[np.ndarray] | None:
        """Plan a path from start_q to target_pose_6d. Returns a list of joint waypoints or None on failure."""
        pass


class LinearPlanner(BasePlanner):
    """Linear interpolation planner using IK for endpoint."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm: Robot,
        step_size: float = 0.1,
    ):
        super().__init__(model, data, arm)
        self.step_size = step_size
        self.jnt_limits = self.model.jnt_range[arm.joint_ids]

    def _unwrap_continuous_joints(self, start_q: np.ndarray, goal_q: np.ndarray) -> np.ndarray:
        """If a joint moves more than Pi, check if wrapping 2*Pi is valid and closer."""
        new_goal = goal_q.copy()
        for i in range(len(start_q)):
            diff = new_goal[i] - start_q[i]
            if abs(diff) > np.pi:
                # Try subtracting/adding 2*pi
                candidate = new_goal[i] - 2 * np.pi if diff > 0 else new_goal[i] + 2 * np.pi
                min_limit, max_limit = self.jnt_limits[i]
                if min_limit <= candidate <= max_limit:
                    if abs(candidate - start_q[i]) < abs(diff):
                        new_goal[i] = candidate
        return new_goal

    def _interpolate_path(self, start_q: np.ndarray, goal_q: np.ndarray) -> list[np.ndarray]:
        """Linearly interpolate between start and goal joint positions."""
        diff = goal_q - start_q
        dist = np.linalg.norm(diff)
        num_steps = int(np.ceil(dist / self.step_size))
        if num_steps < 1:
            return [start_q]
        t_values = np.linspace(0, 1, num_steps + 1)
        return [start_q + t * diff for t in t_values]

    def plan(self, start_q: np.ndarray, target_pose_6d: np.ndarray) -> list[np.ndarray] | None:
        goal_q = self.ik_solver.solve(target_pose_6d)
        if goal_q is None:
            logger.error("IK solver failed to find a solution.")
            return None
        goal_q = self._unwrap_continuous_joints(start_q, goal_q)
        path = self._interpolate_path(start_q, goal_q)
        return path
