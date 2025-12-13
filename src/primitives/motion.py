import numpy as np
from loguru import logger

from src.envs import SingleArmEnv
from src.primitives import Primitive
from src.utils import BasePlanner, LinearPlanner


class PlanAndExecuteTrajectory(Primitive):
    """
    Primitive for planning and executing a trajectory to a target 6D pose (position + orientation).
    Uses a planner to generate a joint-space path to the target pose.
    Steps through the path, sending joint commands to the robot, and optionally settles at the goal.
    Useful for open-loop or planned Cartesian/end-effector motions.
    """

    def __init__(self, env: SingleArmEnv, target_pose: np.ndarray, speed: float = 1.5, settle_time: float = 1.0):
        self.env = env
        self.target_pose = target_pose
        self.speed = speed
        self.settle_time = settle_time

        self.planner: BasePlanner = LinearPlanner(env.model, env.data, env.arm_joint_ids, env.ee_site_id)
        self.path: list[np.ndarray] = []
        self.current_idx: int = 0
        self.done: bool = False
        self.commanded_q: np.ndarray | None = None
        self.reached_goal_time: float | None = None

    def reset(self) -> None:
        self.done = False
        self.path = []
        self.current_idx = 0

        start_q = self.env.data.qpos[self.env.arm_joint_ids].copy()
        self.path = self.planner.plan(start_q, self.target_pose)

        if self.path is None:
            logger.error("Planning failed.")
            self.done = True
        else:
            logger.debug("Plan found with {} waypoints.", len(self.path))
            self.commanded_q = self.env.data.qpos[self.env.arm_joint_ids].copy()

    def _settle(self, final_pose: np.ndarray) -> None:
        if self.reached_goal_time is None:
            self.reached_goal_time = self.env.data.time
            self.commanded_q = final_pose

        if self.env.data.time - self.reached_goal_time >= self.settle_time:
            self.done = True

        self.env.data.ctrl[self.env.arm_ctrl_ids] = final_pose

    def step(self) -> None:
        if self.done or not self.path:
            return

        # If we have reached the end of the path, hold position for settle_time
        if self.current_idx >= len(self.path):
            self._settle(self.path[-1])
            return

        # Else, move towards the next waypoint
        target_q = self.path[self.current_idx]
        error = target_q - self.commanded_q
        dist = np.linalg.norm(error)
        max_dist = self.speed * self.env.model.opt.timestep

        if dist < max_dist:
            self.current_idx += 1
            if self.current_idx >= len(self.path):
                self.commanded_q = self.path[-1]
                return
            target_q = self.path[self.current_idx]
            error = target_q - self.commanded_q
            dist = np.linalg.norm(error)

        if dist > 1e-6:
            velocity = (error / dist) * max_dist
            self.commanded_q += velocity
        else:
            self.commanded_q = target_q

        self.env.data.ctrl[self.env.arm_ctrl_ids] = self.commanded_q

    def is_done(self) -> bool:
        return self.done
