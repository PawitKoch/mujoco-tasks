import numpy as np
from loguru import logger

from src.components import Robot
from src.envs import BaseEnv
from src.primitives import Primitive
from src.planners import BasePlanner, LinearPlanner, RRTPlanner


class GoToPose(Primitive):
    """
    Primitive for planning and executing a trajectory to a target 6D pose (position + orientation).
    Uses a planner to generate a joint-space path to the target pose.
    Steps through the path, sending joint commands to the robot, and optionally settles at the goal.
    Useful for open-loop or planned Cartesian/end-effector motions.
    """

    def __init__(
        self,
        name: str,
        env: BaseEnv,
        arm: Robot,
        target_pose: np.ndarray,
        planner: str = "linear",
        speed: float = 1.5,
        settle_time: float = 1.0,
    ):
        super().__init__(name)
        self.env = env
        self.arm = arm
        self.target_pose = target_pose
        self.speed = speed
        self.settle_time = settle_time

        if planner == "linear":
            self.planner: BasePlanner = LinearPlanner(env.model, env.data, arm)
        elif planner == "rrt":
            self.planner = RRTPlanner(env.model, env.data, arm)
        else:
            raise ValueError(f"Unknown planner type: {planner}")

        self.path: list[np.ndarray] = []
        self.current_idx: int = 0
        self.done: bool = False
        self.commanded_q: np.ndarray | None = None
        self.reached_goal_time: float | None = None

    def reset(self) -> None:
        self.done = False
        self.path = []
        self.current_idx = 0
        self.reached_goal_time = None

        start_q = self.arm.qpos.copy()
        self.path = self.planner.plan(start_q, self.target_pose)

        if self.path is None:
            logger.error("[{}] Planning failed.", self.name)
            self.done = True
        else:
            logger.debug("[{}] Plan found with {} waypoints.", self.name, len(self.path))
            self.commanded_q = self.arm.qpos.copy()

    def _settle(self, final_q: np.ndarray) -> None:
        if self.reached_goal_time is None:
            self.reached_goal_time = self.env.data.time
            self.commanded_q = final_q

        if self.env.data.time - self.reached_goal_time >= self.settle_time:
            self.done = True

        self.arm.set_joint_positions(final_q)

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

        self.arm.set_joint_positions(self.commanded_q)

    def is_done(self) -> bool:
        return self.done


class GoToJointPosition(Primitive):
    """Primitive for planning and executing a path to a direct joint configuration."""

    def __init__(
        self,
        name: str,
        env: BaseEnv,
        arm: Robot,
        target_q: np.ndarray,
        planner: str = "rrt",
        speed: float = 1.5,
        settle_time: float = 1.0,
    ):
        super().__init__(name)
        self.env = env
        self.arm = arm
        self.target_q = target_q
        self.speed = speed
        self.settle_time = settle_time

        # For joint-to-joint planning, we primarily use RRT to avoid self-collisions
        # Linear could be supported if you refactored LinearPlanner.
        if planner == "rrt":
            self.planner = RRTPlanner(env.model, env.data, arm)
        else:
            # You could implement a simple linear interpolation fallback here if needed
            raise ValueError(f"GoToJointPosition currently only supports 'rrt', got: {planner}")

        self.path: list[np.ndarray] = []
        self.current_idx: int = 0
        self.done: bool = False
        self.commanded_q: np.ndarray | None = None
        self.reached_goal_time: float | None = None

    def reset(self) -> None:
        self.done = False
        self.path = []
        self.current_idx = 0
        self.reached_goal_time = None

        start_q = self.arm.qpos.copy()
        self.path = self.planner.plan_to_qpos(start_q, self.target_q)

        if self.path is None:
            logger.error("[{}] Planning failed.", self.name)
            self.done = True
        else:
            logger.debug("[{}] Plan found with {} waypoints.", self.name, len(self.path))
            self.commanded_q = self.arm.qpos.copy()

    def _settle(self, final_q: np.ndarray) -> None:
        if self.reached_goal_time is None:
            self.reached_goal_time = self.env.data.time
            self.commanded_q = final_q

        if self.env.data.time - self.reached_goal_time >= self.settle_time:
            self.done = True

        self.arm.set_joint_positions(final_q)

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

        # If we are close enough to the waypoint, advance index
        if dist < max_dist:
            self.current_idx += 1
            if self.current_idx >= len(self.path):
                self.commanded_q = self.path[-1]
                return
            target_q = self.path[self.current_idx]
            error = target_q - self.commanded_q
            dist = np.linalg.norm(error)

        # Move commanded_q towards target
        if dist > 1e-6:
            velocity = (error / dist) * max_dist
            self.commanded_q += velocity
        else:
            self.commanded_q = target_q

        self.arm.set_joint_positions(self.commanded_q)

    def is_done(self) -> bool:
        return self.done
