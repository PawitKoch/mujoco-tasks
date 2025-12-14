import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R

from src.components import Robot
from src.envs import BaseEnv
from src.primitives import Primitive
from src.planners import BasePlanner, LinearPlanner, RRTPlanner
from src.utils import LevenbergMarquardtIKSolver


class BaseMotionPrimitive(Primitive):
    """
    Base class for motion primitives that follow a planned path in joint space.
    Handles path following, settling, and state management.
    """

    def __init__(self, name: str, env: BaseEnv, arm: Robot, speed: float = 2.0, settle_time: float = 0.5):
        super().__init__(name)
        self.env = env
        self.arm = arm
        self.speed = speed
        self.settle_time = settle_time
        self._success: bool = True

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
        self.path = self.generate_path(start_q)

        if self.path is None:
            logger.error(f"[{self.name}] Planning failed.")
            self.done = True
            self._success = False
        else:
            logger.debug(f"[{self.name}] Plan found with {len(self.path)} waypoints.")
            self._success = True
            self.commanded_q = self.arm.qpos.copy()

    def _settle(self, final_q: np.ndarray) -> None:
        if self.reached_goal_time is None:
            self.reached_goal_time = self.env.data.time
            self.commanded_q = final_q
        if self.env.data.time - self.reached_goal_time >= self.settle_time:
            self.done = True
            self._success = True
        self.arm.set_joint_positions(final_q)

    def step(self) -> None:
        if self.done or not self.path:
            return

        if self.current_idx >= len(self.path):
            self._settle(self.path[-1])
            return

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

    def generate_path(self, start_q: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError


class GoToPose(BaseMotionPrimitive):
    """
    Primitive for planning and executing a trajectory to a target 6D pose (position + orientation).
    Uses a planner to generate a joint-space path to the target pose.
    """

    def __init__(
        self,
        name: str,
        env: BaseEnv,
        arm: Robot,
        target_pose: np.ndarray,
        planner: str = "linear",
        speed: float = 2.0,
        settle_time: float = 0.5,
    ):
        super().__init__(name, env, arm, speed, settle_time)
        self.target_pose = target_pose
        if planner == "linear":
            self.planner: BasePlanner = LinearPlanner(env.model, env.data, arm)
        elif planner == "rrt":
            self.planner = RRTPlanner(env.model, env.data, arm)
        else:
            raise ValueError(f"Unknown planner type: {planner}")

    def generate_path(self, start_q: np.ndarray) -> list[np.ndarray]:
        return self.planner.plan(start_q, self.target_pose)


class GoToJointPosition(BaseMotionPrimitive):
    """Primitive for planning and executing a path to a direct joint configuration."""

    def __init__(
        self,
        name: str,
        env: BaseEnv,
        arm: Robot,
        target_q: np.ndarray,
        planner: str = "rrt",
        speed: float = 2.0,
        settle_time: float = 0.5,
    ):
        super().__init__(name, env, arm, speed, settle_time)
        self.target_q = target_q
        if planner == "rrt":
            self.planner = RRTPlanner(env.model, env.data, arm)
        else:
            raise ValueError(f"GoToJointPosition currently only supports 'rrt', got: {planner}")

    def generate_path(self, start_q: np.ndarray) -> list[np.ndarray]:
        return self.planner.plan_to_qpos(start_q, self.target_q)


class CircularArcMotion(BaseMotionPrimitive):
    """Performs a circular arc motion of the end-effector around a specified center point and axis."""

    def __init__(
        self,
        name: str,
        env: BaseEnv,
        arm: Robot,
        hinge_body_name: str,
        target_angle: float,
        steps: int = 50,
        speed: float = 2.0,
        settle_time: float = 0.5,
    ):
        super().__init__(name, env, arm, speed, settle_time)
        self.hinge_body_id = env.model.body(hinge_body_name).id
        self.hinge_jnt_id = self.env.model.body_jntadr[self.hinge_body_id]
        self.target_angle = target_angle
        self.steps = steps
        self.ik_solver = LevenbergMarquardtIKSolver(env.model, env.data, arm, damping=1e-1)

    def _get_hinge_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the hinge position and axis."""
        hinge_body_pos = self.env.data.xpos[self.hinge_body_id].copy()
        hinge_axis = self.env.data.xaxis[self.hinge_jnt_id].copy()
        return hinge_body_pos, hinge_axis

    def generate_path(self, start_q: np.ndarray) -> list[np.ndarray]:
        # Get hinge geometry
        hinge_pos, hinge_axis = self._get_hinge_geometry()

        # Get current TCP pose and convert to rotation object
        self.arm.set_joint_positions(start_q)
        self.env.forward()
        tcp_pos = self.arm.tcp_pose[:3].copy()
        tcp_quat = self.arm.tcp_pose[3:].copy()
        r_start = R.from_quat([tcp_quat[1], tcp_quat[2], tcp_quat[3], tcp_quat[0]])  # wxyz to xyzw

        # Caclulate waypoints along circular arc
        radius_vec = tcp_pos - hinge_pos
        joint_path = []
        prev_q = start_q.copy()

        for i in range(self.steps):
            # Compute incremental rotation and update radius vector
            fraction = (i + 1) / self.steps
            theta = self.target_angle * fraction
            rot_vec = hinge_axis * theta
            r_step = R.from_rotvec(rot_vec)
            new_radius_vec = r_step.apply(radius_vec)

            # Compute target pose from radius vector
            target_pos = hinge_pos + new_radius_vec
            target_rot = r_step * r_start
            qx, qy, qz, qw = target_rot.as_quat()
            target_quat = np.array([qw, qx, qy, qz])  # xyzw to wxyz (scipy to mujoco)
            target_pose_6d = np.concatenate([target_pos, target_quat])

            # Solve IK for joint angles, using previous q as initial guess
            q_sol = self.ik_solver.solve(
                target_pose_6d, initial_q=prev_q, pos_tol=1e-2, rot_tol=1e-1
            )  # relaxed tolerances
            if q_sol is None:
                logger.error(f"[{self.name}] IK failed at step {i} of circular arc motion.")
                return None

            joint_path.append(q_sol)
            prev_q = q_sol

        return joint_path
