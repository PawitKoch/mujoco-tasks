import numpy as np
import mujoco
from loguru import logger
from enum import Enum, auto

from src.components import Robot
from src.planners import BasePlanner


class RRTPlanner(BasePlanner):
    """
    Bi-directional RRT (RRT-Connect) Planner.
    Grows two trees from start and goal positions until they connect.
    """

    class Status(Enum):
        REACHED = auto()
        ADVANCED = auto()
        TRAPPED = auto()

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm: Robot,
        step_size: float = 0.1,
        max_iter: int = 2000,
        collision_tol: float = 0.005,
    ):
        super().__init__(model, data, arm)
        self.step_size = step_size
        self.max_iter = max_iter
        self.collision_tol = collision_tol
        self.jnt_limits = self.model.jnt_range[arm.joint_ids]

        # Ghost mjData for collision checking
        self._ghost_data = mujoco.MjData(model)

    def _check_collision(self, q: np.ndarray) -> bool:
        """Returns True if the position q is in collision."""
        # Sync ghost data
        self._ghost_data.qpos[:] = self.data.qpos[:]
        self._ghost_data.qpos[self.arm.joint_ids] = q
        mujoco.mj_forward(self.model, self._ghost_data)

        # Check contacts for any collision involving the arm
        for i in range(self._ghost_data.ncon):
            contact = self._ghost_data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            geom1_is_arm = geom1 in self.arm.arm_geom_ids
            geom2_is_arm = geom2 in self.arm.arm_geom_ids
            if not (geom1_is_arm or geom2_is_arm) or (contact.dist >= self.collision_tol):
                continue

            return True
        return False

    def plan(self, start_q: np.ndarray, target_pose: np.ndarray) -> list[np.ndarray] | None:
        """Plans a path to a Cartesian pose (uses IK internally)."""
        # Solve IK for goal
        goal_q = self.ik_solver.solve(target_pose)
        if goal_q is None:
            logger.error("RRT: IK solver failed to find goal joint position.")
            return None

        # Delegate to joint planner
        return self.plan_to_qpos(start_q, goal_q)

    def plan_to_qpos(self, start_q: np.ndarray, target_q: np.ndarray) -> list[np.ndarray] | None:
        """
        Plans a path directly to a specific joint position.
        """
        if self._check_collision(start_q):
            logger.warning("RRT: Start joint position is in collision.")
            return None
        if self._check_collision(target_q):
            logger.warning("RRT: Target joint position is in collision.")
            return None

        # Initialize Trees
        start_tree = [(start_q, -1)]
        goal_tree = [(target_q, -1)]

        # Run RRT Connect
        return self._run_rrt_connect(start_tree, goal_tree)

    def _run_rrt_connect(self, start_tree, goal_tree) -> list[np.ndarray] | None:
        """Main logic for bi-directional RRT."""
        for _ in range(self.max_iter):
            # Grow start tree
            q_rand = self._sample_random_q()
            status_s, q_new_s = self._extend(start_tree, q_rand)

            if status_s != self.Status.TRAPPED:
                # Try to connect goal tree to the new node from start tree
                status_connect = self._connect(goal_tree, q_new_s)
                if status_connect == self.Status.REACHED:
                    return self._construct_path(start_tree, goal_tree)

            # Grow goal tree (Swap roles)
            q_rand = self._sample_random_q()
            status_g, q_new_g = self._extend(goal_tree, q_rand)

            if status_g != self.Status.TRAPPED:
                # Try to connect start tree to the new node from goal tree
                status_connect = self._connect(start_tree, q_new_g)
                if status_connect == self.Status.REACHED:
                    return self._construct_path(start_tree, goal_tree)

        logger.error("RRT: Max iterations reached without finding a path.")
        return None

    def _sample_random_q(self):
        """Uniform sampling within joint limits."""
        return np.random.uniform(self.jnt_limits[:, 0], self.jnt_limits[:, 1])

    def _get_nearest_neighbor(self, tree: list[tuple[np.ndarray, int]], q_query: np.ndarray) -> tuple[int, np.ndarray]:
        """Find the index and config of the nearest node in the tree."""
        distances = [np.linalg.norm(node[0] - q_query) for node in tree]
        idx = np.argmin(distances)
        return idx, tree[idx][0]

    def _steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        """Move from q_from towards q_to by at most step_size."""
        diff = q_to - q_from
        dist = np.linalg.norm(diff)

        if dist < self.step_size:
            return q_to
        else:
            return q_from + (diff / dist) * self.step_size

    def _is_segment_valid(self, q1: np.ndarray, q2: np.ndarray) -> bool:
        """Check for collisions along the line segment between q1 and q2."""
        dist = np.linalg.norm(q2 - q1)
        # Resolution for checking (e.g. every 2 degrees)
        steps = int(np.ceil(dist / 0.05))

        for i in range(1, steps + 1):
            q_interp = q1 + (q2 - q1) * (i / steps)
            if self._check_collision(q_interp):
                return False
        return True

    def _extend(self, tree: list[tuple[np.ndarray, int]], q_target: np.ndarray):
        """
        Tries to extend the tree towards q_target.
        Returns: "REACHED" (hit target), "ADVANCED" (moved closer), "TRAPPED" (collision)
        """
        nearest_idx, q_near = self._get_nearest_neighbor(tree, q_target)
        q_new = self._steer(q_near, q_target)

        if self._is_segment_valid(q_near, q_new):
            tree.append((q_new, nearest_idx))

            if np.array_equal(q_new, q_target):
                return self.Status.REACHED, q_new
            else:
                return self.Status.ADVANCED, q_new

        return self.Status.TRAPPED, None

    def _connect(self, tree: list[tuple[np.ndarray, int]], q_target: np.ndarray) -> str:
        """
        Repeatedly extends tree towards q_target until it reaches or hits obstacle.
        Used for the 'Greedy' connection step in RRT-Connect.
        """
        status = self.Status.ADVANCED
        q_new = None
        while status == self.Status.ADVANCED:
            status, q_new = self._extend(tree, q_target)
        return status

    def _construct_path(
        self, start_tree: list[tuple[np.ndarray, int]], goal_tree: list[tuple[np.ndarray, int]]
    ) -> list[np.ndarray]:
        """Reconstruct the full path from the two trees meeting in the middle."""
        # The last added nodes in both trees are the connection point
        path_from_start = []
        curr_idx = len(start_tree) - 1
        while curr_idx != -1:
            path_from_start.append(start_tree[curr_idx][0])
            curr_idx = start_tree[curr_idx][1]
        path_from_start.reverse()

        path_from_goal = []
        curr_idx = len(goal_tree) - 1
        while curr_idx != -1:
            path_from_goal.append(goal_tree[curr_idx][0])
            curr_idx = goal_tree[curr_idx][1]

        # Note: The connection point is duplicated, remove one.
        return path_from_start + path_from_goal[1:]
