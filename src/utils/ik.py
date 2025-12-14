import mujoco
import numpy as np
from loguru import logger

from src.components import Robot


class LevenbergMarquardtIKSolver:
    """
    Levenberg-Marquardt IK solver for 6D pose (position + orientation).
    Provides one-step update and full solve methods.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm: Robot,
        damping: float = 0.05,
    ):
        self._model = model
        self._data = data
        self.arm = arm
        self._arm_joint_ids = arm.joint_ids
        self._tcp_site_id = arm.tcp_site_id
        self._damping = damping

        # Buffers
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._target_quat = np.zeros(4)
        self._err_quat = np.zeros(4)

    def compute_step(self, error_6d: np.ndarray) -> np.ndarray:
        """
        Calculates one gradient descent step (delta_q) for the arm joints.
        """
        mujoco.mj_jacSite(self._model, self._data, self._jacp, self._jacr, self._tcp_site_id)
        jac_full = np.vstack((self._jacp, self._jacr))
        jac = jac_full[:, self._arm_joint_ids]

        n_joints = jac.shape[1]
        I = np.eye(n_joints)
        product = jac.T @ jac + self._damping * I
        gradient = jac.T @ error_6d

        return np.linalg.solve(product, gradient)

    def solve(
        self, target_pose_6d: np.ndarray, max_iters: int = 100, pos_tol: float = 0.001, rot_tol: float = 0.01
    ) -> np.ndarray | None:
        """
        Solves for absolute joint angles to reach a 6D pose.
        Restores simulation state after calculation.
        Returns: joint angles (np.ndarray) or None if failed.
        """
        initial_q = self._data.qpos.copy()

        target_pos = target_pose_6d[:3]
        if len(target_pose_6d) == 6:
            mujoco.mju_euler2Quat(self._target_quat, target_pose_6d[3:], "xyz")
        else:
            self._target_quat = target_pose_6d[3:]

        success = False
        solution_q = None

        for i in range(max_iters):
            mujoco.mj_fwdPosition(self._model, self._data)
            curr_tcp_pose = self.arm.tcp_pose

            # Position Error
            curr_pos = curr_tcp_pose[:3]
            err_pos = target_pos - curr_pos

            # Rotation Error
            curr_quat = curr_tcp_pose[3:]

            # Check if quaternions are antipodal (dot product < 0)
            # If so, negate target quaternion to ensure shortest rotation
            if np.dot(curr_quat, self._target_quat) < 0.0:
                aligned_target_quat = -self._target_quat
            else:
                aligned_target_quat = self._target_quat

            neg_curr_quat = np.zeros(4)
            mujoco.mju_negQuat(neg_curr_quat, curr_quat)
            mujoco.mju_mulQuat(self._err_quat, aligned_target_quat, neg_curr_quat)
            err_rot = self._err_quat[1:] * 2.0  # Vector part scaled
            error_6d = np.concatenate([err_pos, err_rot])

            if np.linalg.norm(err_pos) < pos_tol and np.linalg.norm(err_rot) < rot_tol:
                success = True
                solution_q = self.arm.qpos.copy()
                logger.debug(f"IK converged in {i+1} iterations.")
                break

            delta_q = self.compute_step(error_6d)
            self._data.qpos[self._arm_joint_ids] += delta_q

        if not success:
            logger.error("IK failed to converge after {} iterations.", max_iters)

        # Restore State
        self._data.qpos[:] = initial_q
        mujoco.mj_fwdPosition(self._model, self._data)

        return solution_q if success else None
