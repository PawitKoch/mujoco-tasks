import mujoco
import numpy as np


class Robot:
    """Component representing a robot arm in the simulation."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arm_body_name: str,
        gripper_body_name: str,
        gripper_act_name: str,
        tcp_site_name: str,
    ):
        self.model = model
        self.data = data

        # Body & Site IDs
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, arm_body_name)
        self.gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gripper_body_name)
        self.tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tcp_site_name)

        # Joint Indices
        all_joint_ids = self.collect_joint_ids(self.body_id)
        gripper_joint_ids = self.collect_joint_ids(self.gripper_body_id)
        self.joint_ids = np.setdiff1d(all_joint_ids, gripper_joint_ids)

        # Actuator Indices
        all_ctrl_ids = np.array([a for a in range(model.nu) if model.actuator_trnid[a, 0] in all_joint_ids], dtype=int)
        self.gripper_ctrl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_act_name)
        self.ctrl_ids = np.setdiff1d(all_ctrl_ids, self.gripper_ctrl_id)

    @property
    def qpos(self) -> np.ndarray:
        """Read current joint positions."""
        return self.data.qpos[self.joint_ids]

    @property
    def tcp_pose(self) -> np.ndarray:
        """Get current TCP 6D pose [x,y,z, qw,qx,qy,qz]."""
        pos = self.data.site_xpos[self.tcp_site_id]
        mat = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return np.concatenate([pos, quat])

    @property
    def gripper_width(self) -> float:
        """Get current gripper width (control value)."""
        return self.data.ctrl[self.gripper_ctrl_id]

    def set_joint_positions(self, q: np.ndarray):
        """Set control setpoints for this arm."""
        self.data.ctrl[self.ctrl_ids] = q

    def set_gripper_width(self, width: float):
        """Set gripper control."""
        self.data.ctrl[self.gripper_ctrl_id] = width

    def collect_joint_ids(self, body_id: int) -> np.ndarray:
        """Recursively collect all joint IDs under the given body ID."""
        joint_ids = []
        for j in range(self.model.njnt):
            if self.model.jnt_bodyid[j] == body_id:
                joint_ids.append(j)
        for child_body_id in range(self.model.nbody):
            if self.model.body_parentid[child_body_id] == body_id:
                joint_ids.extend(self.collect_joint_ids(child_body_id))
        return np.array(joint_ids, dtype=int)
