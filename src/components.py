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
        arm_num_dofs: int = 7,
    ):
        self.model = model
        self.data = data

        # Body & Site IDs
        self.all_body_ids = self.collect_subtree_body_ids(
            model, model.body(arm_body_name).id
        )
        self.body_id = model.body(arm_body_name).id
        self.tcp_site_id = model.site(tcp_site_name).id
        self.gripper_body_ids = self.collect_subtree_body_ids(
            model, model.body(gripper_body_name).id
        )

        # Geom IDs (for collision checking, if needed)
        all_geom_ids = np.array([g for g in range(model.ngeom) if model.geom_bodyid[g] in self.all_body_ids], dtype=int)
        gripper_geom_ids = np.array(
            [g for g in range(model.ngeom) if model.geom_bodyid[g] in self.gripper_body_ids], dtype=int
        )
        self.arm_geom_ids = np.setdiff1d(all_geom_ids, gripper_geom_ids)

        # Joint Indices
        chain_joint_ids = self.collect_joint_ids(self.body_id)
        if len(chain_joint_ids) >= arm_num_dofs:
            self.joint_ids = np.array(chain_joint_ids[:arm_num_dofs], dtype=int)
        else:
            # Fallback if something is weird, though unlikely
            self.joint_ids = np.array(chain_joint_ids, dtype=int)

        # Actuator Indices
        self.ctrl_ids = np.array(
            [
                a
                for a in range(model.nu)
                if model.actuator_trnid[a, 0] in self.joint_ids
                and model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT  # Only joint actuators
            ],
            dtype=int,
        )
        self.gripper_ctrl_id = model.actuator(gripper_act_name).id

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

    def collect_subtree_body_ids(self, model, root_body_id):
        """Recursively collect the root body ID and all its descendants."""
        body_ids = [root_body_id]

        # Iterate over all bodies to find children of the current set
        # (MuJoCo bodies are usually topological, so we can do this efficiently,
        # but a simple recursive search is robust for initialization)
        for i in range(model.nbody):
            if model.body_parentid[i] == root_body_id:
                body_ids.extend(self.collect_subtree_body_ids(model, i))

        return body_ids
