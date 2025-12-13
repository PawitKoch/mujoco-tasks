from dataclasses import dataclass
import mujoco
import numpy as np

from src.envs.base_env import BaseEnv, BaseEnvConfig


@dataclass
class SingleArmEnvConfig(BaseEnvConfig):
    arm_body_name: str
    """Name of the robot body in MJCF."""

    gripper_body_name: str
    """Name of the gripper body in MJCF."""

    gripper_act_name: str
    """Name of the gripper actuator in MJCF."""

    ee_site_name: str
    """Name of the end-effector pose site in MJCF."""


class SingleArmEnv(BaseEnv):
    """Environment class for a single robotic arm."""

    def __init__(self, config: SingleArmEnvConfig):
        super().__init__(config)
        self.config = config
        self.arm_body_name = config.arm_body_name
        self.gripper_body_name = config.gripper_body_name
        self.gripper_act_name = config.gripper_act_name

        self._model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self._data = mujoco.MjData(self._model)

        arm_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self.arm_body_name)
        gripper_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self.gripper_body_name)
        all_joint_ids = np.array(self._collect_joint_ids(arm_body_id), dtype=int)
        gripper_joint_ids = np.array(self._collect_joint_ids(gripper_body_id), dtype=int)
        self.arm_joint_ids = np.setdiff1d(all_joint_ids, gripper_joint_ids)
        ctrl_ids = np.array(
            [a for a in range(self._model.nu) if self._model.actuator_trnid[a, 0] in all_joint_ids], dtype=int
        )
        self.gripper_ctrl_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.gripper_act_name)
        self.arm_ctrl_ids = np.setdiff1d(ctrl_ids, self.gripper_ctrl_id)
        self.ee_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, config.ee_site_name)

    def reset(self):
        """Reset the environment to an initial state."""
        # If a keyframe named 'home' exists, use it; else, just reset data
        key_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self._model, self._data, key_id)
        else:
            mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)

    def _collect_joint_ids(self, body_id):
        """Recursively collect all joint IDs under the given body ID."""
        joint_ids = []
        for j in range(self._model.njnt):
            if self._model.jnt_bodyid[j] == body_id:
                joint_ids.append(j)
        for child_body_id in range(self._model.nbody):
            if self._model.body_parentid[child_body_id] == body_id:
                joint_ids.extend(self._collect_joint_ids(child_body_id))
        return joint_ids

    def forward(self):
        """Perform forward dynamics step."""
        mujoco.mj_forward(self._model, self._data)

    def step(self):
        """Apply action, step and return (observation, done, info)."""
        mujoco.mj_step(self._model, self._data)

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data
