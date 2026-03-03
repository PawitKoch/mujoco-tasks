"""Single arm trying to open a cabinet hinge door."""

from dataclasses import dataclass
import time
from loguru import logger
import mujoco
import mujoco.viewer
import numpy as np
import glfw

from src.envs import BaseEnv, BaseEnvRunner, SingleArmEnv, SingleArmEnvConfig
from src.primitives import PrimitiveSequence, GoToPose, GripperAction, CircularArcMotion


@dataclass
class OpenCabinetEnvConfig(SingleArmEnvConfig):
    handle_site_name: str
    """Name of the handle site in MJCF."""

    hinge_body_name: str
    """Name of the hinge body in MJCF."""

    hinge_joint_name: str
    """Name of the hinge joint in MJCF."""

    hinge_joint_min_init_angle: float = np.deg2rad(5)
    """Minimum initial angle for the hinge joint."""

    hinge_joint_max_init_angle: float = np.deg2rad(10)
    """Maximum initial angle for the hinge joint."""

    hinge_joint_target_angle: float = np.deg2rad(45)
    """Target angle to open the hinge joint to."""


class OpenCabinetEnvRunner(BaseEnvRunner):
    """
    Orchestrates the cabinet opening task: environment reset, cabinet placement, target pose computation,
    primitive sequence creation, and main execution loop.
    """

    def __init__(self, env: BaseEnv | SingleArmEnv, render_dt: float = 0.02):
        super().__init__(env, render_dt)
        self.primitive_seq: PrimitiveSequence | None = None

        self.handle_site_id = self.env.model.site(self.env.config.handle_site_name).id
        self.hinge_joint_id = self.env.model.joint(self.env.config.hinge_joint_name).id
        self.hinge_jnt_qposadr = self.env.model.jnt_qposadr[self.hinge_joint_id]

    def _compute_handle_grasp_pose(self) -> np.ndarray:
        """
        Computes the grasp pose based on specific axis alignment requirements between the gripper and the cabinet handle.
        """
        handle_pos = self.env.data.site_xpos[self.handle_site_id].copy()
        handle_mat = self.env.data.site_xmat[self.handle_site_id].reshape(3, 3)

        site_y = handle_mat[:, 1]  # Normal (Points In)
        site_z = handle_mat[:, 2]  # Vertical (Points Up along handle)
        z_gripper = site_y  # Gripper Z (Approach): Points INTO the door
        x_gripper = -site_z  # Gripper X (Fingers): Aligns with NEGATIVE Site Z
        y_gripper = np.cross(z_gripper, x_gripper)  # Gripper Y (Side): Right-hand rule
        target_mat = np.column_stack((x_gripper, y_gripper, z_gripper))

        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, target_mat.flatten())
        return np.concatenate([handle_pos, target_quat])

    def _randomize_hinge_angle(self):
        """Randomize the hinge joint angle within config bounds."""
        min_angle = self.env.config.hinge_joint_min_init_angle
        max_angle = self.env.config.hinge_joint_max_init_angle
        angle = np.random.uniform(min_angle, max_angle)
        self.env.data.qpos[self.hinge_jnt_qposadr] = angle

    def _build_episode_plan(self) -> PrimitiveSequence:
        arm = self.env.arm
        approach_handle_pose = self._compute_handle_grasp_pose()

        # Compute grasp pose slightly offset (2cm) from handle along gripper approach axis
        grasp_mat = np.zeros(9)
        mujoco.mju_quat2Mat(grasp_mat, approach_handle_pose[3:])
        grasp_mat = grasp_mat.reshape(3, 3)
        z_approach = grasp_mat[:, 2]
        grasp_handle_pose = approach_handle_pose.copy()
        grasp_handle_pose[:3] += 0.02 * z_approach

        return PrimitiveSequence(
            name="OpenCabinetSequence",
            primitives=[
                GoToPose(name="Pregrasp", env=self.env, arm=arm, target_pose=approach_handle_pose, planner="rrt"),
                GoToPose(name="Grasp", env=self.env, arm=arm, target_pose=grasp_handle_pose, planner="linear"),
                GripperAction(name="CloseGripper", env=self.env, arm=arm, cmd=255.0),
                CircularArcMotion(
                    name="OpenHingeDoor",
                    env=self.env,
                    arm=arm,
                    hinge_body_name=self.env.config.hinge_body_name,
                    target_angle=self.env.config.hinge_joint_target_angle,
                    speed=1.5,
                ),
            ],
        )

    def setup_episode(self) -> None:
        self.env.reset()
        self.randomise_object_positions()
        self._randomize_hinge_angle()
        self.env.forward()

        self.primitive_seq = self._build_episode_plan()
        self.primitive_seq.reset()

    def is_done(self) -> bool:
        """Check if the current primitive sequence is done."""
        if self.primitive_seq is not None:
            return self.primitive_seq.is_done()
        return False


if __name__ == "__main__":
    env_config = OpenCabinetEnvConfig(
        mjcf_path="models/mjcf/open_cabinet.xml",
        object_names=["hingecab"],
        object_min_distance_x=0.0,
        object_min_distance_y=0.0,
        object_x_bounds=(0.4, 0.4),
        object_y_bounds=(-0.1, 0),
        object_rz_bounds=(np.deg2rad(-100), np.deg2rad(-90)),
        arm_body_name="xarm7",
        gripper_body_name="xarm_gripper_base_link",
        gripper_act_name="gripper",
        tcp_site_name="link_tcp",
        arm_num_dofs=7,
        handle_site_name="rightdoor_site",
        hinge_body_name="hingerightdoor",
        hinge_joint_name="rightdoorhinge",
    )
    env = SingleArmEnv(env_config)

    runner = OpenCabinetEnvRunner(env, render_dt=0.02)
    runner.run(env.model, env.data)
