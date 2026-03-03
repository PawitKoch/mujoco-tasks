"""Single arm stacking two cubes on a table."""

import mujoco
import numpy as np

from src.envs import BaseEnvRunner, SingleArmEnv, SingleArmEnvConfig
from src.primitives import GoToPose, PrimitiveSequence, GripperAction


class CubeStackingEnvRunner(BaseEnvRunner):
    """
    Orchestrates the cube stacking task: environment reset, cube placement, target pose computation,
    primitive sequence creation, and main execution loop.
    """

    def __init__(self, env: SingleArmEnv, render_dt: float = 0.02):
        super().__init__(env, render_dt)
        self.primitive_seq: PrimitiveSequence | None = None

    def _compute_target_pose(self, target_name: str) -> np.ndarray:
        """Compute the target pose given a target cube name, with gripper pointing down."""
        cube_body_id = self.object_body_name2id[target_name]
        cube_pos = self.env.data.xpos[cube_body_id].copy()
        cube_mat = self.env.data.xmat[cube_body_id].reshape(3, 3)

        target_mat = np.zeros((3, 3))
        target_mat[:, 0] = cube_mat[:, 1]  # Gripper X aligned with cube Y
        target_mat[:, 1] = cube_mat[:, 0]  # Gripper Y aligned with cube X
        target_mat[:, 2] = -cube_mat[:, 2]  # Gripper Z inverted (Down)
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, target_mat.flatten())

        # This should be just above the cube
        target_xyz = cube_pos.copy()
        return np.concatenate([target_xyz, target_quat])

    def setup_episode(self) -> None:
        """Reset environment, place cubes, compute target pose, and create primitive sequence."""
        self.env.reset()
        self.randomise_object_positions()
        self.env.forward()
        arm = self.env.arm
        pregrasp_pose = self._compute_target_pose("red_cube")
        grasp_pose = pregrasp_pose.copy()
        grasp_pose[2] -= 0.05  # Move down to grasp
        place_pose = self._compute_target_pose("green_cube")
        place_pose[2] += 0.05  # Move above to place
        self.primitive_seq = PrimitiveSequence(
            name="CubeStackingSequence",
            primitives=[
                GoToPose(name="Pregrasp", env=self.env, arm=arm, target_pose=pregrasp_pose),
                GoToPose(name="Grasp", env=self.env, arm=arm, target_pose=grasp_pose),
                GripperAction(name="CloseGripper", env=self.env, arm=arm, cmd=255.0),
                GoToPose(name="Lift", env=self.env, arm=arm, target_pose=pregrasp_pose),
                GoToPose(name="Place", env=self.env, arm=arm, target_pose=place_pose),
                GripperAction(name="OpenGripper", env=self.env, arm=arm, cmd=0.0),
            ],
        )
        self.primitive_seq.reset()

    def is_done(self) -> bool:
        """Check if the current primitive sequence is done."""
        if self.primitive_seq is not None:
            return self.primitive_seq.is_done()
        return False


if __name__ == "__main__":
    env_config = SingleArmEnvConfig(
        mjcf_path="models/mjcf/cube_stacking.xml",
        object_names=["red_cube", "green_cube"],
        object_min_distance_x=0.1,
        object_min_distance_y=0.1,
        object_x_bounds=(0, 0.3),
        object_y_bounds=(-0.3, 0.3),
        object_rz_bounds=(0, np.pi / 2),
        arm_body_name="xarm7",
        gripper_body_name="xarm_gripper_base_link",
        gripper_act_name="gripper",
        tcp_site_name="link_tcp",
        arm_num_dofs=7,
    )
    env = SingleArmEnv(env_config)

    runner = CubeStackingEnvRunner(env, render_dt=0.02)
    runner.run()
