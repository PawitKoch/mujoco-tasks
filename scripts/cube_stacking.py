"""Single arm stacking two cubes on a table."""

import time
from loguru import logger
import mujoco
import mujoco.viewer
import numpy as np
import glfw

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
        """Compute the 6D target pose given a target cube name, with gripper pointing down."""
        cube_body_id = self.object_body_name2id[target_name]
        cube_pos = self.env.data.xpos[cube_body_id].copy()
        cube_mat = self.env.data.xmat[cube_body_id].reshape(3, 3)

        # Construct target orientation: gripper Z down, X aligned
        target_mat = np.zeros((3, 3))
        target_mat[:, 0] = cube_mat[:, 0]  # X matches
        target_mat[:, 1] = -cube_mat[:, 1]  # Y inverted
        target_mat[:, 2] = -cube_mat[:, 2]  # Z inverted (Down)
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
                GripperAction(name="OpenGripper", env=self.env, arm=arm, cmd=255.0),
                GoToPose(name="Lift", env=self.env, arm=arm, target_pose=pregrasp_pose),
                GoToPose(name="Place", env=self.env, arm=arm, target_pose=place_pose),
                GripperAction(name="CloseGripper", env=self.env, arm=arm, cmd=0.0),
            ],
        )
        self.primitive_seq.reset()

    def is_done(self) -> bool:
        """Check if the current primitive sequence is done."""
        if self.primitive_seq is not None:
            return self.primitive_seq.is_done()
        return False

    def run(self):
        """Main execution loop: runs episodes, handles rendering and resets."""
        episodes = 0
        reset_requested = False

        def keyboard_callback(keycode):
            nonlocal reset_requested
            if keycode == glfw.KEY_R:
                reset_requested = True

        with mujoco.viewer.launch_passive(self.env.model, self.env.data, key_callback=keyboard_callback) as viewer:
            while viewer.is_running():
                self.setup_episode()
                sim_time = 0.0
                last_render_time = 0.0
                episode_wall_start = time.time()
                while not self.is_done():
                    # Step simulation
                    self.primitive_seq.step()
                    self.env.step()
                    sim_time += self.dt

                    # Render
                    if sim_time - last_render_time >= self.render_dt:
                        viewer.sync()
                        last_render_time = sim_time

                    # Sync to real time
                    wall_time_elapsed = time.time() - episode_wall_start
                    if sim_time > wall_time_elapsed:
                        time.sleep(sim_time - wall_time_elapsed)

                    # Check for manual reset
                    if reset_requested:
                        logger.info("Manual reset requested. Starting new episode.")
                        reset_requested = False
                        break

                episodes += 1
                logger.info("Episode {} finished", episodes)


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
    )
    env = SingleArmEnv(env_config)

    runner = CubeStackingEnvRunner(env, render_dt=0.02)
    runner.run()
