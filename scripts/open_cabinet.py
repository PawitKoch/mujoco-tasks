"""Single arm trying to open a cabinet hinge door."""

import time
from loguru import logger
import mujoco
import mujoco.viewer
import numpy as np
import glfw

from src.envs import BaseEnvRunner, SingleArmEnv, SingleArmEnvConfig


class OpenCabinetEnvRunner(BaseEnvRunner):
    """
    Orchestrates the cabinet opening task: environment reset, cabinet placement, target pose computation,
    primitive sequence creation, and main execution loop.
    """

    def __init__(self, env: SingleArmEnv, render_dt: float = 0.02):
        super().__init__(env, render_dt)

    def setup_episode(self) -> None:
        """Reset environment, place cabinet, compute target pose, and create primitive sequence."""
        self.env.reset()
        self.randomise_object_positions()
        self.env.forward()

    def is_done(self) -> bool:
        """Check if the current primitive sequence is done."""
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
        mjcf_path="models/mjcf/open_cabinet.xml",
        object_names=["hingecab"],
        object_min_distance_x=0.0,
        object_min_distance_y=0.0,
        object_x_bounds=(0.35, 0.45),
        object_y_bounds=(-0.1, 0.1),
        object_rz_bounds=(np.deg2rad(-120), np.deg2rad(-60)),
        arm_body_name="xarm7",
        gripper_body_name="xarm_gripper_base_link",
        gripper_act_name="gripper",
        tcp_site_name="link_tcp",
    )
    env = SingleArmEnv(env_config)

    runner = OpenCabinetEnvRunner(env, render_dt=0.02)
    runner.run()
