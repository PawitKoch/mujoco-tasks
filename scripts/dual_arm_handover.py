"""Dual arm handover of foam brick into red bowl."""

import time
from loguru import logger
import mujoco
import mujoco.viewer
import numpy as np
import glfw

from src.envs import BaseEnvRunner, DualArmEnv, DualArmEnvConfig
from src.primitives import PrimitiveSequence


class DualArmHandoverEnvRunner(BaseEnvRunner):
    """
    Orchestrates the dual arm handover task: environment reset, object placement,
    and main execution loop.
    """

    def __init__(self, env: DualArmEnv, render_dt: float = 0.02):
        super().__init__(env, render_dt)
        self.primitive_seq: PrimitiveSequence | None = None

    def setup_episode(self) -> None:
        """Reset environment, place objects, and create primitive sequence."""
        self.env.reset()
        self.randomise_object_positions()
        self.env.forward()

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
    env_config = DualArmEnvConfig(
        mjcf_path="models/mjcf/dual_arm_handover.xml",
        object_names=["foam_brick", "red_bowl"],
        object_min_distance=0.2,
        object_x_bounds=(-0.2, 0.2),
        object_y_bounds=(-0.2, 0.2),
        object_rz_bounds=(0, np.pi / 2),
        left_arm_body_name="left_arm",
        right_arm_body_name="right_arm",
        left_gripper_body_name="left_xarm_gripper_base_link",
        right_gripper_body_name="right_xarm_gripper_base_link",
        left_gripper_act_name="left_gripper",
        right_gripper_act_name="right_gripper",
        left_tcp_site_name="left_link_tcp",
        right_tcp_site_name="right_link_tcp",
    )
    env = DualArmEnv(env_config)

    runner = DualArmHandoverEnvRunner(env, render_dt=0.02)
    runner.run()
