"""Single arm stacking two cubes on a table."""

from loguru import logger
from dataclasses import dataclass
import mujoco
import mujoco.viewer
import numpy as np
import glfw

from src.envs.single_arm_env import SingleArmEnv, SingleArmEnvConfig


@dataclass
class CubeStackingEnvConfig(SingleArmEnvConfig):
    target_cube_names: list[str]
    """Names of the cubes to be stacked."""
    target_cube_min_distance: float = 0.1
    """Minimum distance between cubes on reset (meters)."""


class CubeStackingEnvRunner:
    def __init__(self, env: SingleArmEnv, render_dt: float = 0.02):
        self.env: SingleArmEnv = env
        self.render_dt = render_dt
        self.dt = self.env.model.opt.timestep

        self.cube_body_ids = [
            mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.env.config.target_cube_names
        ]
        self.cube_jnt_adrs = [
            self.env.model.jnt_qposadr[self.env.model.body_jntadr[body_id]] for body_id in self.cube_body_ids
        ]

    def _generate_random_xy_rz(self):
        x = np.random.uniform(*self.env.config.target_x_bounds)
        y = np.random.uniform(*self.env.config.target_y_bounds)
        rz = np.random.uniform(*self.env.config.target_rz_bounds)
        return np.array([x, y, rz])

    def _sample_cube_positions(self):
        min_dist = self.env.config.target_cube_min_distance
        positions = []
        for cube_name in self.env.config.target_cube_names:
            max_attempts = 10
            candidate = None
            for _ in range(max_attempts):
                candidate = self._generate_random_xy_rz()
                min_ok = all(np.linalg.norm(candidate[:2] - np.array(p)[:2]) >= min_dist for p in positions)
                if min_ok:
                    break
            else:
                logger.warning(
                    "Could not find valid position for %s after %d attempts, using last candidate.",
                    cube_name,
                    max_attempts,
                )
            positions.append(candidate)
        return positions

    def reset(self):
        self.env.reset()
        positions = self._sample_cube_positions()
        for body_id, jnt_adr, candidate in zip(self.cube_body_ids, self.cube_jnt_adrs, positions):
            pos = self.env.model.body_pos[body_id].copy()
            pos[0], pos[1] = candidate[0], candidate[1]
            quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, np.array([0, 0, 1]), candidate[2])
            self.env.data.qpos[jnt_adr : jnt_adr + 3] = pos
            self.env.data.qpos[jnt_adr + 3 : jnt_adr + 7] = quat

    def is_done(self) -> bool:
        return False

    def start(self):
        episodes = 0
        reset_requested = False

        def keyboard_callback(keycode):
            nonlocal reset_requested
            if keycode == glfw.KEY_R:
                reset_requested = True

        with mujoco.viewer.launch_passive(self.env.model, self.env.data, key_callback=keyboard_callback) as viewer:
            while viewer.is_running():
                self.reset()
                sim_time = 0.0
                last_render_time = 0.0
                while not self.is_done():
                    self.env.step()
                    sim_time += self.dt
                    if sim_time - last_render_time >= self.render_dt:
                        viewer.sync()
                        last_render_time = sim_time
                    if reset_requested:
                        logger.info("Manual reset requested. Starting new episode.")
                        reset_requested = False
                        break

                episodes += 1
                logger.info("Episode {} finished", episodes)


if __name__ == "__main__":
    env_config = CubeStackingEnvConfig(
        mjcf_path="models/mjcf/cube_stacking.xml",
        target_x_bounds=(0, 0.3),
        target_y_bounds=(-0.3, 0.3),
        target_rz_bounds=(-np.pi, np.pi),
        arm_name="xarm7",
        gripper_name="gripper",
        ee_site_name="link_tcp",
        target_cube_names=["red_cube", "green_cube"],
        target_cube_min_distance=0.1,
    )
    env = SingleArmEnv(env_config)

    runner = CubeStackingEnvRunner(env, render_dt=0.02)
    runner.start()
