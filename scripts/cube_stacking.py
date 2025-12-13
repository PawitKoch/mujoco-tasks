"""Single arm stacking two cubes on a table."""

import time
from loguru import logger
from dataclasses import dataclass
import mujoco
import mujoco.viewer
import numpy as np
import glfw

from src.envs.single_arm_env import SingleArmEnv, SingleArmEnvConfig
from src.primitives import PlanAndExecuteTrajectory, PrimitiveSequence, GripperAction


@dataclass
class CubeStackingEnvConfig(SingleArmEnvConfig):
    target_cube_names: list[str]
    """Names of the cubes to be stacked."""
    target_cube_min_distance: float = 0.1
    """Minimum distance between cubes on reset (meters)."""


class CubeStackingEnvRunner:
    """
    Orchestrates the cube stacking task: environment reset, cube placement, target pose computation,
    primitive sequence creation, and main execution loop.
    """

    def __init__(self, env: SingleArmEnv, render_dt: float = 0.02):
        self.env: SingleArmEnv = env
        self.render_dt = render_dt
        self.dt = self.env.model.opt.timestep

        # Precompute cube body and joint addresses
        self.cube_body_name2id = {
            name: mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.env.config.target_cube_names
        }
        self.cube_jnt_adrs = [
            self.env.model.jnt_qposadr[self.env.model.body_jntadr[body_id]]
            for body_id in self.cube_body_name2id.values()
        ]
        self.primitive_seq: PrimitiveSequence | None = None

    def _generate_random_xy_rz(self) -> np.ndarray:
        """Sample a random (x, y, rz) within bounds."""
        x = np.random.uniform(*self.env.config.target_x_bounds)
        y = np.random.uniform(*self.env.config.target_y_bounds)
        rz = np.random.uniform(*self.env.config.target_rz_bounds)
        return np.array([x, y, rz])

    def _sample_cube_positions(self) -> list[np.ndarray]:
        """Sample non-overlapping positions for all cubes."""
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

    def _place_cubes(self, positions: list[np.ndarray]) -> None:
        """Set cube positions and orientations in the simulation state."""
        for body_id, jnt_adr, candidate in zip(self.cube_body_name2id.values(), self.cube_jnt_adrs, positions):
            pos = self.env.data.xpos[body_id].copy()
            pos[0], pos[1] = candidate[0], candidate[1]  # Set x, y
            quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, np.array([0, 0, 1]), candidate[2])  # Set rz
            self.env.data.qpos[jnt_adr : jnt_adr + 3] = pos
            self.env.data.qpos[jnt_adr + 3 : jnt_adr + 7] = quat

    def _compute_pose(self, target_name: str) -> np.ndarray:
        """Compute the 6D target pose given a target cube name, with gripper pointing down."""
        cube_body_id = self.cube_body_name2id[target_name]
        cube_pos = self.env.data.xpos[cube_body_id].copy()
        cube_mat = self.env.data.xmat[cube_body_id].reshape(3, 3)

        # Construct target orientation: gripper Z down, X aligned
        target_mat = np.zeros((3, 3))
        target_mat[:, 0] = cube_mat[:, 0]  # X matches
        target_mat[:, 1] = -cube_mat[:, 1]  # Y inverted
        target_mat[:, 2] = -cube_mat[:, 2]  # Z inverted (Down)

        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, target_mat.flatten())

        # This should be above the cube
        target_xyz = cube_pos.copy()
        return np.concatenate([target_xyz, target_quat])

    def setup_episode(self) -> None:
        """Reset environment, place cubes, compute target pose, and create primitive sequence."""
        self.env.reset()
        positions = self._sample_cube_positions()
        self._place_cubes(positions)
        self.env.forward()
        pregrasp_pose = self._compute_pose("red_cube")
        grasp_pose = pregrasp_pose.copy()
        grasp_pose[2] -= 0.05  # Move down to grasp
        place_pose = self._compute_pose("green_cube")
        place_pose[2] += 0.05  # Move above to place
        self.primitive_seq = PrimitiveSequence(
            [
                PlanAndExecuteTrajectory(self.env, pregrasp_pose),
                PlanAndExecuteTrajectory(self.env, grasp_pose),
                GripperAction(self.env, cmd=255.0),
                PlanAndExecuteTrajectory(self.env, pregrasp_pose),
                PlanAndExecuteTrajectory(self.env, place_pose),
                GripperAction(self.env, cmd=0.0),
            ]
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
    env_config = CubeStackingEnvConfig(
        mjcf_path="models/mjcf/cube_stacking.xml",
        target_x_bounds=(0, 0.3),
        target_y_bounds=(-0.3, 0.3),
        target_rz_bounds=(-np.pi, np.pi),
        arm_body_name="xarm7",
        gripper_body_name="xarm_gripper_base_link",
        gripper_act_name="gripper",
        ee_site_name="link_tcp",
        target_cube_names=["red_cube", "green_cube"],
        target_cube_min_distance=0.1,
    )
    env = SingleArmEnv(env_config)

    runner = CubeStackingEnvRunner(env, render_dt=0.02)
    runner.run()
