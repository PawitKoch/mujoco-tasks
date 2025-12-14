from abc import ABC, abstractmethod
from dataclasses import dataclass
from loguru import logger

import mujoco
import numpy as np


@dataclass
class BaseEnvConfig:
    mjcf_path: str
    """Path to the MJCF file defining the environment."""

    object_names: list[str]
    """Names of the objects in the environment."""

    object_min_distance_x: float
    """Minimum distance between object poses on the X axis."""

    object_min_distance_y: float
    """Minimum distance between object poses on the Y axis."""

    object_x_bounds: tuple[float, float]
    """Bounds for target placement in the X axis."""

    object_y_bounds: tuple[float, float]
    """Bounds for target placement in the Y axis."""

    object_rz_bounds: tuple[float, float]
    """Bounds for target rotation around the Z axis."""


class BaseEnv(ABC):
    def __init__(self, config: BaseEnvConfig):
        self._config = config
        self._model = mujoco.MjModel.from_xml_path(config.mjcf_path)
        self._data = mujoco.MjData(self._model)

        self.object_names = config.object_names
        self.object_min_distance_x = config.object_min_distance_x
        self.object_min_distance_y = config.object_min_distance_y
        self.object_x_bounds = config.object_x_bounds
        self.object_y_bounds = config.object_y_bounds
        self.object_rz_bounds = config.object_rz_bounds

    @abstractmethod
    def reset(self):
        """Reset the environment to an initial state."""
        pass

    def forward(self):
        """Perform forward dynamics step."""
        mujoco.mj_forward(self._model, self._data)

    def step(self):
        """Perform a simulation step."""
        mujoco.mj_step(self._model, self._data)

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    @property
    def config(self):
        return self._config


class BaseEnvRunner(ABC):
    def __init__(self, env: BaseEnv, render_dt: float = 0.02):
        self.env: BaseEnv = env
        self.render_dt = render_dt
        self.dt = self.env.model.opt.timestep

        # Precompute object body and joint addresses
        self.object_body_name2id = {
            name: mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in env.object_names
        }
        self.object_jnt_adrs = [
            self.env.model.jnt_qposadr[self.env.model.body_jntadr[body_id]]
            for body_id in self.object_body_name2id.values()
        ]

    @abstractmethod
    def run(self):
        """Run the main execution loop."""
        pass

    @abstractmethod
    def setup_episode(self):
        """Prepare environment and primitives for a new episode."""
        pass

    def _generate_random_xy_rz(self) -> np.ndarray:
        """Sample a random (x, y, rz) within bounds."""
        x = np.random.uniform(*self.env.object_x_bounds)
        y = np.random.uniform(*self.env.object_y_bounds)
        rz = np.random.uniform(*self.env.object_rz_bounds)
        return np.array([x, y, rz])

    def randomise_object_positions(self) -> None:
        """Sample non-overlapping positions for all objects and place them."""
        positions = []
        for object_name in self.env.object_names:
            max_attempts = 100
            candidate = None
            for _ in range(max_attempts):
                candidate = self._generate_random_xy_rz()
                min_ok = all(
                    abs(candidate[0] - p[0]) >= self.env.object_min_distance_x
                    and abs(candidate[1] - p[1]) >= self.env.object_min_distance_y
                    for p in positions
                )
                if min_ok:
                    break
            else:
                logger.warning(
                    "Could not find valid position for {} after {} attempts, using last candidate.",
                    object_name,
                    max_attempts,
                )
            positions.append(candidate)

        for body_id, jnt_adr, candidate in zip(self.object_body_name2id.values(), self.object_jnt_adrs, positions):
            pos = self.env.data.xpos[body_id].copy()
            pos[0], pos[1] = candidate[0], candidate[1]  # Set x, y
            quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, np.array([0, 0, 1]), candidate[2])  # Set rz
            self.env.data.qpos[jnt_adr : jnt_adr + 3] = pos
            self.env.data.qpos[jnt_adr + 3 : jnt_adr + 7] = quat
