from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseEnvConfig:
    mjcf_path: str
    """Path to the MJCF file defining the environment."""

    target_x_bounds: tuple[float, float]
    """Bounds for target placement in the X axis."""

    target_y_bounds: tuple[float, float]
    """Bounds for target placement in the Y axis."""

    target_rz_bounds: tuple[float, float]
    """Bounds for target rotation around the Z axis."""


class BaseEnv(ABC):
    def __init__(self, config: BaseEnvConfig):
        self.mjcf_path = config.mjcf_path
        self.target_x_bounds = config.target_x_bounds
        self.target_y_bounds = config.target_y_bounds
        self.target_rz_bounds = config.target_rz_bounds

    @abstractmethod
    def reset(self):
        """Reset the environment to an initial state."""
        pass

    @abstractmethod
    def step(self):
        """Step the environment."""
        pass
