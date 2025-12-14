from abc import ABC, abstractmethod
from loguru import logger


class Primitive(ABC):
    """
    Abstract base class for all robot primitives.
    A primitive encapsulates a single, atomic robot action or behavior (e.g., move, grasp, open gripper).
    Subclasses must implement reset, step, and is_done.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def reset(self) -> None:
        """Reset the primitive to its initial state."""
        pass

    @abstractmethod
    def step(self) -> None:
        """Perform one step of the primitive."""
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """Check if the primitive has completed its task."""
        pass


class PrimitiveSequence(Primitive):
    """
    Executes a list of primitives in sequence.
    Each primitive is reset and run to completion before the next starts.
    Useful for composing complex behaviors from atomic actions.
    """

    def __init__(self, name: str, primitives: list[Primitive]):
        super().__init__(name)
        self.primitives: list[Primitive] = primitives
        self.current_prim_idx: int = 0
        self.done = False

    def reset(self) -> None:
        self.current_prim_idx = 0
        self.done = False

        # Only reset the FIRST primitive immediately
        if self.primitives:
            logger.info(f"Starting Primitive 0: {self.primitives[0].name}")
            self.primitives[0].reset()

    def step(self) -> None:
        if self.done or not self.primitives:
            return

        active_prim = self.primitives[self.current_prim_idx]
        result = active_prim.step()

        if active_prim.is_done():
            self.current_prim_idx += 1

            if self.current_prim_idx >= len(self.primitives):
                self.done = True
                logger.info("Primitive Sequence Complete.")
                return result

            next_prim = self.primitives[self.current_prim_idx]
            logger.info(f"Switching to Primitive {self.current_prim_idx}: {next_prim.name}")
            next_prim.reset()

        return result

    def is_done(self) -> bool:
        return self.done
