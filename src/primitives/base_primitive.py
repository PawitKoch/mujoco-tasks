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
        self._success: bool = True

    @property
    def success(self) -> bool:
        """Indicates whether the primitive completed successfully."""
        return self._success

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
        self._success = True

        # Only reset the FIRST primitive immediately
        if self.primitives:
            logger.info(f"Starting Primitive 0: {self.primitives[0].name}")
            self.primitives[0].reset()

            if self.primitives[0].is_done() and not self.primitives[0].success:
                self._handle_failure(self.primitives[0])

    def step(self) -> None:
        if self.done or not self.primitives:
            return

        active_prim = self.primitives[self.current_prim_idx]
        active_prim.step()

        if active_prim.is_done():
            if not active_prim.success:  # fail fast
                self._handle_failure(active_prim)
                return

            self.current_prim_idx += 1

            if self.current_prim_idx >= len(self.primitives):
                self.done = True
                self._success = True
                logger.info("Primitive Sequence Complete.")
                return

            next_prim = self.primitives[self.current_prim_idx]
            logger.info(f"Switching to Primitive {self.current_prim_idx}: {next_prim.name}")
            next_prim.reset()

            if next_prim.is_done() and not next_prim.success:  # fail fast for next primitive
                self._handle_failure(next_prim)

    def _handle_failure(self, failed_prim: Primitive) -> None:
        logger.error(f"[{self.name}] Aborted! Primitive '{failed_prim.name}' reported failure.")
        self._success = False
        self.done = True

    def is_done(self) -> bool:
        return self.done
