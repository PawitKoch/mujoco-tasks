import numpy as np
from loguru import logger

from src.components import Robot
from src.envs import SingleArmEnv
from src.primitives import Primitive


class GripperAction(Primitive):
    """
    Primitive for controlling the gripper (open/close) over a specified duration.
    Interpolates the gripper actuator command from the current value to the target command.
    Useful for open-loop gripper actions in manipulation tasks.
    """

    def __init__(self, name: str, env: SingleArmEnv, arm: Robot, cmd: float, duration=0.5):
        super().__init__(name)
        self.arm = arm
        self.cmd = cmd  # 0 = closed, 255 = open (usually) depends on your gripper
        self.steps = int(duration / env.model.opt.timestep)
        self.cmd_range = np.linspace(self.arm.gripper_width, self.cmd, self.steps)
        self.step_idx = 0
        self.done = False

        # Find the gripper actuator ID
        self.act_id = self.arm.gripper_ctrl_id

    def reset(self):
        self.done = False
        self.step_idx = 0
        self.arm.set_gripper_width(self.cmd_range[self.step_idx])

    def step(self):
        if self.step_idx < self.steps:
            self.arm.set_gripper_width(self.cmd_range[self.step_idx])
            self.step_idx += 1
        else:
            self.arm.set_gripper_width(self.cmd)
            logger.debug("Gripper action complete.")
            self.done = True
        return

    def is_done(self):
        return self.done
