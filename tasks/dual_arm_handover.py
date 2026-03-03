"""Dual arm handover of foam brick into red bowl."""

import time
from dataclasses import dataclass
from loguru import logger
import mujoco
import mujoco.viewer
import numpy as np
import glfw

from src.components import Robot
from src.envs import BaseEnvRunner, DualArmEnv, DualArmEnvConfig
from src.primitives import Primitive, PrimitiveSequence, GoToPose, GoToJointPosition, GripperAction


@dataclass
class PickPoses:
    """Dataclass to hold pick-related poses."""

    approach: np.ndarray
    grasp: np.ndarray
    lift: np.ndarray


@dataclass
class HandoverPoses:
    """Dataclass to hold handover-related poses."""

    giver_meet: np.ndarray
    giver_retreat: np.ndarray
    receiver_pre: np.ndarray
    receiver_meet: np.ndarray


@dataclass
class PlacePoses:
    """Dataclass to hold place-related poses."""

    pre_place: np.ndarray
    final_place: np.ndarray


@dataclass
class EpisodePlan:
    """Dataclass to hold all episode poses."""

    pick: PickPoses
    handover: HandoverPoses
    place: PlacePoses


class DualArmHandoverEnvRunner(BaseEnvRunner):
    """
    Orchestrates the dual arm handover task, including environment reset, role assignment,
    """

    def __init__(self, env: DualArmEnv, render_dt: float = 0.02):
        super().__init__(env, render_dt)
        self.primitive_seq: PrimitiveSequence | None = None
        self.giver: Robot | None = None
        self.receiver: Robot | None = None

    def _assign_roles(self) -> None:
        """Determines which arm is Giver vs Receiver based on distance to the brick."""
        brick_pos = self.env.data.xpos[self.object_body_name2id["foam_brick"]]
        left_tcp = self.env.left_arm.tcp_pose[:3]
        right_tcp = self.env.right_arm.tcp_pose[:3]

        dist_left = np.linalg.norm(brick_pos - left_tcp)
        dist_right = np.linalg.norm(brick_pos - right_tcp)

        if dist_left < dist_right:
            logger.info("Role Assignment: LEFT=Giver, RIGHT=Receiver")
            self.giver = self.env.left_arm
            self.receiver = self.env.right_arm
        else:
            logger.info("Role Assignment: RIGHT=Giver, LEFT=Receiver")
            self.giver = self.env.right_arm
            self.receiver = self.env.left_arm

    def _compute_target_pose(self, target_name: str) -> np.ndarray:
        """Compute standard top-down grasping pose (Gripper Y aligned with Object X)."""
        body_id = self.object_body_name2id[target_name]
        body_pos = self.env.data.xpos[body_id].copy()
        body_mat = self.env.data.xmat[body_id].reshape(3, 3)

        target_mat = np.zeros((3, 3))
        target_mat[:, 0] = body_mat[:, 1]  # Gripper X aligned with Object Y
        target_mat[:, 1] = body_mat[:, 0]  # Gripper Y aligned with Object X
        target_mat[:, 2] = -body_mat[:, 2]  # Gripper Z inverted (Down)

        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, target_mat.flatten())
        return np.concatenate([body_pos, quat])

    def _compute_handover_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculates orthogonal handover poses (Giver=Vertical, Receiver=Horizontal)."""
        base_giver = self.env.data.xpos[self.giver.body_id]
        base_receiver = self.env.data.xpos[self.receiver.body_id]

        # Meeting Point
        midpoint = (base_giver + base_receiver) / 2
        midpoint[2] += 0.45

        # Orientation Vectors
        direction = base_receiver - base_giver
        direction[2] = 0
        z_giver = direction / np.linalg.norm(direction)  # Points to receiver
        z_receiver = -z_giver  # Points to giver

        # Giver: Fingers Vertical (Closing Y = Up)
        y_giver = np.array([0, 0, 1])
        x_giver = np.cross(y_giver, z_giver)
        mat_giver = np.column_stack((x_giver, y_giver, z_giver))

        # Receiver: Fingers Horizontal (Closing Y = Side)
        y_receiver = np.cross(z_receiver, np.array([0, 0, -1]))
        x_receiver = np.cross(y_receiver, z_receiver)
        mat_receiver = np.column_stack((x_receiver, y_receiver, z_receiver))

        # 3. Final Poses
        q_giver = np.zeros(4)
        q_receiver = np.zeros(4)
        mujoco.mju_mat2Quat(q_giver, mat_giver.flatten())
        mujoco.mju_mat2Quat(q_receiver, mat_receiver.flatten())

        engagement_offset = 0.05
        pose_giver = np.concatenate([midpoint, q_giver])
        pose_receiver = np.concatenate([midpoint + engagement_offset * z_receiver, q_receiver])

        return pose_giver, pose_receiver

    def _calculate_episode_plan(self) -> EpisodePlan:
        """Centralizes all pose calculations for the episode."""
        # Pick Poses
        base_pick = self._compute_target_pose("foam_brick")
        pick_poses = PickPoses(
            approach=base_pick.copy(),
            grasp=base_pick.copy(),
            lift=base_pick.copy(),
        )
        pick_poses.grasp[2] -= 0.05
        pick_poses.lift[2] += 0.1

        # Handover Poses
        h_giver, h_receiver = self._compute_handover_poses()

        # Calculate approach/retreat vectors based on orientation
        mat_h_recv = np.zeros(9)
        mujoco.mju_quat2Mat(mat_h_recv, h_receiver[3:])
        z_axis_recv = mat_h_recv.reshape(3, 3)[:, 2]

        mat_h_giver = np.zeros(9)
        mujoco.mju_quat2Mat(mat_h_giver, h_giver[3:])
        z_axis_giver = mat_h_giver.reshape(3, 3)[:, 2]

        handover_poses = HandoverPoses(
            giver_meet=h_giver,
            giver_retreat=h_giver.copy(),
            receiver_pre=h_receiver.copy(),
            receiver_meet=h_receiver,
        )
        handover_poses.receiver_pre[:3] -= 0.1 * z_axis_recv
        handover_poses.giver_retreat[:3] -= 0.1 * z_axis_giver

        # Place Poses
        base_place = self._compute_target_pose("red_bowl")
        pre_place = base_place.copy()
        pre_place[2] += 0.1  # Above bowl
        place_poses = PlacePoses(pre_place=pre_place, final_place=base_place)
        return EpisodePlan(pick_poses, handover_poses, place_poses)

    def _build_pick_phase(self, poses: EpisodePlan) -> list[Primitive]:
        return [
            GoToPose("Pick_Approach", self.env, self.giver, poses.pick.approach, "rrt"),
            GoToPose("Pick_Lower", self.env, self.giver, poses.pick.grasp, "linear"),
            GripperAction("Pick_Grasp", self.env, self.giver, 255.0),
            GoToPose("Pick_Lift", self.env, self.giver, poses.pick.lift, "linear"),
        ]

    def _build_handover_phase(self, poses: EpisodePlan) -> list[Primitive]:
        return [
            # Move both to positions
            GoToPose("Giver_To_Handover", self.env, self.giver, poses.handover.giver_meet, "rrt"),
            GoToPose("Recv_To_Ready", self.env, self.receiver, poses.handover.receiver_pre, "rrt"),
            # Execute Transfer
            GripperAction("Recv_Open", self.env, self.receiver, 0.0),
            GoToPose("Recv_Approach", self.env, self.receiver, poses.handover.receiver_meet, "linear"),
            GripperAction("Recv_Grasp", self.env, self.receiver, 255.0),
            GripperAction("Giver_Release", self.env, self.giver, 0.0),
            # Retreats
            GoToPose("Giver_Retreat", self.env, self.giver, poses.handover.giver_retreat, "linear"),
        ]

    def _build_place_phase(self, poses: EpisodePlan) -> list[Primitive]:
        safe_qpos = self.env.home_qpos.copy()  # Safe joint configuration before placing
        return [
            GoToJointPosition("Recv_To_Safe", self.env, self.receiver, safe_qpos),
            GoToPose("Recv_To_Preplace", self.env, self.receiver, poses.place.pre_place, "rrt"),
            GripperAction("Recv_Release", self.env, self.receiver, 0.0),
            GoToPose("Recv_Lift", self.env, self.receiver, poses.place.pre_place, "linear"),
        ]

    def setup_episode(self) -> None:
        """Reset environment, assign roles, calc poses, and build primitives."""
        self.env.reset()
        self.randomise_object_positions()
        self.env.forward()

        self._assign_roles()
        plan = self._calculate_episode_plan()
        sequence = []
        sequence.extend(self._build_pick_phase(plan))
        sequence.extend(self._build_handover_phase(plan))
        sequence.extend(self._build_place_phase(plan))

        self.primitive_seq = PrimitiveSequence("DualArmHandover", sequence)
        self.primitive_seq.reset()

    def is_done(self) -> bool:
        return self.primitive_seq.is_done() if self.primitive_seq else False


if __name__ == "__main__":
    env_config = DualArmEnvConfig(
        mjcf_path="models/mjcf/dual_arm_handover.xml",
        object_names=["foam_brick", "red_bowl"],
        object_min_distance_x=0.5,
        object_min_distance_y=0.1,
        object_x_bounds=(-0.5, 0.5),
        object_y_bounds=(-0.25, 0.25),
        object_rz_bounds=(0, np.pi),
        left_arm_body_name="left_arm",
        right_arm_body_name="right_arm",
        left_gripper_body_name="left_xarm_gripper_base_link",
        right_gripper_body_name="right_xarm_gripper_base_link",
        left_gripper_act_name="left_gripper",
        right_gripper_act_name="right_gripper",
        left_tcp_site_name="left_link_tcp",
        right_tcp_site_name="right_link_tcp",
        left_arm_num_dofs=7,
        right_arm_num_dofs=7,
    )
    env = DualArmEnv(env_config)

    runner = DualArmHandoverEnvRunner(env, render_dt=0.02)
    runner.run(env.model, env.data)
