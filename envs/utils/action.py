from typing import Literal
from .transforms import _tolist
import numpy as np
import sapien


class ArmTag:
    _instances = {}

    def __new__(cls, value):
        if isinstance(value, ArmTag):
            return value
        if isinstance(value, str) and value in ["left", "right"]:
            value = value.lower()
            if value in cls._instances:
                return cls._instances[value]
            instance = super().__new__(cls)
            cls._instances[value] = instance
            return instance
        raise ValueError(f"Invalid arm tag: {value}. Must be 'left' or 'right'.")

    def __init__(self, value):
        if isinstance(value, str):
            self.arm = value.lower()

    @property
    def opposite(self):
        return ArmTag("right") if self.arm == "left" else ArmTag("left")

    def __eq__(self, other):
        if isinstance(other, ArmTag):
            return self.arm == other.arm
        if isinstance(other, str):
            return self.arm == other.lower()
        return False

    def __hash__(self):
        return hash(self.arm)

    def __repr__(self):
        return f"ArmTag('{self.arm}')"

    def __str__(self):
        return self.arm


class Action:
    arm_tag: ArmTag
    action: Literal["move", "move_joint", "gripper", "move_head", "move_torso"]
    target_pose: list = None
    target_joint_pos: list = None
    target_gripper_pos: float = None
    target_head_delta: list = None
    target_torso_delta: list = None

    def __init__(
        self,
        arm_tag: ArmTag | Literal["left", "right"],
        action: Literal["move", "move_joint", "open", "close", "gripper", "move_head", "move_torso"],
        target_pose: sapien.Pose | list | np.ndarray = None,
        target_joint_pos: list | np.ndarray | tuple = None,
        target_gripper_pos: float = None,
        target_head_delta: list | np.ndarray | tuple = None,
        target_torso_delta: list | np.ndarray | tuple = None,
        **args,
    ):
        self.arm_tag = ArmTag(arm_tag)
        if action == "move":
            self.action = "move"
            assert (target_pose is not None), "target_pose cannot be None for move action."
            self.target_pose = _tolist(target_pose)
        elif action == "move_joint":
            self.action = "move_joint"
            if target_joint_pos is None:
                target_joint_pos = target_pose
            assert (target_joint_pos is not None), "target_joint_pos cannot be None for move_joint action."
            target_joint_pos = np.array(target_joint_pos, dtype=np.float64).reshape(-1)
            if target_joint_pos.shape[0] == 0:
                raise ValueError("target_joint_pos must contain at least one value for move_joint action.")
            self.target_joint_pos = target_joint_pos.tolist()
        elif action == "move_head":
            self.action = "move_head"
            if target_head_delta is None:
                target_head_delta = target_pose
            assert (target_head_delta is not None), "target_head_delta cannot be None for move_head action."
            target_head_delta = np.array(target_head_delta, dtype=np.float64).reshape(-1)
            if target_head_delta.shape[0] != 2:
                raise ValueError(
                    f"target_head_delta must have 2 values: [delta_joint1, delta_joint2], got shape {target_head_delta.shape}"
                )
            self.target_head_delta = target_head_delta.tolist()
        elif action == "move_torso":
            self.action = "move_torso"
            if target_torso_delta is None:
                target_torso_delta = target_pose
            assert (target_torso_delta is not None), "target_torso_delta cannot be None for move_torso action."
            target_torso_delta = np.array(target_torso_delta, dtype=np.float64).reshape(-1)
            if target_torso_delta.shape[0] != 1:
                raise ValueError(
                    f"target_torso_delta must have 1 value: [delta_joint], got shape {target_torso_delta.shape}"
                )
            self.target_torso_delta = target_torso_delta.tolist()
        else:
            if action == "open":
                self.action = "gripper"
                self.target_gripper_pos = (target_gripper_pos if target_gripper_pos is not None else 1.0)
            elif action == "close":
                self.action = "gripper"
                self.target_gripper_pos = (target_gripper_pos if target_gripper_pos is not None else 0.0)
            elif action == "gripper":
                self.action = "gripper"
            else:
                raise ValueError(
                    f"Invalid action: {action}. Must be one of "
                    "'move', 'move_joint', 'move_head', 'move_torso', 'open', 'close', 'gripper'."
                )
            assert (self.target_gripper_pos is not None), "target_gripper_pos cannot be None for gripper action."
        self.args = args

    def __str__(self):
        result = f"{self.arm_tag}: {self.action}"
        if self.action == "move":
            result += f"({self.target_pose})"
        elif self.action == "move_joint":
            result += f"({self.target_joint_pos})"
        elif self.action == "move_head":
            result += f"({self.target_head_delta})"
        elif self.action == "move_torso":
            result += f"({self.target_torso_delta})"
        else:
            result += f"({self.target_gripper_pos})"
        if self.args:
            result += f"    {self.args}"
        return result
