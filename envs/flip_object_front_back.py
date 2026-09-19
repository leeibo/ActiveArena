import numpy as np
import sapien.core as sapien
import transforms3d as t3d

from ._base_task import Base_Task
from .utils import *


class flip_object_front_back(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"
    ROTATE_SCAN_SCENE_R = 0.62
    ROTATE_SCAN_SCENE_Z_BIAS = 0.88
    ROTATE_SCAN_SCENE_FALLBACK_THETAS = (0.90, -0.90)

    HALF_SIZE = (0.040, 0.018, 0.026)
    MAIN_COLOR = (0.70, 0.70, 0.70)
    FRONT_COLOR = (0.92, 0.16, 0.12)
    BACK_COLOR = (0.12, 0.35, 0.92)

    PICK_PRE_GRASP_DIS = 0.09
    PICK_GRASP_DIS = 0.01
    LIFT_Z = 0.04
    FLIP_CONTACT_POINT_ID = 3
    FLIP_WRIST_JOINT_INDEX = 6
    FLIP_WRIST_DELTA = -np.pi
    FLIP_HOLD_STEPS = 8
    SUCCESS_FRONT_DOT = -0.85

    def setup_demo(self, is_test=False, **kwargs):
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        super()._init_task_env_(is_test=is_test, **kwargs)

    def _create_front_back_box(self, pose):
        scene, pose = preprocess(self, pose)
        entity = sapien.Entity()
        entity.set_name("front_back_flip_object")
        entity.set_pose(pose)

        half_size = np.array(self.HALF_SIZE, dtype=np.float32)
        rigid_component = sapien.physx.PhysxRigidDynamicComponent()
        rigid_component.attach(
            sapien.physx.PhysxCollisionShapeBox(
                half_size=half_size,
                material=scene.default_physical_material,
            )
        )

        render_component = sapien.render.RenderBodyComponent()
        render_component.attach(
            sapien.render.RenderShapeBox(
                half_size,
                sapien.render.RenderMaterial(base_color=[*self.MAIN_COLOR, 1.0]),
            )
        )

        face_half_size = np.array([0.0012, self.HALF_SIZE[1] * 0.82, self.HALF_SIZE[2] * 0.82], dtype=np.float32)
        for sign, color in (
            (1.0, self.FRONT_COLOR),
            (-1.0, self.BACK_COLOR),
        ):
            marker = sapien.render.RenderShapeBox(
                face_half_size,
                sapien.render.RenderMaterial(base_color=[*color, 1.0]),
            )
            marker.set_local_pose(
                sapien.Pose(
                    [sign * (self.HALF_SIZE[0] + 0.0010), 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                )
            )
            render_component.attach(marker)

        entity.add_component(rigid_component)
        entity.add_component(render_component)
        scene.add_entity(entity)

        data = {
            "center": [0.0, 0.0, 0.0],
            "extents": list(self.HALF_SIZE),
            "scale": list(self.HALF_SIZE),
            "target_pose": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            "contact_points_pose": [
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [0.0, 0.0, -1.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            "transform_matrix": np.eye(4).tolist(),
            "functional_matrix": [
                [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -1.0], [0.0, 0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            ],
            "contact_points_description": [],
            "contact_points_group": [[0, 1, 2, 3]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the marked box."],
        }
        return Actor(entity, data)

    def _front_axis_world(self):
        pose = self.flip_object.get_pose()
        rot = t3d.quaternions.quat2mat(np.array(pose.q, dtype=np.float64))
        axis = np.array(rot[:, 0], dtype=np.float64)
        return axis / max(np.linalg.norm(axis), 1e-9)

    def _up_axis_world(self):
        pose = self.flip_object.get_pose()
        rot = t3d.quaternions.quat2mat(np.array(pose.q, dtype=np.float64))
        axis = np.array(rot[:, 2], dtype=np.float64)
        return axis / max(np.linalg.norm(axis), 1e-9)

    def _target_theta(self, point):
        local = world_to_robot(
            np.array(point, dtype=np.float64).reshape(3).tolist(),
            self.robot_root_xy,
            self.robot_yaw,
        )
        return float(local[1])

    def _focus_held_object_for_flip(self, arm_tag):
        focus_point = np.array(self.flip_object.get_pose().p, dtype=np.float64).reshape(3)
        focus_point[2] += 0.02
        self._set_rotate_subtask_state(
            subtask_idx=1,
            stage=2,
            focus_object_key="A",
            search_target_keys=["A"],
            action_target_keys=["A"],
            info_complete=0,
            camera_mode=2,
            camera_target_theta=self._target_theta(focus_point),
        )
        self.delay(4, save_freq=1)

    def _get_arm_joint_state(self, arm_tag):
        if ArmTag(arm_tag) == "left":
            return np.array(self.robot.get_left_arm_real_jointState()[:-1], dtype=np.float64)
        return np.array(self.robot.get_right_arm_real_jointState()[:-1], dtype=np.float64)

    def _make_front_back_flip_joint_target(self, arm_tag):
        target = self._get_arm_joint_state(arm_tag)
        joint_idx = int(self.FLIP_WRIST_JOINT_INDEX)
        if 0 <= joint_idx < target.shape[0]:
            target[joint_idx] += float(self.FLIP_WRIST_DELTA)
        arm_joints = self.robot.left_arm_joints if ArmTag(arm_tag) == "left" else self.robot.right_arm_joints
        for idx, joint in enumerate(arm_joints):
            if idx >= target.shape[0]:
                break
            target[idx] = self.robot._clip_joint_target_to_limits(joint, target[idx])
        return target.tolist()

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.flip_object,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "flip_object_front_back",
                    "instruction_idx": 1,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "object_front_back_flipped",
                    "next_subtask_id": -1,
                }
            ],
        )

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        # Keep the original sampling stream stable while forcing the object
        # into the right-arm workspace.
        np.random.rand()
        side = -1.0
        while True:
            pose = rand_pose_cyl(
                rlim=[0.42, 0.50],
                thetalim=rotate_theta_side(self, side=side),
                zlim=[0.74 + self.HALF_SIZE[2] + 0.002, 0.74 + self.HALF_SIZE[2] + 0.002],
                robot_root_xy=self.robot_root_xy,
                robot_yaw_rad=self.robot_yaw,
                qpos=[1.0, 0.0, 0.0, 0.0],
                rotate_rand=True,
                rotate_lim=[0.0, 0.0, np.pi / 5.0],
            )
            cyl = world_to_robot(pose.p.tolist(), self.robot_root_xy, self.robot_yaw)
            if abs(cyl[1]) >= 0.30:
                break

        self.flip_object = self._create_front_back_box(pose)
        self.flip_object.set_mass(0.025)
        self.initial_front_axis = self._front_axis_world()
        self.initial_up_axis = self._up_axis_world()
        self.initial_object_xy = np.array(self.flip_object.get_pose().p[:2], dtype=np.float64)
        self.initial_object_z = float(self.flip_object.get_pose().p[2])
        self.add_prohibit_area(self.flip_object, padding=0.08)
        self._configure_rotate_subtask_plan()

    def play_once(self):
        self.begin_rotate_subtask(1)
        self._set_rotate_subtask_state(
            subtask_idx=1,
            stage=1,
            focus_object_key="A",
            search_target_keys=["A"],
            action_target_keys=["A"],
            info_complete=0,
            camera_mode=1,
            camera_target_theta=self._target_theta(self.flip_object.get_pose().p),
        )
        self.delay(4, save_freq=1)

        arm_tag = ArmTag("right")

        self.enter_rotate_action_stage(1, focus_object_key="A")
        if not self.move(
            self.grasp_actor(
                self.flip_object,
                arm_tag=arm_tag,
                pre_grasp_dis=float(self.PICK_PRE_GRASP_DIS),
                grasp_dis=float(self.PICK_GRASP_DIS),
                contact_point_id=int(self.FLIP_CONTACT_POINT_ID),
            )
        ):
            self.info["info"] = {"{A}": "front-back marked block", "{a}": str(arm_tag)}
            return self.info

        self._set_carried_object_keys(["A"])
        if not self.move(self.move_by_displacement(arm_tag=arm_tag, z=float(self.LIFT_Z), move_axis="world")):
            self.info["info"] = {"{A}": "front-back marked block", "{a}": str(arm_tag)}
            return self.info

        self._focus_held_object_for_flip(arm_tag)

        if not self.move([Action(arm_tag, "move_joint", target_joint_pos=self._make_front_back_flip_joint_target(arm_tag))]):
            self.info["info"] = {"{A}": "front-back marked block", "{a}": str(arm_tag)}
            return self.info

        self.delay(int(self.FLIP_HOLD_STEPS), save_freq=1)
        self.complete_rotate_subtask(1, carried_after=[])

        self.info["info"] = {
            "{A}": "front-back marked block",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        current_front_axis = self._front_axis_world()
        current_front_xy = np.array(current_front_axis[:2], dtype=np.float64)
        initial_front_xy = np.array(self.initial_front_axis[:2], dtype=np.float64)
        current_front_xy /= max(np.linalg.norm(current_front_xy), 1e-9)
        initial_front_xy /= max(np.linalg.norm(initial_front_xy), 1e-9)
        front_reversed = float(np.dot(current_front_xy, initial_front_xy)) < float(self.SUCCESS_FRONT_DOT)
        return bool(front_reversed)
