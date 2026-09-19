from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np


class place_container_plate_rotate_view(Base_Task):
    ROTATE_TABLE_SHAPE = "fan"
    OBJECT_RLIM = (0.38, 0.47)
    OBJECT_MIN_DISTANCE = 0.15
    OBJECT_SAMPLE_TRIES = 120

    def check_success(self):
        container_pose = self.container.get_pose().p
        target_pose = self.plate.get_pose().p
        eps = np.array([0.05, 0.05, 0.03])
        return (np.all(abs(container_pose[:3] - target_pose) < eps) and self.is_left_gripper_open()
                and self.is_right_gripper_open())

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.plate,
                "B": self.container,
            },
            subtask_defs=[
                {
                    "id": 1,
                    "name": "pick_container",
                    "instruction_idx": 1,
                    "search_target_keys": ["B"],
                    "action_target_keys": ["B"],
                    "required_carried_keys": [],
                    "carry_keys_after_done": ["B"],
                    "allow_stage2_from_memory": True,
                    "done_when": "container_grasped",
                    "next_subtask_id": 2,
                },
                {
                    "id": 2,
                    "name": "place_container_on_plate",
                    "instruction_idx": 2,
                    "search_target_keys": ["A"],
                    "action_target_keys": ["A", "B"],
                    "required_carried_keys": ["B"],
                    "carry_keys_after_done": [],
                    "allow_stage2_from_memory": True,
                    "done_when": "container_on_plate",
                    "next_subtask_id": -1,
                },
            ]
        )

    def setup_demo(self, **kwags):
        kwags = prepare_rotate_task_kwargs(self, kwags)
        super()._init_task_env_(**kwags)

    def _sample_object_pose(self, z=0.741):
        return rand_pose_cyl(
            rlim=list(self.OBJECT_RLIM),
            thetalim=rotate_theta_center(self),
            zlim=[float(z), float(z)],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
        )

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()

        while True:
            container_pose = self._sample_object_pose()
            if abs(container_pose.p[0]) < 0.2:
                continue
            break

        id_list = {"002_bowl": [1, 2, 3, 5], "021_cup": [1, 2, 3, 4, 5, 6, 7]}
        self.actor_name = str(np.random.choice(["002_bowl", "021_cup"]))
        self.container_id = int(np.random.choice(id_list[self.actor_name]))
        self.container = create_actor(
            self,
            pose=container_pose,
            modelname=self.actor_name,
            model_id=self.container_id,
            convex=True,
        )

        self.plate_id = 0
        for _ in range(int(self.OBJECT_SAMPLE_TRIES)):
            plate_pose = self._sample_object_pose()
            if float(np.linalg.norm(container_pose.p[:2] - plate_pose.p[:2])) >= float(self.OBJECT_MIN_DISTANCE):
                break
        else:
            raise RuntimeError("Failed to sample non-overlapping container and plate poses")
        self.plate = create_actor(
            self,
            pose=plate_pose,
            modelname="003_plate",
            scale=[0.025, 0.025, 0.025],
            is_static=True,
            convex=True,
        )
        self.container.set_mass(0.1)
        self.add_prohibit_area(self.container, padding=0.1)
        self.add_prohibit_area(self.plate, padding=0.1)
        self._configure_rotate_subtask_plan()

    def play_once(self):
        container_key = self.search_and_focus_rotate_subtask(
            1,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )

        container_pose = self.container.get_pose().p
        arm_tag = ArmTag("right" if container_pose[0] > 0 else "left")

        self.enter_rotate_action_stage(1, focus_object_key=(container_key or "B"))
        self.move(
            self.grasp_actor(
                self.container,
                arm_tag=arm_tag,
                contact_point_id=[0, 2][int(arm_tag == "left")],
                pre_grasp_dis=0.1,
                grasp_dis=-0.02,
                gripper_pos=-0.1,
            )
        )
        self._set_carried_object_keys(["B"])
        self.move(self.move_by_displacement(arm_tag, z=0.1, move_axis="arm"))
        self.complete_rotate_subtask(1, carried_after=["B"])

        plate_key = self.search_and_focus_rotate_subtask(
            2,
            scan_r=0.62,
            scan_z=0.88 + self.table_z_bias,
            joint_name_prefer="astribot_torso_joint_2",
        )
        target_pose = self.plate.get_functional_point(0)
        self.enter_rotate_action_stage(2, focus_object_key=(plate_key or "A"))
        self.move(
            self.place_actor(
                self.container,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.12,
                dis=0.03,
                constrain="free",
            )
        )
        self._set_carried_object_keys([])
        self.move(self.move_by_displacement(arm_tag, z=0.08, move_axis="arm"))
        self.complete_rotate_subtask(2, carried_after=[])

        self.info["info"] = {
            "{A}": "plate",
            "{B}": self._natural_model_label(self.actor_name, fallback="container"),
            "{a}": str(arm_tag),
        }
        return self.info
