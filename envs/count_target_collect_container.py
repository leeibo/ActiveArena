import numpy as np
import sapien.core as sapien

from ._base_task import Base_Task
from ._info_task_helpers import PolarCountLayoutMixin
from .utils import *


class count_target_collect_container(PolarCountLayoutMixin, Base_Task):
    """Count target blocks, then collect exactly those targets into a container."""

    ROTATE_TABLE_SHAPE = "fan"
    ROTATE_LOWER_LAYER_KEEP_HEAD_HOME = True
    TARGET_COLOR = (0.10, 0.80, 0.20)
    DISTRACTOR_COLORS = (
        (0.90, 0.20, 0.20),
        (0.20, 0.45, 0.92),
        (0.92, 0.74, 0.18),
    )
    BLOCK_HALF_SIZE = 0.021
    TARGET_COUNT_RANGE = (1, 3)
    DISTRACTOR_COUNT = 3
    TARGET_COUNT_OVERRIDE_CONFIG_KEY = "count_collect_target_count_override"
    COUNT_OBJECT_RLIM = (0.46, 0.66)
    COUNT_OBJECT_THETA_RATIO = 0.58
    COUNT_SLOT_COUNT = 7

    CONTAINER_LABEL = "collection bin"
    CONTAINER_R = 0.55
    # Keep the bin visually separated from the counted objects.  Earlier
    # prototypes placed the bin near the first two target slots, which made
    # x>1 demos accidentally succeed before collection and caused the second
    # grasp to collide with the already placed block.
    CONTAINER_THETA = 0.90
    CONTAINER_HALF_SIZE = (0.10, 0.08, 0.010)
    CONTAINER_WALL_THICKNESS = 0.012
    CONTAINER_WALL_HEIGHT = 0.055
    CONTAINER_RELEASE_Z = 0.83
    CONTAINER_SLOT_OFFSETS = (
        (-0.025, 0.000),
        (0.025, 0.000),
        (0.000, 0.030),
    )
    CONTAINER_SUCCESS_MARGIN = 0.035

    SCAN_R = 0.62
    SCAN_Z_BIAS = 0.88
    SCAN_JOINT_NAME = "astribot_torso_joint_2"
    ROTATE_SCAN_SCENE_FALLBACK_THETAS = (0.72, 0.24, -0.24, -0.72)

    PICK_ARM = "left"
    PICK_PRE_GRASP_DIS = 0.09
    PICK_GRASP_DIS = 0.01
    PICK_LIFT_Z = 0.12
    PLACE_PRE_LIFT_Z = 0.08
    PLACE_RETREAT_Z = 0.10

    def setup_demo(self, **kwargs):
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        override = kwargs.get(self.TARGET_COUNT_OVERRIDE_CONFIG_KEY, None)
        self.count_collect_target_count_override = None if override is None else int(override)
        super()._init_task_env_(**kwargs)

    def _create_count_block(self, key, pose, color, label):
        block = create_box(
            scene=self,
            pose=pose,
            half_size=(self.BLOCK_HALF_SIZE, self.BLOCK_HALF_SIZE, self.BLOCK_HALF_SIZE),
            color=tuple(float(v) for v in color),
            name=str(key),
        )
        block.set_mass(0.03)
        self.object_layers[str(key)] = "lower"
        self.object_labels[str(key)] = str(label)
        self.add_prohibit_area(block, padding=0.04)
        return block

    def _create_container(self):
        point = place_point_cyl(
            [float(self.CONTAINER_R), float(self.CONTAINER_THETA), 0.741],
            robot_root_xy=self.robot_root_xy,
            robot_yaw_rad=self.robot_yaw,
            ret="list",
        )
        center = np.array(point, dtype=np.float64).reshape(3)
        bottom_z = 0.74 + float(self.CONTAINER_HALF_SIZE[2])
        center[2] = bottom_z
        # Use visual-only bin geometry so the information-gathering task is not
        # dominated by fine collision tuning against thin container walls.  A
        # tiny static anchor at the center is registered as object C for
        # rotate-view annotations and target focusing.
        create_visual_box(
            scene=self,
            pose=sapien.Pose(center.tolist(), [1, 0, 0, 0]),
            half_size=self.CONTAINER_HALF_SIZE,
            color=(0.55, 0.55, 0.55),
            name="collection_bin_bottom",
        )
        container = create_box(
            scene=self,
            pose=sapien.Pose([float(center[0]), float(center[1]), 0.745], [1, 0, 0, 0]),
            half_size=(0.012, 0.012, 0.005),
            color=(0.25, 0.25, 0.25),
            is_static=True,
            name="collection_bin_anchor",
        )

        hx, hy, hz = [float(v) for v in self.CONTAINER_HALF_SIZE]
        wall_t = float(self.CONTAINER_WALL_THICKNESS)
        wall_h = float(self.CONTAINER_WALL_HEIGHT)
        wall_z = 0.74 + 2.0 * hz + wall_h
        wall_specs = [
            ([center[0] - hx - wall_t, center[1], wall_z], [wall_t, hy + wall_t, wall_h], "left"),
            ([center[0] + hx + wall_t, center[1], wall_z], [wall_t, hy + wall_t, wall_h], "right"),
            ([center[0], center[1] - hy - wall_t, wall_z], [hx + wall_t, wall_t, wall_h], "front"),
            ([center[0], center[1] + hy + wall_t, wall_z], [hx + wall_t, wall_t, wall_h], "back"),
        ]
        self.container_walls = []
        for wall_center, wall_half_size, suffix in wall_specs:
            wall = create_visual_box(
                scene=self,
                pose=sapien.Pose(wall_center, [1, 0, 0, 0]),
                half_size=wall_half_size,
                color=(0.42, 0.42, 0.42),
                name=f"collection_bin_{suffix}_wall",
            )
            self.container_walls.append(wall)

        self.object_layers["C"] = "lower"
        self.object_labels["C"] = self.CONTAINER_LABEL
        self.add_prohibit_area(container, padding=0.08)
        return container

    def _configure_rotate_subtask_plan(self):
        registry = {}
        for key, block in self.target_blocks.items():
            registry[key] = block
        for key, block in self.distractor_blocks.items():
            registry[key] = block
        registry["C"] = self.container

        subtask_defs = [
            {
                "id": 1,
                "name": "count_green_target_blocks",
                "instruction_idx": 1,
                "search_target_keys": list(self.target_blocks.keys()) + list(self.distractor_blocks.keys()),
                "action_target_keys": [],
                "required_carried_keys": [],
                "carry_keys_after_done": [],
                "allow_stage2_from_memory": False,
                "done_when": "target_count_known",
                "next_subtask_id": 2,
            }
        ]
        next_id = 2
        for idx, block_key in enumerate(self.target_blocks.keys(), start=1):
            pick_id = next_id
            place_id = pick_id + 1
            next_id += 2
            is_last = idx == len(self.target_blocks)
            subtask_defs.extend(
                [
                    {
                        "id": pick_id,
                        "name": f"pick_target_block_{idx}",
                        "instruction_idx": 2,
                        "search_target_keys": [block_key],
                        "action_target_keys": [block_key],
                        "required_carried_keys": [],
                        "carry_keys_after_done": [block_key],
                        "allow_stage2_from_memory": True,
                        "done_when": "target_block_grasped",
                        "next_subtask_id": place_id,
                    },
                    {
                        "id": place_id,
                        "name": f"place_target_block_{idx}_into_container",
                        "instruction_idx": 3,
                        "search_target_keys": ["C"],
                        "action_target_keys": [block_key, "C"],
                        "required_carried_keys": [block_key],
                        "carry_keys_after_done": [],
                        "allow_stage2_from_memory": True,
                        "done_when": "target_block_in_container",
                        "next_subtask_id": -1 if is_last else next_id,
                    },
                ]
            )

        self.configure_rotate_subtask_plan(
            object_registry=registry,
            subtask_defs=subtask_defs,
            task_instruction=(
                "Count the green target blocks on the table, then collect exactly "
                "those target blocks into the collection bin."
            ),
        )

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.object_layers = {}
        self.object_labels = {}
        self.container = self._create_container()

        if getattr(self, "count_collect_target_count_override", None) is None:
            self.target_count = int(np.random.randint(self.TARGET_COUNT_RANGE[0], self.TARGET_COUNT_RANGE[1] + 1))
        else:
            lo, hi = self.TARGET_COUNT_RANGE
            self.target_count = int(np.clip(int(self.count_collect_target_count_override), int(lo), int(hi)))

        self.target_blocks = {}
        self.distractor_blocks = {}
        self.used_pick_arms = []
        slots = self._polar_count_slots(int(self.target_count) + int(self.DISTRACTOR_COUNT))
        r_min = min(float(slot[0]) for slot in slots)

        def target_slot_key(slot):
            theta = float(slot[1])
            theta_rank = 0 if abs(theta) < 1e-6 else (1 if theta > 0.0 else 2)
            return (abs(float(slot[0]) - r_min), theta_rank, abs(theta))

        target_slots = sorted(slots, key=target_slot_key)[: self.target_count]
        distractor_slots = [slot for slot in slots if slot not in target_slots][: int(self.DISTRACTOR_COUNT)]

        for idx, (r, theta) in enumerate(target_slots, start=1):
            key = f"T{idx}"
            self.target_blocks[key] = self._create_count_block(
                key=key,
                pose=self._block_pose_from_cyl(r, theta),
                color=self.TARGET_COLOR,
                label="green target block",
            )

        for idx, (r, theta) in enumerate(distractor_slots, start=1):
            key = f"D{idx}"
            color = self.DISTRACTOR_COLORS[(idx - 1) % len(self.DISTRACTOR_COLORS)]
            self.distractor_blocks[key] = self._create_count_block(
                key=key,
                pose=self._block_pose_from_cyl(r, theta),
                color=color,
                label="distractor block",
            )

        self._configure_rotate_subtask_plan()

    def _get_rotate_object_layer(self, object_key):
        return self.object_layers.get(str(object_key), "lower")

    def _scan_all_blocks_for_count(self):
        scene_objects = list(self.target_blocks.values()) + list(self.distractor_blocks.values())
        self._scan_count_objects_fixed_head(
            subtask_idx=1,
            target_keys=list(self.target_blocks.keys()),
            all_keys=list(self.target_blocks.keys()) + list(self.distractor_blocks.keys()),
            objects=scene_objects,
        )

    def _container_target_pose(self, slot_idx):
        center = np.array(self.container.get_pose().p, dtype=np.float64).reshape(3)
        slot = self.CONTAINER_SLOT_OFFSETS[int(slot_idx) % len(self.CONTAINER_SLOT_OFFSETS)]
        # Offset in world XY is sufficient for this tabletop demo; the basket is
        # near the front-left and the slots stay inside its visible opening.
        target = center.copy()
        target[0] += float(slot[0])
        target[1] += float(slot[1])
        target[2] = float(self.CONTAINER_RELEASE_Z)
        return target.tolist() + [1.0, 0.0, 0.0, 0.0]

    def _current_arm_ee_pose(self, arm_tag):
        if ArmTag(arm_tag) == "left":
            return np.array(self.robot.get_left_ee_pose(), dtype=np.float64).reshape(7)
        return np.array(self.robot.get_right_ee_pose(), dtype=np.float64).reshape(7)

    def _move_carried_block_center_to(self, arm_tag, block, target_center):
        target_center = np.array(target_center, dtype=np.float64).reshape(3)
        ee_pose = self._current_arm_ee_pose(arm_tag)
        block_center = np.array(block.get_pose().p, dtype=np.float64).reshape(3)
        target_ee_pose = ee_pose.copy()
        target_ee_pose[:3] += target_center - block_center
        # Release from above the basket instead of driving into the basket
        # volume.  The latter is often rejected by cuRobo because the static
        # basket collision is directly below the gripper, while a high release
        # keeps the path collision-free and lets the block drop into the
        # container.
        target_ee_pose[2] += float(self.PLACE_PRE_LIFT_Z)
        if not self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=target_ee_pose.tolist())):
            return False
        return True

    def _pick_arm_for_block(self, block):
        theta = float(world_to_robot(block.get_pose().p.tolist(), self.robot_root_xy, self.robot_yaw)[1])
        return ArmTag("right" if theta < -1e-6 else self.PICK_ARM)

    def _pick_target_block(self, subtask_idx, block_key):
        block = self.target_blocks[str(block_key)]
        arm_tag = self._pick_arm_for_block(block)
        arm_name = str(arm_tag)
        if arm_name not in self.used_pick_arms:
            self.used_pick_arms.append(arm_name)
        self._focus_lower_object_fixed_head(
            subtask_idx=subtask_idx,
            object_key=block_key,
            obj=block,
            search_target_keys=[block_key],
            action_target_keys=[block_key],
        )
        self.enter_rotate_action_stage(subtask_idx, focus_object_key=block_key)
        if not self.move(
            self.grasp_actor(
                block,
                arm_tag=arm_tag,
                pre_grasp_dis=float(self.PICK_PRE_GRASP_DIS),
                grasp_dis=float(self.PICK_GRASP_DIS),
            )
        ):
            return arm_tag
        self._set_carried_object_keys([block_key])
        if not self.move(self.move_by_displacement(arm_tag=arm_tag, z=float(self.PICK_LIFT_Z))):
            return arm_tag
        self.complete_rotate_subtask(subtask_idx, carried_after=[block_key])
        return arm_tag

    def _place_target_block(self, arm_tag, subtask_idx, block_key, slot_idx):
        block = self.target_blocks[str(block_key)]
        self._focus_lower_object_fixed_head(
            subtask_idx=subtask_idx,
            object_key="C",
            obj=self.container,
            search_target_keys=["C"],
            action_target_keys=[block_key, "C"],
        )
        self.enter_rotate_action_stage(subtask_idx, focus_object_key="C")
        target_pose = np.array(self._container_target_pose(slot_idx), dtype=np.float64).reshape(7)
        if not self._block_in_container(block):
            if not self._move_carried_block_center_to(arm_tag, block, target_pose[:3]):
                return False
        if not self.move(self.open_gripper(arm_tag)):
            return False
        self._set_carried_object_keys([])
        if not self.move(self.back_to_origin(arm_tag)):
            return False
        self.delay(10)
        self.complete_rotate_subtask(subtask_idx, carried_after=[])
        return True

    def play_once(self):
        self._scan_all_blocks_for_count()
        if not self.plan_success:
            self.info["info"] = self._build_info()
            return self.info

        for idx, block_key in enumerate(list(self.target_blocks.keys())):
            pick_subtask_idx = 2 + 2 * idx
            place_subtask_idx = pick_subtask_idx + 1
            arm_tag = self._pick_target_block(pick_subtask_idx, block_key)
            if not self.plan_success:
                break
            if not self._place_target_block(arm_tag, place_subtask_idx, block_key, idx):
                self.plan_success = False
                break
        self.info["info"] = self._build_info()
        return self.info

    def _build_info(self):
        used_arms = getattr(self, "used_pick_arms", None) or [self.PICK_ARM]
        return {
            "{A}": "green target blocks",
            "{B}": self.CONTAINER_LABEL,
            "{a}": "/".join(used_arms),
            "{x}": int(getattr(self, "target_count", 0)),
        }

    def _block_in_container(self, block):
        block_pose = np.array(block.get_pose().p, dtype=np.float64).reshape(3)
        center = np.array(self.container.get_pose().p, dtype=np.float64).reshape(3)
        half_size = np.array(self.CONTAINER_HALF_SIZE[:2], dtype=np.float64).reshape(2)
        margin = float(self.CONTAINER_SUCCESS_MARGIN)
        return bool(np.all(np.abs(block_pose[:2] - center[:2]) <= half_size + margin))

    def check_success(self):
        targets_ok = all(self._block_in_container(block) for block in self.target_blocks.values())
        distractors_out = all(not self._block_in_container(block) for block in self.distractor_blocks.values())
        gripper_open = self.is_left_gripper_open() and self.is_right_gripper_open()
        return bool(targets_ok and distractors_out and gripper_open)
