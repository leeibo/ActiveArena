import hashlib
import json
import numpy as np
import sapien.core as sapien
import transforms3d as t3d
import tempfile
from datetime import date, timedelta
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import left_check_pose
from ._info_task_helpers import BacksidePatchBlockMixin
from .utils import *


class check_cola_date(BacksidePatchBlockMixin, Base_Task):
    """Pick up a 071_can asset, inspect its backside date label, and sort it."""

    ROTATE_TABLE_SHAPE = "fan"
    ROTATE_LOWER_LAYER_KEEP_HEAD_HOME = True

    CAN_MODEL_ID = 3
    CAN_MODEL_NAME = "071_can"
    CAN_QPOS = [0, 0, 0.707, 0.707]
    CAN_Z = 0.755
    BLOCK_HALF_SIZE = 0.048
    CAN_MASS = 0.1
    PICK_PRE_GRASP_DIS = 0.20
    PICK_GRASP_DIS = -0.02
    LIFT_Z = 0.10
    CAN_CYL = (0.43, 0.)
    INSPECT_HOLD_STEPS = 8
    RETURN_AFTER_INSPECT = True

    REFERENCE_DATE = "2026-06-15"
    CURRENT_DATE_RANGE = ("2023-01-01", "2026-12-31")
    SHELF_LIFE_DAYS = 365
    EXPIRED_PRODUCTION_AGE_DAYS = (366, 730)
    VALID_PRODUCTION_AGE_DAYS = (1, 365)
    EXPIRED_SAMPLE_RATE = 0.5

    LABEL_ARC_DEG = 45.0
    LABEL_RADIUS = 0.029
    LABEL_CENTER_DEG = 180.0
    LABEL_FLIP_U = False
    LABEL_FLIP_V = True
    LABEL_VERTICAL_TEXTURE = True
    LABEL_TEXT = "PROD DATE"
    LABEL_FONT_SIZE = 76
    DATE_FONT_SIZE = 150

    PAD_Z = 0.741
    PAD_HALF_SIZE = (0.065, 0.055, 0.0005)
    PAD_TARGETS = {
        "expired": (0.42, 0.4),
        "valid": (0.42, -0.4),
    }
    PAD_COLORS = {
        "expired": (0.90, 0.12, 0.10),
        "valid": (0.10, 0.25, 0.95),
    }
    PAD_KEYS = {
        "expired": "RED",
        "valid": "BLUE",
    }
    PAD_FOOTPRINT_TOL = 0.01
    PLACE_PRE_DIS = 0.07
    PLACE_DIS = 0.01
    PLACE_RETREAT_Z = 0.08

    def setup_demo(self, **kwargs):
        kwargs = prepare_rotate_task_kwargs(self, kwargs)
        self.production_date_override = kwargs.pop("can_production_date_override", None)
        self.current_date_override = kwargs.pop("can_current_date_override", None)
        self.expired_override = kwargs.pop("can_expired_override", None)
        super()._init_task_env_(**kwargs)

    @staticmethod
    def _load_font(size):
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
        return ImageFont.load_default()

    @classmethod
    def _fit_font(cls, text, max_width, max_height, start_size, min_size=10):
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        for size in range(int(start_size), int(min_size) - 1, -1):
            font = cls._load_font(size)
            bbox = probe.textbbox((0, 0), text, font=font)
            if bbox[2] - bbox[0] <= max_width and bbox[3] - bbox[1] <= max_height:
                return font
        return cls._load_font(min_size)

    @classmethod
    def _draw_centered(cls, draw, xyxy, text, font, fill):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x0, y0, x1, y1 = xyxy
        x = x0 + (x1 - x0 - text_w) * 0.5
        y = y0 + (y1 - y0 - text_h) * 0.5 - bbox[1]
        draw.text((x, y), text, font=font, fill=fill)

    @classmethod
    def _make_date_label_texture(cls, path, production_date):
        image = Image.new("RGBA", (768, 512), (246, 247, 239, 255))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((18, 18, 750, 494), radius=34, outline=(32, 32, 32, 255), width=10)
        label_font = cls._fit_font(str(cls.LABEL_TEXT), 680, 108, int(cls.LABEL_FONT_SIZE))
        date_font = cls._fit_font(str(production_date), 680, 266, int(cls.DATE_FONT_SIZE))
        cls._draw_centered(draw, (44, 58, 724, 166), str(cls.LABEL_TEXT), label_font, (32, 32, 32, 255))
        cls._draw_centered(draw, (44, 174, 724, 440), str(production_date), date_font, (10, 10, 10, 255))
        if bool(cls.LABEL_VERTICAL_TEXTURE):
            image = image.rotate(90, expand=True)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path

    @staticmethod
    def _can_asset_file(modeldir, subdir, model_id):
        root = Path(modeldir) / str(subdir)
        for name in (f"base{model_id}.glb", f"textured{model_id}.obj"):
            path = root / name
            if path.exists():
                return path
        return None

    @classmethod
    def _local_cylinder_label_mesh(cls, model_data):
        scale = np.array(model_data.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        center = np.array(model_data.get("center", [0.0, 0.0, 0.0]), dtype=np.float64) * scale
        extents = np.array(model_data.get("extents", [1.0, 1.0, 1.0]), dtype=np.float64) * scale

        radius = float(cls.LABEL_RADIUS)
        y_half = float(extents[1] * 0.38)
        side_angle = np.deg2rad(float(cls.LABEL_CENTER_DEG))
        arc = np.deg2rad(float(cls.LABEL_ARC_DEG))
        theta_segments, y_segments = 24, 4
        thetas = np.linspace(side_angle - arc * 0.5, side_angle + arc * 0.5, theta_segments + 1)
        ys = np.linspace(float(center[1]) - y_half, float(center[1]) + y_half, y_segments + 1)

        vertices, normals, uvs = [], [], []
        for j, y in enumerate(ys):
            v = j / max(1, y_segments)
            for i, theta in enumerate(thetas):
                u = i / max(1, theta_segments)
                uu = u
                vv = v
                if bool(cls.LABEL_FLIP_U):
                    uu = 1.0 - uu
                if bool(cls.LABEL_FLIP_V):
                    vv = 1.0 - vv
                normal = np.array([np.cos(theta), 0.0, np.sin(theta)], dtype=np.float64)
                vertices.append([
                    float(center[0]) + radius * float(normal[0]),
                    float(y),
                    float(center[2]) + radius * float(normal[2]),
                ])
                normals.append(normal.tolist())
                uvs.append([float(uu), float(vv)])

        row = len(thetas)
        triangles = []
        for j in range(y_segments):
            for i in range(theta_segments):
                v00 = j * row + i
                v10 = j * row + i + 1
                v01 = (j + 1) * row + i
                v11 = (j + 1) * row + i + 1
                triangles.extend(([v00, v01, v10], [v10, v01, v11]))
        return vertices, normals, uvs, triangles

    @classmethod
    def _write_label_obj(cls, obj_path, texture_path, model_data):
        vertices, normals, uvs, triangles = cls._local_cylinder_label_mesh(model_data)
        obj_path = Path(obj_path)
        mtl_path = obj_path.with_suffix(".mtl")
        texture_path = Path(texture_path)

        with open(mtl_path, "w", encoding="utf-8") as file:
            file.write("newmtl can_date_label\n")
            file.write("Ka 1.0 1.0 1.0\n")
            file.write("Kd 1.0 1.0 1.0\n")
            file.write("Ks 0.0 0.0 0.0\n")
            file.write("d 1.0\n")
            file.write(f"map_Kd {texture_path.name}\n")

        with open(obj_path, "w", encoding="utf-8") as file:
            file.write(f"mtllib {mtl_path.name}\n")
            file.write("usemtl can_date_label\n")
            for vertex in vertices:
                file.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
            for uv in uvs:
                file.write(f"vt {uv[0]:.8f} {uv[1]:.8f}\n")
            for normal in normals:
                file.write(f"vn {normal[0]:.8f} {normal[1]:.8f} {normal[2]:.8f}\n")
            for tri in triangles:
                a, b, c = [idx + 1 for idx in tri]
                file.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
                file.write(f"f {c}/{c}/{c} {b}/{b}/{b} {a}/{a}/{a}\n")
        return obj_path

    def _label_asset_path(self, production_date, model_data):
        payload = {
            "model_id": int(self.CAN_MODEL_ID),
            "production_date": str(production_date),
            "arc_deg": float(self.LABEL_ARC_DEG),
            "radius": float(self.LABEL_RADIUS),
            "center_deg": float(self.LABEL_CENTER_DEG),
            "flip_u": bool(self.LABEL_FLIP_U),
            "flip_v": bool(self.LABEL_FLIP_V),
            "vertical": bool(self.LABEL_VERTICAL_TEXTURE),
            "scale": model_data.get("scale"),
            "center": model_data.get("center"),
            "extents": model_data.get("extents"),
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        base = Path(tempfile.gettempdir()) / "activearena_can_date_label_task" / digest
        base.mkdir(parents=True, exist_ok=True)
        texture_path = self._make_date_label_texture(base / "production_date.png", production_date)
        obj_path = base / "production_date_label.obj"
        return self._write_label_obj(obj_path, texture_path, model_data)

    def _make_can_asset(self, pose, production_date):
        scene, pose = preprocess(self, pose)
        model_id = int(self.CAN_MODEL_ID)
        modeldir = Path("assets/objects") / self.CAN_MODEL_NAME
        json_file_path = modeldir / f"model_data{model_id}.json"
        with open(json_file_path, "r", encoding="utf-8") as file:
            model_data = json.load(file)
        scale = model_data["scale"]

        collision_file = self._can_asset_file(modeldir, "collision", model_id) or self._can_asset_file(modeldir, ".", model_id)
        visual_file = self._can_asset_file(modeldir, "visual", model_id) or self._can_asset_file(modeldir, ".", model_id)
        if collision_file is None or visual_file is None:
            raise FileNotFoundError(f"Missing {self.CAN_MODEL_NAME}/base{model_id} asset")

        label_file = self._label_asset_path(production_date, model_data)
        builder = scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        builder.add_multiple_convex_collisions_from_file(filename=str(collision_file), scale=scale)
        builder.add_visual_from_file(filename=str(visual_file), scale=scale)
        builder.add_visual_from_file(filename=str(label_file), scale=[1.0, 1.0, 1.0])
        can_entity = builder.build(name="inspect_backside_can")
        can_entity.set_pose(pose)

        can = Actor(can_entity, model_data)
        can.set_mass(float(self.CAN_MASS))
        return can

    def load_actors(self):
        self.robot_root_xy, self.robot_yaw = self._get_robot_root_xy_yaw()
        self.object_layers = {}
        self.object_labels = {}
        self.target_pads = {}
        self.pad_centers = {}
        for label, cyl in self.PAD_TARGETS.items():
            key = self.PAD_KEYS[label]
            pad = self._make_target_pad(label, cyl)
            self.target_pads[label] = pad
            self.pad_centers[label] = np.array(pad.get_pose().p, dtype=np.float64).reshape(3)
            self.object_layers[key] = "lower"
            self.object_labels[key] = f"{'red expired' if label == 'expired' else 'blue valid'} area"

        self.reference_date = self._sample_current_date()
        self.production_date = self._sample_production_date(self.reference_date)
        self.can_expired = self._is_expired(self.production_date, self.reference_date)
        self.target_label = "expired" if self.can_expired else "valid"
        self.target_pad_key = self.PAD_KEYS[self.target_label]
        self.can_backside_inspected = False
        self.can_placed = False
        self.can_id = int(self.CAN_MODEL_ID)
        self.can = self._make_can_asset(
            self._pose_from_cyl(
                self.CAN_CYL,
                z=float(self.CAN_Z),
                qpos=list(self.CAN_QPOS),
                quat_frame="cyl_legacy",
                rotate_rand=False,
                rotate_lim=[0.0, 0.0, 0.0],
            ),
            self.production_date,
        )
        self.object_layers["A"] = "lower"
        self.object_labels["A"] = f"071 can with production date {self.production_date}"
        self.initial_can_z = float(self.can.get_pose().p[2])
        self.add_prohibit_area(self.can, padding=0.06)
        self._configure_rotate_subtask_plan()

    @staticmethod
    def _random_date_between(start_date, end_date):
        start = date.fromisoformat(str(start_date))
        end = date.fromisoformat(str(end_date))
        if end < start:
            raise ValueError(f"Invalid date range: {start_date} > {end_date}")
        offset = int(np.random.randint((end - start).days + 1))
        return start + timedelta(days=offset)

    def _sample_current_date(self):
        if self.current_date_override is not None:
            return str(self.current_date_override)
        return self._random_date_between(*self.CURRENT_DATE_RANGE).isoformat()

    def _sample_expired_label(self):
        if self.expired_override is not None:
            if isinstance(self.expired_override, str):
                return self.expired_override.strip().lower() in {"1", "true", "yes", "y", "expired"}
            return bool(self.expired_override)
        return bool(np.random.rand() < float(self.EXPIRED_SAMPLE_RATE))

    def _sample_production_date(self, reference_date):
        if self.production_date_override is not None:
            return str(self.production_date_override)
        age_range = self.EXPIRED_PRODUCTION_AGE_DAYS if self._sample_expired_label() else self.VALID_PRODUCTION_AGE_DAYS
        age_days = int(np.random.randint(int(age_range[0]), int(age_range[1]) + 1))
        produced = date.fromisoformat(str(reference_date)) - timedelta(days=age_days)
        return produced.isoformat()

    def _is_expired(self, production_date, reference_date=None):
        produced = date.fromisoformat(str(production_date))
        reference = date.fromisoformat(str(reference_date or getattr(self, "reference_date", self.REFERENCE_DATE)))
        return bool(produced + timedelta(days=int(self.SHELF_LIFE_DAYS)) < reference)

    def _make_target_pad(self, label, cyl):
        pose = self._pose_from_cyl(
            cyl,
            z=float(self.PAD_Z),
            qpos=[1.0, 0.0, 0.0, 0.0],
            quat_frame="cyl",
            rotate_rand=False,
            rotate_lim=[0.0, 0.0, 0.0],
        )
        pad = create_box(
            scene=self,
            pose=pose,
            half_size=self.PAD_HALF_SIZE,
            color=self.PAD_COLORS[label],
            name=f"{label}_can_sort_area",
            is_static=True,
        )
        self.add_prohibit_area(pad, padding=0.08)
        return pad

    def _configure_rotate_subtask_plan(self):
        self.configure_rotate_subtask_plan(
            object_registry={
                "A": self.can,
                self.PAD_KEYS["expired"]: self.target_pads["expired"],
                self.PAD_KEYS["valid"]: self.target_pads["valid"],
            },
            subtask_defs=[
                dict(id=1, name="pick_cola_can", instruction_idx=1,
                     search_target_keys=["A"], action_target_keys=["A"], required_carried_keys=[],
                     carry_keys_after_done=["A"], allow_stage2_from_memory=True,
                     done_when="cola_can_grasped", next_subtask_id=2),
                dict(id=2, name="inspect_cola_backside_date", instruction_idx=2,
                     search_target_keys=[], action_target_keys=["A"], required_carried_keys=["A"],
                     carry_keys_after_done=["A"], allow_stage2_from_memory=False,
                     done_when="cola_backside_date_seen", next_subtask_id=3),
                dict(id=3, name="restore_cola_after_inspection", instruction_idx=3,
                     search_target_keys=[], action_target_keys=["A"], required_carried_keys=["A"],
                     carry_keys_after_done=["A"], allow_stage2_from_memory=False,
                     done_when="cola_pose_restored_after_date_inspection", next_subtask_id=4),
                dict(id=4, name="place_cola_by_expiry", instruction_idx=4,
                     search_target_keys=[self.target_pad_key], action_target_keys=["A", self.target_pad_key],
                     required_carried_keys=["A"], carry_keys_after_done=[], allow_stage2_from_memory=True,
                     done_when="cola_sorted_by_expiry", next_subtask_id=-1),
            ],
        )

    def _get_rotate_object_layer(self, object_key):
        return self.object_layers.get(str(object_key), "lower")

    def _pick_and_lift_can(self, subtask_idx):
        self.enter_rotate_action_stage(subtask_idx, focus_object_key="A")
        if not self.move(
            self.grasp_actor(
                self.can,
                arm_tag=self.ARM,
                pre_grasp_dis=float(self.PICK_PRE_GRASP_DIS),
                contact_point_id=12,
                grasp_dis=float(self.PICK_GRASP_DIS),
                gripper_pos=0.2
            )
        ):
            self.plan_success = False
            return False
        self._set_carried_object_keys(["A"])
        if not self.move(self.move_by_displacement(arm_tag=self.ARM, z=float(self.LIFT_Z), move_axis="world")):
            self.plan_success = False
            return False

        self.can_inspection_return_joint_state = np.array(self.robot.get_left_arm_real_jointState()[:-1], dtype=np.float64).tolist()
        self.complete_rotate_subtask(subtask_idx, carried_after=["A"])
        return True

    def _inspect_can_backside(self, subtask_idx):
        if int(getattr(self, "current_subtask_idx", 0)) != int(subtask_idx):
            self.begin_rotate_subtask(subtask_idx)
        self.enter_rotate_action_stage(subtask_idx, focus_object_key="A")
        if not self.move((self.ARM, [Action(self.ARM, "move_joint", target_pose=left_check_pose)])):
            self.plan_success = False
            return False
        self.delay(int(self.INSPECT_HOLD_STEPS), save_freq=1)
        self.complete_rotate_subtask(subtask_idx, carried_after=["A"])
        self.can_backside_inspected = True
        return True

    def _restore_can_after_inspection(self, subtask_idx):
        if int(getattr(self, "current_subtask_idx", 0)) != int(subtask_idx):
            self.begin_rotate_subtask(subtask_idx)
        self.enter_rotate_action_stage(subtask_idx, focus_object_key="A")
        return_state = getattr(self, "can_inspection_return_joint_state", None)
        if return_state is None:
            self.plan_success = False
            return False
        if bool(self.RETURN_AFTER_INSPECT):
            if not self.move((self.ARM, [Action(self.ARM, "move_joint", target_joint_pos=return_state)])):
                self.plan_success = False
                return False

        self.complete_rotate_subtask(subtask_idx, carried_after=["A"])
        return True

    def _place_can_in_target_area(self, subtask_idx):
        pad = self.target_pads[self.target_label]
        pad_pos = np.array(pad.get_pose().p, dtype=np.float64).reshape(3)
        self.enter_rotate_action_stage(subtask_idx, focus_object_key=self.target_pad_key)

        target_pose = pad_pos.copy()
        target_pose[2] = float(pad_pos[2]) + float(self.CAN_Z - self.PAD_Z)
        target_pose = target_pose.tolist() 
        if not self.move(
            self.place_actor(
                self.can,
                arm_tag=self.ARM,
                target_pose=target_pose,
                pre_dis=float(self.PLACE_PRE_DIS),
                dis=float(self.PLACE_DIS),
                constrain="free",
            )
        ):
            self.plan_success = False
            return False
        self._set_carried_object_keys([])
        self.delay(10)
        self.move(self.move_by_displacement(arm_tag=self.ARM, z=float(self.PLACE_RETREAT_Z), move_axis="arm"))
        self.complete_rotate_subtask(subtask_idx, carried_after=[])
        self.can_placed = True
        return True

    def play_once(self):
        found = self.search_and_focus_rotate_subtask(1, scan_r=self.SCAN_R, scan_z=self.SCAN_Z_BIAS + self.table_z_bias, joint_name_prefer=self.SCAN_JOINT_NAME)
        if found is None or not self._pick_and_lift_can(1):
            self.plan_success = False
        else:
            if not self._inspect_can_backside(2):
                self.plan_success = False
            if self.plan_success and not self._restore_can_after_inspection(3):
                self.plan_success = False
        if self.plan_success:
            self.search_and_focus_rotate_subtask(
                4,
                scan_r=self.SCAN_R,
                scan_z=self.SCAN_Z_BIAS + self.table_z_bias,
                joint_name_prefer=self.SCAN_JOINT_NAME,
            )
            if not self._place_can_in_target_area(4):
                self.plan_success = False
        self.info["info"] = {
            "{A}": self._natural_model_label(self.CAN_MODEL_NAME, fallback="cola can"),
            "{B}": "red expired area",
            "{C}": "blue valid area",
            "{D}": f"{self.SHELF_LIFE_DAYS} days",
            "{E}": self.reference_date,
            "{F}": self.production_date,
            "{G}": "expired" if self.can_expired else "valid",
            "{a}": str(self.ARM),
        }
        return self.info

    def check_success(self):
        return bool(
            self._can_center_on_target_pad()
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )

    def _can_center_on_target_pad(self):
        pad_pose = self.target_pads[self.target_label].get_pose()
        can_pos = np.array(self.can.get_pose().p, dtype=np.float64).reshape(3)
        pad_pos = np.array(pad_pose.p, dtype=np.float64).reshape(3)
        pad_rot = t3d.quaternions.quat2mat(np.array(pad_pose.q, dtype=np.float64))
        local_xy = (pad_rot.T @ (can_pos - pad_pos))[:2]
        half_xy = np.array(self.PAD_HALF_SIZE[:2], dtype=np.float64)
        return bool(np.all(np.abs(local_xy) <= half_xy + float(self.PAD_FOOTPRINT_TOL)))
