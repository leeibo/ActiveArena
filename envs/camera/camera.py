import sapien.core as sapien
import numpy as np
import pdb
from PIL import Image, ImageColor
import open3d as o3d
import json
import transforms3d as t3d
import cv2
import torch
import yaml
import trimesh
import math
from .._GLOBAL_CONFIGS import CONFIGS_PATH
import os
from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig

try:
    import pytorch3d.ops as torch3d_ops

    def fps(points, num_points=1024, use_cuda=True):
        K = [num_points]
        if use_cuda:
            points = torch.from_numpy(points).cuda()
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.cpu().numpy()
        else:
            points = torch.from_numpy(points)
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.numpy()

        return sampled_points, indices

except:
    print("missing pytorch3d")

    def fps(points, num_points=1024, use_cuda=True):
        print("fps error: missing pytorch3d")
        exit()


class Camera:

    def __init__(self, bias=0, random_head_camera_dis=0, **kwags):
        """ """
        self.pcd_crop = kwags.get("pcd_crop", False)
        self.pcd_down_sample_num = kwags.get("pcd_down_sample_num", 0)
        self.pcd_crop_bbox = kwags.get("bbox", [[-0.6, -0.35, 0.7401], [0.6, 0.35, 2]])
        self.pcd_crop_bbox[0][2] += bias
        self.table_z_bias = bias
        self.random_head_camera_dis = random_head_camera_dis

        self.static_camera_config = []
        self.camera_runtime_specs = {}
        self.camera_args = kwags["camera"]
        self.head_camera_type = kwags["camera"].get("head_camera_type", "D435")
        self.wrist_camera_type = kwags["camera"].get("wrist_camera_type", "D435")

        self.collect_head_camera = kwags["camera"].get("collect_head_camera", True)
        self.collect_wrist_camera = kwags["camera"].get("collect_wrist_camera", True)
        raw_top_view_camera_configs = kwags["camera"].get("top_view_cameras", None)
        self.collect_top_view_camera = bool(
            kwags["camera"].get("collect_top_view_camera", False)
            or kwags.get("data_type", {}).get("top_view", False)
            or raw_top_view_camera_configs is not None
        )
        if raw_top_view_camera_configs is None:
            top_view_camera_config = dict(kwags["camera"].get("top_view_camera", {}))
            top_view_camera_config.setdefault("name", "top_view_camera")
            self.top_view_camera_configs = [top_view_camera_config] if self.collect_top_view_camera else []
        elif isinstance(raw_top_view_camera_configs, dict):
            self.top_view_camera_configs = []
            for camera_name, camera_config in raw_top_view_camera_configs.items():
                camera_config = dict(camera_config or {})
                camera_config.setdefault("name", str(camera_name))
                self.top_view_camera_configs.append(camera_config)
        else:
            self.top_view_camera_configs = []
            for idx, camera_config in enumerate(raw_top_view_camera_configs):
                camera_config = dict(camera_config or {})
                camera_config.setdefault("name", "top_view_camera" if idx == 0 else f"top_view_camera_{idx}")
                self.top_view_camera_configs.append(camera_config)
        self.top_view_cameras = []
        self.top_view_camera_names = []
        # Head-mounted camera (URDF link camera_head/head_camera), moving with robot head.
        self.collect_head_link_camera = kwags["camera"].get("collect_head_link_camera", True)
        self.has_head_link_camera = bool(kwags.get("has_head_link_camera", False))
        self.camera_head = None

        # embodiment = kwags.get('embodiment')
        # embodiment_config_path = os.path.join(CONFIGS_PATH, '_embodiment_config.yml')
        # with open(embodiment_config_path, 'r', encoding='utf-8') as f:
        #     embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
        # robot_file = embodiment_types[embodiment]['file_path']
        # if robot_file is None:
        #     raise "No embodiment files"

        # robot_config_file = os.path.join(robot_file, 'config.yml')
        # with open(robot_config_file, 'r', encoding='utf-8') as f:
        #     embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        # TODO
        self.static_camera_info_list = kwags["left_embodiment_config"]["static_camera_list"]
        self.static_camera_num = len(self.static_camera_info_list)

    def load_camera(self, scene):
        """
        Add cameras and set camera parameters
            - Including four cameras: left, right, front, head.
        """
        near, far = 0.1, 100
        camera_config_path = os.path.join(CONFIGS_PATH, "_camera_config.yml")

        assert os.path.isfile(camera_config_path), "task config file is missing"

        with open(camera_config_path, "r", encoding="utf-8") as f:
            camera_args = yaml.load(f.read(), Loader=yaml.FullLoader)

        # sensor_mount_actor = scene.create_actor_builder().build_kinematic()

        # camera_args = get_camera_config()
        def create_camera(camera_info, random_head_camera_dis=0):
            if camera_info["type"] not in camera_args.keys():
                raise ValueError(f"Camera type {camera_info['type']} not supported")

            camera_config = camera_args[camera_info["type"]]
            cam_pos = np.array(camera_info["position"])
            vector = np.random.randn(3)
            random_dir = vector / np.linalg.norm(vector)
            cam_pos = cam_pos + random_dir * np.random.uniform(low=0, high=random_head_camera_dis)
            cam_forward = np.array(camera_info["forward"]) / np.linalg.norm(np.array(camera_info["forward"]))
            cam_left = np.array(camera_info["left"]) / np.linalg.norm(np.array(camera_info["left"]))
            up = np.cross(cam_forward, cam_left)
            mat44 = np.eye(4)
            mat44[:3, :3] = np.stack([cam_forward, cam_left, up], axis=1)
            mat44[:3, 3] = cam_pos

            # ========================= sensor camera =========================
            # sensor_config = StereoDepthSensorConfig()
            # sensor_config.rgb_resolution = (camera_config['w'], camera_config['h'])

            camera = scene.add_camera(
                name=camera_info["name"],
                width=camera_config["w"],
                height=camera_config["h"],
                fovy=np.deg2rad(camera_config["fovy"]),
                near=near,
                far=far,
            )
            camera.entity.set_pose(sapien.Pose(mat44))

            # ========================= sensor camera =========================
            # sensor_camera = StereoDepthSensor(
            #     sensor_config,
            #     sensor_mount_actor,
            #     sapien.Pose(mat44)
            # )
            # camera.entity.set_pose(sapien.Pose(camera_info['position']))
            # return camera, sensor_camera, camera_config
            return camera, camera_config

        def register_camera_spec(camera_name, camera_config):
            if camera_config is None:
                return
            self.camera_runtime_specs[str(camera_name)] = {
                "w": int(camera_config["w"]),
                "h": int(camera_config["h"]),
                "fovy_deg": float(camera_config["fovy"]),
                "fovy_rad": float(np.deg2rad(camera_config["fovy"])),
                "near": float(near),
                "far": float(far),
            }

        # ================================= wrist camera =================================
        if self.collect_wrist_camera:
            wrist_camera_config = camera_args[self.wrist_camera_type]
            self.left_camera = scene.add_camera(
                name="left_camera",
                width=wrist_camera_config["w"],
                height=wrist_camera_config["h"],
                fovy=np.deg2rad(wrist_camera_config["fovy"]),
                near=near,
                far=far,
            )
            self.right_camera = scene.add_camera(
                name="right_camera",
                width=wrist_camera_config["w"],
                height=wrist_camera_config["h"],
                fovy=np.deg2rad(wrist_camera_config["fovy"]),
                near=near,
                far=far,
            )
            register_camera_spec("left_camera", wrist_camera_config)
            register_camera_spec("right_camera", wrist_camera_config)

        if self.collect_head_link_camera and self.has_head_link_camera:
            head_link_camera_config = camera_args[self.head_camera_type]
            self.camera_head = scene.add_camera(
                name="camera_head",
                width=head_link_camera_config["w"],
                height=head_link_camera_config["h"],
                fovy=np.deg2rad(head_link_camera_config["fovy"]),
                near=near,
                far=far,
            )
            register_camera_spec("camera_head", head_link_camera_config)

        # ================================= sensor camera =================================
        # sensor_config = StereoDepthSensorConfig()
        # sensor_config.rgb_resolution = (wrist_camera_config['w'], wrist_camera_config['h'])
        # self.left_sensor_camera = StereoDepthSensor(
        #     sensor_config,
        #     sensor_mount_actor,
        #     sapien.Pose([0,0,0],[1,0,0,0])
        # )

        # self.right_sensor_camera = StereoDepthSensor(
        #     sensor_config,
        #     sensor_mount_actor,
        #     sapien.Pose([0,0,0],[1,0,0,0])
        # )

        # ================================= static camera =================================
        self.head_camera_id = None
        self.static_camera_list = []
        # self.static_sensor_camera_list = []
        self.static_camera_name = []
        # static camera list
        for i, camera_info in enumerate(self.static_camera_info_list):
            if camera_info.get("forward") == None:
                camera_info["forward"] = (-1 * np.array(camera_info["position"])).tolist()
            if camera_info.get("left") == None:
                camera_info["left"] = [
                    -camera_info["forward"][1],
                    camera_info["forward"][0],
                ] + [0]

            if camera_info["name"] == "camera_head" and self.camera_head is not None:
                print("[Camera] skip static camera_head; using URDF head-mounted camera_head instead")
                continue

            if camera_info["name"] == "head_camera":
                if self.collect_head_camera:
                    self.head_camera_id = i
                    camera_info["type"] = self.head_camera_type
                    # camera, sensor_camera, camera_config = create_camera(camera_info)
                    camera, camera_config = create_camera(camera_info,
                                                          random_head_camera_dis=self.random_head_camera_dis)
                    self.static_camera_list.append(camera)
                    self.static_camera_name.append(camera_info["name"])
                    # self.static_sensor_camera_list.append(sensor_camera)
                    self.static_camera_config.append(camera_config)
                    register_camera_spec(camera_info["name"], camera_config)
                    # ================================= sensor camera =================================
                    # camera_config = get_camera_config(camera_info['type'])
                    # cam_pos = np.array(camera_info['position'])
                    # cam_forward = np.array(camera_info['forward']) / np.linalg.norm(np.array(camera_info['forward']))
                    # cam_left = np.array(camera_info['left']) / np.linalg.norm(np.array(camera_info['left']))
                    # up = np.cross(cam_forward, cam_left)
                    # mat44 = np.eye(4)
                    # mat44[:3, :3] = np.stack([cam_forward, cam_left, up], axis=1)
                    # mat44[:3, 3] = cam_pos
                    # sensor_config = StereoDepthSensorConfig()
                    # sensor_config.rgb_resolution = (camera_config['w'], camera_config['h'])

                    # self.head_sensor = StereoDepthSensor(
                    #     sensor_config,
                    #     sensor_mount_actor,
                    #     sapien.Pose(mat44)
                    # )
            else:
                # camera, sensor_camera, camera_config = create_camera(camera_info)
                camera, camera_config = create_camera(camera_info)
                self.static_camera_list.append(camera)
                self.static_camera_name.append(camera_info["name"])
                # self.static_sensor_camera_list.append(sensor_camera)
                self.static_camera_config.append(camera_config)
                register_camera_spec(camera_info["name"], camera_config)

        # observer camera
        observer_camera_args = dict(self.camera_args.get("observer_camera", {}))
        observer_width = int(observer_camera_args.get("w", 320))
        observer_height = int(observer_camera_args.get("h", 240))
        observer_fovy = float(observer_camera_args.get("fovy", 93))
        observer_near = float(observer_camera_args.get("near", near))
        observer_far = float(observer_camera_args.get("far", far))
        self.observer_camera = scene.add_camera(
            name="observer_camera",
            width=observer_width,
            height=observer_height,
            fovy=np.deg2rad(observer_fovy),
            near=observer_near,
            far=observer_far,
        )
        observer_cam_pos = np.array(observer_camera_args.get("position", [0.0, 0.23, 1.33]), dtype=np.float64)
        observer_cam_forward = np.array(observer_camera_args.get("forward", [0, -1, -1.02]), dtype=np.float64)
        observer_cam_forward = observer_cam_forward / np.linalg.norm(observer_cam_forward)
        # observer_cam_left = np.array([1,-1, 0])
        observer_cam_left = np.array(observer_camera_args.get("left", [1, 0, 0]), dtype=np.float64)
        observer_cam_left = observer_cam_left / np.linalg.norm(observer_cam_left)
        observer_up = np.cross(observer_cam_forward, observer_cam_left)
        observer_mat44 = np.eye(4)
        observer_mat44[:3, :3] = np.stack([observer_cam_forward, observer_cam_left, observer_up], axis=1)
        observer_mat44[:3, 3] = observer_cam_pos
        self.observer_camera.entity.set_pose(sapien.Pose(observer_mat44))
        register_camera_spec(
            "observer_camera",
            {
                "w": observer_width,
                "h": observer_height,
                "fovy": observer_fovy,
                "near": observer_near,
                "far": observer_far,
            },
        )

        for top_view_idx, top_view_args in enumerate(self.top_view_camera_configs):
            top_view_name = str(top_view_args.get("name", "top_view_camera" if top_view_idx == 0 else f"top_view_camera_{top_view_idx}"))
            top_view_width = int(top_view_args.get("w", 512))
            top_view_height = int(top_view_args.get("h", 512))
            top_view_fovy = float(top_view_args.get("fovy", 78))
            top_view_camera = scene.add_camera(
                name=top_view_name,
                width=top_view_width,
                height=top_view_height,
                fovy=np.deg2rad(top_view_fovy),
                near=float(top_view_args.get("near", near)),
                far=float(top_view_args.get("far", far)),
            )
            top_view_pos = np.array(top_view_args.get("position", [0.0, 0.0, 2.0]), dtype=np.float64)
            top_view_target = np.array(
                top_view_args.get("target", [0.0, 0.0, 0.74 + self.table_z_bias]),
                dtype=np.float64,
            )
            top_view_forward = top_view_target - top_view_pos
            top_view_forward = top_view_forward / np.linalg.norm(top_view_forward)
            top_view_left = np.array(top_view_args.get("left", [1.0, 0.0, 0.0]), dtype=np.float64)
            top_view_left = top_view_left / np.linalg.norm(top_view_left)
            top_view_up = np.cross(top_view_forward, top_view_left)
            top_view_mat44 = np.eye(4)
            top_view_mat44[:3, :3] = np.stack([top_view_forward, top_view_left, top_view_up], axis=1)
            top_view_mat44[:3, 3] = top_view_pos
            top_view_camera.entity.set_pose(sapien.Pose(top_view_mat44))
            self.top_view_cameras.append(top_view_camera)
            self.top_view_camera_names.append(top_view_name)
            if top_view_idx == 0 or top_view_name == "top_view_camera":
                self.top_view_camera = top_view_camera
            register_camera_spec(
                top_view_name,
                {
                    "w": top_view_width,
                    "h": top_view_height,
                    "fovy": top_view_fovy,
                },
            )

        # world pcd camera
        self.world_camera1 = scene.add_camera(
            name="world_camera1",
            width=640,
            height=480,
            fovy=np.deg2rad(50),
            near=near,
            far=far,
        )
        world_cam_pos = np.array([0.4, -0.4, 1.6])
        world_cam_forward = np.array([-1, 1, -1.4])
        world_cam_left = np.array([-1, -1, 0])
        world_cam_up = np.cross(world_cam_forward, world_cam_left)
        world_cam_mat44 = np.eye(4)
        world_cam_mat44[:3, :3] = np.stack([world_cam_forward, world_cam_left, world_cam_up], axis=1)
        world_cam_mat44[:3, 3] = world_cam_pos
        self.world_camera1.entity.set_pose(sapien.Pose(world_cam_mat44))
        register_camera_spec("world_camera1", {"w": 640, "h": 480, "fovy": 50})

        self.world_camera2 = scene.add_camera(
            name="world_camera2",
            width=640,
            height=480,
            fovy=np.deg2rad(50),
            near=near,
            far=far,
        )
        world_cam_pos = np.array([-0.4, -0.4, 1.6])
        world_cam_forward = np.array([1, 1, -1.4])
        world_cam_left = np.array([-1, 1, 0])
        world_cam_up = np.cross(world_cam_forward, world_cam_left)
        world_cam_mat44 = np.eye(4)
        world_cam_mat44[:3, :3] = np.stack([world_cam_forward, world_cam_left, world_cam_up], axis=1)
        world_cam_mat44[:3, 3] = world_cam_pos
        self.world_camera2.entity.set_pose(sapien.Pose(world_cam_mat44))
        register_camera_spec("world_camera2", {"w": 640, "h": 480, "fovy": 50})

    def update_picture(self):
        # camera
        if self.collect_wrist_camera:
            self.left_camera.take_picture()
            self.right_camera.take_picture()
        if self.camera_head is not None:
            self.camera_head.take_picture()
        for camera in self.top_view_cameras:
            camera.take_picture()

        for camera in self.static_camera_list:
            camera.take_picture()

        # ================================= sensor camera =================================
        # self.head_sensor.take_picture()
        # self.head_sensor.compute_depth()

    def update_wrist_camera(self, left_pose, right_pose, head_pose=None):
        """
        Update rendering to refresh the camera's RGBD information
        (rendering must be updated even when disabled, otherwise data cannot be collected).
        """
        if self.collect_wrist_camera:
            self.left_camera.entity.set_pose(left_pose)
            self.right_camera.entity.set_pose(right_pose)
        if self.camera_head is not None and head_pose is not None:
            self.camera_head.entity.set_pose(head_pose)

    def get_config(self) -> dict:
        res = {}

        def _get_config(camera):
            camera_intrinsic_cv = camera.get_intrinsic_matrix()
            camera_extrinsic_cv = camera.get_extrinsic_matrix()
            camera_model_matrix = camera.get_model_matrix()
            return {
                "intrinsic_cv": camera_intrinsic_cv,
                "extrinsic_cv": camera_extrinsic_cv,
                "cam2world_gl": camera_model_matrix,
            }

        if self.collect_wrist_camera:
            res["left_camera"] = _get_config(self.left_camera)
            res["right_camera"] = _get_config(self.right_camera)
        if self.camera_head is not None:
            res["camera_head"] = _get_config(self.camera_head)
        for camera, camera_name in zip(self.top_view_cameras, self.top_view_camera_names):
            res[camera_name] = _get_config(camera)

        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = _get_config(camera)
            else:
                res[camera_name] = _get_config(camera)
        # ================================= sensor camera =================================
        # res['head_sensor'] = res['head_camera']
        # print(res)
        return res

    def get_config_by_name(self, camera_name) -> dict:
        camera = self.get_camera_by_name(camera_name)
        if camera is None:
            return None
        camera_intrinsic_cv = camera.get_intrinsic_matrix()
        camera_extrinsic_cv = camera.get_extrinsic_matrix()
        camera_model_matrix = camera.get_model_matrix()
        return {
            "intrinsic_cv": camera_intrinsic_cv,
            "extrinsic_cv": camera_extrinsic_cv,
            "cam2world_gl": camera_model_matrix,
        }

    def get_rgb(self) -> dict:
        rgba = self.get_rgba()
        rgb = {}
        for camera_name, camera_data in rgba.items():
            rgb[camera_name] = {}
            rgb[camera_name]["rgb"] = camera_data["rgba"][:, :, :3]  # Exclude alpha channel
        return rgb

    def get_camera_runtime_spec(self, camera_name):
        return self.camera_runtime_specs.get(str(camera_name))

    def get_camera_by_name(self, camera_name):
        camera_name = str(camera_name)
        if camera_name == "left_camera":
            return getattr(self, "left_camera", None)
        if camera_name == "right_camera":
            return getattr(self, "right_camera", None)
        if camera_name == "camera_head":
            return getattr(self, "camera_head", None)
        if camera_name == "observer_camera":
            return getattr(self, "observer_camera", None)
        if camera_name == "world_camera1":
            return getattr(self, "world_camera1", None)
        if camera_name == "world_camera2":
            return getattr(self, "world_camera2", None)
        for camera, name in zip(self.top_view_cameras, self.top_view_camera_names):
            if str(name) == camera_name:
                return camera
        for camera, name in zip(self.static_camera_list, self.static_camera_name):
            if str(name) == camera_name:
                return camera
        return None

    def get_rgb_by_name(self, camera_name) -> dict:
        camera = self.get_camera_by_name(camera_name)
        if camera is None:
            return None
        camera.take_picture()
        camera_rgba = camera.get_picture("Color")
        camera_rgb_img = (camera_rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
        return {"rgb": camera_rgb_img}
    
    # Get Camera RGBA
    def get_rgba(self) -> dict:

        def _get_rgba(camera):
            camera_rgba = camera.get_picture("Color")
            camera_rgba_img = (camera_rgba * 255).clip(0, 255).astype("uint8")
            return camera_rgba_img

        # ================================= sensor camera =================================
        # def _get_sensor_rgba(sensor):
        #     camera_rgba = sensor.get_rgb()
        #     camera_rgba_img = (camera_rgba * 255).clip(0, 255).astype("uint8")[:,:,:3]
        #     return camera_rgba_img

        res = {}

        if self.collect_wrist_camera:
            res["left_camera"] = {}
            res["right_camera"] = {}
            res["left_camera"]["rgba"] = _get_rgba(self.left_camera)
            res["right_camera"]["rgba"] = _get_rgba(self.right_camera)
        if self.camera_head is not None:
            res["camera_head"] = {}
            res["camera_head"]["rgba"] = _get_rgba(self.camera_head)
        for camera, camera_name in zip(self.top_view_cameras, self.top_view_camera_names):
            res[camera_name] = {}
            res[camera_name]["rgba"] = _get_rgba(camera)

        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = {}
                    res[camera_name]["rgba"] = _get_rgba(camera)
            else:
                res[camera_name] = {}
                res[camera_name]["rgba"] = _get_rgba(camera)
        # ================================= sensor camera =================================
        # res['head_sensor']['rgb'] = _get_sensor_rgba(self.head_sensor)

        return res

    def get_observer_rgb(self) -> dict:
        rgb = self.get_rgb_by_name("observer_camera")
        if rgb is None:
            return None
        return rgb["rgb"]

    # Get Camera Segmentation
    def get_segmentation(self, level="mesh") -> dict:

        def _get_segmentation(camera, level="mesh"):
            # visual_id is the unique id of each visual shape
            seg_labels = camera.get_picture("Segmentation")  # [H, W, 4]
            colormap = sorted(set(ImageColor.colormap.values()))
            color_palette = np.array([ImageColor.getrgb(color) for color in colormap], dtype=np.uint8)
            if level == "mesh":
                label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
            elif level == "actor":
                label0_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
            return color_palette[label0_image]

        res = {
            # 'left_camera':{},
            # 'right_camera':{}
        }

        if self.collect_wrist_camera:
            res["left_camera"] = {}
            res["right_camera"] = {}
            res["left_camera"][f"{level}_segmentation"] = _get_segmentation(self.left_camera, level=level)
            res["right_camera"][f"{level}_segmentation"] = _get_segmentation(self.right_camera, level=level)
        if self.camera_head is not None:
            res["camera_head"] = {}
            res["camera_head"][f"{level}_segmentation"] = _get_segmentation(self.camera_head, level=level)

        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = {}
                    res[camera_name][f"{level}_segmentation"] = _get_segmentation(camera, level=level)
            else:
                res[camera_name] = {}
                res[camera_name][f"{level}_segmentation"] = _get_segmentation(camera, level=level)
        return res

    # Get Camera Depth
    def get_depth(self) -> dict:

        def _get_depth(camera):
            position = camera.get_picture("Position")
            depth = -position[..., 2]
            depth_image = (depth * 1000.0).astype(np.float64)
            return depth_image

        def _get_sensor_depth(sensor):
            depth = sensor.get_depth()
            depth = (depth * 1000.0).astype(np.float64)
            return depth

        res = {}
        rgba = self.get_rgba()

        if self.collect_wrist_camera:
            res["left_camera"] = {}
            res["right_camera"] = {}
            res["left_camera"]["depth"] = _get_depth(self.left_camera)
            res["right_camera"]["depth"] = _get_depth(self.right_camera)
            res["left_camera"]["depth"] *= rgba["left_camera"]["rgba"][:, :, 3] / 255
            res["right_camera"]["depth"] *= rgba["right_camera"]["rgba"][:, :, 3] / 255
        if self.camera_head is not None:
            res["camera_head"] = {}
            res["camera_head"]["depth"] = _get_depth(self.camera_head)
            res["camera_head"]["depth"] *= rgba["camera_head"]["rgba"][:, :, 3] / 255
        
        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = {}
                    res[camera_name]["depth"] = _get_depth(camera)
                    res[camera_name]["depth"] *= rgba[camera_name]["rgba"][:, :, 3] / 255
            else:
                res[camera_name] = {}
                res[camera_name]["depth"] = _get_depth(camera)
                res[camera_name]["depth"] *= rgba[camera_name]["rgba"][:, :, 3] / 255
        # res['head_sensor']['depth'] = _get_sensor_depth(self.head_sensor)

        return res

    # Get World PointCloud
    def get_world_pcd(self):
        self.world_camera1.take_picture()
        self.world_camera2.take_picture()

        def _get_camera_pcd(camera, color=True):
            rgba = camera.get_picture_cuda("Color").torch()  # [H, W, 4]
            position = camera.get_picture_cuda("Position").torch()
            model_matrix = camera.get_model_matrix()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_matrix = torch.tensor(model_matrix, dtype=torch.float32).to(device)

            # Extract valid three-dimensional points and corresponding color data.
            valid_mask = position[..., 3] < 1
            points_opengl = position[..., :3][valid_mask]
            points_color = rgba[valid_mask][:, :3]
            # Transform into the world coordinate system.
            points_world = (torch.bmm(
                points_opengl.view(1, -1, 3),
                model_matrix[:3, :3].transpose(0, 1).view(-1, 3, 3),
            ).squeeze(1) + model_matrix[:3, 3])

            # Format color data.
            points_color = torch.clamp(points_color, 0, 1)
            points_world = points_world.squeeze(0)

            # Convert the tensor back to a NumPy array for use with Open3D.
            points_world_np = points_world.cpu().numpy()
            points_color_np = points_color.cpu().numpy()
            # print(points_world_np.shape, points_color_np.shape)

            res_pcd = (np.hstack((points_world_np, points_color_np)) if color else points_world_np)
            return res_pcd

        pcd1 = _get_camera_pcd(self.world_camera1, color=True)
        pcd2 = _get_camera_pcd(self.world_camera2, color=True)
        res_pcd = np.vstack((pcd1, pcd2))

        return res_pcd
        pcd_array, index = fps(res_pcd[:, :3], 2000)
        index = index.detach().cpu().numpy()[0]

        return pcd_array

    # Get Camera PointCloud
    def get_pcd(self, if_combine=False):

        def _get_camera_pcd(camera, point_num=0):
            rgba = camera.get_picture_cuda("Color").torch()  # [H, W, 4]
            position = camera.get_picture_cuda("Position").torch()
            model_matrix = camera.get_model_matrix()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_matrix = torch.tensor(model_matrix, dtype=torch.float32).to(device)

            # Extract valid three-dimensional points and corresponding color data.
            valid_mask = position[..., 3] < 1
            points_opengl = position[..., :3][valid_mask]
            points_color = rgba[valid_mask][:, :3]
            # Transform into the world coordinate system.
            points_world = (torch.bmm(
                points_opengl.view(1, -1, 3),
                model_matrix[:3, :3].transpose(0, 1).view(-1, 3, 3),
            ).squeeze(1) + model_matrix[:3, 3])

            # Format color data.
            points_color = torch.clamp(points_color, 0, 1)

            points_world = points_world.squeeze(0)

            # If crop is needed
            if self.pcd_crop:
                min_bound = torch.tensor(self.pcd_crop_bbox[0], dtype=torch.float32).to(device)
                max_bound = torch.tensor(self.pcd_crop_bbox[1], dtype=torch.float32).to(device)
                inside_bounds_mask = (points_world.squeeze(0) >= min_bound).all(dim=1) & (points_world.squeeze(0)
                                                                                          <= max_bound).all(dim=1)
                points_world = points_world[inside_bounds_mask]
                points_color = points_color[inside_bounds_mask]

            # Convert the tensor back to a NumPy array for use with Open3D.
            points_world_np = points_world.cpu().numpy()
            points_color_np = points_color.cpu().numpy()

            if point_num > 0:
                # points_world_np,index = fps(points_world_np,point_num)
                index = index.detach().cpu().numpy()[0]
                points_color_np = points_color_np[index, :]

            return np.hstack((points_world_np, points_color_np))

        if self.head_camera_id is None:
            print("No head camera in static camera list, pointcloud save error!")
            return None

        combined_pcd = np.array([])

        # Merge pointcloud
        if if_combine:
            # combined_pcd = np.vstack((head_pcd , left_pcd , right_pcd, front_pcd))
            if self.collect_wrist_camera:
                combined_pcd = np.vstack((
                    _get_camera_pcd(self.left_camera),
                    _get_camera_pcd(self.right_camera),
                ))
            for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
                if camera_name == "head_camera":
                    if self.collect_head_camera:
                        combined_pcd = np.vstack((combined_pcd, _get_camera_pcd(camera)))
                else:
                    combined_pcd = np.vstack((combined_pcd, _get_camera_pcd(camera)))
        elif self.collect_head_camera:
            combined_pcd = _get_camera_pcd(self.static_camera_list[self.head_camera_id])
        
        def pad_array(numpy_array, target_num):
            current_num = numpy_array.shape[0]
            if current_num < target_num:
                if current_num == 0:
                    numpy_array = np.zeros((target_num, 6))
                else:
                    pad_num = target_num - current_num
                    padding = np.zeros((pad_num, 6))
                    numpy_array = np.vstack([numpy_array, padding])
            return numpy_array

        combined_pcd = pad_array(combined_pcd, self.pcd_down_sample_num)

        pcd_array, index = combined_pcd[:, :3], np.array(range(len(combined_pcd)))

        if self.pcd_down_sample_num > 0:
            pcd_array, index = fps(combined_pcd[:, :3], self.pcd_down_sample_num)
            index = index.detach().cpu().numpy()[0]

        return combined_pcd[index]
