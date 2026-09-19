import sapien.core as sapien
import numpy as np
import transforms3d as t3d
import sapien.physx as sapienp
from .create_actor import *

import re
import json
from pathlib import Path


def _objaverse_model_dir(base_dir: Path, model_name: str, model_id: str) -> Path:
    return Path(base_dir) / str(model_name) / str(model_id)


def _objaverse_model_exists(base_dir: Path, model_name: str, model_id: str) -> bool:
    return (_objaverse_model_dir(base_dir, model_name, model_id) / "model.urdf").exists()


def get_all_cluttered_objects():
    cluttered_objects_info = {}
    cluttered_objects_name = []

    # load from cluttered_objects
    objaverse_root = Path("./assets/objects/objaverse")
    cluttered_objects_config = json.load(open(objaverse_root / "list.json", "r", encoding="utf-8"))
    for model_name, model_ids in cluttered_objects_config["list_of_items"].items():
        params = {}
        valid_model_ids = []
        for model_id in model_ids:
            if not _objaverse_model_exists(objaverse_root, model_name, model_id):
                continue
            model_full_name = f"{model_name}_{model_id}"
            params[model_id] = {
                "z_max": cluttered_objects_config["z_max"][model_full_name],
                "radius": cluttered_objects_config["radius"][model_full_name],
                "z_offset": cluttered_objects_config["z_offset"][model_full_name],
            }
            valid_model_ids.append(model_id)
        if len(valid_model_ids) == 0:
            continue
        cluttered_objects_name.append(model_name)
        cluttered_objects_info[model_name] = {
            "ids": valid_model_ids,
            "type": "urdf",
            "root": f"objects/objaverse/{model_name}",
            "params": params,
        }

    # load from objects
    objects_dir = Path("./assets/objects")
    for model_dir in objects_dir.iterdir():
        if not model_dir.is_dir():
            continue
        if re.search(r"^(\d+)_(.*)", model_dir.name) is None:
            continue
        model_name = model_dir.name
        model_id_list, params = [], {}
        for model_cfg in model_dir.iterdir():
            if model_cfg.is_dir() or model_cfg.suffix != ".json":
                continue

            # get model id
            model_id = re.search(r"model_data(\d+)", model_cfg.name)
            if not model_id:
                continue
            model_id = model_id.group(1)

            try:
                # get model params
                model_config: dict = json.load(open(model_cfg, "r", encoding="utf-8"))
                if "center" not in model_config or "extents" not in model_config:
                    continue
                if model_config.get("stable", False) is False:
                    continue
                center = model_config["center"]
                extents = model_config["extents"]
                scale = model_config.get("scale", [1.0, 1.0, 1.0])
                # 0: x, 1: z, 2: y
                params[model_id] = {
                    "z_max": (extents[1] + center[1]) * scale[1],
                    "radius": max(extents[0] * scale[0], extents[2] * scale[2]) / 2,
                    "z_offset": 0,
                }
                model_id_list.append(model_id)
            except Exception as e:
                print(f"Error loading model config {model_cfg}: {e}")
        if len(model_id_list) == 0:
            continue
        cluttered_objects_name.append(model_name)
        model_id_list.sort()
        cluttered_objects_info[model_name] = {
            "ids": model_id_list,
            "type": "glb",
            "root": f"objects/{model_name}",
            "params": params,
        }

    same_obj = json.load(open(Path("./assets/objects/same.json"), "r", encoding="utf-8"))
    cluttered_objects_name = list(cluttered_objects_name)
    cluttered_objects_name.sort()
    return cluttered_objects_info, cluttered_objects_name, same_obj


cluttered_objects_info, cluttered_objects_list, same_obj = get_all_cluttered_objects()


def expand_clutter_model_names(model_names):
    global same_obj

    expanded = set()
    for model_name in model_names or []:
        if model_name is None:
            continue
        model_name = str(model_name)
        if not model_name:
            continue
        expanded.add(model_name)
        if same_obj.get(model_name) is not None:
            expanded.update(str(item) for item in same_obj[model_name])
    return expanded


def get_available_cluttered_objects(entity_on_scene: list, include_models=None, exclude_models=None):
    global cluttered_objects_info, cluttered_objects_list, same_obj

    model_in_use = expand_clutter_model_names(entity_on_scene)
    model_in_use.update(expand_clutter_model_names(exclude_models))

    if include_models is None:
        available_models = set(cluttered_objects_list)
    else:
        available_models = set(cluttered_objects_list).intersection(expand_clutter_model_names(include_models))
    available_models -= model_in_use
    available_models = list(available_models)
    available_models.sort()
    return available_models, cluttered_objects_info


def check_overlap(radius, x, y, area):
    if x <= area[0]:
        dx = area[0] - x
    elif area[0] < x and x < area[2]:
        dx = 0
    elif x >= area[2]:
        dx = x - area[2]
    if y <= area[1]:
        dy = area[1] - y
    elif area[1] < y and y < area[3]:
        dy = 0
    elif y >= area[3]:
        dy = y - area[3]

    return dx * dx + dy * dy <= radius * radius


def _wrap_to_pi(theta):
    theta = float(theta)
    while theta <= -np.pi:
        theta += 2.0 * np.pi
    while theta > np.pi:
        theta -= 2.0 * np.pi
    return theta


def _point_in_fan_region(x, y, fan_region):
    if fan_region is None:
        return False
    center_xy = np.array(fan_region["center_xy"], dtype=np.float64).reshape(2)
    dx = float(x) - float(center_xy[0])
    dy = float(y) - float(center_xy[1])
    radius = float(np.hypot(dx, dy))
    theta = float(np.arctan2(dy, dx))
    center_theta = float(fan_region["center_theta_rad"])
    half_angle = float(fan_region["angle_rad"]) * 0.5
    theta_err = abs(_wrap_to_pi(theta - center_theta))
    return bool(radius >= float(fan_region["inner_radius"]) and radius <= float(fan_region["outer_radius"]) and theta_err <= half_angle + 1e-6)


def _sample_fan_xy(fan_region, point_radius, prefer_back=True):
    center_xy = np.array(fan_region["center_xy"], dtype=np.float64).reshape(2)
    radius_floor = float(fan_region.get("radius_floor", 0.0))
    inner_radius = max(float(fan_region["inner_radius"]), radius_floor) + float(point_radius)
    outer_radius = float(fan_region["outer_radius"]) - float(point_radius)
    if outer_radius <= inner_radius:
        return None

    theta_min = float(fan_region["center_theta_rad"]) - 0.5 * float(fan_region["angle_rad"])
    theta_max = float(fan_region["center_theta_rad"]) + 0.5 * float(fan_region["angle_rad"])
    candidate_count = int(fan_region.get("candidate_count", 6 if prefer_back else 1))
    radial_bias_power = float(fan_region.get("radial_bias_power", 0.35 if prefer_back else 1.0))
    candidates = []
    for _ in range(max(candidate_count, 1)):
        theta = float(np.random.uniform(theta_min, theta_max))
        u = float(np.random.uniform(0.0, 1.0))
        if prefer_back:
            u = u ** radial_bias_power
        radius_sq = inner_radius * inner_radius + (outer_radius * outer_radius - inner_radius * inner_radius) * u
        radius = float(np.sqrt(max(radius_sq, 0.0)))
        x = float(center_xy[0] + radius * np.cos(theta))
        y = float(center_xy[1] + radius * np.sin(theta))
        candidates.append((radius, x, y))
    candidates.sort(key=lambda item: item[0], reverse=bool(prefer_back))
    for _, x, y in candidates:
        if _point_in_fan_region(x=x, y=y, fan_region={
            "center_xy": center_xy,
            "inner_radius": max(float(fan_region["inner_radius"]), radius_floor) + float(point_radius),
            "outer_radius": float(fan_region["outer_radius"]) - float(point_radius),
            "center_theta_rad": float(fan_region["center_theta_rad"]),
            "angle_rad": float(fan_region["angle_rad"]),
        }):
            return x, y
    return None


def _is_valid_cluttered_xy(x, y, new_obj_radius, size_dict, prohibited_area, xlim, ylim, fan_region, z_max):
    is_overlap = False
    for area in prohibited_area:
        if check_overlap(new_obj_radius, x, y, area):
            is_overlap = True
            break
    if is_overlap:
        return False

    distances = np.sqrt((np.array([sub_list[0] for sub_list in size_dict]) - x)**2 +
                        (np.array([sub_list[1] for sub_list in size_dict]) - y)**2)
    max_distances = np.array([sub_list[3] + new_obj_radius + 0.005 for sub_list in size_dict])

    if y - new_obj_radius < 0 and z_max > 0.05:
        return False
    if fan_region is None:
        if (x - new_obj_radius < -0.6 or x + new_obj_radius > 0.6 or y - new_obj_radius < -0.34
                or y + new_obj_radius > 0.34):
            return False
        if not np.all(distances > max_distances) or not (y + new_obj_radius < ylim[1]):
            return False
        return True

    if not _point_in_fan_region(x=x, y=y, fan_region=fan_region):
        return False
    if not np.all(distances > max_distances):
        return False
    return True


def rand_pose_cluttered(
    xlim: np.ndarray,
    ylim: np.ndarray,
    zlim: np.ndarray,
    ylim_prop=False,
    rotate_rand=False,
    rotate_lim=[0, 0, 0],
    qpos=[1, 0, 0, 0],
    size_dict=None,
    obj_radius=0.1,
    z_offset=0.001,
    z_max=0,
    prohibited_area=None,
    obj_margin=0.005,
    fan_region=None,
    prefer_back=True,
) -> sapien.Pose:
    if len(xlim) < 2 or xlim[1] < xlim[0]:
        xlim = np.array([xlim[0], xlim[0]])
    if len(ylim) < 2 or ylim[1] < ylim[0]:
        ylim = np.array([ylim[0], ylim[0]])
    if len(zlim) < 2 or zlim[1] < zlim[0]:
        zlim = np.array([zlim[0], zlim[0]])

    times = 0
    while True:
        times += 1
        if times > 100:
            return False, None
        new_obj_radius = obj_radius + obj_margin
        if fan_region is None:
            x = np.random.uniform(xlim[0], xlim[1])
            y = np.random.uniform(ylim[0], ylim[1])
        else:
            sampled_xy = _sample_fan_xy(fan_region=fan_region, point_radius=new_obj_radius, prefer_back=prefer_back)
            if sampled_xy is None:
                continue
            x, y = sampled_xy
        if _is_valid_cluttered_xy(
            x=x,
            y=y,
            new_obj_radius=new_obj_radius,
            size_dict=size_dict,
            prohibited_area=prohibited_area,
            xlim=xlim,
            ylim=ylim,
            fan_region=fan_region,
            z_max=z_max,
        ):
            break

    z = np.random.uniform(zlim[0], zlim[1])
    z = z - z_offset

    rotate = qpos
    if rotate_rand:
        angles = [0, 0, 0]
        for i in range(3):
            angles[i] = np.random.uniform(-rotate_lim[i], rotate_lim[i])
        rotate_quat = t3d.euler.euler2quat(angles[0], angles[1], angles[2])
        rotate = t3d.quaternions.qmult(rotate, rotate_quat)

    return True, sapien.Pose([x, y, z], rotate)


def rand_create_cluttered_actor(
    scene,
    modelname: str,
    modelid: str,
    modeltype: str,
    xlim: np.ndarray,
    ylim: np.ndarray,
    zlim: np.ndarray,
    ylim_prop=False,
    rotate_rand=False,
    rotate_lim=[0, 0, 0],
    qpos=None,
    scale=(1, 1, 1),
    convex=True,
    is_static=False,
    size_dict=None,
    obj_radius=0.1,
    z_offset=0.001,
    z_max=0,
    fix_root_link=True,
    prohibited_area=None,
    fan_region=None,
    prefer_back=True,
) -> tuple[bool, Actor | None]:

    if qpos is None:
        if modeltype == "glb":
            qpos = [0.707107, 0.707107, 0, 0]
            rotate_lim = [rotate_lim[0], rotate_lim[2], rotate_lim[1]]
        else:
            qpos = [1, 0, 0, 0]

    success, obj_pose = rand_pose_cluttered(
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        ylim_prop=ylim_prop,
        rotate_rand=rotate_rand,
        rotate_lim=rotate_lim,
        qpos=qpos,
        size_dict=size_dict,
        obj_radius=obj_radius,
        z_offset=z_offset,
        z_max=z_max,
        prohibited_area=prohibited_area,
        fan_region=fan_region,
        prefer_back=prefer_back,
    )

    if not success:
        return False, None

    if modeltype == "urdf":
        obj = create_cluttered_urdf_obj(
            scene=scene,
            pose=obj_pose,
            modelname=f"objects/objaverse/{modelname}/{modelid}",
            scale=scale if isinstance(scale, float) else scale[0],
            fix_root_link=fix_root_link,
        )
        if obj is None:
            return False, None
        else:
            return True, obj
    else:
        obj = create_actor(
            scene=scene,
            pose=obj_pose,
            modelname=modelname,
            model_id=modelid,
            scale=scale,
            convex=convex,
            is_static=is_static,
        )
        if obj is None:
            return False, None
        else:
            return True, obj


def create_cluttered_urdf_obj(scene, pose: sapien.Pose, modelname: str, scale=1.0, fix_root_link=True) -> Actor:
    scene, pose = preprocess(scene, pose)
    modeldir = Path("assets") / modelname

    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = scale
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = False
    object: sapien.Articulation = loader.load_multiple(str(modeldir / "model.urdf"))[1][0]
    object.set_pose(pose)

    if isinstance(object, sapien.physx.PhysxArticulation):
        return ArticulationActor(object, None)
    else:
        return Actor(object, None)
