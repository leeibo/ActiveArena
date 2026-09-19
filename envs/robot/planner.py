import json
import mplib.planner
import mplib
import numpy as np
import pdb
import traceback
import numpy as np
import sapien
import toppra as ta
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
import transforms3d as t3d
import random
import os
import tempfile
import envs._GLOBAL_CONFIGS as CONFIGS
# left_pose = json.load(open('script/calibration/ik_left_identity.json'))
# right_pose = json.load(open('script/calibration/ik_right_identity.json'))
# left_pose = left_pose['success_points']
# right_pose = right_pose['success_points']

# left_pose_list = random.sample(left_pose, 1000)
# right_pose_list = random.sample(right_pose, 1000)


def _plan_gripper_profile(
    now_val,
    target_val,
    dt=1 / 250,
    max_vel=4.0,
    acc=30.0,
    min_steps=6,
    max_steps=180,
):
    now_val = float(now_val)
    target_val = float(target_val)
    dis_val = target_val - now_val
    dis_abs = abs(dis_val)
    if dis_abs < 1e-9:
        return {"num_step": 1, "per_step": 0.0, "result": np.array([target_val], dtype=np.float64)}

    dt = max(float(dt), 1e-6)
    max_vel = max(float(max_vel), 1e-6)
    acc = max(float(acc), 1e-6)
    min_steps = max(int(min_steps), 1)
    max_steps = max(int(max_steps), min_steps)

    t_acc_nom = max_vel / acc
    d_acc_nom = 0.5 * acc * (t_acc_nom**2)
    if dis_abs <= 2.0 * d_acc_nom:
        t_acc = np.sqrt(dis_abs / acc)
        t_flat = 0.0
        v_peak = acc * t_acc
    else:
        t_acc = t_acc_nom
        t_flat = (dis_abs - 2.0 * d_acc_nom) / max_vel
        v_peak = max_vel

    d_acc = 0.5 * acc * (t_acc**2)
    total_time = max(2.0 * t_acc + t_flat, dt)
    num_step = int(np.ceil(total_time / dt))
    num_step = int(np.clip(num_step, min_steps, max_steps))
    times = np.linspace(total_time / num_step, total_time, num=num_step, dtype=np.float64)

    s = np.zeros_like(times, dtype=np.float64)
    t_switch = t_acc + t_flat
    for i, t in enumerate(times):
        if t <= t_acc:
            s[i] = 0.5 * acc * (t**2)
        elif t <= t_switch:
            s[i] = d_acc + v_peak * (t - t_acc)
        else:
            t_dec = t - t_switch
            s[i] = d_acc + v_peak * t_flat + v_peak * t_dec - 0.5 * acc * (t_dec**2)
    s = np.clip(s, 0.0, dis_abs)
    s[-1] = dis_abs

    direction = 1.0 if dis_val >= 0 else -1.0
    vals = now_val + direction * s
    vals[-1] = target_val
    per_step = dis_val / num_step
    return {"num_step": num_step, "per_step": per_step, "result": vals}


def create_rgb_axis_marker(
    scene: sapien.Scene,
    axis_len=0.1,
    axis_radius=0.003,
    name="target_pose_marker",
):
    mat_x = sapien.render.RenderMaterial()
    mat_x.set_base_color([1.0, 0.0, 0.0, 1.0])

    mat_y = sapien.render.RenderMaterial()
    mat_y.set_base_color([0.0, 1.0, 0.0, 1.0])

    mat_z = sapien.render.RenderMaterial()
    mat_z.set_base_color([0.0, 0.0, 1.0, 1.0])

    builder = scene.create_actor_builder()

    def quat_from_axis_angle(axis, angle):
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        s = np.sin(angle / 2.0)
        return [np.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s]

    q_x = [1.0, 0.0, 0.0, 0.0]
    q_y = quat_from_axis_angle([0, 0, 1], np.pi / 2)
    q_z = quat_from_axis_angle([0, 1, 0], -np.pi / 2)
    half = axis_len / 2

    builder.add_capsule_visual(
        pose=sapien.Pose([half, 0, 0], q_x),
        radius=axis_radius, half_length=half, material=mat_x,
    )
    builder.add_capsule_visual(
        pose=sapien.Pose([0, half, 0], q_y),
        radius=axis_radius, half_length=half, material=mat_y,
    )
    builder.add_capsule_visual(
        pose=sapien.Pose([0, 0, half], q_z),
        radius=axis_radius, half_length=half, material=mat_z,
    )
    return builder.build_static(name=name)


try:
    # ********************** CuroboPlanner (optional) **********************
    from curobo.types.math import Pose as CuroboPose
    import time
    from curobo.types.robot import JointState, RobotConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.motion_gen import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
        PoseCostMetric,
    )
    from curobo.util import logger
    from curobo.util_file import load_yaml
    import torch
    import yaml
    from curobo.util import logger
    logger.setup_logger(level="error", logger_name="curobo")

    def _get_nested_cfg_value(data, key_path):
        cur = data
        for key in key_path:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return cur

    def _set_nested_cfg_value(data, key_path, value):
        cur = data
        for key in key_path[:-1]:
            if key not in cur or not isinstance(cur[key], dict):
                cur[key] = {}
            cur = cur[key]
        cur[key_path[-1]] = value

    def _resolve_curobo_file_path(raw_path, yml_dir):
        if raw_path is None or not isinstance(raw_path, str):
            return raw_path

        resolved = str(raw_path).strip()
        if resolved == "":
            return raw_path

        repo_root = CONFIGS.ROOT_PATH.rstrip("/")
        for src, dst in {
            "${ASSETS_PATH}": repo_root,
            "$ASSETS_PATH": repo_root,
            "${ROOT_PATH}": repo_root,
            "$ROOT_PATH": repo_root,
        }.items():
            resolved = resolved.replace(src, dst)
        resolved = os.path.expanduser(os.path.expandvars(resolved))

        candidates = [resolved]
        if not os.path.isabs(resolved):
            candidates.append(os.path.abspath(os.path.join(yml_dir, resolved)))

        if "/assets/" in resolved:
            suffix = resolved.split("/assets/", 1)[1]
            candidates.append(os.path.abspath(os.path.join(CONFIGS.ASSETS_PATH, suffix)))

        if "/RoboTwin_Astribot/" in resolved:
            suffix = resolved.split("/RoboTwin_Astribot/", 1)[1]
            candidates.append(os.path.abspath(os.path.join(CONFIGS.ROOT_PATH, suffix)))

        candidates.append(os.path.abspath(os.path.join(yml_dir, os.path.basename(resolved))))

        seen = set()
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            norm = os.path.abspath(candidate)
            if norm in seen:
                continue
            seen.add(norm)
            if os.path.exists(norm):
                return norm
        return resolved

    def _prepare_curobo_config_file(yml_path):
        with open(yml_path, "r", encoding="utf-8") as f:
            yml_data = yaml.safe_load(f)

        yml_dir = os.path.dirname(os.path.abspath(yml_path))
        changed = False
        for key_path in [
            ("robot_cfg", "kinematics", "urdf_path"),
            ("robot_cfg", "kinematics", "collision_spheres"),
            ("robot_cfg", "kinematics", "asset_root_path"),
        ]:
            raw_value = _get_nested_cfg_value(yml_data, key_path)
            resolved_value = _resolve_curobo_file_path(raw_value, yml_dir)
            if resolved_value != raw_value:
                _set_nested_cfg_value(yml_data, key_path, resolved_value)
                changed = True

        if not changed:
            return os.path.abspath(yml_path), yml_data

        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yml",
            prefix="curobo_resolved_",
            delete=False,
            encoding="utf-8",
        )
        try:
            yaml.safe_dump(yml_data, tmp, sort_keys=False)
        finally:
            tmp.close()
        return tmp.name, yml_data

    class CuroboPlanner:

        def __init__(
            self,
            robot_origion_pose,
            active_joints_name,
            all_joints,
            yml_path=None,
            verbose=False,
        ):
            super().__init__()
            ta.setup_logging("CRITICAL")  # hide logging
            logger.setup_logger(level="error", logger_name="'curobo")
            self.verbose = bool(verbose)

            if yml_path != None:
                self.original_yml_path = os.path.abspath(yml_path)
                self.yml_path, yml_data = _prepare_curobo_config_file(self.original_yml_path)
            else:
                raise ValueError("[Planner.py]: CuroboPlanner yml_path is None!")
            self.robot_origion_pose = robot_origion_pose
            self.active_joints_name = active_joints_name
            self.all_joints = all_joints

            self._root_to_base_chain = self._extract_urdf_root_to_base_chain(yml_data)
            self.T_root_to_base = np.eye(4, dtype=np.float64)
            self.T_base_to_root = np.eye(4, dtype=np.float64)
            self.update_base_frame(curr_joint_pos=None)
            self._log(f"[CuroboPlanner] T_root_to_base:\n{np.round(self.T_root_to_base, 6)}")
            self._log(f"[CuroboPlanner] T_base_to_root:\n{np.round(self.T_base_to_root, 6)}")

            self._table_dims = [0.7, 2.0, 0.04]
            self._table_world_pose = [0.0, 0.0, 0.72, 1.0, 0.0, 0.0, 0.0]
            self._extra_world_cuboids = []
            self._world_update_warned = False
            world_config = self._build_world_config()

            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.yml_path,
                world_config,
                interpolation_dt=1 / 250,
                num_trajopt_seeds=1,
            )

            self.motion_gen = MotionGen(motion_gen_config)
            self.motion_gen.warmup()

            config_data = load_yaml(self.yml_path)
            urdf_file = config_data["robot_cfg"]["kinematics"]["urdf_path"]
            base_link_name = config_data["robot_cfg"]["kinematics"]["base_link"]
            ee_link_name = config_data["robot_cfg"]["kinematics"]["ee_link"]
            tensor_args = TensorDeviceType()
            fk_robot_cfg = RobotConfig.from_basic(urdf_file, base_link_name, ee_link_name, tensor_args)
            self.fk_model = CudaRobotModel(fk_robot_cfg.kinematics)
            self._fk_arm_name = base_link_name.replace("_base_link", "")
            self._log(f"[CuroboPlanner] FK model ready for {self._fk_arm_name}, dof={self.fk_model.get_dof()}")

            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.yml_path,
                world_config,
                interpolation_dt=1 / 250,
                num_trajopt_seeds=1,
                num_graph_seeds=1,
            )
            self.motion_gen_batch = MotionGen(motion_gen_config)
            self.motion_gen_batch.warmup(batch=CONFIGS.ROTATE_NUM)

        _trans_debug_count = 0

        def _log(self, msg: str):
            if self.verbose:
                print(msg)

        @staticmethod
        def _to_pose7(pose_like):
            arr = np.array(pose_like, dtype=np.float64).reshape(-1)
            if arr.shape[0] != 7:
                raise ValueError(f"Pose must have 7 values [x,y,z,qw,qx,qy,qz], got shape {arr.shape}.")
            return arr.tolist()

        def _world_pose_to_base_pose7(self, world_pose7):
            world_pose7 = self._to_pose7(world_pose7)
            world_base_pose = np.concatenate([
                np.array(self.robot_origion_pose.p, dtype=np.float64),
                np.array(self.robot_origion_pose.q, dtype=np.float64),
            ])
            root_p, root_q = self._trans_from_world_to_base(world_base_pose, np.array(world_pose7, dtype=np.float64))
            T_in_root = np.eye(4, dtype=np.float64)
            T_in_root[:3, :3] = t3d.quaternions.quat2mat(root_q)
            T_in_root[:3, 3] = root_p
            T_in_base = self.T_base_to_root @ T_in_root
            return list(T_in_base[:3, 3].tolist()) + list(t3d.quaternions.mat2quat(T_in_base[:3, :3]).tolist())

        def _build_world_config(self):
            table_pose = self._world_pose_to_base_pose7(self._table_world_pose)
            cuboid_cfg = {
                "table": {
                    "dims": list(np.array(self._table_dims, dtype=np.float64).reshape(-1)[:3].tolist()),
                    "pose": table_pose,
                }
            }
            for idx, item in enumerate(self._extra_world_cuboids):
                name = str(item.get("name", f"obj_{idx}"))
                dims = np.array(item.get("dims", [0.1, 0.1, 0.1]), dtype=np.float64).reshape(-1)
                if dims.shape[0] < 3:
                    continue
                dims = np.maximum(dims[:3], 1e-3)
                pose_world = item.get("pose", None)
                if pose_world is None:
                    continue
                try:
                    pose_base = self._world_pose_to_base_pose7(pose_world)
                except Exception:
                    continue
                cuboid_cfg[name] = {"dims": list(dims.tolist()), "pose": pose_base}
            return {"cuboid": cuboid_cfg}

        def _apply_world_config(self, world_config):
            ok = True
            for motion_gen in [self.motion_gen, self.motion_gen_batch]:
                try:
                    if hasattr(motion_gen, "update_world"):
                        motion_gen.update_world(world_config)
                    elif hasattr(motion_gen, "update_world_obstacles"):
                        motion_gen.update_world_obstacles(world_config)
                    else:
                        ok = False
                except Exception as e:
                    ok = False
                    self._log(f"[CuroboPlanner] update_world failed: {e}")
            if (not ok) and (not self._world_update_warned):
                self._world_update_warned = True
                self._log("[CuroboPlanner] world obstacle update API is unavailable; using init-world only.")
            return ok

        def refresh_world_obstacles(self, curr_joint_pos=None):
            if curr_joint_pos is not None:
                self.update_base_frame(curr_joint_pos)
            world_config = self._build_world_config()
            self._apply_world_config(world_config)
            return world_config

        def set_world_extra_cuboids(self, cuboids, curr_joint_pos=None):
            parsed = []
            if cuboids is None:
                cuboids = []
            for idx, item in enumerate(cuboids):
                if not isinstance(item, dict):
                    continue
                pose = item.get("pose", None)
                dims = item.get("dims", None)
                if pose is None or dims is None:
                    continue
                try:
                    pose7 = self._to_pose7(pose)
                    dims3 = np.maximum(np.array(dims, dtype=np.float64).reshape(-1)[:3], 1e-3)
                    if dims3.shape[0] != 3:
                        continue
                except Exception:
                    continue
                parsed.append({
                    "name": str(item.get("name", f"obj_{idx}")),
                    "pose": pose7,
                    "dims": dims3.tolist(),
                })
            self._extra_world_cuboids = parsed
            self.refresh_world_obstacles(curr_joint_pos=curr_joint_pos)
            self._log(f"[CuroboPlanner] extra world cuboids set: {len(self._extra_world_cuboids)}")
            return len(self._extra_world_cuboids)

        def _trans_world_to_curobo_frame(self, target_gripper_pose):
            """Transform a world-frame sapien.Pose to the CuRobo base_link frame."""
            world_base_pose = np.concatenate([
                np.array(self.robot_origion_pose.p),
                np.array(self.robot_origion_pose.q),
            ])
            world_target_pose = np.concatenate([
                np.array(target_gripper_pose.p),
                np.array(target_gripper_pose.q),
            ])
            root_p, root_q = self._trans_from_world_to_base(world_base_pose, world_target_pose)

            T_in_root = np.eye(4)
            T_in_root[:3, :3] = t3d.quaternions.quat2mat(root_q)
            T_in_root[:3, 3] = root_p
            T_in_base = self.T_base_to_root @ T_in_root

            if self.verbose and CuroboPlanner._trans_debug_count < 2:
                CuroboPlanner._trans_debug_count += 1
                self._log(f"[_trans] world target p={list(np.round(np.array(target_gripper_pose.p), 4))}")
                self._log(f"[_trans] in ROOT frame  p={list(np.round(root_p, 4))}")
                self._log(f"[_trans] in BASE frame  p={list(np.round(T_in_base[:3, 3], 4))}")

            return T_in_base[:3, 3], t3d.quaternions.mat2quat(T_in_base[:3, :3])

        def _fk_verify(self, joint_angles_list, target_pose_p, target_pose_q, target_world_pose, label=""):
            """Use FK to verify the planned result and print diagnostics."""
            q = torch.tensor(joint_angles_list, dtype=torch.float32).cuda().reshape(1, -1)
            fk_state = self.fk_model.get_state(q)
            fk_pos = fk_state.ee_position[0].cpu().numpy()
            fk_quat = fk_state.ee_quaternion[0].cpu().numpy()

            pos_err = np.linalg.norm(fk_pos - np.array(target_pose_p))
            self._log(f"[FK-{label}] {self._fk_arm_name}")
            self._log(f"  target  in base_link: p={list(np.round(target_pose_p, 5))}, q={list(np.round(target_pose_q, 5))}")
            self._log(f"  FK      in base_link: p={list(np.round(fk_pos, 5))}, q={list(np.round(fk_quat, 5))}")
            self._log(f"  pos error (base_link): {pos_err:.6f} m")

            T_fk_base = np.eye(4)
            T_fk_base[:3, :3] = t3d.quaternions.quat2mat(fk_quat)
            T_fk_base[:3, 3] = fk_pos
            T_fk_root = self.T_root_to_base @ T_fk_base
            fk_root_p = T_fk_root[:3, 3]
            fk_root_q = t3d.quaternions.mat2quat(T_fk_root[:3, :3])

            robot_p = np.array(self.robot_origion_pose.p)
            robot_q = np.array(self.robot_origion_pose.q)
            wRb = t3d.quaternions.quat2mat(robot_q)
            fk_world_p = wRb @ fk_root_p + robot_p
            fk_world_q = t3d.quaternions.mat2quat(wRb @ T_fk_root[:3, :3])

            target_world_p = np.array(target_world_pose.p)
            world_err = np.linalg.norm(fk_world_p - target_world_p)
            self._log(f"  target  in world:     p={list(np.round(target_world_p, 5))}")
            self._log(f"  FK      in world:     p={list(np.round(fk_world_p, 5))}, q={list(np.round(fk_world_q, 5))}")
            self._log(f"  pos error (world):    {world_err:.6f} m")
            return fk_world_p, world_err

        def plan_path(
            self,
            curr_joint_pos,
            target_gripper_pose,
            constraint_pose=None,
            arms_tag=None,
        ):  
            self.update_base_frame(curr_joint_pos)
            self.refresh_world_obstacles(curr_joint_pos=None)
            target_pose_p, target_pose_q = self._trans_world_to_curobo_frame(target_gripper_pose)
            ## Temporarily add the successful xyz coordinates ##
            # target_pose_p = [0.35, 0.23, 0.09]  # Example: using 0.35, 0.23, 0.09
            # target_pose_q = [1., 0., 0., 0.]
            # ## End temporary addition ## 
            # # goal_pose_of_gripper = CuroboPose.from_list(list(target_pose_p) + list(target_pose_q))
            # print(f'[plan_path] {self._fk_arm_name} target_pose_p: {list(np.round(target_pose_p, 5))} target_pose_q: {list(np.round(target_pose_q, 5))}')
            # Remove the hardcoded position and quaternion
            # target_pose_p = np.array([0.25, 0.3, 0.09] )
            # target_pose_q = np.array([1.0, 0.0, 0.0, 0.0])
            # print('[debug]: target_pose_q: ', target_pose_q)
            goal_pose_of_ee = CuroboPose.from_list(list(target_pose_p) + list(target_pose_q))

            joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            joint_angles = [round(angle, 5) for angle in joint_angles]
            start_joint_states = JointState.from_position(
                torch.tensor(joint_angles).cuda().reshape(1, -1),
                joint_names=self.active_joints_name,
            )
            try:
                vq, vq_status = self.motion_gen.check_start_state(start_joint_states)
                self._log(f"[plan_path] {self._fk_arm_name} check_start_state: valid={vq}, status={vq_status}")
            except Exception as diag_e:
                self._log(f"[plan_path] {self._fk_arm_name} check_start_state error: {diag_e}")

            self._fk_verify(joint_angles, target_pose_p, target_pose_q, target_gripper_pose, label="START")

            plan_config = MotionGenPlanConfig(max_attempts=10)
            if constraint_pose is not None:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
                )
                plan_config.pose_cost_metric = pose_cost_metric
            self._log(f"[plan_path] {self._fk_arm_name} planning...")
            result = self.motion_gen.plan_single(start_joint_states, goal_pose_of_ee, plan_config)
            
            res_result = dict()
            if result.success.item() == False:
                res_result["status"] = "Fail"
                self._log(f"[plan_path] {self._fk_arm_name} FAILED")
                return res_result
            else:
                res_result["status"] = "Success"
                res_result["position"] = np.array(result.interpolated_plan.position.to("cpu"))
                res_result["velocity"] = np.array(result.interpolated_plan.velocity.to("cpu"))

                final_joints = res_result["position"][-1].tolist()
                fk_world_p, world_err = self._fk_verify(
                    final_joints, target_pose_p, target_pose_q, target_gripper_pose, label="END"
                )
                self._log(
                    f"[plan_path] {self._fk_arm_name} SUCCESS, "
                    f"{res_result['position'].shape[0]} steps, world_err={world_err:.6f}m"
                )
                return res_result

        def plan_batch(
            self,
            curr_joint_pos,
            target_gripper_pose_list,
            constraint_pose=None,
            arms_tag=None,
        ):
            """
            Plan a batch of trajectories for multiple target poses.

            Input:
                - curr_joint_pos: List of current joint angles (1 x n)
                - target_gripper_pose_list: List of target poses [sapien.Pose, sapien.Pose, ...]

            Output:
                - result['status']: numpy array of string values indicating "Success"/"Fail" for each pose
                - result['position']: numpy array of joint positions with shape (n x m x l)
                  where n is number of target poses, m is number of waypoints, l is number of joints
                - result['velocity']: numpy array of joint velocities with same shape as position
            """

            self.update_base_frame(curr_joint_pos)
            self.refresh_world_obstacles(curr_joint_pos=None)
            num_poses = len(target_gripper_pose_list)
            self._log(f"[plan_batch] num_poses={num_poses}")
            poses_list = []
            for idx, target_gripper_pose in enumerate(target_gripper_pose_list):
                p, q = self._trans_world_to_curobo_frame(target_gripper_pose)
                poses_list.append(list(p) + list(q))
                if idx == 0:
                    self._log(
                        f"[plan_batch] first target in base_link: "
                        f"p={list(np.round(p, 4))}, q={list(np.round(q, 4))}"
                    )

            poses_cuda = torch.tensor(poses_list, dtype=torch.float32).cuda()
            goal_pose_of_ee = CuroboPose(poses_cuda[:, :3], poses_cuda[:, 3:])
            joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            joint_angles = [round(angle, 5) for angle in joint_angles]  # avoid the precision problem
            self._log(f"[plan_batch] start joints ({len(joint_angles)}): {[round(a,4) for a in joint_angles]}")
            joint_angles_cuda = (torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1))

            # diagnose start state with single-target MotionGen
            single_start = JointState.from_position(joint_angles_cuda.clone(), joint_names=self.active_joints_name)
            try:
                vq, vq_status = self.motion_gen.check_start_state(single_start)
                self._log(f"[plan_batch] check_start_state: valid={vq}, status={vq_status}")
            except Exception as diag_e:
                self._log(f"[plan_batch] check_start_state error: {diag_e}")

            joint_angles_cuda = torch.cat([joint_angles_cuda] * num_poses, dim=0)
            start_joint_states = JointState.from_position(joint_angles_cuda, joint_names=self.active_joints_name)
            # plan
            plan_config = MotionGenPlanConfig(max_attempts=10)
            if constraint_pose is not None:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
                )
                plan_config.pose_cost_metric = pose_cost_metric

            try:
                result = self.motion_gen_batch.plan_batch(start_joint_states, goal_pose_of_ee, plan_config)
            except Exception as e:
                import traceback
                print(f"[plan_batch] EXCEPTION: {e}")
                traceback.print_exc()
                return {"status": ["Failure" for i in range(10)]}

            # output
            res_result = dict()
            # Convert boolean success values to "Success"/"Failure" strings
            success_array = result.success.cpu().numpy()
            status_array = np.array(["Success" if s else "Failure" for s in success_array], dtype=object)
            res_result["status"] = status_array
            n_success = np.sum(success_array)
            self._log(f"[plan_batch] {n_success}/{len(success_array)} succeeded")
            if n_success == 0:
                try:
                    self._log(f"[plan_batch] valid_query: {result.valid_query}")
                    if hasattr(result, 'status') and result.status is not None:
                        self._log(f"[plan_batch] status: {result.status}")
                except: pass

            if np.all(res_result["status"] == "Failure"):
                return res_result

            res_result["position"] = np.array(result.interpolated_plan.position.to("cpu"))
            res_result["velocity"] = np.array(result.interpolated_plan.velocity.to("cpu"))
            return res_result

        def plan_grippers(self, now_val, target_val):
            return _plan_gripper_profile(
                now_val=now_val,
                target_val=target_val,
                dt=1 / 250,
                max_vel=4.0,
                acc=30.0,
                min_steps=6,
                max_steps=180,
            )

        @staticmethod
        def _extract_urdf_root_to_base_chain(yml_data):
            """
            Parse URDF and extract the ordered joint chain from URDF root link to CuRobo base_link.
            Each chain item contains joint name/type/origin/axis.
            """
            import xml.etree.ElementTree as ET

            urdf_path = yml_data['robot_cfg']['kinematics']['urdf_path']
            base_link = yml_data['robot_cfg']['kinematics']['base_link']

            tree = ET.parse(urdf_path)
            urdf_root = tree.getroot()

            parent_map = {}
            joint_map = {}
            for j in urdf_root.findall('.//joint'):
                parent_link = j.find('parent').get('link')
                child_link = j.find('child').get('link')
                parent_map[child_link] = parent_link
                joint_map[child_link] = j

            chain = []
            current = base_link
            while current in parent_map:
                j = joint_map[current]
                joint_name = j.get("name", "")
                joint_type = j.get("type", "fixed")
                origin = j.find('origin')
                xyz = [float(v) for v in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()]
                rpy = [float(v) for v in (origin.get('rpy', '0 0 0') if origin is not None else '0 0 0').split()]
                axis_tag = j.find('axis')
                axis_default = "1 0 0" if joint_type in ["revolute", "continuous", "prismatic"] else "0 0 0"
                axis = [float(v) for v in (axis_tag.get('xyz', axis_default) if axis_tag is not None else axis_default).split()]
                chain.append({
                    "name": str(joint_name),
                    "type": str(joint_type),
                    "xyz": xyz,
                    "rpy": rpy,
                    "axis": axis,
                })
                current = parent_map[current]
            chain.reverse()
            return chain

        @staticmethod
        def _joint_motion_transform(joint_type, axis, value):
            T = np.eye(4, dtype=np.float64)
            jt = str(joint_type).lower()
            val = float(value)
            if jt in ["revolute", "continuous"]:
                axis = np.array(axis, dtype=np.float64).reshape(3)
                norm = float(np.linalg.norm(axis))
                if norm > 1e-12:
                    axis = axis / norm
                    T[:3, :3] = t3d.axangles.axangle2mat(axis, val)
            elif jt == "prismatic":
                axis = np.array(axis, dtype=np.float64).reshape(3)
                norm = float(np.linalg.norm(axis))
                if norm > 1e-12:
                    axis = axis / norm
                    T[:3, 3] = axis * val
            return T

        @staticmethod
        def _compute_root_to_base_transform_from_chain(chain, joint_state_map):
            T = np.eye(4, dtype=np.float64)
            for item in chain:
                xyz = item["xyz"]
                rpy = item["rpy"]
                T_joint = np.eye(4, dtype=np.float64)
                T_joint[:3, :3] = t3d.euler.euler2mat(rpy[0], rpy[1], rpy[2], axes='sxyz')
                T_joint[:3, 3] = xyz
                q = float(joint_state_map.get(item["name"], 0.0))
                T_motion = CuroboPlanner._joint_motion_transform(item["type"], item["axis"], q)
                # URDF joint transform: parent->child = origin * joint_motion
                T = T @ T_joint @ T_motion
            return T

        @staticmethod
        def _compute_urdf_root_to_base_transform(yml_data):
            """Compute root->base transform from URDF with zero joint positions."""
            chain = CuroboPlanner._extract_urdf_root_to_base_chain(yml_data)
            return CuroboPlanner._compute_root_to_base_transform_from_chain(chain, {})

        def _qpos_to_joint_map(self, curr_joint_pos):
            if curr_joint_pos is None:
                return {}
            if isinstance(curr_joint_pos, dict):
                out = {}
                for k, v in curr_joint_pos.items():
                    try:
                        out[str(k)] = float(np.array(v, dtype=np.float64).reshape(-1)[0])
                    except Exception:
                        continue
                return out
            try:
                q = np.array(curr_joint_pos, dtype=np.float64).reshape(-1)
            except Exception:
                return {}
            num = min(len(self.all_joints), q.shape[0])
            return {str(self.all_joints[i]): float(q[i]) for i in range(num)}

        def update_base_frame(self, curr_joint_pos=None):
            """
            Update URDF-root <-> CuRobo-base transforms using current articulation state.
            Must be called before world/base pose conversions when torso/base-chain joints move.
            """
            joint_state_map = self._qpos_to_joint_map(curr_joint_pos)
            self.T_root_to_base = self._compute_root_to_base_transform_from_chain(
                self._root_to_base_chain, joint_state_map
            )
            self.T_base_to_root = np.linalg.inv(self.T_root_to_base)
            return self.T_root_to_base, self.T_base_to_root

        def _trans_from_world_to_base(self, base_pose, target_pose):
            '''
                transform target pose from world frame to base frame
                base_pose: np.array([x, y, z, qw, qx, qy, qz])
                target_pose: np.array([x, y, z, qw, qx, qy, qz])
            '''
            base_p, base_q = base_pose[0:3], base_pose[3:]
            target_p, target_q = target_pose[0:3], target_pose[3:]
            rel_p = target_p - base_p
            wRb = t3d.quaternions.quat2mat(base_q)
            wRt = t3d.quaternions.quat2mat(target_q)
            result_p = wRb.T @ rel_p
            result_q = t3d.quaternions.mat2quat(wRb.T @ wRt)
            return result_p, result_q
    
except Exception as e:
    print('[planner.py]: Something wrong happened when importing CuroboPlanner! Please check if Curobo is installed correctly. If the problem still exists, you can install Curobo from https://github.com/NVlabs/curobo manually.')
    print('Exception traceback:')
    traceback.print_exc()


# ********************** MplibPlanner **********************
class MplibPlanner:
    # links=None, joints=None
    def __init__(
        self,
        urdf_path,
        srdf_path,
        move_group,
        robot_origion_pose,
        robot_entity,
        planner_type="mplib_RRT",
        scene=None,
    ):
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging

        links = [link.get_name() for link in robot_entity.get_links()]
        joints = [joint.get_name() for joint in robot_entity.get_active_joints()]

        if scene is None:
            self.planner = mplib.Planner(
                urdf=urdf_path,
                srdf=srdf_path,
                move_group=move_group,
                user_link_names=links,
                user_joint_names=joints,
                use_convex=False,
            )
            self.planner.set_base_pose(robot_origion_pose)
        else:
            planning_world = SapienPlanningWorld(scene, [robot_entity])
            self.planner = SapienPlanner(planning_world, move_group)

        self.planner_type = planner_type
        self.plan_step_lim = 2500
        self.TOPP = self.planner.TOPP

    def show_info(self):
        print("joint_limits", self.planner.joint_limits)
        print("joint_acc_limits", self.planner.joint_acc_limits)

    def plan_pose(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        try_times=2,
        log=True,
    ):
        result = {}
        result["status"] = "Fail"

        now_try_times = 1
        while result["status"] != "Success" and now_try_times < try_times:
            result = self.planner.plan_pose(
                goal_pose=target_pose,
                current_qpos=np.array(now_qpos),
                time_step=1 / 250,
                planning_time=5,
                # rrt_range=0.05
                # =================== mplib 0.1.1 ===================
                # use_point_cloud=use_point_cloud,
                # use_attach=use_attach,
                # planner_name="RRTConnect"
            )
            now_try_times += 1

        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm planning failed ({result['status']}) !")
        else:
            n_step = result["position"].shape[0]
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm planning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_screw(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        log=False,
    ):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        result = self.planner.plan_screw(
            goal_pose=target_pose,
            current_qpos=now_qpos,
            time_step=1 / 250,
            # =================== mplib 0.1.1 ===================
            # use_point_cloud=use_point_cloud,
            # use_attach=use_attach,
        )

        # plan fail
        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm planning failed ({result['status']}) !")
            # return result
        else:
            n_step = result["position"].shape[0]
            # plan step lim
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm planning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_path(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        log=True,
    ):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        if self.planner_type == "mplib_RRT":
            result = self.plan_pose(
                now_qpos,
                target_pose,
                use_point_cloud,
                use_attach,
                arms_tag,
                try_times=10,
                log=log,
            )
        elif self.planner_type == "mplib_screw":
            result = self.plan_screw(now_qpos, target_pose, use_point_cloud, use_attach, arms_tag, log)

        return result

    def plan_grippers(self, now_val, target_val):
        return _plan_gripper_profile(
            now_val=now_val,
            target_val=target_val,
            dt=1 / 250,
            max_vel=4.0,
            acc=30.0,
            min_steps=6,
            max_steps=180,
        )
