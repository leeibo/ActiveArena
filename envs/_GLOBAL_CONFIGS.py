# global configs
import os

ROOT_PATH = os.path.abspath(__file__)
ROOT_PATH = ROOT_PATH[:ROOT_PATH.rfind("/")]
ROOT_PATH = ROOT_PATH[:ROOT_PATH.rfind("/") + 1]

ASSETS_PATH = os.path.join(ROOT_PATH, "assets/")
EMBODIMENTS_PATH = os.path.join(ASSETS_PATH, "embodiments/")
TEXTURES_PATH = os.path.join(ASSETS_PATH, "background_texture/")
CONFIGS_PATH = os.path.join(ROOT_PATH, "task_config/")
SCRIPT_PATH = os.path.join(ROOT_PATH, "script/")
DESCRIPTION_PATH = os.path.join(ROOT_PATH, "description/")

# Euler angles in world coordinates
# t3d.euler.quat2euler(quat) returns (theta_x, theta_y, theta_z)
# theta_y controls the pitch, and theta_z controls rotation around the axis perpendicular to the tabletop plane
GRASP_DIRECTION_DIC = {
    "left": [0, 0, 0, -1],
    "front_left": [-0.383, 0, 0, -0.924],
    "front": [-0.707, 0, 0, -0.707],
    "front_right": [-0.924, 0, 0, -0.383],
    "right": [-1, 0, 0, 0],
    "top_down": [-0.5, 0.5, -0.5, -0.5],
    "down_right": [-0.707, 0, -0.707, 0],
    "down_left": [0, 0.707, 0, -0.707],
    "top_down_little_left": [-0.353523, 0.61239, -0.353524, -0.61239],
    "top_down_little_right": [-0.61239, 0.353523, -0.61239, -0.353524],
    "left_arm_perf": [-0.853532, 0.146484, -0.353542, -0.3536],
    "right_arm_perf": [-0.353518, 0.353564, -0.14642, -0.853568],
}
left_check_pose = [-1.5, -0.7, 0.4, 1.8, 2.6, -0.76, -0.7]
WORLD_DIRECTION_DIC = {
    "left": [0, -0.707, 0, 0.707],  # -z  -y  -x
    "front": [0.5, -0.5, 0.5, 0.5],  # y   z   -x
    "right": [0.707, 0, 0.707, 0],  # z   y   -x
    "top_down": [0, 0.707, -0.707, 0],  # -x  -y  -z
}

ROTATE_NUM = 10

# Shared upper-layer geometry for fan_double rotate tasks. The lower fan uses
# the same fan geometry as ordinary rotate tasks; these profiles only describe
# the second layer and its support. When fan_double_upper_align_with_lower is
# true, runtime config expands the upper theta range to match fan_angle_deg.
ROTATE_FAN_DOUBLE_LAYER_CONFIGS = {
    "centered": {
        "fan_double_upper_outer_radius": 0.8,
        "fan_double_upper_inner_radius": 0.6,
        "fan_double_layer_gap": 0.35,
        "fan_double_upper_align_with_lower": True,
        "fan_double_upper_theta_start_deg": -75.0,
        "fan_double_upper_theta_end_deg": 75.0,
        "fan_double_support_theta_deg": 0.0,
        "fan_double_upper_theta_offset_deg": 0.0,
        "fan_double_upper_collision_under_padding": 0.08,
    },
    "left_support": {
        "fan_double_upper_outer_radius": 0.8,
        "fan_double_upper_inner_radius": 0.6,
        "fan_double_layer_gap": 0.35,
        "fan_double_upper_align_with_lower": True,
        "fan_double_upper_theta_start_deg": -75.0,
        "fan_double_upper_theta_end_deg": 75.0,
        "fan_double_support_theta_deg": 0.0,
        "fan_double_upper_theta_offset_deg": 0.0,
        "fan_double_upper_collision_under_padding": 0.08,
    },
}
