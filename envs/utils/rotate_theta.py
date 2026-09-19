import numpy as np

from envs._GLOBAL_CONFIGS import ROTATE_FAN_DOUBLE_LAYER_CONFIGS


DEFAULT_ROTATE_THETA_SHARED_RATIO = 1.0
DEFAULT_SIDE_INNER_RATIO = 0.55
DEFAULT_FIXED_RATIO = 0.75
DEFAULT_MIXED_NEAR_RATIO = 0.45
DEFAULT_SCAN_THETA_UNIT_DEG = 15.0
DEFAULT_SCAN_QUANTIZE_MODE = "outward"
DEFAULT_SCAN_MIN_STEPS = 1
DEFAULT_SCAN_STRATEGY = "object_coverage"
DEFAULT_SCAN_SEQUENCE_STEPS = (4, -4, 2, -2, 0)
DEFAULT_SCAN_ORDER = "left_to_right"
DEFAULT_SCAN_VISIBILITY_MODE = "aabb"
DEFAULT_SCAN_VISIBILITY_CAMERA_NAME = "camera_head"
DEFAULT_SCAN_HORIZONTAL_MARGIN_DEG = 5.0
DEFAULT_SCAN_VERTICAL_MARGIN_DEG = 3.0
DEFAULT_SCAN_AABB_VISIBLE_RATIO_THRESHOLD = 0.4
DEFAULT_SCAN_REFERENCE_R = 0.63
DEFAULT_SCAN_JOINT_NAME = "astribot_torso_joint_2"
DEFAULT_STAGE1_THETA_UNIT_DEG = 45.0
DEFAULT_STAGE2_MAX_THETA_STEP_DEG = 30.0
DEFAULT_STAGE2_CENTER_TOL_DEG = 3.0

ROTATE_FAN_TABLE_CONFIG_REQUIRED_KEYS = (
    "fan_center_on_robot",
    "fan_outer_radius",
    "fan_inner_radius",
    "fan_angle_deg",
    "fan_center_deg",
    "rotate_object_margin_deg",
    "rotate_theta_reference_fan_angle_deg",
    "rotate_theta_shared_ratio",
)

ROTATE_TABLE_CONFIG_REQUIRED_KEYS = {
    "fan": ROTATE_FAN_TABLE_CONFIG_REQUIRED_KEYS,
    "fan_double": ROTATE_FAN_TABLE_CONFIG_REQUIRED_KEYS,
}

FAN_DOUBLE_LAYER_CONFIG_ALIASES = {
    "fan_double_centered": "centered",
    "centered": "centered",
    "fan_double_left_support": "left_support",
    "left_support": "left_support",
}


def _normalize_fan_double_layer_config_key(config_key):
    config_key = str(config_key).strip().lower()
    if config_key in FAN_DOUBLE_LAYER_CONFIG_ALIASES:
        return FAN_DOUBLE_LAYER_CONFIG_ALIASES[config_key]
    return config_key


def _get_task_fan_double_layer_config_key(task, kwargs):
    config_key = kwargs.get(
        "rotate_fan_double_layer_config_key",
        kwargs.get("fan_double_layer_config_key", None),
    )
    if config_key is None:
        config_key = getattr(task, "ROTATE_FAN_DOUBLE_LAYER_CONFIG_KEY", None)
    if config_key is None:
        legacy_table_key = str(getattr(task, "ROTATE_TABLE_CONFIG_KEY", "")).strip().lower()
        if legacy_table_key.startswith("fan_double_"):
            config_key = legacy_table_key
    if config_key is None:
        config_key = "centered"
    return _normalize_fan_double_layer_config_key(config_key)


def apply_rotate_fan_double_layer_config(task, kwargs):
    """Merge global second-layer geometry into fan_double task kwargs."""
    kwargs = dict(kwargs)
    table_shape = str(getattr(task, "ROTATE_TABLE_SHAPE", kwargs.get("table_shape", "fan"))).lower()
    if table_shape != "fan_double":
        return kwargs

    config_key = _get_task_fan_double_layer_config_key(task, kwargs)
    if config_key not in ROTATE_FAN_DOUBLE_LAYER_CONFIGS:
        task_name = getattr(task, "task_name", None) or task.__class__.__name__
        available = ", ".join(sorted(ROTATE_FAN_DOUBLE_LAYER_CONFIGS.keys()))
        raise KeyError(
            f"Unknown fan_double layer config for {task_name}: {config_key!r}. "
            f"Available configs: {available}"
        )

    layer_config = dict(ROTATE_FAN_DOUBLE_LAYER_CONFIGS[config_key])
    if bool(layer_config.pop("fan_double_upper_align_with_lower", False)):
        fan_angle_deg = float(kwargs.get("fan_angle_deg", layer_config.get("fan_angle_deg", 150.0)))
        half_angle_deg = 0.5 * max(fan_angle_deg, 0.0)
        layer_config["fan_double_upper_theta_start_deg"] = -half_angle_deg
        layer_config["fan_double_upper_theta_end_deg"] = half_angle_deg
        layer_config["fan_double_upper_theta_offset_deg"] = 0.0
    for key, value in layer_config.items():
        kwargs[key] = value

    kwargs["fan_double_lower_outer_radius"] = kwargs.get("fan_outer_radius", 0.9)
    kwargs["fan_double_lower_inner_radius"] = kwargs.get("fan_inner_radius", 0.3)
    kwargs["fan_double_layer_config_key"] = config_key
    kwargs["rotate_fan_double_layer_config_key"] = config_key
    return kwargs


def prepare_rotate_task_kwargs(task, kwargs):
    """Apply the task-selected table shape and validate YAML-provided geometry config."""
    kwargs = dict(kwargs)
    table_shape = str(getattr(task, "ROTATE_TABLE_SHAPE", kwargs.get("table_shape", "fan"))).lower()
    kwargs["table_shape"] = table_shape
    kwargs = apply_rotate_fan_double_layer_config(task, kwargs)

    required_keys = ROTATE_TABLE_CONFIG_REQUIRED_KEYS.get(table_shape, ())
    missing_keys = [key for key in required_keys if key not in kwargs]
    if missing_keys and bool(kwargs.get("validate_rotate_table_config", True)):
        task_name = getattr(task, "task_name", None) or task.__class__.__name__
        profile = kwargs.get("task_table_config_key", getattr(task, "ROTATE_TABLE_CONFIG_KEY", table_shape))
        raise KeyError(
            f"Missing rotate table config for {task_name} ({profile}): "
            f"{', '.join(missing_keys)}"
        )

    return init_rotate_theta_bounds(task, kwargs)


def _normalize_theta_range(theta_lim):
    arr = np.array(theta_lim, dtype=np.float64).reshape(-1)
    if arr.shape[0] == 0:
        return np.array([0.0, 0.0], dtype=np.float64)
    if arr.shape[0] == 1:
        return np.array([arr[0], arr[0]], dtype=np.float64)
    lo, hi = float(arr[0]), float(arr[1])
    if hi < lo:
        lo, hi = hi, lo
    return np.array([lo, hi], dtype=np.float64)


def init_rotate_theta_bounds(
    task,
    kwargs,
    default_fan_angle_deg=220.0,
    default_object_margin_deg=10.0,
    default_reference_fan_angle_deg=220.0,
    min_object_half_deg=5.0,
    default_scan_theta_unit_deg=DEFAULT_SCAN_THETA_UNIT_DEG,
):
    """
    Initialize per-task theta adaptation states for rotate-view tasks.

    Runtime config keys:
      - fan_angle_deg
      - rotate_object_margin_deg
      - rotate_theta_reference_fan_angle_deg
      - rotate_min_object_half_deg
      - rotate_theta_shared_ratio
      - rotate_scan_theta_unit_deg
      - rotate_scan_quantize_mode
      - rotate_scan_min_steps
      - rotate_scan_large_swing_first
    """
    fan_angle_deg = float(kwargs.get("fan_angle_deg", default_fan_angle_deg))
    object_margin_deg = float(kwargs.get("rotate_object_margin_deg", default_object_margin_deg))
    reference_fan_angle_deg = float(
        kwargs.get("rotate_theta_reference_fan_angle_deg", default_reference_fan_angle_deg)
    )
    min_half_deg = float(kwargs.get("rotate_min_object_half_deg", min_object_half_deg))

    table_half_rad = np.deg2rad(max(fan_angle_deg * 0.5, 0.0))
    object_half_rad = max(table_half_rad - np.deg2rad(max(object_margin_deg, 0.0)), np.deg2rad(max(min_half_deg, 0.0)))
    ref_object_half_rad = max(
        np.deg2rad(max(reference_fan_angle_deg * 0.5 - object_margin_deg, min_half_deg)),
        1e-6,
    )

    task.rotate_table_theta_half_rad = float(table_half_rad)
    task.rotate_object_theta_half_rad = float(object_half_rad)
    task.rotate_theta_scale = float(object_half_rad / ref_object_half_rad)
    task.rotate_fan_angle_deg = float(fan_angle_deg)
    task.rotate_object_margin_deg = float(object_margin_deg)
    task.rotate_theta_reference_fan_angle_deg = float(reference_fan_angle_deg)
    shared_ratio = float(kwargs.get("rotate_theta_shared_ratio", DEFAULT_ROTATE_THETA_SHARED_RATIO))
    shared_ratio = float(np.clip(shared_ratio, 0.0, 1.0))
    task.rotate_theta_shared_ratio = shared_ratio
    task.rotate_theta_shared_half_rad = float(task.rotate_object_theta_half_rad * shared_ratio)
    scan_theta_unit_deg = max(float(kwargs.get("rotate_scan_theta_unit_deg", default_scan_theta_unit_deg)), 0.0)
    task.rotate_scan_theta_unit_deg = float(scan_theta_unit_deg)
    task.rotate_scan_theta_unit_rad = float(np.deg2rad(scan_theta_unit_deg))
    task.rotate_scan_quantize_mode = str(kwargs.get("rotate_scan_quantize_mode", DEFAULT_SCAN_QUANTIZE_MODE))
    task.rotate_scan_min_steps = max(int(kwargs.get("rotate_scan_min_steps", DEFAULT_SCAN_MIN_STEPS)), 0)
    task.rotate_scan_large_swing_first = bool(kwargs.get("rotate_scan_large_swing_first", True))
    task.rotate_scan_strategy = str(kwargs.get("rotate_scan_strategy", DEFAULT_SCAN_STRATEGY)).lower()
    task.rotate_scan_order = str(kwargs.get("rotate_scan_order", DEFAULT_SCAN_ORDER)).lower()
    task.rotate_scan_visibility_mode = str(
        kwargs.get("rotate_scan_visibility_mode", DEFAULT_SCAN_VISIBILITY_MODE)
    ).lower()
    task.rotate_scan_visibility_camera_name = str(
        kwargs.get("rotate_scan_visibility_camera_name", DEFAULT_SCAN_VISIBILITY_CAMERA_NAME)
    )
    task.rotate_scan_horizontal_margin_deg = max(
        float(kwargs.get("rotate_scan_horizontal_margin_deg", DEFAULT_SCAN_HORIZONTAL_MARGIN_DEG)),
        0.0,
    )
    task.rotate_scan_vertical_margin_deg = max(
        float(kwargs.get("rotate_scan_vertical_margin_deg", DEFAULT_SCAN_VERTICAL_MARGIN_DEG)),
        0.0,
    )
    task.rotate_scan_horizontal_margin_rad = float(np.deg2rad(task.rotate_scan_horizontal_margin_deg))
    task.rotate_scan_vertical_margin_rad = float(np.deg2rad(task.rotate_scan_vertical_margin_deg))
    task.rotate_scan_aabb_visible_ratio_threshold = float(
        np.clip(
            float(
                kwargs.get(
                    "rotate_scan_aabb_visible_ratio_threshold",
                    DEFAULT_SCAN_AABB_VISIBLE_RATIO_THRESHOLD,
                )
            ),
            0.0,
            1.0,
        )
    )
    task.rotate_scan_reference_r = max(float(kwargs.get("rotate_scan_reference_r", DEFAULT_SCAN_REFERENCE_R)), 0.0)
    task.rotate_scan_joint_name_prefer = str(kwargs.get("rotate_scan_joint_name_prefer", DEFAULT_SCAN_JOINT_NAME))
    rotate_stage1_theta_unit_deg = max(
        float(kwargs.get("rotate_stage1_theta_unit_deg", DEFAULT_STAGE1_THETA_UNIT_DEG)),
        0.0,
    )
    task.rotate_stage1_theta_unit_deg = float(rotate_stage1_theta_unit_deg)
    task.rotate_stage1_theta_unit_rad = float(np.deg2rad(rotate_stage1_theta_unit_deg))
    rotate_stage2_max_theta_step_deg = max(
        float(kwargs.get("rotate_stage2_max_theta_step_deg", DEFAULT_STAGE2_MAX_THETA_STEP_DEG)),
        0.0,
    )
    task.rotate_stage2_max_theta_step_deg = float(rotate_stage2_max_theta_step_deg)
    task.rotate_stage2_max_theta_step_rad = float(np.deg2rad(rotate_stage2_max_theta_step_deg))
    rotate_stage2_center_tol_deg = max(
        float(kwargs.get("rotate_stage2_center_tol_deg", DEFAULT_STAGE2_CENTER_TOL_DEG)),
        0.0,
    )
    task.rotate_stage2_center_tol_deg = float(rotate_stage2_center_tol_deg)
    task.rotate_stage2_center_tol_rad = float(np.deg2rad(rotate_stage2_center_tol_deg))
    scan_sequence_steps = kwargs.get("rotate_scan_sequence_steps", DEFAULT_SCAN_SEQUENCE_STEPS)
    if isinstance(scan_sequence_steps, str):
        scan_sequence_steps = [item.strip() for item in scan_sequence_steps.split(",") if item.strip()]
    parsed_steps = []
    for step in np.array(scan_sequence_steps, dtype=np.float64).reshape(-1).tolist():
        parsed_steps.append(int(np.round(step)))
    if len(parsed_steps) == 0:
        parsed_steps = list(DEFAULT_SCAN_SEQUENCE_STEPS)
    task.rotate_scan_sequence_steps = tuple(parsed_steps)

    return kwargs


def quantize_theta_to_unit(
    theta_rad,
    unit_rad,
    mode: str = DEFAULT_SCAN_QUANTIZE_MODE,
    min_steps: int = 0,
    max_abs_rad=None,
):
    theta = float(theta_rad)
    unit = max(float(unit_rad), 0.0)
    if unit <= 1e-9:
        if max_abs_rad is None:
            return theta
        return float(np.clip(theta, -float(max_abs_rad), float(max_abs_rad)))

    abs_theta = abs(theta)
    raw_steps = abs_theta / unit
    mode = str(mode).lower()
    if mode == "nearest":
        steps = int(np.round(raw_steps))
    elif mode == "inward":
        steps = int(np.floor(raw_steps + 1e-9))
    else:
        steps = int(np.ceil(raw_steps - 1e-9))

    if abs_theta > 1e-9:
        steps = max(steps, int(min_steps))

    if max_abs_rad is not None:
        max_steps = int(np.floor(max(float(max_abs_rad), 0.0) / unit + 1e-9))
        steps = min(steps, max_steps)

    if steps <= 0:
        return 0.0
    return float(np.copysign(steps * unit, theta))


def quantize_scan_thetas_for_task(task, theta_list):
    thetas = np.array(theta_list, dtype=np.float64).reshape(-1)
    if thetas.shape[0] == 0:
        return []

    unit_rad = float(getattr(task, "rotate_scan_theta_unit_rad", 0.0))
    mode = str(getattr(task, "rotate_scan_quantize_mode", DEFAULT_SCAN_QUANTIZE_MODE))
    min_steps = int(getattr(task, "rotate_scan_min_steps", DEFAULT_SCAN_MIN_STEPS))
    max_abs_rad = float(getattr(task, "rotate_object_theta_half_rad", np.pi))

    quantized = []
    for theta in thetas.tolist():
        snapped = quantize_theta_to_unit(
            theta,
            unit_rad=unit_rad,
            mode=mode,
            min_steps=min_steps,
            max_abs_rad=max_abs_rad,
        )
        if not any(abs(snapped - existing) < 1e-6 for existing in quantized):
            quantized.append(snapped)

    if bool(getattr(task, "rotate_scan_large_swing_first", True)):
        quantized.sort(key=lambda val: (-abs(val), -val))
    return quantized


def sort_scan_thetas(theta_list, order=DEFAULT_SCAN_ORDER):
    thetas = np.array(theta_list, dtype=np.float64).reshape(-1)
    if thetas.shape[0] == 0:
        return []

    ordered = [float(val) for val in thetas.tolist()]
    order = str(order).lower()
    if order in ("left_to_right", "left-right", "ltr"):
        ordered.sort(reverse=True)
    elif order in ("right_to_left", "right-left", "rtl"):
        ordered.sort()
    elif order in ("center_out", "center-out"):
        ordered.sort(key=lambda val: (abs(val), -val))
    elif order in ("outside_in", "outside-in", "large_swing_first"):
        ordered.sort(key=lambda val: (-abs(val), -val))
    return ordered


def sort_scan_thetas_for_task(task, theta_list):
    return sort_scan_thetas(theta_list, order=getattr(task, "rotate_scan_order", DEFAULT_SCAN_ORDER))


def build_scan_theta_search_sequence_for_task(task):
    unit_rad = float(getattr(task, "rotate_scan_theta_unit_rad", 0.0))
    max_abs_rad = float(getattr(task, "rotate_object_theta_half_rad", np.pi))
    if unit_rad <= 1e-9:
        return [0.0]

    max_step = int(np.floor(max(max_abs_rad, 0.0) / unit_rad + 1e-9))
    if max_step <= 0:
        return [0.0]

    sequence = []
    for raw_step in getattr(task, "rotate_scan_sequence_steps", DEFAULT_SCAN_SEQUENCE_STEPS):
        step = int(raw_step)
        if step > 0:
            step = min(step, max_step)
        elif step < 0:
            step = -min(abs(step), max_step)
        snapped = float(step * unit_rad)
        if not any(abs(snapped - existing) < 1e-6 for existing in sequence):
            sequence.append(snapped)

    if len(sequence) == 0:
        return [float(max_step * unit_rad), float(-max_step * unit_rad), 0.0]
    return sequence


def rotate_theta_half(task):
    half = float(getattr(task, "rotate_theta_shared_half_rad", np.pi))
    return max(0.0, half)


def rotate_theta_center(task):
    half = rotate_theta_half(task)
    return [-half, half]


def rotate_theta_side(task, side=1.0):
    side_sign = 1.0 if float(side) >= 0.0 else -1.0
    outer = rotate_theta_half(task)
    inner = outer * DEFAULT_SIDE_INNER_RATIO
    if side_sign > 0:
        return [inner, outer]
    return [-outer, -inner]


def rotate_theta_fixed(task, side=1.0):
    sign = 1.0 if float(side) >= 0.0 else -1.0
    val = rotate_theta_half(task) * DEFAULT_FIXED_RATIO * sign
    return [val, val]


def rotate_theta_mixed(task, side=1.0):
    far = rotate_theta_half(task)
    near = far * DEFAULT_MIXED_NEAR_RATIO
    if float(side) >= 0.0:
        return [-far, near]
    return [-near, far]


def adapt_rotate_theta_range(task, theta_lim):
    """
    Adapt an old hard-coded theta range (designed around reference fan angle)
    to current fan-angle workspace.
    """
    lo, hi = _normalize_theta_range(theta_lim)
    scale = float(getattr(task, "rotate_theta_scale", 1.0))
    half = float(getattr(task, "rotate_object_theta_half_rad", np.pi))

    lo = float(np.clip(lo * scale, -half, half))
    hi = float(np.clip(hi * scale, -half, half))

    if hi < lo:
        mid = float(np.clip(0.5 * (lo + hi), -half, half))
        lo, hi = mid, mid

    return [lo, hi]
