from ._base_task import Base_Task, PutBlockFanDoubleMixin


class put_block_on_upper_hard(PutBlockFanDoubleMixin, Base_Task):
    ROTATE_TABLE_SHAPE = "fan_double"
    ROTATE_TABLE_CONFIG_KEY = "fan"
    ROTATE_FAN_DOUBLE_LAYER_CONFIG_KEY = "left_support"
    ROTATE_LOWER_LAYER_KEEP_HEAD_HOME = True

    BLOCK_COUNT = 2
    BLOCK_LAYER_SEQUENCE = ("lower", "lower")
    BLOCK_LAYER_SPECS = {
        "lower": {
            "r_min": 0.42,
            "r_max": 0.43,
            "theta_min_deg": -55.0,
            "theta_max_deg": 55.0,
        },
        "upper": {
            "inner_margin": 0.05,
            "outer_margin": 0.07,
            "max_cyl_r": 0.68,
            "theta_shrink": 0.96,
        },
    }

    PLATE_LAYER = "upper"
    PLATE_LOWER_LAYER_PROB = 0.2
    PLATE_LAYER_SPECS = {
        "lower": {
            "r": 0.44,
            "theta_deg": -60.0,
            "z_offset": 0.015,
            "qpos": [0.5, 0.5, 0.5, 0.5],
            "scale": [0.025, 0.025, 0.025],
        },
        "upper": {
            "r": 0.70,
            "theta_deg": 0.0,
            "z_offset": 0.0,
            "qpos": [0.5, 0.5, 0.5, 0.5],
            "scale": [0.025, 0.025, 0.025],
        },
    }
    PLATE_PLACE_SLOT_OFFSETS = 0.025

    LOWER_PLACE_PRE_DIS = 0.15
    RETURN_TO_HOMESTATE_AFTER_PLACE = False
    UPPER_PICK_ENTRY_Z_OFFSET = 0.06
    UPPER_PICK_PRE_GRASP_DIS = 0.06
    UPPER_PLACE_LATERAL_ESCAPE_DIS = 0.20
    RETURN_TO_RELEASE_ENTRY_AFTER_INTERMEDIATE_UPPER_PLACE = True
    END_AFTER_FINAL_DIRECT_RELEASE = True
