from ._match_backside_blocks import MatchBacksideBlocksBase


class match_backside_two_blocks(MatchBacksideBlocksBase):
    """Inspect two gray blocks and move each to the pad matching its backside color."""

    BLOCK_COUNT = 2
    COLOR_SAMPLE_COUNT = 2

    TASK_INSTRUCTION = (
        "Inspect the backside colors of two gray blocks and place each block on "
        "the matching colored pad."
    )
    INFO_BLOCKS = "two gray backside-marked blocks"
    INFO_PADS = "two matching color pads"

    def check_success(self):
        return bool(
            super().check_success()
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
