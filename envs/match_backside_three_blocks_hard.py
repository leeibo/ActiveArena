from ._info_task_helpers import BACKSIDE_RGB_COLOR_SPECS
from ._match_backside_blocks import MatchBacksideBlocksBase


class match_backside_three_blocks_hard(MatchBacksideBlocksBase):
    """Hard three-pad/three-block backside color matching; at most two blocks share a color."""

    BLOCK_COUNT = 3
    COLOR_SPECS = BACKSIDE_RGB_COLOR_SPECS
    UNIQUE_LABEL_RATE = 0.55
    MAX_LABEL_REPEAT = 2
    PRE_INSPECT_DELAY = 2

    TASK_INSTRUCTION = (
        "Inspect three gray blocks and place each on the pad matching its hidden "
        "backside color; at most two blocks share a color."
    )
    INFO_BLOCKS = "three gray backside-marked blocks"
    INFO_PADS = "three matching color pads"
