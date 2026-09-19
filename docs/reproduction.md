# ActiveArena reproduction

This document records the portable path used for the AAAI 2027 ActiveArena
experiments. The simulator and the policy code are separate projects:
`ActiveArena` provides the vendored simulator/task snapshot and `ActiveArena-VLA`
serves or trains the policy. Keep the three repositories as siblings, or set the
paths explicitly. The exact simulator and asset revisions are recorded in
[`UPSTREAM_SNAPSHOTS.md`](UPSTREAM_SNAPSHOTS.md).

## Evaluation

Create the `activearena-sim` environment and install the pinned assets following
the repository README. Install ActiveArena-VLA's Python dependencies in a
separate environment,
and place the Qwen3-VL-2B-Instruct base model at
`ActiveArena-VLA/playground/Pretrained_models/ActiveArena/Qwen3-VL-2B-Instruct` (or set
`ACTIVEARENA_VLA_BASE_VLM`). Download or copy `ActiveArena-VLA-weights` beside the two
code repositories and verify it with:

```bash
cd ../ActiveArena-VLA-weights
./verify.sh
```

A fixed-seed evaluation then starts from the ActiveArena directory:

```bash
export ACTIVEARENA_VLA_ROOT=/path/to/ActiveArena-VLA
export ACTIVEARENA_VLA_WEIGHTS_ROOT=/path/to/ActiveArena-VLA-weights
export ACTIVEARENA_PYTHON=/path/to/conda/envs/activearena-sim/bin/python
export ACTIVEARENA_VLA_PYTHON=/path/to/conda/envs/activearena-vla/bin/python
GPU_LIST=0 TASK_LIMIT=1 EVAL_TEST_NUM=1 DRY_RUN=1 \
  bash ./eval_seed.sh oft_subtask_action_12_ws info_gathering_demo
```

This is a configuration check only; it does not run the simulator or model
inference. For a one-episode smoke run, remove `DRY_RUN=1`. For the full paper
protocol, also remove `TASK_LIMIT=1` and `EVAL_TEST_NUM=1`:

```bash
GPU_LIST=0 EVAL_TEST_NUM=50 EARLY_STOP_ZERO_EPISODES=0 \
  bash ./eval_seed.sh oft_subtask_action_12_ws info_gathering_demo
GPU_LIST=0 EVAL_TEST_NUM=50 EARLY_STOP_ZERO_EPISODES=0 \
  bash ./eval_seed.sh oft_subtask_action_12_ws info_gathering_randomized
```

Both settings contain the same 35 tasks as the training data registry. Each
task's committed JSON contains 100 unique, ordered valid seeds; the paper
protocol evaluates the first 50 entries without skipping unsuccessful policy
episodes. `EVAL_TEST_NUM=50` and `EARLY_STOP_ZERO_EPISODES=0` are the launcher
defaults. A positive early-stop threshold is for debugging only: the stopped
task is incomplete (exit status 75), has no full-run success rate, and is not
imported as completed by `RESUME_FROM_RUN`. Legacy
`early_stop_assumed_result.txt` rows are also excluded when resuming.

The supported bundles are
`oft_instruction_action_12_ws`, `oft_subtask_action_12_wos`, and
`oft_subtask_action_12_ws`; each is the 100,000-step checkpoint in the weights
repository. `GPU_LIST`, `TASK_LIST`, `EVAL_TEST_NUM`, and `RESUME_FROM_RUN` make
runs reproducible and resumable. Full launcher details are in
[`ACTIVEARENA_EVAL_LAUNCH.md`](../ACTIVEARENA_EVAL_LAUNCH.md).

## Data and training

Use the real conversion tool in the sibling ActiveArena-VLA project:

```bash
cd ../ActiveArena-VLA
pip install -r tools/requirements-data.txt
python tools/convert_astribot_to_lerobot.py \
  --raw-root ../ActiveArena/data \
  --output-root playground/dataset/ActiveArena_Astribot_lerobot \
  --config info_gathering_demo__info_gathering_demo --dry-run
```

Remove `--dry-run` to convert. The complete field contract, 19-to-18 joint
mapping, RGB convention, and training commands are in the sibling project's
[`docs/data-conversion.md`](../../ActiveArena-VLA/docs/data-conversion.md).
Keep fixed evaluation episodes separate from training demonstrations. The
release conversion check validated one real episode; complete training and
GPU simulator evaluation were not rerun during release preparation.
