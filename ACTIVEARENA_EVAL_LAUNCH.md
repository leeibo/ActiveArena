# ActiveArena-VLA 评测启动指南

本文说明如何启动 `eval_seed.sh` 支持的各个 ActiveArena-VLA
模型。先按 [安装指南](README.md) 配置仿真环境和资源。统一入口为
`bash eval_seed.sh MODEL_NAME [TASK_CONFIG]`，第二个参数可以是
`info_gathering_demo`（默认）或 `info_gathering_randomized`，各自使用对应的固定种子列表。
下文示例使用 `info_gathering_randomized`。换机器后需要重新设置仓库、Python 环境和
模型资源路径。

## 1. 目录和环境变量

推荐将两个代码仓库和权重仓库放在同一父目录下：

```text
workspace/
├── ActiveArena/
├── ActiveArena-VLA/
└── ActiveArena-VLA-weights/
```

也可以放在任意位置，此时显式设置以下变量：

```bash
export ACTIVEARENA_ROOT=/path/to/ActiveArena
export ACTIVEARENA_VLA_ROOT=/path/to/ActiveArena-VLA
export ACTIVEARENA_VLA_WEIGHTS_ROOT=/path/to/ActiveArena-VLA-weights

export ACTIVEARENA_PYTHON=/path/to/conda/envs/activearena-sim/bin/python
export ACTIVEARENA_VLA_PYTHON=/path/to/conda/envs/activearena-vla/bin/python

```

如果两个仓库互为同级目录，`ACTIVEARENA_VLA_ROOT` 可以省略。两个 Python 变量也可以
省略，启动脚本会尝试查找名为 `activearena-sim` 和`activearena-vla`的 conda 环境。

脚本从权重包中每个模型的 `config.yaml` 解析 base VLM；模型代码、权重和基础
VLM 建议保持以下目录结构：

```text
$ACTIVEARENA_VLA_ROOT/                         # ActiveArena-VLA code
├── playground/Pretrained_models/
│   ├── ActiveArena/Qwen3-VL-2B-Instruct/
│   ├── ActiveArena/Qwen3-VL-2B-Instruct-Action/
│   └── fast/
$ACTIVEARENA_VLA_WEIGHTS_ROOT/                # ActiveArena-VLA-weights
└── <MODEL_NAME>/
    ├── config.yaml
    ├── config.full.yaml
    ├── dataset_statistics.json
    └── checkpoints/steps_100000_pytorch_model.pt
```

如果 base VLM 位于其他目录，可覆盖：

```bash
export ACTIVEARENA_VLA_BASE_VLM=/path/to/ActiveArena/Qwen3-VL-2B-Instruct
```

通常不要全局设置 `ACTIVEARENA_VLA_BASE_VLM`，让脚本分别读取各模型的
`config.yaml`；只有模型配置里的路径在新机器上不可用时才覆盖它。

## 2. 推荐启动方式

先进入 ActiveArena 仓库：

```bash
cd "$ACTIVEARENA_ROOT"
```

建议在一个持久化的 tmux 会话里运行总启动脚本：

```bash
tmux new-session -s activearena-eval
```

进入 tmux 后运行：

```bash
MODEL=oft_subtask_action_12_ws
GPU_LIST="0,1,2,3" \
EVAL_TEST_NUM=50 \
EARLY_STOP_ZERO_EPISODES=0 \
ACTIVEARENA_VLA_ROOT="$ACTIVEARENA_VLA_ROOT" \
ACTIVEARENA_VLA_WEIGHTS_ROOT="$ACTIVEARENA_VLA_WEIGHTS_ROOT" \
ACTIVEARENA_ROOT="$ACTIVEARENA_ROOT" \
ACTIVEARENA_PYTHON="$ACTIVEARENA_PYTHON" \
ACTIVEARENA_VLA_PYTHON="$ACTIVEARENA_VLA_PYTHON" \
bash ./eval_seed.sh "$MODEL" info_gathering_randomized
```

总启动脚本会为每张卡启动一个策略服务和一个 ActiveArena worker，并在所有 worker
之间共享 35 个任务的队列。端口规则如下：

- Action server: `19000 + GPU ID`

正式默认是 `EVAL_TEST_NUM=50`、`EARLY_STOP_ZERO_EPISODES=0`：每个任务执行完
50 个回合，不提前将剩余回合假定为失败。demo 和 randomized 各有 35 份固定种子
JSON，每份保存 100 个互不重复的有效种子；正式协议按文件顺序取前 50 个。
两种设置的任务集合与训练 registry 的 35 个任务完全一致。

仅调试时可以设置正数 `EARLY_STOP_ZERO_EPISODES`。触发后任务会标记为未完成
（退出状态 75），不生成完整成功率，也不会被恢复功能当作已完成任务导入。
旧版生成的 `early_stop_assumed_result.txt` 也会在恢复时排除。

不要只运行 ActiveArena-VLA 里的 `run_policy_server.sh`。该脚本只会把模型加载到显卡并
监听端口，不会启动 `eval_policy.py`，因此不会真正执行 ActiveArena 任务。

启动后可用下面的命令确认服务端和评测端都存在：

```bash
pgrep -af 'server_policy.py|script/eval_policy.py'
nvidia-smi
```

## 3. 支持的三个模型

本仓库只携带下面三个模型的 ActiveArena 适配配置。权重包位于
`ActiveArena-VLA-weights/`，每个模型固定使用 100,000 step。先统一设置 GPU 和 episodes：

```bash
export GPU_LIST="0,1,2,3"
export EVAL_TEST_NUM=50
export EARLY_STOP_ZERO_EPISODES=0
cd "$ACTIVEARENA_ROOT"
```

| 模型 | Step | 启动命令 |
|---|---:|---|
| `oft_instruction_action_12_ws` | 100,000 | `bash ./eval_seed.sh oft_instruction_action_12_ws info_gathering_randomized` |
| `oft_subtask_action_12_wos` | 100,000 | `bash ./eval_seed.sh oft_subtask_action_12_wos info_gathering_randomized` |
| `oft_subtask_action_12_ws` | 100,000 | `bash ./eval_seed.sh oft_subtask_action_12_ws info_gathering_randomized` |

## 4. 启动前检查

启动器会检查 checkpoint、Python、base VLM、随机种子列表和 CUDA 工具链。OFT
权重使用连续动作头，不需要 FAST tokenizer。可以先做一次不启动服务和评测进程的配置检查：

```bash
GPU_LIST="0" \
EVAL_TEST_NUM=50 \
DRY_RUN=1 \
bash ./eval_seed.sh oft_subtask_action_12_ws info_gathering_randomized
```

`DRY_RUN=1` 只验证依赖和配置，不会执行模型推理或真实仿真，也不证明论文结果已复现。

正式启动前还应确认目标端口没有被旧服务占用：

```bash
ss -ltnp | grep -E ':1900[0-9]\b' || true
```

如果端口已被同一模型的孤立策略服务占用，应先正常停止旧服务，再从总启动脚本
重新启动。不要在同一端口上叠加第二个服务。

## 5. 日志、报告和恢复

每轮输出位于 `logs/eval_seed/<TASK_CONFIG>/<MODEL_NAME>/<RUN_ID>/`，例如：

```text
$ACTIVEARENA_ROOT/logs/eval_seed/info_gathering_randomized/<MODEL_NAME>/<RUN_ID>/
├── report.md
├── summary.tsv
├── servers.tsv
├── tasks/
├── servers/
├── gpu_consoles/
└── eval_result/
```

正式报告需确认 `Episodes per task: 50`、`Zero-success early stop: 0`，并同时满足：

```text
Progress: 35/35
Failed: 0
```

需要从上一轮继续时，可以导入上一轮中成功的任务：

```bash
MODEL=oft_subtask_action_12_ws
PREVIOUS_RUN="$ACTIVEARENA_ROOT/logs/eval_seed/info_gathering_randomized/$MODEL/<RUN_ID>"

GPU_LIST="0,1,2,3" \
EVAL_TEST_NUM=50 \
RESUME_FROM_RUN="$PREVIOUS_RUN" \
bash ./eval_seed.sh "$MODEL" info_gathering_randomized
```

只有 checkpoint、任务配置、种子列表、episodes 数量和早停设置都相同时才应使用
`RESUME_FROM_RUN`，否则新报告会混入不同评测设置的结果。

调试时可限制任务数量：

```bash
GPU_LIST="0" TASK_LIMIT=1 EVAL_TEST_NUM=1 \
bash ./eval_seed.sh oft_subtask_action_12_ws info_gathering_randomized
```

使用 demo 配置时，模型名和环境变量保持相同，第二个参数改为：

```bash
bash ./eval_seed.sh "$MODEL" info_gathering_demo
```
