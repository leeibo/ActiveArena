# ActiveArena

ActiveArena is the simulator, task suite, dataset tooling, and fixed-seed evaluator for active visual perception in robotic manipulation. It ships a frozen simulator snapshot with a controllable Astribot S1 embodiment, large workspaces, hidden evidence, process-level annotations, and ID/OOD protocols.

<p align="center">
  <img src="docs/images/teaser.png" alt="ActiveArena benchmark rollout" width="96%">
</p>

<p align="center"><b>35 tasks · 2 families · 5 categories · active viewpoints · memory</b></p>

| Project | Purpose |
| --- | --- |
| **ActiveArena** | Simulation, collection, assets, task configs and evaluation |
| [ActiveArena-VLA](https://github.com/leeibo/ActiveArena-VLA) | Training and serving the released VLA policies |
| [ActiveArena-VLA weights](https://huggingface.co/leeibo/ActiveArena-VLA) | Three 100,000-step checkpoint bundles and SHA-256 manifest |
| [Project website](https://leeibo.github.io/ActiveArena) | Visual overview, videos and release instructions |

Public release links: [simulator code](https://github.com/leeibo/ActiveArena), [VLA code](https://github.com/leeibo/ActiveArena-VLA), [training data](https://huggingface.co/datasets/leeibo/ActiveArena-Data), and [static assets](https://huggingface.co/datasets/leeibo/ActiveArena-Assets).

> **Release status.** This checkout is the public-facing benchmark code. Large model bundles are published through the [ActiveArena-VLA Hugging Face repository](https://huggingface.co/leeibo/ActiveArena-VLA). The source archive at the workspace root contains the AAAI LaTeX files; its bundled submission PDFs are formatting templates and are not linked as the paper.

## Benchmark at a glance

ActiveArena evaluates language-conditioned manipulation when the initial observation is insufficient. The Astribot action vector has 18 dimensions covering both arms, grippers, torso, and head. Tasks require the policy to alternate between information acquisition and task execution while retaining evidence across viewpoints.

The 35 tasks span two families—visual search and interactive information acquisition—and five categories:

- **SS — Single-Object Search:** locate one target and manipulate it.
- **SL — Single-Object Loop:** search across regions and transport one object.
- **ML — Multi-Object Loop:** complete several perception–action loops.
- **MD — Multi-Object Decision:** compare candidates using acquired evidence.
- **IA — Interactive Information Acquisition:** interact with the scene to reveal hidden information.

The paper's training protocol uses 100 ID trajectories per task (581.2k frames / 10.76 hours). Those raw demonstrations and converted training datasets are not included in this code checkout. This release contains the collection and annotation tooling plus fixed lists of 50 ID and 50 OOD evaluation seeds. The evaluator can run either the demonstration or randomized scene configuration.

<p align="center">
  <img src="docs/images/benchmark.png" alt="ActiveArena benchmark tasks and simulator with controllable head and torso" width="100%">
</p>

The paper studies 13 VLA configurations. The companion [ActiveArena-VLA](https://github.com/leeibo/ActiveArena-VLA) release packages three OFT training recipes and checkpoint bundles.

## Install the ActiveArena simulator

Clone the simulator and its two public release companions into one workspace:

```bash
mkdir -p activearena-release && cd activearena-release
git clone https://github.com/leeibo/ActiveArena.git
git clone https://github.com/leeibo/ActiveArena-VLA.git
cd ActiveArena
```

The checkpoints are large Git-LFS files and are downloaded separately below.

Use Linux with Python 3.10, an NVIDIA GPU, a CUDA toolkit compatible with the pinned PyTorch build, Vulkan rendering, and `ffmpeg`. The release includes a pinned environment definition and vendored simulator source:

```bash
conda env create -f environment.yml
conda activate activearena-sim
sudo apt-get update
sudo apt-get install -y git curl unzip ffmpeg libvulkan1 mesa-vulkan-drivers vulkan-tools \
  libx11-6 libxext6 libxrender1 libxfixes3 libxrandr2 libxi6 build-essential
bash script/_install.sh
python -c "import torch, sapien, mplib, curobo, yaml, h5py; print('ActiveArena imports OK')"
```

The cuRobo installation and evaluation launcher require the CUDA compiler (`nvcc`). If CUDA is installed outside the conda environment, set `CUDA_HOME` for installation and `ACTIVEARENA_CUDA_HOME` for evaluation to that toolkit directory. Use the separate `activearena-vla` environment for the policy code, as described in the [ActiveArena-VLA install guide](https://github.com/leeibo/ActiveArena-VLA#install).

Download the pinned base assets, then the ActiveArena additions:

```bash
bash script/_download_assets.sh
python script/download_activearena_assets.py
```

The downloader verifies checksums and installs the button object and Astribot embodiment into `assets/`. The exact source and asset revisions are recorded in [docs/UPSTREAM_SNAPSHOTS.md](docs/UPSTREAM_SNAPSHOTS.md).

`script/_download_assets.sh` downloads the three pinned base archives from the
RoboTwin2.0 dataset snapshot. `script/download_activearena_assets.py` downloads
the two ActiveArena additions from the public [ActiveArena-Assets dataset](https://huggingface.co/datasets/leeibo/ActiveArena-Assets).
Both commands can be retried safely and validate the expected archive layout
before removing downloaded archives or exposing an asset directory to the
simulator.

## Collect data

```bash
bash collect_data.sh count_target_press_button info_gathering_demo 0
bash collect_data.sh count_target_press_button info_gathering_randomized 0
```

The arguments are task name, task configuration, and GPU ID. Collection writes HDF5 trajectories, videos, scene metadata, and episode instructions below `data/<task>/<config>__<difficulty_tag>/`. Set `episode_num` in `task_config/*.yml` to control the number of episodes.

For training, convert the collected trajectories to the LeRobot layout expected by [ActiveArena-VLA](https://github.com/leeibo/ActiveArena-VLA). The required fields and setup are recorded in [docs/reproduction.md](docs/reproduction.md).

The released converted LeRobot demonstrations are available from the [ActiveArena-Data dataset](https://huggingface.co/datasets/leeibo/ActiveArena-Data). They are consumed by ActiveArena-VLA; they are not simulator HDF5 collection output:

```bash
huggingface-cli download leeibo/ActiveArena-Data \
  --repo-type dataset \
  --local-dir /path/to/ActiveArena-VLA/playground/dataset/ActiveArena_Astribot_lerobot
```

## Evaluate a released policy

Place the matching checkpoint bundle beside the two repositories. Download the
base VLM from [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
into the path expected by the released configs (or set
`ACTIVEARENA_VLA_BASE_VLM` to another local snapshot):

```bash
cd ../ActiveArena-VLA
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir playground/Pretrained_models/ActiveArena/Qwen3-VL-2B-Instruct
cd ../ActiveArena
```

Download the released action bundles from [leeibo/ActiveArena-VLA](https://huggingface.co/leeibo/ActiveArena-VLA)
with the Hugging Face CLI (this avoids leaving Git-LFS pointer files in place),
then verify their manifest:

```bash
mkdir -p ../ActiveArena-VLA-weights
huggingface-cli download leeibo/ActiveArena-VLA \
  --local-dir ../ActiveArena-VLA-weights
cd ../ActiveArena-VLA-weights
./verify.sh
cd ../ActiveArena
```

Run a dry check before starting a policy server. The launcher uses the
`activearena-sim` environment by default. For a custom environment, set
`ACTIVEARENA_PYTHON` (the legacy `ROBOTWIN_PYTHON` alias remains accepted):

```bash
GPU_LIST=0 DRY_RUN=1 \
  bash eval_seed.sh oft_subtask_action_12_ws info_gathering_demo
```

After the dry check passes, a one-task, one-episode smoke run is:

```bash
GPU_LIST=0 TASK_LIMIT=1 EVAL_TEST_NUM=1 TMUX_MONITOR=0 \
  bash eval_seed.sh oft_subtask_action_12_ws info_gathering_demo
```

This starts a real policy server and simulator worker. It requires an NVIDIA
GPU, `nvcc`, the installed cuRobo source tree, the base VLM, and the matching
checkpoint. `DRY_RUN=1` only validates paths and metadata; it does not test
inference.

Run all 35 tasks with fixed seeds after the dependencies are ready. The launcher starts the policy servers and evaluation workers:

```bash
GPU_LIST=0,1,2,3 EVAL_TEST_NUM=50 \
  bash eval_seed.sh oft_subtask_action_12_ws info_gathering_randomized
```

The three supported configurations are `oft_instruction_action_12_ws`, `oft_subtask_action_12_wos`, and `oft_subtask_action_12_ws`, all at step 100,000. Logs and reports are written to `logs/eval_seed/<task_config>/<model>/<run_id>/`. Use `TASK_LIMIT=1 EVAL_TEST_NUM=1` for a smoke run. Full environment variables and recovery instructions are in [ACTIVEARENA_EVAL_LAUNCH.md](ACTIVEARENA_EVAL_LAUNCH.md).

## Acknowledgements

ActiveArena builds on and gratefully acknowledges the following open-source projects:

- [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) for the simulator infrastructure and robot-manipulation foundation.
- [RMBench](https://github.com/RoboTwin-Platform/RMBench) for the articulated information-gathering assets used by the button tasks.
- [StarVLA](https://github.com/starVLA/starVLA) for the vision-language-action training and policy-serving foundation used by ActiveArena-VLA.

Please follow the original projects' licenses and citation requirements when using the corresponding code or assets.

## Asset layout

```text
assets/
├── background_texture/                  # pinned base simulation assets
├── objects/005_button/                  # ActiveArena object archive
└── embodiments/astribot_descriptions_texture/
                                         # Astribot URDF, meshes and cuRobo files
```

`python script/download_activearena_assets.py --assets-dir /path/to/assets` supports a custom cache. When the repository moves, rerun the downloader so absolute URDF and collision paths are rewritten.

## Reproduction and release notes

- [docs/reproduction.md](docs/reproduction.md) — end-to-end environment, data, model, and evaluation checklist.
- [docs/UPSTREAM_SNAPSHOTS.md](docs/UPSTREAM_SNAPSHOTS.md) — immutable simulator, asset, and source-build revisions.
- [docs/media.md](docs/media.md) — provenance and dimensions of the website demo clips (when present).
- [ACTIVEARENA_EVAL_LAUNCH.md](ACTIVEARENA_EVAL_LAUNCH.md) — multi-GPU policy-server/evaluator orchestration.
- [LICENSE](LICENSE) — benchmark repository license and upstream attribution.

Do not commit downloaded assets, checkpoints, API keys, or local result logs. The included task seeds and configuration files are the reproducibility surface; external assets and base models remain separately licensed.

## Citation

```bibtex
@misc{activearena_manuscript,
  title = {ActiveArena: Benchmarking and Understanding Active Perception in Robotic Manipulation},
  note  = {Manuscript submitted for review at AAAI 2027}
}
```

The supplied source does not establish the manuscript's author list, public URL, or DOI. This temporary entry deliberately omits those fields and does not imply acceptance. Replace it with the verified manuscript citation when available. Contributions and bug reports are welcome through the issue tracker of the public repository.
