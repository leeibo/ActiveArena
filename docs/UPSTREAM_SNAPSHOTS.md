# ActiveArena reproducibility snapshots

ActiveArena vendors the simulator and task code needed by the benchmark. The
following immutable revisions define the release used for the paper and keep
future upstream changes from silently changing an installation.

| Component | Revision | How it is used |
| --- | --- | --- |
| Simulator/task source | `RoboTwin_Astribot@53e5fc8` | The vendored `envs/`, `description/`, `code_gen/`, and simulator utilities in this checkout were copied from this commit. |
| Base asset dataset | `TianxingChen/RoboTwin2.0@981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042` | `assets/_download.py` downloads the three base archives at this revision. |
| ActiveArena asset dataset | `leeibo/ActiveArena-Assets@819632dfd545b569657e195cc47aae3af7fce5f1` | `script/download_activearena_assets.py` downloads the button and Astribot archives at this revision and verifies SHA-256 digests. |
| cuRobo | `NVlabs/curobo@d64c4b005459db10c5dd867d8b30a87d5bda9bdb` | `script/_install.sh` checks out this exact source revision. |
| PyTorch3D | `facebookresearch/pytorch3d@89653419d0973396f3eff1a381ba09a07fffc2ed` | `script/_install.sh` installs this exact source revision. |

The Python package versions are listed in [`script/requirements.txt`](../script/requirements.txt)
and the environment entry point is [`environment.yml`](../environment.yml).
The source snapshot is intentionally kept in this repository instead of being
loaded from an upstream checkout at runtime.

## Stable environment interface

Create the simulator environment with the release name:

```bash
conda env create -f environment.yml
conda activate activearena-sim
bash script/_install.sh
```

Launchers use the following `ACTIVEARENA_*` variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `ACTIVEARENA_ROOT` | ActiveArena checkout | Directory containing `eval_seed.sh` |
| `ACTIVEARENA_PYTHON` | Python executable for simulation workers | Python in `activearena-sim` |
| `ACTIVEARENA_ENV_NAME` | Conda environment used for discovery | `activearena-sim` |
| `ACTIVEARENA_CUROBO_SRC` | cuRobo source tree | `$ACTIVEARENA_ROOT/envs/curobo/src` |
| `ACTIVEARENA_CUDA_HOME` | CUDA toolkit containing `nvcc` | Environment prefix |
| `ACTIVEARENA_TORCH_EXTENSIONS_DIR` | Torch extension cache | `logs/torch_extensions` |

`ROBOTWIN_*` variables remain accepted as deprecated aliases so existing local
scripts continue to work. They are only compatibility names; new commands
should use the `ACTIVEARENA_*` interface. Internal compatibility symbols such
as the `envs` Python package and the action-order constant are unchanged because
they are part of the saved policy/evaluation protocol.
