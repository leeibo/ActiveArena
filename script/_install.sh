#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Keep source dependencies immutable. These revisions correspond to the
# versions recorded in docs/UPSTREAM_SNAPSHOTS.md.
PYTORCH3D_COMMIT="89653419d0973396f3eff1a381ba09a07fffc2ed"
CUROBO_COMMIT="d64c4b005459db10c5dd867d8b30a87d5bda9bdb"
CUROBO_DIR="${REPO_ROOT}/envs/curobo"

echo "Installing the ActiveArena simulation packages ..."
python -m pip install -r script/requirements.txt

echo "Installing pinned PyTorch3D (${PYTORCH3D_COMMIT}) ..."
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_COMMIT}" --no-build-isolation

echo "Adjusting code in sapien/wrapper/urdf_loader.py ..."
# SAPIEN_LOCATION points into the activearea-sim environment.
SAPIEN_LOCATION=$(python -m pip show sapien | awk '/^Location:/{print $2}')/sapien
# Adjust some code in wrapper/urdf_loader.py
URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
# ----------- before -----------
# 667         with open(urdf_file, "r") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + "srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
# ----------- after  -----------
# 667         with open(urdf_file, "r", encoding="utf-8") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + "srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r", encoding="utf-8") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
sed -i -E \
  -e 's/("r")(\))( as)/\1, encoding="utf-8") as/g' \
  "${URDF_LOADER}"


echo "Adjusting code in mplib/planner.py ..."
# MPLIB_LOCATION points into the activearea-sim environment.
MPLIB_LOCATION=$(python -m pip show mplib | awk '/^Location:/{print $2}')/mplib

# Adjust some code in planner.py
# ----------- before -----------
# 807             if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
# ----------- after  ----------- 
# 807             if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
PLANNER=$MPLIB_LOCATION/planner.py
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "${PLANNER}"

echo "Installing Curobo ..."
if [[ ! -d "${CUROBO_DIR}/.git" ]]; then
  rm -rf "${CUROBO_DIR}"
  git clone --filter=blob:none https://github.com/NVlabs/curobo.git "${CUROBO_DIR}"
fi
git -C "${CUROBO_DIR}" fetch --depth 1 origin "${CUROBO_COMMIT}"
git -C "${CUROBO_DIR}" checkout --detach "${CUROBO_COMMIT}"
python -m pip install -e "${CUROBO_DIR}" --no-build-isolation
python -m pip install warp-lang==1.12.0 setuptools==69.5.1

echo "Installation basic environment complete!"
echo -e "You need to:"
echo -e "    1. \033[34m\033[1m(Important!)\033[0m Download assets from huggingface."
echo -e "    2. Install requirements for running baselines. (Optional)"
echo "See README.md for more instructions."
