#!/usr/bin/env bash
set -Eeuo pipefail

# Resolve paths from this script so it also works when invoked by absolute path
# from outside the repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}/assets"
python _download.py

extract_base_archive() {
    local archive="$1"
    local expected_dir="$2"
    if [[ ! -f "${archive}" ]]; then
        echo "Downloaded asset archive is missing: ${REPO_ROOT}/assets/${archive}" >&2
        return 1
    fi

    # -o keeps retries non-interactive when an earlier extraction left files
    # behind. Keep the archive until extraction and the layout check succeed so
    # an interrupted run can be resumed safely.
    unzip -q -o "${archive}" -d "${REPO_ROOT}/assets"
    if [[ ! -d "${REPO_ROOT}/assets/${expected_dir}" ]]; then
        echo "Asset archive has an unexpected layout: ${archive} (missing ${expected_dir}/)" >&2
        return 1
    fi
    rm -f "${archive}"
}

extract_base_archive background_texture.zip background_texture
extract_base_archive embodiments.zip embodiments
extract_base_archive objects.zip objects

cd ..
echo "Configuring Path ..."
python ./script/update_embodiment_config_path.py
