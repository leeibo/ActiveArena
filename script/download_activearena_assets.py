#!/usr/bin/env python3
"""Install ActiveArena's supplementary assets (Python standard library + curl)."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DATASET_REVISION = "819632dfd545b569657e195cc47aae3af7fce5f1"
BASE_URL = f"https://huggingface.co/datasets/leeibo/ActiveArena-Assets/resolve/{ASSET_DATASET_REVISION}"
PACKAGES = (
    {
        "archive": "005_button.zip",
        "sha256": "a6340c04fbd107e6728c21bfff3be58f6c6862093a7acb197ae41f06322b0a63",
        "parent": "objects",
        "root": "005_button",
        "required": ("10124/mobility.urdf", "10124/model_data.json"),
    },
    {
        "archive": "astribot_descriptions_texture_20260630_174911.zip",
        "sha256": "49be39c500e763d6d0194d4449751967bff16df5a2c044f1f53f7e9b7adffcb7",
        "parent": "embodiments",
        "root": "astribot_descriptions_texture",
        "required": (
            "config.yml", "astribot_whole_body_maniskill_merged.urdf",
            "curobo_left.yml", "curobo_right.yml",
            "collision_left.yml", "collision_right.yml",
        ),
    },
)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(package, cache_dir):
    archive = cache_dir / package["archive"]
    if archive.is_file() and checksum(archive) == package["sha256"]:
        print(f"Using verified cache: {archive}", flush=True)
        return archive
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required; install it before downloading assets")
    partial = archive.with_suffix(".zip.part")
    print(f"Downloading {package['archive']}", flush=True)
    subprocess.run([
        "curl", "-L", "--fail", "--retry", "3", "--connect-timeout", "30",
        "--max-time", "1800", "--output", str(partial),
        f"{BASE_URL}/{package['archive']}",
    ], check=True)
    if checksum(partial) != package["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch: {partial}; asset was not installed")
    partial.replace(archive)
    return archive


def check_layout(root, package):
    for name in package["required"]:
        if not (root / name).is_file():
            raise RuntimeError(f"Incomplete asset: missing {root / name}")
    if package["parent"] == "embodiments" and not (root / "meshes").is_dir():
        raise RuntimeError(f"Incomplete robot asset: missing {root / 'meshes'}")


def install(package, assets_dir, cache_dir):
    parent = assets_dir / package["parent"]
    target = parent / package["root"]
    if target.exists():
        check_layout(target, package)
        print(f"Already installed: {target}", flush=True)
        return
    archive = download(package, cache_dir)
    parent.mkdir(parents=True, exist_ok=True)
    # Stage and validate before making the asset visible to collection scripts.
    with tempfile.TemporaryDirectory(prefix=".activearena-extract-", dir=parent) as tmp:
        stage = Path(tmp)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                name = PurePosixPath(member.filename)
                if (name.is_absolute() or ".." in name.parts
                        or "\\" in member.filename
                        or stat.S_ISLNK(member.external_attr >> 16)):
                    raise RuntimeError(f"Unsafe archive entry: {member.filename}")
                if not name.parts:
                    continue
                if (name.parts[0] == "__MACOSX" or name.name.startswith("._")
                        or name.name == ".DS_Store" or ".bak" in name.name):
                    continue
                if name.parts[0] != package["root"]:
                    raise RuntimeError(f"Unexpected archive directory: {member.filename}")
                bundle.extract(member, stage)
        check_layout(stage / package["root"], package)
        (stage / package["root"]).rename(target)
    print(f"Installed: {target}", flush=True)


def configure_robot_paths(assets_dir):
    robot = assets_dir / "embodiments" / "astribot_descriptions_texture"
    for arm in ("left", "right"):
        config = robot / f"curobo_{arm}.yml"
        content = config.read_text(encoding="utf-8")
        for field, filename in (
            ("urdf_path", "astribot_whole_body_maniskill_merged.urdf"),
            ("collision_spheres", f"collision_{arm}.yml"),
        ):
            content, count = re.subn(
                rf"(?m)^(\s*{field}:)[^\n]*$",
                lambda match: f"{match[1]} {json.dumps(str(robot / filename))}",
                content,
            )
            if count != 1:
                raise RuntimeError(f"Expected one {field} field in {config}, found {count}")
        config.write_text(content, encoding="utf-8")
    print(f"Configured cuRobo paths for {robot}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=REPO_ROOT / "assets")
    parser.add_argument("--cache-dir", type=Path, help="Default: <assets-dir>/.downloads")
    args = parser.parse_args()
    assets_dir = args.assets_dir.expanduser().resolve()
    cache_dir = (args.cache_dir or assets_dir / ".downloads").expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        for package in PACKAGES:
            install(package, assets_dir, cache_dir)
        configure_robot_paths(assets_dir)
    except (OSError, RuntimeError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Asset installation failed: {exc}\n")


if __name__ == "__main__":
    main()
