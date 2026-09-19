#!/usr/bin/env python3
"""
Batch update URDF link masses from a YAML config.

Default usage (from repo root, after activating the activearena-sim environment):
  python script/update_urdf_link_mass.py --dry-run
  python script/update_urdf_link_mass.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple

import yaml


DEFAULT_URDF = "assets/embodiments/astribot_descriptions/robot_mimic.urdf"
DEFAULT_CONFIG = "assets/embodiments/astribot_descriptions/mass_config.yml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update URDF link masses from YAML config.")
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="URDF file path to modify.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file path.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output URDF path. If omitted, overwrite --urdf.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned changes, do not write file.",
    )
    parser.add_argument(
        "--allow-missing-links",
        action="store_true",
        help="Do not fail when a config link is missing in URDF.",
    )
    parser.add_argument(
        "--allow-missing-inertial",
        action="store_true",
        help="Auto-create <inertial><mass> if a target link has no inertial block.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Backup suffix for in-place mode. Set empty string to disable backup.",
    )
    return parser.parse_args()


def _load_mass_map(cfg_path: Path) -> Dict[str, float]:
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Config is empty: {cfg_path}")

    # Support either:
    # 1) top-level mapping: {link_name: mass, ...}
    # 2) nested mapping: {links: {link_name: mass, ...}}
    links = data.get("links") if isinstance(data, dict) and "links" in data else data

    if not isinstance(links, dict):
        raise ValueError(
            "Config format error: expected mapping or 'links' mapping. "
            f"Got type={type(links).__name__}"
        )

    mass_map: Dict[str, float] = {}
    for link_name, mass_value in links.items():
        if not isinstance(link_name, str):
            raise ValueError(f"Invalid link name type: {type(link_name).__name__}")
        if not isinstance(mass_value, (int, float)):
            raise ValueError(f"Invalid mass type for '{link_name}': {type(mass_value).__name__}")
        mass = float(mass_value)
        if mass <= 0:
            raise ValueError(f"Mass must be > 0 for '{link_name}', got {mass_value}")
        mass_map[link_name] = mass

    if not mass_map:
        raise ValueError("No mass entries found in config.")
    return mass_map


def _ensure_inertial_and_mass(link_el: ET.Element, allow_create: bool) -> ET.Element:
    inertial_el = link_el.find("inertial")
    if inertial_el is None:
        if not allow_create:
            raise ValueError(
                f"Link '{link_el.get('name')}' has no <inertial>. "
                "Use --allow-missing-inertial to auto-create."
            )
        inertial_el = ET.SubElement(link_el, "inertial")
        ET.SubElement(inertial_el, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(
            inertial_el,
            "inertia",
            {"ixx": "1e-6", "ixy": "0", "ixz": "0", "iyy": "1e-6", "iyz": "0", "izz": "1e-6"},
        )

    mass_el = inertial_el.find("mass")
    if mass_el is None:
        if not allow_create:
            raise ValueError(
                f"Link '{link_el.get('name')}' has <inertial> but no <mass>. "
                "Use --allow-missing-inertial to auto-create."
            )
        mass_el = ET.SubElement(inertial_el, "mass")
    return mass_el


def _plan_updates(
    root: ET.Element, mass_map: Dict[str, float], allow_missing_inertial: bool
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, ET.Element]]:
    # planned_changes[link_name] = (old_mass, new_mass)
    planned_changes: Dict[str, Tuple[float, float]] = {}
    mass_nodes: Dict[str, ET.Element] = {}

    links_by_name = {link.get("name"): link for link in root.findall("link")}

    for link_name, new_mass in mass_map.items():
        link_el = links_by_name.get(link_name)
        if link_el is None:
            continue
        mass_el = _ensure_inertial_and_mass(link_el, allow_create=allow_missing_inertial)

        old_raw = mass_el.get("value", "nan")
        try:
            old_mass = float(old_raw)
        except ValueError:
            old_mass = float("nan")

        planned_changes[link_name] = (old_mass, new_mass)
        mass_nodes[link_name] = mass_el

    return planned_changes, mass_nodes


def _fmt_mass(v: float) -> str:
    return f"{v:.12g}"


def main() -> int:
    args = _parse_args()
    urdf_path = Path(args.urdf)
    cfg_path = Path(args.config)
    output_path = Path(args.output) if args.output else urdf_path

    if not urdf_path.exists():
        print(f"[ERROR] URDF not found: {urdf_path}", file=sys.stderr)
        return 2
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}", file=sys.stderr)
        return 2

    try:
        mass_map = _load_mass_map(cfg_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse config: {e}", file=sys.stderr)
        return 2

    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERROR] Failed to parse URDF: {e}", file=sys.stderr)
        return 2

    planned_changes, mass_nodes = _plan_updates(
        root, mass_map, allow_missing_inertial=args.allow_missing_inertial
    )
    found_links = set(planned_changes.keys())
    requested_links = set(mass_map.keys())
    missing_links = sorted(requested_links - found_links)

    if missing_links and not args.allow_missing_links:
        print(
            "[ERROR] Some links in config were not found in URDF:\n  - "
            + "\n  - ".join(missing_links),
            file=sys.stderr,
        )
        return 2

    print(f"[INFO] URDF: {urdf_path}")
    print(f"[INFO] Config: {cfg_path}")
    print(f"[INFO] Requested links: {len(requested_links)}")
    print(f"[INFO] Matched links: {len(found_links)}")
    if missing_links:
        print(f"[WARN] Missing links ({len(missing_links)}): {', '.join(missing_links)}")

    for link_name in sorted(planned_changes.keys()):
        old_mass, new_mass = planned_changes[link_name]
        print(f"[PLAN] {link_name}: {old_mass} -> {new_mass}")

    if args.dry_run:
        print("[INFO] Dry-run enabled, no file written.")
        return 0

    for link_name, (_, new_mass) in planned_changes.items():
        mass_nodes[link_name].set("value", _fmt_mass(new_mass))

    # Backup only for in-place mode.
    if output_path.resolve() == urdf_path.resolve() and args.backup_suffix:
        backup_path = urdf_path.with_name(urdf_path.name + args.backup_suffix)
        shutil.copy2(urdf_path, backup_path)
        print(f"[INFO] Backup saved: {backup_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8")
    print(f"[INFO] Updated URDF written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
