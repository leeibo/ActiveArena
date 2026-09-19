#!/usr/bin/env python3
"""Plot configuration-level ActiveArena workspace and view comparisons."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle, Wedge


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "report" / "figures"

ORIGINAL_TABLE_LENGTH_M = 1.2
ORIGINAL_TABLE_WIDTH_M = 0.7
ORIGINAL_VFOV_DEG = 37.0
ORIGINAL_ASPECT = 320 / 240

FAN_INNER_RADIUS_M = 0.3
FAN_OUTER_RADIUS_M = 0.9
FAN_ANGLE_DEG = 180.0
UPPER_INNER_RADIUS_M = 0.6
UPPER_OUTER_RADIUS_M = 0.8
UPPER_LAYER_GAP_M = 0.35
ASTRIBOT_VFOV_DEG = 60.0
ASTRIBOT_ASPECT = 512 / 384
HEAD_YAW_LIMIT_DEG = 90.0
HEAD_PITCH_LIMIT_DEG = 69.9
TORSO_YAW_LIMIT_DEG = 68.8
OPERATIONAL_AZIMUTH_DEG = 180.0
OPERATIONAL_HEAD_J2_MIN_RAD = 0.47
OPERATIONAL_HEAD_J2_MAX_RAD = 1.22

NAVY = "#183153"
BLUE = "#2378B5"
CYAN = "#6BC4D6"
RED = "#D65245"
GOLD = "#E0A92E"
LIGHT = "#E8EEF3"
GRAY = "#687785"
WHITE = "#FFFFFF"


def horizontal_fov_deg(vertical_fov_deg: float, aspect: float) -> float:
    half_v = math.radians(vertical_fov_deg) / 2
    return math.degrees(2 * math.atan(aspect * math.tan(half_v)))


def rectangular_solid_angle_sr(horizontal_deg: float, vertical_deg: float) -> float:
    half_h = math.radians(horizontal_deg) / 2
    half_v = math.radians(vertical_deg) / 2
    return 4 * math.asin(math.sin(half_h) * math.sin(half_v))


def style_axis(ax, title: str) -> None:
    ax.set_title(title, loc="left", color=NAVY, fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(colors=GRAY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#CAD4DC")


def plot_workspace(output_dir: Path) -> Path:
    original_area = ORIGINAL_TABLE_LENGTH_M * ORIGINAL_TABLE_WIDTH_M
    fan_area = 0.5 * math.radians(FAN_ANGLE_DEG) * (
        FAN_OUTER_RADIUS_M**2 - FAN_INNER_RADIUS_M**2
    )
    upper_area = 0.5 * math.radians(FAN_ANGLE_DEG) * (
        UPPER_OUTER_RADIUS_M**2 - UPPER_INNER_RADIUS_M**2
    )

    fig = plt.figure(figsize=(14.2, 5.4), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.28)

    ax = fig.add_subplot(grid[0, 0])
    style_axis(ax, "A  ActiveArena: local rectangular table")
    ax.add_patch(
        Rectangle(
            (-ORIGINAL_TABLE_LENGTH_M / 2, 0),
            ORIGINAL_TABLE_LENGTH_M,
            ORIGINAL_TABLE_WIDTH_M,
            facecolor=LIGHT,
            edgecolor=NAVY,
            linewidth=2,
        )
    )
    ax.add_patch(Wedge((0, -0.12), 0.14, 0, 180, facecolor=NAVY, edgecolor="none"))
    ax.annotate("fixed base", (0, -0.16), xytext=(0, -0.31), ha="center", color=GRAY)
    ax.annotate("1.2 m", (-0.6, 0.75), (0, 0.75), ha="center", color=NAVY,
                arrowprops=dict(arrowstyle="|-|", color=NAVY))
    ax.annotate("0.7 m", (0.68, 0), (0.68, 0.7), va="center", rotation=90, color=NAVY,
                arrowprops=dict(arrowstyle="|-|", color=NAVY))
    ax.text(0, 0.35, f"single level\n{original_area:.2f} m$^2$", ha="center", va="center",
            color=NAVY, fontsize=14, fontweight="bold")
    ax.set(xlim=(-0.85, 0.85), ylim=(-0.36, 0.9), aspect="equal", xlabel="x (m)", ylabel="y (m)")

    ax = fig.add_subplot(grid[0, 1])
    style_axis(ax, "B  Astribot: body-centered sector")
    ax.add_patch(Wedge((0, 0), FAN_OUTER_RADIUS_M, 0, 180,
                       width=FAN_OUTER_RADIUS_M - FAN_INNER_RADIUS_M,
                       facecolor=CYAN, edgecolor=BLUE, linewidth=2, alpha=0.8))
    ax.add_patch(Wedge((0, 0), UPPER_OUTER_RADIUS_M, 0, 180,
                       width=UPPER_OUTER_RADIUS_M - UPPER_INNER_RADIUS_M,
                       facecolor=GOLD, edgecolor="#A87912", linewidth=1.6, alpha=0.68))
    ax.add_patch(Wedge((0, -0.03), 0.18, 0, 180, facecolor=NAVY, edgecolor="none"))
    ax.add_patch(Arc((0, 0), 1.25, 1.25, theta1=0, theta2=180, color=RED, linewidth=2))
    ax.text(0, 0.53, "180 deg", ha="center", color=RED, fontsize=13, fontweight="bold")
    ax.annotate("0.9 m", (0, 0), (0.64, 0.64), color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=1.8))
    ax.text(0, 0.28, f"lower {fan_area:.2f} m$^2$", ha="center", color=NAVY,
            fontsize=13, fontweight="bold")
    ax.text(0, 0.70, f"upper {upper_area:.2f} m$^2$", ha="center", color="#75550E", fontsize=10)
    ax.set(xlim=(-1.0, 1.0), ylim=(-0.22, 1.02), aspect="equal", xlabel="x (m)", ylabel="y (m)")

    ax = fig.add_subplot(grid[0, 2])
    style_axis(ax, "C  Surface and height gain")
    labels = ["original\nsingle", "ours\nlower", "ours\ntwo-level"]
    values = [original_area, fan_area, fan_area + upper_area]
    colors = [LIGHT, CYAN, GOLD]
    bars = ax.bar(labels, values, color=colors, edgecolor=[NAVY, BLUE, "#A87912"], linewidth=1.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.2f} m$^2$",
                ha="center", color=NAVY, fontweight="bold")
    ax.text(1, fan_area * 0.52, "+35%", ha="center", color=NAVY, fontsize=16, fontweight="bold")
    ax.text(2, (fan_area + upper_area) * 0.52, "+87%", ha="center", color=NAVY,
            fontsize=16, fontweight="bold")
    ax.annotate(f"second level: +{UPPER_LAYER_GAP_M:.2f} m", xy=(2, fan_area + upper_area),
                xytext=(1.3, 1.78), color="#75550E",
                arrowprops=dict(arrowstyle="->", color="#A87912"))
    ax.set_ylim(0, 2.03)
    ax.set_ylabel("configured support surface (m$^2$)")
    ax.grid(axis="y", color="#E6EBEF", linewidth=0.8)

    fig.suptitle("Workspace geometry: configuration-level comparison", x=0.04, y=1.01,
                 ha="left", fontsize=20, color=NAVY, fontweight="bold")
    fig.text(0.04, 0.012,
             "Area is geometric support surface, not collision-aware IK reachability. "
             "The upper sector uses r=0.6-0.8 m and the configured 180 deg alignment.",
             color=GRAY, fontsize=9)
    output = output_dir / "workspace_geometry_comparison.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def _angular_patch(ax, center_yaw, center_pitch, hfov, vfov, **kwargs):
    left = math.radians(center_yaw - hfov / 2)
    bottom = math.radians(center_pitch - vfov / 2)
    return ax.add_patch(Rectangle((left, bottom), math.radians(hfov), math.radians(vfov), **kwargs))


def plot_spherical_view(output_dir: Path) -> Path:
    original_hfov = horizontal_fov_deg(ORIGINAL_VFOV_DEG, ORIGINAL_ASPECT)
    astribot_hfov = horizontal_fov_deg(ASTRIBOT_VFOV_DEG, ASTRIBOT_ASPECT)
    original_sr = rectangular_solid_angle_sr(original_hfov, ORIGINAL_VFOV_DEG)
    astribot_sr = rectangular_solid_angle_sr(astribot_hfov, ASTRIBOT_VFOV_DEG)

    fig = plt.figure(figsize=(14.2, 7.5), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 0.36], hspace=0.28, wspace=0.18)

    for index, (title, full_sweep) in enumerate([
        ("A  Frozen baseline: fixed D435", False),
        ("B  Astribot: active head + torso", True),
    ]):
        ax = fig.add_subplot(grid[0, index], projection="mollweide")
        ax.set_title(title, loc="left", color=NAVY, fontsize=14, fontweight="bold", pad=14)
        ax.grid(color="#D8E0E6", linewidth=0.7)
        ax.set_facecolor("#F7F9FA")
        longitude_ticks = np.radians([-120, -60, 0, 60, 120])
        ax.set_xticks(longitude_ticks)
        ax.set_xticklabels(["120 deg", "60 deg", "0 deg", "60 deg", "120 deg"],
                           color=GRAY, fontsize=8)
        ax.tick_params(colors=GRAY, labelsize=8)
        if not full_sweep:
            _angular_patch(ax, 0, -18, original_hfov, ORIGINAL_VFOV_DEG,
                           facecolor=RED, edgecolor="#A73831", linewidth=2, alpha=0.82)
            ax.text(0, math.radians(20), f"one fixed view\n{original_hfov:.1f} x {ORIGINAL_VFOV_DEG:.0f} deg",
                    ha="center", color=NAVY, fontsize=12, fontweight="bold")
        else:
            operational_elevation_max = min(
                math.pi / 2,
                -OPERATIONAL_HEAD_J2_MIN_RAD + math.radians(ASTRIBOT_VFOV_DEG) / 2,
            )
            ax.add_patch(Rectangle((-math.pi / 2, -math.pi / 2), math.pi,
                                   operational_elevation_max + math.pi / 2,
                                   facecolor=CYAN, edgecolor=GOLD, linewidth=1.5, alpha=0.42))
            _angular_patch(ax, 0, -18, astribot_hfov, ASTRIBOT_VFOV_DEG,
                           facecolor=BLUE, edgecolor=NAVY, linewidth=2, alpha=0.88)
            ax.text(0, math.radians(20), f"instantaneous view\n{astribot_hfov:.1f} x {ASTRIBOT_VFOV_DEG:.0f} deg",
                    ha="center", color=NAVY, fontsize=11, fontweight="bold")
            ax.text(math.radians(122), math.radians(54), "180-deg task-constrained\ndirection envelope",
                    ha="center", color=NAVY, fontsize=10, fontweight="bold")

    ax = fig.add_subplot(grid[1, :])
    ax.axis("off")
    columns = [0.03, 0.29, 0.55, 0.82]
    metrics = [
        ("Instantaneous FOV", f"{original_hfov:.1f} x {ORIGINAL_VFOV_DEG:.0f} deg", f"{astribot_hfov:.1f} x {ASTRIBOT_VFOV_DEG:.0f} deg"),
        ("Solid angle / sphere", f"{original_sr:.2f} sr / {100*original_sr/(4*math.pi):.1f}%", f"{astribot_sr:.2f} sr / {100*astribot_sr/(4*math.pi):.1f}%"),
        ("Vertical operation", "fixed", "head J2 0.47-1.22 rad\nabout 93 deg envelope"),
        ("Horizontal operation", "fixed", "180 deg\ntable sector"),
    ]
    for x, (name, original, ours) in zip(columns, metrics):
        ax.text(x, 0.86, name, transform=ax.transAxes, color=GRAY, fontsize=9, fontweight="bold")
        ax.text(x, 0.53, original, transform=ax.transAxes, color=RED, fontsize=11)
        ax.text(x, 0.18, ours, transform=ax.transAxes, color=BLUE, fontsize=11, fontweight="bold")
    ax.text(0.0, 0.53, "Original", transform=ax.transAxes, color=RED, fontsize=10,
            fontweight="bold", ha="right")
    ax.text(0.0, 0.18, "Ours", transform=ax.transAxes, color=BLUE, fontsize=10,
            fontweight="bold", ha="right")

    fig.suptitle("View-space projection on the viewing sphere", x=0.04, y=0.99,
                 ha="left", fontsize=20, color=NAVY, fontweight="bold")
    fig.text(0.04, 0.015,
             "Blue rectangle: one frame. Cyan: benchmark task-constrained direction envelope "
             "(180-deg table sector, configured vertical scan). Occlusion and range limits are excluded.",
             color=GRAY, fontsize=9)
    output = output_dir / "spherical_view_coverage_comparison.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def _plot_envelope_globe(ax, active: bool) -> None:
    lon = np.linspace(-math.pi, math.pi, 181)
    lat = np.linspace(-math.pi / 2, math.pi / 2, 91)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x = np.cos(lat_grid) * np.cos(lon_grid)
    y = np.cos(lat_grid) * np.sin(lon_grid)
    z = np.sin(lat_grid)

    base = np.array(to_rgba("#E9EEF2", 0.72))
    if active:
        accent = np.array(to_rgba(BLUE, 0.94))
        hfov = math.radians(horizontal_fov_deg(ASTRIBOT_VFOV_DEG, ASTRIBOT_ASPECT))
        vfov = math.radians(ASTRIBOT_VFOV_DEG)
    else:
        accent = np.array(to_rgba(RED, 0.94))
        hfov = math.radians(horizontal_fov_deg(ORIGINAL_VFOV_DEG, ORIGINAL_ASPECT))
        vfov = math.radians(ORIGINAL_VFOV_DEG)

    # Put the nominal forward direction on the visible hemisphere for the chosen camera view.
    center_lon = -math.pi / 2
    center_lat = math.radians(-18)
    lon_delta = np.angle(np.exp(1j * (lon_grid - center_lon)))
    instantaneous = (np.abs(lon_delta) <= hfov / 2) & (np.abs(lat_grid - center_lat) <= vfov / 2)
    facecolors = np.empty((*x.shape, 4), dtype=float)
    facecolors[:] = base
    if active:
        elevation_min = max(-math.pi / 2, -OPERATIONAL_HEAD_J2_MAX_RAD - vfov / 2)
        elevation_max = min(math.pi / 2, -OPERATIONAL_HEAD_J2_MIN_RAD + vfov / 2)
        operational = (
            (np.abs(lon_delta) <= math.radians(OPERATIONAL_AZIMUTH_DEG) / 2)
            & (lat_grid >= elevation_min)
            & (lat_grid <= elevation_max)
        )
        facecolors[operational] = np.array(to_rgba(CYAN, 0.72))
    facecolors[instantaneous] = accent
    ax.plot_surface(x, y, z, facecolors=facecolors, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False)

    grid_color = "#AEBBC5" if not active else "#FFFFFF"
    grid_alpha = 0.64 if not active else 0.58
    for longitude in np.radians(np.arange(-150, 181, 30)):
        lat_line = np.linspace(-math.pi / 2, math.pi / 2, 180)
        ax.plot(np.cos(lat_line) * np.cos(longitude),
                np.cos(lat_line) * np.sin(longitude), np.sin(lat_line),
                color=grid_color, alpha=grid_alpha, linewidth=0.65)
    for latitude in np.radians(np.arange(-60, 61, 30)):
        lon_line = np.linspace(-math.pi, math.pi, 240)
        ax.plot(np.cos(latitude) * np.cos(lon_line),
                np.cos(latitude) * np.sin(lon_line), np.full_like(lon_line, np.sin(latitude)),
                color=grid_color, alpha=grid_alpha, linewidth=0.65)

    if active:
        elevation_min = max(-math.pi / 2, -OPERATIONAL_HEAD_J2_MAX_RAD - vfov / 2)
        elevation_max = min(math.pi / 2, -OPERATIONAL_HEAD_J2_MIN_RAD + vfov / 2)
        ring_lon = np.linspace(center_lon - math.pi / 2, center_lon + math.pi / 2, 220)
        radius = 1.055
        ax.plot(radius * np.cos(elevation_max) * np.cos(ring_lon),
                radius * np.cos(elevation_max) * np.sin(ring_lon),
                np.full_like(ring_lon, radius * np.sin(elevation_max)),
                color=GOLD, linewidth=2.2, alpha=0.95)
        ring_lat = np.linspace(elevation_min, elevation_max, 160)
        for boundary_lon in (center_lon - math.pi / 2, center_lon + math.pi / 2):
            ax.plot(radius * np.cos(ring_lat) * np.cos(boundary_lon),
                    radius * np.cos(ring_lat) * np.sin(boundary_lon), radius * np.sin(ring_lat),
                    color=GOLD, linewidth=2.2, alpha=0.95)

    ax.view_init(elev=13, azim=-90)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_zlim(-1.08, 1.08)
    ax.set_axis_off()


def plot_view_envelope_hero(output_dir: Path) -> Path:
    fig = plt.figure(figsize=(13.2, 4.4), facecolor="white")
    left = fig.add_axes([0.02, 0.01, 0.43, 0.98], projection="3d")
    right = fig.add_axes([0.55, 0.01, 0.43, 0.98], projection="3d")
    _plot_envelope_globe(left, active=False)
    _plot_envelope_globe(right, active=True)
    fig.text(0.50, 0.52, ">", ha="center", va="center", fontsize=52,
             color=NAVY, fontweight="bold")
    fig.text(0.25, 0.06, "FIXED ENVELOPE", ha="center", fontsize=11,
             color=RED, fontweight="bold")
    fig.text(0.75, 0.06, "TASK-CONSTRAINED ENVELOPE", ha="center", fontsize=11,
             color=BLUE, fontweight="bold")
    output = output_dir / "view_envelope_globes.png"
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def _draw_plan_view(ax, active: bool) -> None:
    if active:
        ax.add_patch(Wedge((0, 0), 1.0, 0, 180, width=0.66,
                           facecolor="#EEF2F4", edgecolor="#B9C5CD", linewidth=1.2))
    else:
        ax.add_patch(Rectangle((-0.78, 0.16), 1.56, 0.72,
                               facecolor="#EEF2F4", edgecolor="#B9C5CD", linewidth=1.2))
        ax.text(0, 0.84, "1.2 × 0.7 m table", ha="center", va="center",
                fontsize=9, color=GRAY, fontweight="bold")
    ax.add_patch(Wedge((0, -0.01), 0.15, 0, 180, facecolor=NAVY, edgecolor="none"))
    if active:
        ax.add_patch(Wedge((0, 0), 0.96, 0, 180, width=0.60,
                           facecolor=CYAN, edgecolor=BLUE, linewidth=1.5, alpha=0.55))
        half = horizontal_fov_deg(ASTRIBOT_VFOV_DEG, ASTRIBOT_ASPECT) / 2
        ax.add_patch(Wedge((0, 0), 0.92, 90 - half, 90 + half,
                           facecolor=BLUE, edgecolor=NAVY, linewidth=1.5, alpha=0.86))
        ax.add_patch(FancyArrowPatch(
            (-0.78, 0.55), (0.78, 0.55), connectionstyle="arc3,rad=-0.42",
            arrowstyle="<->", mutation_scale=13, linewidth=2.1, color=GOLD,
        ))
        ax.text(0, 0.99, "180° active horizontal coverage", ha="center", va="bottom",
                fontsize=11, color=BLUE, fontweight="bold")
        ax.text(0, 0.54, "75.2°\ncurrent view", ha="center", va="center",
                fontsize=10, color=WHITE, fontweight="bold")
        ax.text(0, -0.16, "controllable head + torso", ha="center", va="top",
                fontsize=9, color=BLUE, fontweight="bold")
    else:
        half = horizontal_fov_deg(ORIGINAL_VFOV_DEG, ORIGINAL_ASPECT) / 2
        ax.add_patch(Wedge((0, 0), 0.92, 90 - half, 90 + half,
                           facecolor=RED, edgecolor="#A83C34", linewidth=1.5, alpha=0.84))
        ax.text(0, 0.57, "48.1°\nfixed view", ha="center", va="center",
                fontsize=10, color=WHITE, fontweight="bold")
        ax.text(0, 0.99, "single fixed camera", ha="center", va="bottom",
                fontsize=11, color=RED, fontweight="bold")
        ax.text(0, -0.16, "no active view control", ha="center", va="top",
                fontsize=9, color=RED, fontweight="bold")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-0.25, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_side_view(ax) -> None:
    camera = np.array([-0.95, 0.82])
    ax.scatter([camera[0]], [camera[1]], s=75, color=NAVY, zorder=5)
    ax.text(camera[0], camera[1] + 0.12, "head camera", ha="center", color=NAVY,
            fontsize=9, fontweight="bold")
    ax.plot([-0.12, 1.0], [0.22, 0.22], color=BLUE, linewidth=5, solid_capstyle="butt")
    ax.plot([-0.12, 1.0], [0.62, 0.62], color=GOLD, linewidth=5, solid_capstyle="butt")
    ax.text(1.02, 0.22, "lower level", va="center", fontsize=9, color=BLUE, fontweight="bold")
    ax.text(1.02, 0.62, "upper level  (+0.35 m)", va="center", fontsize=9,
            color="#8B6610", fontweight="bold")
    for target, color in [((-0.02, 0.22), BLUE), ((-0.02, 0.62), GOLD)]:
        ax.add_patch(FancyArrowPatch(tuple(camera), target, arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.8, color=color))
    ax.text(-0.30, 0.02, "configured vertical inspection", ha="center", fontsize=10,
            color=NAVY, fontweight="bold")
    ax.text(-0.30, -0.17, "head J2: 0.47–1.22 rad   ·   vertical FOV: 60°", ha="center",
            fontsize=9, color=GRAY)
    ax.set_xlim(-1.22, 1.62)
    ax.set_ylim(-0.28, 1.05)
    ax.axis("off")


def plot_operational_view_coverage(output_dir: Path) -> Path:
    fig = plt.figure(figsize=(12.8, 5.2), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.52], hspace=0.02, wspace=0.16)
    original = fig.add_subplot(grid[0, 0])
    active = fig.add_subplot(grid[0, 1])
    side = fig.add_subplot(grid[1, :])
    _draw_plan_view(original, active=False)
    _draw_plan_view(active, active=True)
    _draw_side_view(side)
    fig.text(0.255, 0.98, "ORIGINAL ROBOTWIN", ha="center", va="top",
             fontsize=11, color=GRAY, fontweight="bold")
    fig.text(0.745, 0.98, "ACTIVEARENA-SIM", ha="center", va="top",
             fontsize=11, color=NAVY, fontweight="bold")
    fig.lines.append(plt.Line2D([0.5, 0.5], [0.42, 0.94], transform=fig.transFigure,
                                color="#D6DEE4", linewidth=1.0))
    output = output_dir / "operational_view_coverage.png"
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def plot_operational_view_plan(output_dir: Path) -> Path:
    fig = plt.figure(figsize=(12.8, 3.9), facecolor="white")
    original = fig.add_axes([0.02, 0.02, 0.45, 0.90])
    active = fig.add_axes([0.53, 0.02, 0.45, 0.90])
    _draw_plan_view(original, active=False)
    _draw_plan_view(active, active=True)
    fig.text(0.245, 0.99, "ORIGINAL ROBOTWIN", ha="center", va="top",
             fontsize=11, color=GRAY, fontweight="bold")
    fig.text(0.755, 0.99, "ACTIVEARENA-SIM", ha="center", va="top",
             fontsize=11, color=NAVY, fontweight="bold")
    fig.lines.append(plt.Line2D([0.5, 0.5], [0.05, 0.93], transform=fig.transFigure,
                                color="#D6DEE4", linewidth=1.0))
    output = output_dir / "operational_view_plan.png"
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def plot_operational_view_side(output_dir: Path) -> Path:
    fig = plt.figure(figsize=(12.8, 2.3), facecolor="white")
    side = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    _draw_side_view(side)
    output = output_dir / "operational_view_side.png"
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def _sector_surface(
    ax, theta_min, theta_max, r_min, r_max, z, color, alpha, edge=None, zorder=1
):
    theta = np.linspace(math.radians(theta_min), math.radians(theta_max), 100)
    radius = np.linspace(r_min, r_max, 3)
    theta_grid, radius_grid = np.meshgrid(theta, radius)
    x = radius_grid * np.cos(theta_grid)
    y = radius_grid * np.sin(theta_grid)
    z_grid = np.full_like(x, z)
    ax.plot_surface(x, y, z_grid, color=color, alpha=alpha, linewidth=0,
                    shade=False, zorder=zorder)
    if edge:
        ax.plot(r_max * np.cos(theta), r_max * np.sin(theta), np.full_like(theta, z),
                color=edge, linewidth=1.5, zorder=zorder + 0.2)


def _rect_surface(
    ax, x_min, x_max, y_min, y_max, z, color, alpha, edge=None, zorder=1
):
    x = np.linspace(x_min, x_max, 2)
    y = np.linspace(y_min, y_max, 2)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = np.full_like(x_grid, z)
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha,
                    linewidth=0, shade=False, zorder=zorder)
    if edge:
        corners = [
            (x_min, y_min), (x_max, y_min), (x_max, y_max),
            (x_min, y_max), (x_min, y_min),
        ]
        ax.plot([p[0] for p in corners], [p[1] for p in corners],
                [z] * len(corners), color=edge, linewidth=1.5, zorder=zorder + 0.2)


def _frustum_sweep_surface(
    ax, camera, theta_min, theta_max, r_end, z_end, color, alpha, zorder=4
):
    theta = np.linspace(math.radians(theta_min), math.radians(theta_max), 120)
    blend = np.linspace(0.0, 1.0, 12)
    theta_grid, blend_grid = np.meshgrid(theta, blend)
    target_x = r_end * np.cos(theta_grid)
    target_y = r_end * np.sin(theta_grid)
    x = (1 - blend_grid) * camera[0] + blend_grid * target_x
    y = (1 - blend_grid) * camera[1] + blend_grid * target_y
    z = (1 - blend_grid) * camera[2] + blend_grid * z_end
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0,
                    shade=False, zorder=zorder)


def _frustum_outline(
    ax, camera, center_deg, hfov_deg, r_end, z_end, color, linewidth=1.6, zorder=10
):
    half = hfov_deg / 2
    endpoints = []
    for theta_deg in (center_deg - half, center_deg + half):
        theta = math.radians(theta_deg)
        endpoint = np.array([r_end * math.cos(theta), r_end * math.sin(theta), z_end])
        endpoints.append(endpoint)
        ax.plot([camera[0], endpoint[0]], [camera[1], endpoint[1]], [camera[2], endpoint[2]],
                color=color, linewidth=linewidth, alpha=0.92, zorder=zorder)
    ax.plot([endpoints[0][0], endpoints[1][0]], [endpoints[0][1], endpoints[1][1]],
            [z_end, z_end], color=color, linewidth=linewidth, alpha=0.92, zorder=zorder)


def _style_3d_envelope_axis(ax):
    ax.view_init(elev=30, azim=-64)
    ax.set_proj_type("ortho")
    ax.set_xlim(-0.08, 1.03)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-0.04, 1.02)
    ax.set_box_aspect((1.1, 1.8, 0.9))
    ax.set_axis_off()


def plot_view_frustum_envelope_3d(output_dir: Path) -> Path:
    original_hfov = horizontal_fov_deg(ORIGINAL_VFOV_DEG, ORIGINAL_ASPECT)
    active_hfov = horizontal_fov_deg(ASTRIBOT_VFOV_DEG, ASTRIBOT_ASPECT)
    camera = np.array([0.0, 0.0, 0.92])

    fig = plt.figure(figsize=(13.2, 4.7), facecolor="white")
    left = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
    right = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)

    # Original: one fixed frustum over a neutral local support plane.
    _rect_surface(left, 0.18, 0.88, -0.60, 0.60, 0.0,
                  "#E8EDF0", 0.68, "#BBC6CD", zorder=1)
    # Keep the complete fixed-view footprint strictly on the tabletop. The
    # small positive z offset avoids ambiguous alpha sorting in Matplotlib 3D.
    original_hit_radius = 0.82
    original_hit_z = 0.025
    _sector_surface(left, -original_hfov / 2, original_hfov / 2,
                    0.28, original_hit_radius, original_hit_z,
                    RED, 0.58, "#A83C34", zorder=3)
    _frustum_sweep_surface(left, camera, -original_hfov / 2, original_hfov / 2,
                           original_hit_radius, original_hit_z, RED, 0.12, zorder=5)
    _frustum_outline(left, camera, 0, original_hfov,
                     original_hit_radius, original_hit_z, "#A83C34", 2.0, zorder=12)
    left.scatter([camera[0]], [camera[1]], [camera[2]], s=34, color=NAVY,
                 depthshade=False, zorder=14)
    _style_3d_envelope_axis(left)

    # ActiveArena: the current dark-blue frustum sits inside a 180-degree swept envelope.
    _sector_surface(right, -90, 90, 0.28, 0.95, 0.0, CYAN, 0.54, BLUE)
    _sector_surface(right, -90, 90, 0.58, 0.82, 0.35, "#F4E6B9",
                    0.72, GOLD)
    _frustum_sweep_surface(right, camera, -90, 90, 0.95, 0.0, CYAN, 0.12)
    _frustum_sweep_surface(right, camera, -90, 90, 0.82, 0.35, GOLD, 0.13)
    _frustum_sweep_surface(right, camera, -active_hfov / 2, active_hfov / 2,
                           0.92, 0.0, BLUE, 0.25)
    for center in (-68, -34, 0, 34, 68):
        _frustum_outline(right, camera, center, active_hfov, 0.92, 0.0,
                         BLUE if center == 0 else "#72B9C8", 1.8 if center == 0 else 0.85)
    for center in (-60, 0, 60):
        _frustum_outline(right, camera, center, active_hfov, 0.82, 0.35,
                         GOLD, 1.7 if center == 0 else 0.9)
    right.scatter([camera[0]], [camera[1]], [camera[2]], s=34, color=NAVY,
                  depthshade=False, zorder=14)
    _style_3d_envelope_axis(right)

    fig.text(0.25, 0.95, "ORIGINAL: FIXED VIEW ON A RECTANGULAR TABLE", ha="center", fontsize=12,
             color=RED, fontweight="bold")
    fig.text(0.75, 0.95, "ACTIVEARENA-SIM: SWEPT VIEW ENVELOPE", ha="center", fontsize=12,
             color=BLUE, fontweight="bold")
    fig.text(0.25, 0.08, "fixed envelope:  Ω = 0.519 sr  (H 48.1°  ·  V 37°)", ha="center",
             fontsize=10, color=GRAY)
    fig.text(0.75, 0.08, "task-effective swept envelope:  Ω = 3.310 sr", ha="center",
             fontsize=10, color=GRAY)
    output = output_dir / "view_frustum_envelope_3d.png"
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(plot_workspace(OUTPUT_DIR))
    print(plot_spherical_view(OUTPUT_DIR))
    print(plot_view_envelope_hero(OUTPUT_DIR))
    print(plot_operational_view_coverage(OUTPUT_DIR))
    print(plot_operational_view_plan(OUTPUT_DIR))
    print(plot_operational_view_side(OUTPUT_DIR))
    print(plot_view_frustum_envelope_3d(OUTPUT_DIR))


if __name__ == "__main__":
    main()
