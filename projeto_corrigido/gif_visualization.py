#!/usr/bin/env python3
"""
gif_visualization.py

Criterion covered: GIF visualization.

This module renders fixed-size 2D + 3D frames and creates animated GIFs.
It also contains the huge-scale representative GIF mode for 10^8 / 10^9 particles.
That mode does NOT execute direct N-body physics; it creates a weighted density
visualization where every sampled point represents many real particles.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
import matplotlib.patheffects as path_effects

import imageio.v2 as imageio
from PIL import Image

Array = np.ndarray


def stellar_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "stellar_heat_modular",
        ["#4d6dff", "#28d9ff", "#f7fbff", "#ffd86b", "#ff842f", "#ff3f94"],
    )


def nebula_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "deep_space_nebula_modular",
        ["#02030a", "#071129", "#15195a", "#32116a", "#77267d", "#ff9f1c"],
    )


def make_background(seed: int, count: int, extent: float) -> Dict[str, Any]:
    """Reusable background star field and nebula texture."""
    rng = np.random.default_rng(int(seed) + 100_000)
    bx = rng.uniform(-extent, extent, int(count))
    by = rng.uniform(-extent, extent, int(count))
    bz = rng.uniform(-0.35 * extent, 0.35 * extent, int(count))
    bs = rng.lognormal(mean=-0.25, sigma=0.55, size=int(count))
    bs = np.clip(bs, 0.18, 3.8)

    grid_size = 360
    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    xx, yy = np.meshgrid(x, y)
    nebula = np.zeros((grid_size, grid_size), dtype=np.float64)
    for _ in range(10):
        cx = rng.uniform(-0.90 * extent, 0.90 * extent)
        cy = rng.uniform(-0.90 * extent, 0.90 * extent)
        sx = rng.uniform(0.15 * extent, 0.46 * extent)
        sy = rng.uniform(0.12 * extent, 0.38 * extent)
        amp = rng.uniform(0.25, 0.95)
        nebula += amp * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2.0)
    nebula += 0.05 * rng.random((grid_size, grid_size))
    nebula /= max(float(nebula.max()), 1e-12)
    return {"stars": (bx, by, bz, bs), "nebula": nebula, "extent": extent}


def _fixed_figure(args):
    fig = plt.figure(figsize=(float(args.fig_width), float(args.fig_height)), dpi=int(args.dpi), facecolor="#02030a")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.065)
    ax2d = fig.add_subplot(grid[0, 0])
    ax3d = fig.add_subplot(grid[0, 1], projection="3d")
    for ax in (ax2d, ax3d):
        ax.set_facecolor("#02030a")
    return fig, ax2d, ax3d


def render_direct_frame(
    snapshot: Dict[str, Array],
    history: deque,
    frame_path: Path,
    year: float,
    frame_index: int,
    total_frames: int,
    args,
    mpi_info: Dict[str, Any],
    diagnostics: Dict[str, Any],
    background: Dict[str, Any],
) -> None:
    """Render one physical simulation frame with fixed layout."""
    pos = snapshot["pos"]
    vel = snapshot["vel"]
    mass = snapshot["mass"]
    speed = np.linalg.norm(vel, axis=1) if len(vel) else np.zeros(0)
    radius = np.linalg.norm(pos, axis=1) if len(pos) else np.zeros(0)

    extent = float(args.galaxy_radius) * 1.42
    bg_extent = float(background["extent"])
    bx, by, bz, bs = background["stars"]
    nebula = background["nebula"]
    star_cmap = stellar_colormap()

    fig, ax2d, ax3d = _fixed_figure(args)

    ax2d.imshow(
        nebula,
        extent=[-bg_extent, bg_extent, -bg_extent, bg_extent],
        origin="lower",
        cmap=nebula_colormap(),
        alpha=0.34,
        interpolation="bilinear",
        zorder=0,
    )
    ax2d.scatter(bx, by, s=bs * 2.1, c="#ffffff", alpha=0.22, linewidths=0, zorder=1)
    ax2d.scatter(bx[::7], by[::7], s=bs[::7] * 8.0, c="#9fc5ff", alpha=0.08, linewidths=0, zorder=1)
    ax3d.scatter(bx, by, bz, s=bs * 0.70, c="#ffffff", alpha=0.11, linewidths=0)

    # 2D density map: clearer than raw scatter for many particles.
    if len(pos) > 0:
        ax2d.hist2d(
            pos[:, 0],
            pos[:, 1],
            bins=int(args.density_bins),
            range=[[-extent, extent], [-extent, extent]],
            cmap="magma",
            norm=LogNorm(vmin=1),
            alpha=0.52,
            zorder=2,
        )

    # Orbital trails.
    if len(history) > 1 and int(args.trail_length) > 0:
        hist = list(history)
        max_segments = min(len(hist) - 1, int(args.trail_length))
        for h in range(max_segments):
            prev = hist[-max_segments + h]["pos"][:, :2]
            curr = hist[-max_segments + h + 1]["pos"][:, :2]
            segments = np.stack([prev, curr], axis=1)
            age = (h + 1) / max_segments
            lc = LineCollection(
                segments,
                colors=(0.45, 0.80, 1.0, 0.020 + 0.15 * age),
                linewidths=0.18 + 0.52 * age,
                zorder=3,
            )
            ax2d.add_collection(lc)

    # Particle layers: glow + crisp stars.
    if len(pos) > 0:
        inner = np.percentile(radius, 3)
        outer = np.percentile(radius, 98)
        norm = Normalize(vmin=inner, vmax=max(outer, inner + 1e-9))
        color_values = norm(radius)
        mass_norm = (mass - np.min(mass)) / (np.ptp(mass) + 1e-12)
        speed_norm = speed / (np.percentile(speed, 96) + 1e-12)
        sizes = np.clip(5.0 + 21.0 * mass_norm + 8.0 * speed_norm, 3.0, 40.0)

        ax2d.scatter(pos[:, 0], pos[:, 1], s=sizes * 5.2, c=color_values, cmap=star_cmap, alpha=0.075, linewidths=0, zorder=4)
        ax2d.scatter(pos[:, 0], pos[:, 1], s=sizes * 1.8, c=color_values, cmap=star_cmap, alpha=0.16, linewidths=0, zorder=5)
        ax2d.scatter(pos[:, 0], pos[:, 1], s=sizes, c=color_values, cmap=star_cmap, alpha=0.94, linewidths=0.04, edgecolors="#fff7d8", zorder=6)

        rng = np.random.default_rng(int(args.seed) + frame_index * 17)
        n3 = min(int(args.max_3d_points), len(pos))
        idx3 = rng.choice(len(pos), size=n3, replace=False)
        floor_z = -extent * 0.38
        ax3d.scatter(pos[idx3, 0], pos[idx3, 1], np.full(n3, floor_z), s=sizes[idx3] * 0.20, c=color_values[idx3], cmap=star_cmap, alpha=0.08, linewidths=0)
        ax3d.scatter(pos[idx3, 0], pos[idx3, 1], pos[idx3, 2], s=sizes[idx3] * 0.90, c=color_values[idx3], cmap=star_cmap, alpha=0.90, linewidths=0, depthshade=True)

    # Core glow: three layers.
    ax2d.scatter([0], [0], s=1800, c="#ff8a00", alpha=0.050, linewidths=0, zorder=7)
    ax2d.scatter([0], [0], s=720, c="#ffd36a", alpha=0.15, linewidths=0, zorder=8)
    ax2d.scatter([0], [0], s=155, c="#fff7c2", alpha=1.0, edgecolors="#ff9f1c", linewidths=1.1, zorder=9)
    ax3d.scatter([0], [0], [0], s=220, c="#fff7c2", alpha=1.0, edgecolors="#ff9f1c", linewidths=0.9)
    ax3d.scatter([0], [0], [0], s=1100, c="#ff8a00", alpha=0.045, linewidths=0)

    ax2d.set_xlim(-extent, extent)
    ax2d.set_ylim(-extent, extent)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_title("2D density + orbital trails", color="white", fontsize=13, pad=10)
    ax2d.set_xlabel("x", color="#cfd8ff")
    ax2d.set_ylabel("y", color="#cfd8ff")
    ax2d.tick_params(colors="#8fa0d6", labelsize=8)
    ax2d.grid(color="#27395f", alpha=0.22, linewidth=0.6)

    ax3d.set_xlim(-extent, extent)
    ax3d.set_ylim(-extent, extent)
    ax3d.set_zlim(-extent * 0.40, extent * 0.40)
    ax3d.set_title("3D perspective", color="white", fontsize=13, pad=10)
    ax3d.set_xlabel("x", color="#cfd8ff", labelpad=-2)
    ax3d.set_ylabel("y", color="#cfd8ff", labelpad=-2)
    ax3d.set_zlabel("z", color="#cfd8ff", labelpad=-2)
    ax3d.tick_params(colors="#8fa0d6", labelsize=7)
    progress = frame_index / max(1, total_frames - 1)
    azim = 25.0 + 90.0 * progress
    elev = 24.0 + 4.0 * np.sin(2.0 * np.pi * progress)
    ax3d.view_init(elev=elev, azim=azim)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.set_facecolor((0.02, 0.02, 0.07, 0.18))
        axis.pane.set_edgecolor((0.25, 0.28, 0.45, 0.25))
    ax3d.grid(True)

    title = "Parallel Galaxy Simulation — Newtonian N-body with MPI"
    st = fig.suptitle(title, color="#f8fbff", fontsize=17, y=0.988, fontweight="bold")
    st.set_path_effects([path_effects.withStroke(linewidth=3, foreground="#02030a")])

    hud = (
        f"year {year:,.0f}  •  N={len(pos)}  •  ranks={mpi_info.get('size')}  •  frame {frame_index + 1}/{total_frames}\n"
        f"Ek={diagnostics.get('kinetic', 0.0):.3e}  Ep={diagnostics.get('potential', 0.0):.3e}  "
        f"E={diagnostics.get('total_energy', 0.0):.3e}  drift={diagnostics.get('energy_drift_percent', 0.0):+.2f}%  "
        f"virial={diagnostics.get('virial_ratio', 0.0):.3f}"
    )
    fig.text(
        0.022,
        0.035,
        hud,
        color="#e8efff",
        fontsize=9.2,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.50", facecolor="#05091c", edgecolor="#37508a", alpha=0.72),
    )
    fig.text(
        0.66,
        0.035,
        "MPI: bcast • scatter • isend/irecv ring • gather • reduce • allreduce • Barrier",
        color="#b7c8ff",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#05091c", edgecolor="#283a68", alpha=0.58),
    )

    fig.subplots_adjust(left=0.052, right=0.985, top=0.925, bottom=0.135)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(frame_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def read_frames_same_shape(frame_paths: List[Path], target_size: Optional[Tuple[int, int]] = None) -> List[Array]:
    pil_frames = [Image.open(p).convert("RGBA") for p in frame_paths]
    if target_size is None:
        width = max(img.size[0] for img in pil_frames)
        height = max(img.size[1] for img in pil_frames)
        target_size = (width, height)
    arrays = []
    for img in pil_frames:
        if img.size != target_size:
            canvas = Image.new("RGBA", target_size, (2, 3, 10, 255))
            x = (target_size[0] - img.size[0]) // 2
            y = (target_size[1] - img.size[1]) // 2
            canvas.paste(img, (x, y))
            img = canvas
        arrays.append(np.asarray(img))
    return arrays


def create_gif(frame_paths: List[Path], gif_path: Path, fps: int) -> None:
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    images = read_frames_same_shape(frame_paths)
    imageio.mimsave(str(gif_path), images, duration=1.0 / max(1, int(fps)), loop=0)


# ---------------------------------------------------------------------------
# Huge-scale representative GIF mode.
# ---------------------------------------------------------------------------

def generate_scale_rank_sample(local_n: int, args, rank: int) -> Dict[str, Array]:
    """Generate a weighted visual sample for 10^8 / 10^9 particles.

    The sample is not a direct simulation. It is a density proxy where each sample
    point represents many physical particles.
    """
    rng = np.random.default_rng(int(args.seed) + 2009 * int(rank))
    local_n = int(local_n)
    n_halo = int(round(local_n * float(args.halo_fraction)))
    n_disk = local_n - n_halo

    r = rng.gamma(shape=2.0, scale=float(args.galaxy_radius) / 5.0, size=n_disk)
    r = np.clip(r, 0.2, float(args.galaxy_radius))
    arm = rng.integers(0, max(1, int(args.spiral_arms)), size=n_disk)
    theta = 2.0 * np.pi * arm / max(1, int(args.spiral_arms))
    theta += float(args.arm_twist) * (r / float(args.galaxy_radius))
    theta += rng.normal(0.0, 0.16 + 0.0045 * r, size=n_disk)
    z = rng.normal(0.0, float(args.thickness) * (0.20 + 0.80 * r / float(args.galaxy_radius)), size=n_disk)

    if n_halo > 0:
        rh = float(args.galaxy_radius) * (rng.random(n_halo) ** (1.0 / 3.0))
        cost = rng.uniform(-1.0, 1.0, size=n_halo)
        ph = rng.uniform(0.0, 2.0 * np.pi, size=n_halo)
        r_halo_xy = rh * np.sqrt(np.maximum(0.0, 1.0 - cost * cost))
        z_halo = rh * cost * 0.70
        r = np.concatenate([r, r_halo_xy])
        theta = np.concatenate([theta, ph])
        z = np.concatenate([z, z_halo])

    temp = (1.0 - np.clip(r / float(args.galaxy_radius), 0.0, 1.0)).astype(np.float32)
    return {"r": r.astype(np.float32), "theta0": theta.astype(np.float32), "z": z.astype(np.float32), "temp": temp}


def scale_positions_at(sample: Dict[str, Array], args, frame_idx: int, nframes: int) -> Tuple[Array, Array, Array, Array]:
    r = sample["r"].astype(np.float32)
    theta0 = sample["theta0"].astype(np.float32)
    z = sample["z"].astype(np.float32)
    temp = sample["temp"].astype(np.float32)
    progress = frame_idx / max(1, nframes - 1)

    omega = float(args.rotation_factor) * (1.0 / np.sqrt((r / 38.0) ** 2 + 0.28))
    theta = theta0 + progress * 2.7 * omega
    breathing = 1.0 + 0.030 * np.sin(2.0 * np.pi * progress + r / 24.0)
    rr = r * breathing
    warp = 0.075 * float(args.thickness) * np.sin(theta0 * 1.7 + 2.0 * np.pi * progress) * (r / float(args.galaxy_radius))
    x = rr * np.cos(theta)
    y = rr * np.sin(theta)
    zz = z + warp
    return x.astype(np.float32), y.astype(np.float32), zz.astype(np.float32), temp


def render_scale_frame(
    all_samples: List[Dict[str, Array]],
    frame_path: Path,
    frame_idx: int,
    nframes: int,
    args,
    rank_count: int,
    represented_particles: int,
) -> Dict[str, Any]:
    xs: List[Array] = []
    ys: List[Array] = []
    zs: List[Array] = []
    temps: List[Array] = []
    for sample in all_samples:
        x, y, z, temp = scale_positions_at(sample, args, frame_idx, nframes)
        xs.append(x); ys.append(y); zs.append(z); temps.append(temp)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    temp = np.concatenate(temps)

    lim = float(args.galaxy_radius) * 1.20
    year = float(args.years) * frame_idx / max(1, nframes - 1)
    sample_n = len(x)
    represented_per_sample = float(represented_particles) / max(1, sample_n)
    weights = np.full(sample_n, represented_per_sample, dtype=np.float64)

    fig, ax2d, ax3d = _fixed_figure(args)
    star_cmap = stellar_colormap()

    ax2d.set_title("2D weighted density — all particles represented", color="white", fontsize=13, pad=10)
    ax2d.hist2d(
        x,
        y,
        bins=int(args.density_bins),
        range=[[-lim, lim], [-lim, lim]],
        weights=weights,
        cmap="magma",
        norm=LogNorm(vmin=max(1.0, represented_per_sample)),
        alpha=0.96,
    )

    rng = np.random.default_rng(int(args.seed) + 777 + frame_idx)
    overlay_n = min(int(args.scale_overlay_points), sample_n)
    idx = rng.choice(sample_n, size=overlay_n, replace=False)
    sizes = 0.35 + 4.8 * np.clip(temp[idx], 0.0, 1.0) ** 2
    ax2d.scatter(x[idx], y[idx], s=sizes * 3.0, c=temp[idx], cmap=star_cmap, alpha=0.085, linewidths=0)
    ax2d.scatter(x[idx], y[idx], s=sizes, c=temp[idx], cmap=star_cmap, alpha=0.42, linewidths=0)

    # Core glow.
    ax2d.scatter([0], [0], s=1900, c="#ff8a00", alpha=0.055, linewidths=0)
    ax2d.scatter([0], [0], s=720, c="#ffd36a", alpha=0.16, linewidths=0)
    ax2d.scatter([0], [0], s=145, c="#fff7c2", alpha=1.0, edgecolors="#ff9f1c", linewidths=1.1)

    ax2d.set_xlim(-lim, lim)
    ax2d.set_ylim(-lim, lim)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_xlabel("x", color="#cfd8ff")
    ax2d.set_ylabel("y", color="#cfd8ff")
    ax2d.tick_params(colors="#8fa0d6", labelsize=8)
    ax2d.grid(color="#27395f", alpha=0.20, linewidth=0.6)

    n3 = min(int(args.max_3d_points), sample_n)
    idx3 = rng.choice(sample_n, size=n3, replace=False)
    ax3d.scatter(x[idx3], y[idx3], z[idx3], s=1.2 + 5.5 * temp[idx3], c=temp[idx3], cmap=star_cmap, alpha=0.55, linewidths=0, depthshade=True)
    ax3d.scatter([0], [0], [0], s=180, c="#fff7c2", alpha=1.0, edgecolors="#ff9f1c", linewidths=0.8)
    ax3d.scatter([0], [0], [0], s=900, c="#ff8a00", alpha=0.045, linewidths=0)
    ax3d.set_title("3D sampled proxy", color="white", fontsize=13, pad=10)
    ax3d.set_xlim(-lim, lim); ax3d.set_ylim(-lim, lim); ax3d.set_zlim(-lim * 0.38, lim * 0.38)
    ax3d.set_xlabel("x", color="#cfd8ff", labelpad=-2)
    ax3d.set_ylabel("y", color="#cfd8ff", labelpad=-2)
    ax3d.set_zlabel("z", color="#cfd8ff", labelpad=-2)
    ax3d.tick_params(colors="#8fa0d6", labelsize=7)
    progress = frame_idx / max(1, nframes - 1)
    ax3d.view_init(elev=23.0 + 4.0 * np.sin(2.0 * np.pi * progress), azim=25.0 + 90.0 * progress)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.set_facecolor((0.02, 0.02, 0.07, 0.16))
        axis.pane.set_edgecolor((0.25, 0.28, 0.45, 0.25))
    ax3d.grid(True)

    title = "Galaxy scale GIF — weighted density proxy"
    st = fig.suptitle(title, color="#f8fbff", fontsize=17, y=0.988, fontweight="bold")
    st.set_path_effects([path_effects.withStroke(linewidth=3, foreground="#02030a")])
    text = (
        f"represented particles: {represented_particles:,}  •  visual sample: {sample_n:,}  •  ranks={rank_count}\n"
        f"each plotted sample represents ≈ {represented_per_sample:,.0f} particles  •  year {year:,.0f}/{float(args.years):,.0f}\n"
        "NOT a direct O(N²) N-body run: this GIF represents all particles through weighted density."
    )
    fig.text(
        0.025,
        0.030,
        text,
        color="#e8efff",
        fontsize=9.2,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.50", facecolor="#05091c", edgecolor="#37508a", alpha=0.76),
    )
    fig.subplots_adjust(left=0.052, right=0.985, top=0.925, bottom=0.135)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(frame_path, facecolor=fig.get_facecolor())
    plt.close(fig)

    return {
        "frame": frame_idx,
        "visual_year": f"{year:.6f}",
        "represented_particles": represented_particles,
        "visual_sample_particles": sample_n,
        "represented_per_sample": f"{represented_per_sample:.6f}",
        "mode": "weighted_density_proxy_not_direct_nbody",
    }
