#!/usr/bin/env python3
"""
galaxy_mpi_visualscale_final.py

Representative large-scale galaxy GIF generator for PA Project 2.

This version focuses on visual quality and stable GIF generation:
- fixed canvas/layout for every frame;
- 2D density map instead of a raw scatter cloud;
- reduced 3D point count for readability;
- smooth 3D camera rotation from 25 to 115 degrees by default;
- multi-layer glow around the central core;
- no tight_layout() and no bbox_inches="tight".

Important: this is a representative visual proxy, not a direct physical
all-pairs N-body simulation for 1e8/1e9 particles.

MPI usage:
- bcast(): distribute visualization parameters.
- gather(): collect rank-local samples on rank 0 for plotting.
- reduce(): verify total generated sample size.
- Barrier(): synchronize timing.

Run examples:
  mpiexec -n 4 python galaxy_mpi_visualscale_final.py --represented-particles 100000000 --sample-particles 120000 --frames 30 --output-dir visual_1e8_final
  mpiexec -n 4 python galaxy_mpi_visualscale_final.py --represented-particles 1000000000 --sample-particles 120000 --max-3d-points 12000 --frames 30 --output-dir visual_1e9_final
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import imageio.v2 as imageio
from PIL import Image

try:
    from mpi4py import MPI
except Exception:  # pragma: no cover
    MPI = None


FLOAT = np.float32
SPACE_BG = "#02030A"
AX_BG = "#03040A"


def get_comm():
    """Return MPI communicator, with a safe serial fallback for preview runs."""
    if MPI is None:
        class DummyComm:
            def Get_rank(self): return 0
            def Get_size(self): return 1
            def bcast(self, x, root=0): return x
            def gather(self, x, root=0): return [x]
            def reduce(self, x, op=None, root=0): return x
            def Barrier(self): return None
        return DummyComm(), 0, 1
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a representative large-scale galaxy GIF without a full N-body simulation."
    )
    p.add_argument("--represented-particles", type=int, default=100_000_000,
                   help="Particle count represented conceptually, e.g. 100000000 or 1000000000.")
    p.add_argument("--sample-particles", type=int, default=120_000,
                   help="Actual sampled particles used for the density view. 120000 is a good final value.")
    p.add_argument("--years", type=float, default=20_000.0,
                   help="Displayed simulated years for the visual proxy.")
    p.add_argument("--frames", type=int, default=30,
                   help="Number of frames in the GIF.")
    p.add_argument("--fps", type=int, default=12,
                   help="GIF frames per second.")
    p.add_argument("--radius", type=float, default=120.0,
                   help="Galaxy radius used for the representative sample.")
    p.add_argument("--thickness", type=float, default=7.0,
                   help="Vertical thickness of the disk.")
    p.add_argument("--spiral-arms", type=int, default=4,
                   help="Number of spiral arms.")
    p.add_argument("--arm-twist", type=float, default=4.8,
                   help="How tightly the arms wind around the core.")
    p.add_argument("--halo-fraction", type=float, default=0.10,
                   help="Fraction of sampled points in a diffuse halo.")
    p.add_argument("--rotation-factor", type=float, default=0.95,
                   help="Visual angular speed multiplier; not a physical integration parameter.")
    p.add_argument("--output-dir", type=str, default="visual_scale_final",
                   help="Output directory.")
    p.add_argument("--gif-name", type=str, default="galaxy_visual_scale_final.gif")
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--fig-width", type=float, default=13.0)
    p.add_argument("--fig-height", type=float, default=6.5)
    p.add_argument("--seed", type=int, default=2120622)
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--max-3d-points", type=int, default=12_000,
                   help="Maximum sample points drawn in 3D per frame. Use 12000-15000 for clean output.")
    p.add_argument("--density-bins", type=int, default=300,
                   help="2D density grid resolution.")
    p.add_argument("--density-smooth-passes", type=int, default=2,
                   help="Number of light smoothing passes over the 2D density grid.")
    p.add_argument("--overlay-points", type=int, default=5_000,
                   help="Small sparkle overlay on top of the density map. Keep low to avoid raw scatter clutter.")
    p.add_argument("--camera-azim-start", type=float, default=25.0,
                   help="Initial azimuth for the 3D camera.")
    p.add_argument("--camera-azim-end", type=float, default=115.0,
                   help="Final azimuth for the 3D camera.")
    p.add_argument("--camera-elev", type=float, default=24.0,
                   help="Fixed elevation for the 3D camera.")
    return p.parse_args()


def split_counts(total: int, size: int) -> np.ndarray:
    base = total // size
    rem = total % size
    return np.array([base + (1 if r < rem else 0) for r in range(size)], dtype=np.int64)


def generate_rank_sample(n: int, args, rank: int) -> Dict[str, np.ndarray]:
    """Generate one MPI rank's statistical galaxy sample."""
    rng = np.random.default_rng(args.seed + 1009 * rank)
    n_halo = int(round(n * np.clip(args.halo_fraction, 0.0, 0.5)))
    n_disk = n - n_halo

    # Disk: gamma distribution creates a dense core and an extended disk.
    r = rng.gamma(shape=2.0, scale=args.radius / 5.0, size=n_disk).astype(FLOAT)
    r = np.clip(r, 0.2, args.radius).astype(FLOAT)

    arm = rng.integers(0, max(1, args.spiral_arms), size=n_disk)
    base_theta = 2.0 * np.pi * arm / max(1, args.spiral_arms)
    theta = base_theta + args.arm_twist * (r / args.radius) + rng.normal(0.0, 0.18 + 0.006 * r, size=n_disk)
    theta = theta.astype(FLOAT)
    z = rng.normal(0.0, args.thickness * (0.25 + 0.75 * r / args.radius), size=n_disk).astype(FLOAT)

    if n_halo > 0:
        rh = args.radius * (rng.random(n_halo) ** (1.0 / 3.0)).astype(FLOAT)
        cost = rng.uniform(-1.0, 1.0, n_halo).astype(FLOAT)
        ph = rng.uniform(0.0, 2.0 * np.pi, n_halo).astype(FLOAT)
        halo_r_xy = rh * np.sqrt(np.maximum(0.0, 1.0 - cost * cost))
        r = np.concatenate([r, halo_r_xy.astype(FLOAT)])
        theta = np.concatenate([theta, ph.astype(FLOAT)])
        z = np.concatenate([z, (rh * cost).astype(FLOAT)])
        is_halo = np.concatenate([np.zeros(n_disk, dtype=bool), np.ones(n_halo, dtype=bool)])
    else:
        is_halo = np.zeros(n_disk, dtype=bool)

    # Visual intensity: hot/bright near the core, cooler in the outer disk.
    mass = (1.0 + 2.5 * np.exp(-r / 28.0) + rng.lognormal(mean=-1.2, sigma=0.45, size=r.shape[0])).astype(FLOAT)
    temp = (1.0 - np.clip(r / args.radius, 0, 1)).astype(FLOAT)
    return {"r": r, "theta0": theta, "z": z, "mass": mass, "temp": temp, "halo": is_halo}


def positions_at(sample: Dict[str, np.ndarray], args, frame_idx: int, nframes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = sample["r"]
    theta0 = sample["theta0"]
    z = sample["z"]
    temp = sample["temp"]
    mass = sample["mass"]

    progress = frame_idx / max(1, nframes - 1)

    # Differential rotation: purely visual, inner radii rotate faster.
    omega = args.rotation_factor * (1.0 / np.sqrt((r / 38.0) ** 2 + 0.28))
    theta = theta0 + progress * 2.7 * omega

    # Gentle breathing and warp for depth; not a physical N-body update.
    breathing = 1.0 + 0.035 * np.sin(2.0 * np.pi * progress + r / 24.0)
    rr = r * breathing
    warp = 0.08 * args.thickness * np.sin(theta0 * 1.7 + 2.0 * np.pi * progress) * (r / args.radius)

    x = rr * np.cos(theta)
    y = rr * np.sin(theta)
    zz = z + warp.astype(FLOAT)
    return x.astype(FLOAT), y.astype(FLOAT), zz.astype(FLOAT), temp, mass


def smooth_density(field: np.ndarray, passes: int) -> np.ndarray:
    """Small separable blur; avoids scipy dependency."""
    if passes <= 0:
        return field
    out = field.astype(np.float32, copy=True)
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    for _ in range(passes):
        out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=out)
        out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=out)
    return out


def set_dark_axes(ax):
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_color("#303449")
    ax.tick_params(colors="#B8BED8", labelsize=8)
    ax.xaxis.label.set_color("#C9D2FF")
    ax.yaxis.label.set_color("#C9D2FF")
    ax.title.set_color("#F0F3FF")
    ax.grid(True, alpha=0.12, linewidth=0.5)


def draw_core_glow_2d(ax):
    """Three visible glow layers around the galactic core."""
    ax.scatter([0], [0], s=2600, c="#FF7A00", alpha=0.035, linewidths=0, zorder=8)
    ax.scatter([0], [0], s=1000, c="#FFD166", alpha=0.095, linewidths=0, zorder=9)
    ax.scatter([0], [0], s=250, c="#FFF0B3", alpha=0.95, edgecolors="#FFB347", linewidths=0.7, zorder=10)
    ax.scatter([0], [0], s=45, c="#FFFFFF", alpha=1.0, linewidths=0, zorder=11)


def draw_core_glow_3d(ax3):
    ax3.scatter([0], [0], [0], s=850, c="#FF7A00", alpha=0.045, linewidths=0)
    ax3.scatter([0], [0], [0], s=300, c="#FFD166", alpha=0.15, linewidths=0)
    ax3.scatter([0], [0], [0], s=90, c="#FFF0B3", alpha=1.0, edgecolors="#FFB347", linewidths=0.5)


def choose_weighted_indices(rng: np.random.Generator, temp: np.ndarray, n: int) -> np.ndarray:
    """Choose points biased slightly toward bright/core particles, without replacing."""
    n = min(n, len(temp))
    if n <= 0:
        return np.array([], dtype=np.int64)
    weights = 0.20 + np.clip(temp, 0.0, 1.0) ** 1.4
    weights = weights / np.sum(weights)
    return rng.choice(len(temp), size=n, replace=False, p=weights)


def make_frame(frame_idx: int, nframes: int, all_samples, args, out_path: Path, rank_count: int):
    xs, ys, zs, temps, masses = [], [], [], [], []
    for s in all_samples:
        x, y, z, t, m = positions_at(s, args, frame_idx, nframes)
        xs.append(x); ys.append(y); zs.append(z); temps.append(t); masses.append(m)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    temp = np.concatenate(temps)
    mass = np.concatenate(masses)

    lim = args.radius * 1.18
    year = args.years * frame_idx / max(1, nframes - 1)
    progress = frame_idx / max(1, nframes - 1)

    # Fixed figure, fixed axes, fixed manual margins: all frames keep the same pixel dimensions.
    fig = plt.figure(figsize=(args.fig_width, args.fig_height), dpi=args.dpi, facecolor=SPACE_BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.10)

    ax = fig.add_subplot(gs[0, 0])
    set_dark_axes(ax)
    ax.set_title("2D density view — smooth sampled density", pad=10, fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")

    # 2D density: histogram + smoothing + interpolated image. This reads as density,
    # not as a raw scatter of many unrelated dots.
    weights = 0.65 + 1.35 * np.clip(temp, 0.0, 1.0) + 0.18 * np.clip(mass, 0.0, np.percentile(mass, 99))
    density, xedges, yedges = np.histogram2d(
        x, y,
        bins=args.density_bins,
        range=[[-lim, lim], [-lim, lim]],
        weights=weights,
    )
    density = smooth_density(density.T, args.density_smooth_passes)
    positive = density[density > 0]
    if positive.size:
        vmax = max(float(np.percentile(positive, 99.75)), 2.0)
        ax.imshow(
            np.ma.masked_less_equal(density, 0),
            extent=[-lim, lim, -lim, lim],
            origin="lower",
            cmap="magma",
            norm=LogNorm(vmin=max(0.65, float(np.min(positive))), vmax=vmax),
            interpolation="bicubic",
            alpha=0.98,
            zorder=2,
        )

    # Small sparkle layer only, not the main visualization.
    rng = np.random.default_rng(args.seed + 777 + frame_idx)
    overlay_n = min(args.overlay_points, len(x))
    idx = choose_weighted_indices(rng, temp, overlay_n)
    if idx.size:
        sizes = 0.75 + 4.5 * np.clip(temp[idx], 0, 1) ** 2
        ax.scatter(x[idx], y[idx], s=sizes * 3.4, c=temp[idx], cmap="cool", alpha=0.055, linewidths=0, zorder=4)
        ax.scatter(x[idx], y[idx], s=sizes, c=temp[idx], cmap="cool", alpha=0.36, linewidths=0, zorder=5)

    draw_core_glow_2d(ax)

    ax3 = fig.add_subplot(gs[0, 1], projection="3d")
    ax3.set_facecolor(AX_BG)
    ax3.set_title("3D sample view — reduced point cloud", pad=8, fontsize=12, color="#F0F3FF")
    ax3.set_xlim(-lim, lim)
    ax3.set_ylim(-lim, lim)
    ax3.set_zlim(-lim * 0.35, lim * 0.35)
    ax3.set_xlabel("x", color="#C9D2FF")
    ax3.set_ylabel("y", color="#C9D2FF")
    ax3.set_zlabel("z", color="#C9D2FF")
    ax3.tick_params(colors="#B8BED8", labelsize=7)
    ax3.grid(True, alpha=0.12)

    azim = args.camera_azim_start + (args.camera_azim_end - args.camera_azim_start) * progress
    ax3.view_init(elev=args.camera_elev, azim=azim)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.pane.set_facecolor((0.02, 0.02, 0.06, 0.0))
        axis.pane.set_edgecolor((0.25, 0.28, 0.45, 0.25))

    n3 = min(args.max_3d_points, len(x))
    idx3 = choose_weighted_indices(rng, temp, n3)
    if idx3.size:
        sizes3 = 1.2 + 4.6 * np.clip(temp[idx3], 0, 1)
        ax3.scatter(
            x[idx3], y[idx3], z[idx3],
            s=sizes3,
            c=temp[idx3],
            cmap="plasma",
            alpha=0.48,
            linewidths=0,
            depthshade=True,
        )
    draw_core_glow_3d(ax3)

    represented = f"{args.represented_particles:,}"
    sampled = f"{args.sample_particles:,}"
    txt = (
        "REPRESENTATIVE VISUALIZATION — NOT FULL N-BODY SIMULATION\n"
        f"represented particles: {represented} | plotted sample: {sampled} | 3D points/frame: {n3:,} | MPI ranks: {rank_count}\n"
        f"year: {year:,.0f} / {args.years:,.0f} | frames: {nframes} | method: smoothed density + analytic rotation\n"
        f"camera azimuth: {azim:.1f}° | reason: plotting/simulating all {represented} particles directly is not feasible on a laptop"
    )
    fig.text(
        0.018,
        0.018,
        txt,
        color="#DDE5FF",
        fontsize=9,
        bbox=dict(facecolor="#080B18", alpha=0.78, boxstyle="round,pad=0.55", edgecolor="#313A68"),
    )

    fig.suptitle("Galaxy scale proxy GIF — fixed layout final", color="#FFFFFF", fontsize=15, y=0.985)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.925, bottom=0.145)

    # Never use bbox_inches="tight" here. It can change PNG dimensions frame-to-frame.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_csv(path: Path, args, rank_count: int, total_sample: int, runtime: float):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "represented_particles", "sample_particles", "max_3d_points", "mpi_ranks",
            "frames", "years", "runtime_seconds", "camera_azim_start", "camera_azim_end", "note",
        ])
        w.writerow([
            args.represented_particles, total_sample, args.max_3d_points, rank_count,
            args.frames, args.years, f"{runtime:.6f}",
            args.camera_azim_start, args.camera_azim_end,
            "Representative sampled visual proxy; not a direct all-pairs physical simulation.",
        ])


def read_frames_same_shape(frame_paths, target_size=None):
    """Read PNG frames and force the same pixel size for GIF encoding."""
    pil_frames = [Image.open(path).convert("RGBA") for path in frame_paths]
    if target_size is None:
        target_size = (
            max(img.size[0] for img in pil_frames),
            max(img.size[1] for img in pil_frames),
        )

    fixed = []
    for img in pil_frames:
        if img.size != target_size:
            canvas = Image.new("RGBA", target_size, (2, 3, 10, 255))
            xoff = (target_size[0] - img.size[0]) // 2
            yoff = (target_size[1] - img.size[1]) // 2
            canvas.paste(img, (xoff, yoff))
            img = canvas
        fixed.append(np.asarray(img))
    return fixed


def main():
    comm, rank, size = get_comm()
    args = parse_args()

    if args.sample_particles < 1:
        raise ValueError("--sample-particles must be positive")
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")
    if args.density_bins < 16:
        raise ValueError("--density-bins must be at least 16")
    if args.max_3d_points < 1:
        raise ValueError("--max-3d-points must be positive")

    if rank == 0:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = out_dir / "frames_visual_scale_final"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        config = vars(args).copy()
    else:
        out_dir = None
        frames_dir = None
        config = None

    # bcast: all ranks use the same visualization parameters.
    config = comm.bcast(config, root=0)
    args = argparse.Namespace(**config)
    if rank != 0:
        out_dir = Path(args.output_dir)
        frames_dir = out_dir / "frames_visual_scale_final"

    sample_counts = split_counts(args.sample_particles, size)
    local_n = int(sample_counts[rank])

    comm.Barrier()
    t0 = time.perf_counter()
    local_sample = generate_rank_sample(local_n, args, rank)

    # reduce: verify the total plotted sample size.
    local_count = int(local_sample["r"].shape[0])
    total_sample = comm.reduce(local_count, op=MPI.SUM if MPI is not None else None, root=0)

    if rank == 0:
        print("=== Galaxy visual scale proxy GIF: final visual version ===")
        print(f"Represented particles: {args.represented_particles:,}")
        print(f"Actual 2D sample: {args.sample_particles:,} | 3D points/frame: {args.max_3d_points:,}")
        print(f"Ranks: {size} | sample counts: {sample_counts.tolist()}")
        print("2D method: smoothed log-density map + small sparkle overlay")
        print(f"3D camera: azimuth {args.camera_azim_start:g}° -> {args.camera_azim_end:g}°")
        print("MPI methods used: bcast, gather, reduce, Barrier")

    frame_paths = []
    for frame_idx in range(args.frames):
        # gather: rank 0 collects the rank-local samples for plotting each frame.
        gathered = comm.gather(local_sample, root=0)
        if rank == 0:
            frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
            make_frame(frame_idx, args.frames, gathered, args, frame_path, size)
            frame_paths.append(frame_path)
            visual_year = args.years * frame_idx / max(1, args.frames - 1)
            print(f"Frame {frame_idx + 1:03d}/{args.frames} | visual year={visual_year:,.0f}")

    comm.Barrier()
    runtime = time.perf_counter() - t0

    if rank == 0:
        gif_path = out_dir / args.gif_name
        images = read_frames_same_shape(frame_paths)
        imageio.mimsave(gif_path, images, fps=args.fps)
        csv_path = out_dir / "visual_scale_results_final.csv"
        last_frame = out_dir / "last_frame_visual_scale_final.png"
        write_csv(csv_path, args, size, int(total_sample), runtime)
        shutil.copyfile(frame_paths[-1], last_frame)
        if not args.keep_frames:
            shutil.rmtree(frames_dir)
        print("\n=== Done ===")
        print(f"GIF: {gif_path}")
        print(f"Last frame: {last_frame}")
        print(f"CSV: {csv_path}")
        print(f"Runtime: {runtime:.3f} seconds")
        print("Use this GIF only as a representative scale visualization in the report.")


if __name__ == "__main__":
    main()
