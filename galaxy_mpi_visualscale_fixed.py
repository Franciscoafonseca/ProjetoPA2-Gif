#!/usr/bin/env python3
"""
galaxy_mpi_visualscale_fixed.py

Representative large-scale galaxy GIF generator for the PA Project 2. Fixed-frame version.

This file is intentionally NOT a full N-body simulation for 1e8/1e9 particles.
It generates a visual proxy: a statistically sampled spiral galaxy and an
analytic rotation animation. The output should be described in the report as a
representative visualization of the requested scale, not as a physical direct
all-pairs simulation.

Why this exists:
- Plotting 1e8 points is not useful or feasible on a normal laptop.
- Direct all-pairs N-body for 1e8 particles requires ~1e16 interactions per
  force evaluation.
- A readable GIF should use a sample/density representation.

MPI usage:
- bcast(): distribute visualization parameters.
- gather(): collect rank-local samples on rank 0 for plotting.
- reduce(): sum local sample counts and represented particles.
- Barrier(): synchronize timing.

Run examples:
mpiexec -n 4 python galaxy_mpi_visualscale_fixed.py --represented-particles 100000000 --sample-particles 80000 --frames 30 --output-dir visual_1e8
mpiexec -n 4 python galaxy_mpi_visualscale_fixed.py --represented-particles 1000000000 --sample-particles 100000 --frames 30 --output-dir visual_1e9
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Headless-safe plotting backend
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


def get_comm():
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
                   help="Particle count represented by the visualization, e.g. 100000000 or 1000000000.")
    p.add_argument("--sample-particles", type=int, default=80_000,
                   help="Actual sampled particles plotted in the GIF. Keep 30k-150k for normal laptops.")
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
    p.add_argument("--output-dir", type=str, default="visual_scale_1e8",
                   help="Output directory.")
    p.add_argument("--gif-name", type=str, default="galaxy_visual_scale.gif")
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--fig-width", type=float, default=13.0)
    p.add_argument("--fig-height", type=float, default=6.5)
    p.add_argument("--seed", type=int, default=2120622)
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--max-3d-points", type=int, default=9000,
                   help="Maximum sample points drawn in 3D panel per frame.")
    p.add_argument("--density-bins", type=int, default=260,
                   help="2D density grid resolution.")
    return p.parse_args()


def split_counts(total: int, size: int) -> np.ndarray:
    base = total // size
    rem = total % size
    return np.array([base + (1 if r < rem else 0) for r in range(size)], dtype=np.int64)


def generate_rank_sample(n: int, args, rank: int) -> Dict[str, np.ndarray]:
    """Generate a rank-local statistical galaxy sample.

    We store radius/theta/z/mass/temperature-like scalar. Frames are obtained
    by analytic rotation, so we do not repeatedly allocate large arrays.
    """
    rng = np.random.default_rng(args.seed + 1009 * rank)
    n_halo = int(round(n * args.halo_fraction))
    n_disk = n - n_halo

    # Disk radius: gamma-like distribution for dense core + extended disk.
    r = rng.gamma(shape=2.0, scale=args.radius / 5.0, size=n_disk).astype(FLOAT)
    r = np.clip(r, 0.2, args.radius).astype(FLOAT)

    arm = rng.integers(0, max(1, args.spiral_arms), size=n_disk)
    base_theta = 2.0 * np.pi * arm / max(1, args.spiral_arms)
    theta = base_theta + args.arm_twist * (r / args.radius) + rng.normal(0.0, 0.18 + 0.006 * r, size=n_disk)
    theta = theta.astype(FLOAT)
    z = rng.normal(0.0, args.thickness * (0.25 + 0.75 * r / args.radius), size=n_disk).astype(FLOAT)

    # Halo: sparse, diffuse background around the disk.
    if n_halo > 0:
        rh = args.radius * (rng.random(n_halo) ** (1.0 / 3.0)).astype(FLOAT)
        cost = rng.uniform(-1.0, 1.0, n_halo).astype(FLOAT)
        ph = rng.uniform(0.0, 2.0 * np.pi, n_halo).astype(FLOAT)
        # Store halo in polar disk variables approximately for rendering.
        halo_r_xy = rh * np.sqrt(np.maximum(0.0, 1.0 - cost * cost))
        halo_theta = ph
        halo_z = rh * cost
        r = np.concatenate([r, halo_r_xy.astype(FLOAT)])
        theta = np.concatenate([theta, halo_theta.astype(FLOAT)])
        z = np.concatenate([z, halo_z.astype(FLOAT)])
        is_halo = np.concatenate([np.zeros(n_disk, dtype=bool), np.ones(n_halo, dtype=bool)])
    else:
        is_halo = np.zeros(n_disk, dtype=bool)

    # Visual mass/intensity: brighter near core and in spiral arms.
    mass = (1.0 + 2.5 * np.exp(-r / 28.0) + rng.lognormal(mean=-1.2, sigma=0.45, size=r.shape[0])).astype(FLOAT)
    temperature = (1.0 - np.clip(r / args.radius, 0, 1)).astype(FLOAT)

    return {"r": r, "theta0": theta, "z": z, "mass": mass, "temp": temperature, "halo": is_halo}


def positions_at(sample: Dict[str, np.ndarray], args, frame_idx: int, nframes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = sample["r"]
    theta0 = sample["theta0"]
    z = sample["z"]
    temp = sample["temp"]

    progress = frame_idx / max(1, nframes - 1)
    # Differential rotation: inner radii rotate faster; purely visual proxy.
    omega = args.rotation_factor * (1.0 / np.sqrt((r / 38.0) ** 2 + 0.28))
    theta = theta0 + progress * 2.7 * omega

    # Gentle breathing/warp only for readability. Not an N-body update.
    breathing = 1.0 + 0.035 * np.sin(2.0 * np.pi * progress + r / 24.0)
    rr = r * breathing
    warp = 0.08 * args.thickness * np.sin(theta0 * 1.7 + 2.0 * np.pi * progress) * (r / args.radius)

    x = rr * np.cos(theta)
    y = rr * np.sin(theta)
    zz = z + warp.astype(FLOAT)
    return x.astype(FLOAT), y.astype(FLOAT), zz.astype(FLOAT), temp


def set_dark_axes(ax):
    ax.set_facecolor("#03040A")
    for spine in ax.spines.values():
        spine.set_color("#303449")
    ax.tick_params(colors="#B8BED8", labelsize=8)
    ax.xaxis.label.set_color("#C9D2FF")
    ax.yaxis.label.set_color("#C9D2FF")
    ax.title.set_color("#F0F3FF")
    ax.grid(True, alpha=0.12, linewidth=0.5)


def make_frame(frame_idx: int, nframes: int, all_samples, args, out_path: Path, rank_count: int):
    xs, ys, zs, temps = [], [], [], []
    for s in all_samples:
        x, y, z, t = positions_at(s, args, frame_idx, nframes)
        xs.append(x); ys.append(y); zs.append(z); temps.append(t)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    temp = np.concatenate(temps)

    # Fixed axis range keeps the motion readable.
    lim = args.radius * 1.18
    year = args.years * frame_idx / max(1, nframes - 1)

    fig = plt.figure(figsize=(args.fig_width, args.fig_height), dpi=args.dpi, facecolor="#02030A")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.10)

    ax = fig.add_subplot(gs[0, 0])
    set_dark_axes(ax)
    ax.set_title("2D density view — representative scale sample", pad=10, fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")

    # Density map is more honest/readable than trying to draw every point.
    h = ax.hist2d(x, y, bins=args.density_bins, range=[[-lim, lim], [-lim, lim]],
                  cmap="magma", norm=LogNorm(vmin=1, vmax=max(2, len(x) // 2500)), alpha=0.96)

    # Sparkle overlay from a subset: visual stars without clutter.
    rng = np.random.default_rng(args.seed + 777 + frame_idx)
    overlay_n = min(6000, len(x))
    idx = rng.choice(len(x), size=overlay_n, replace=False)
    sizes = 1.0 + 5.0 * np.clip(temp[idx], 0, 1) ** 2
    ax.scatter(x[idx], y[idx], s=sizes, c=temp[idx], cmap="cool", alpha=0.34, linewidths=0)
    ax.scatter([0], [0], s=110, c="#FFF0B3", alpha=0.95, edgecolors="#FFB347", linewidths=0.5)

    ax3 = fig.add_subplot(gs[0, 1], projection="3d")
    ax3.set_facecolor("#03040A")
    ax3.set_title("3D sample view", pad=8, fontsize=12, color="#F0F3FF")
    ax3.set_xlim(-lim, lim); ax3.set_ylim(-lim, lim); ax3.set_zlim(-lim * 0.35, lim * 0.35)
    ax3.set_xlabel("x", color="#C9D2FF"); ax3.set_ylabel("y", color="#C9D2FF"); ax3.set_zlabel("z", color="#C9D2FF")
    ax3.tick_params(colors="#B8BED8", labelsize=7)
    ax3.grid(True, alpha=0.12)
    ax3.view_init(elev=24, azim=35 + 110 * frame_idx / max(1, nframes - 1))
    # Transparent panes
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.pane.set_facecolor((0.02, 0.02, 0.06, 0.0))
        axis.pane.set_edgecolor((0.25, 0.28, 0.45, 0.25))

    n3 = min(args.max_3d_points, len(x))
    idx3 = rng.choice(len(x), size=n3, replace=False)
    ax3.scatter(x[idx3], y[idx3], z[idx3], s=1.7 + 5.0 * temp[idx3], c=temp[idx3], cmap="plasma", alpha=0.42, linewidths=0)
    ax3.scatter([0], [0], [0], s=80, c="#FFF0B3", alpha=1.0)

    represented = f"{args.represented_particles:,}"
    sampled = f"{args.sample_particles:,}"
    txt = (
        f"REPRESENTATIVE VISUALIZATION — NOT FULL N-BODY SIMULATION\n"
        f"represented particles: {represented} | plotted sample: {sampled} | MPI ranks: {rank_count}\n"
        f"year: {year:,.0f} / {args.years:,.0f} | frames: {nframes} | method: sampled density + analytic rotation\n"
        f"reason: plotting/simulating all {represented} particles directly is not feasible on a laptop"
    )
    fig.text(0.018, 0.018, txt, color="#DDE5FF", fontsize=9,
             bbox=dict(facecolor="#080B18", alpha=0.78, boxstyle="round,pad=0.55", edgecolor="#313A68"))

    fig.suptitle("Galaxy scale proxy GIF", color="#FFFFFF", fontsize=15, y=0.985)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.925, bottom=0.145)
    # Do not use bbox_inches="tight": it can create PNGs with different shapes,
    # which breaks imageio when assembling a GIF. Fixed canvas = fixed GIF frames.
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_csv(path: Path, args, rank_count: int, total_sample: int, runtime: float):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["represented_particles", "sample_particles", "mpi_ranks", "frames", "years", "runtime_seconds", "note"])
        w.writerow([
            args.represented_particles, total_sample, rank_count, args.frames, args.years,
            f"{runtime:.6f}",
            "Representative sampled visual proxy; not a direct all-pairs physical simulation."
        ])


def read_frames_same_shape(frame_paths, target_size=None):
    """Read PNG frames and force all of them to exactly the same pixel size.

    Matplotlib 3D axes and text can sometimes create images with slightly different
    dimensions depending on labels/ticks. GIF encoders require every frame to have
    the same shape. This function pads/resizes through Pillow so imageio receives
    consistent RGBA arrays.
    """
    pil_frames = []
    for path in frame_paths:
        img = Image.open(path).convert("RGBA")
        pil_frames.append(img)

    if target_size is None:
        # Use the largest observed canvas, so no frame gets cropped.
        width = max(img.size[0] for img in pil_frames)
        height = max(img.size[1] for img in pil_frames)
        target_size = (width, height)

    fixed = []
    for img in pil_frames:
        if img.size != target_size:
            canvas = Image.new("RGBA", target_size, (2, 3, 10, 255))
            x = (target_size[0] - img.size[0]) // 2
            y = (target_size[1] - img.size[1]) // 2
            canvas.paste(img, (x, y))
            img = canvas
        fixed.append(np.asarray(img))
    return fixed


def main():
    comm, rank, size = get_comm()
    args = parse_args()

    # Sanity limits for readable GIFs.
    if args.sample_particles < 1:
        raise ValueError("--sample-particles must be positive")
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")

    if rank == 0:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = out_dir / "frames_visual_scale"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        config = vars(args).copy()
    else:
        out_dir = None
        frames_dir = None
        config = None

    # bcast used appropriately: all ranks need the same configuration.
    config = comm.bcast(config, root=0)
    # Rebuild args namespace so all ranks have same values.
    args = argparse.Namespace(**config)
    if rank != 0:
        out_dir = Path(args.output_dir)
        frames_dir = out_dir / "frames_visual_scale"

    sample_counts = split_counts(args.sample_particles, size)
    local_n = int(sample_counts[rank])

    comm.Barrier()
    t0 = time.perf_counter()
    local_sample = generate_rank_sample(local_n, args, rank)

    # reduce used to verify sample size. Represented count is conceptual scale.
    local_count = int(local_sample["r"].shape[0])
    total_sample = comm.reduce(local_count, op=MPI.SUM if MPI is not None else None, root=0)

    if rank == 0:
        print("=== Galaxy visual scale proxy GIF ===")
        print(f"Represented particles: {args.represented_particles:,}")
        print(f"Actual plotted sample: {args.sample_particles:,} | ranks: {size} | sample counts: {sample_counts.tolist()}")
        print("This is a representative visualization, not a direct all-pairs N-body simulation.")
        print("MPI methods used: bcast, gather, reduce, Barrier")

    frame_paths = []
    for frame_idx in range(args.frames):
        # gather used appropriately: rank 0 is responsible for plotting.
        gathered = comm.gather(local_sample, root=0)
        if rank == 0:
            frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
            make_frame(frame_idx, args.frames, gathered, args, frame_path, size)
            frame_paths.append(frame_path)
            print(f"Frame {frame_idx + 1:03d}/{args.frames} | visual year={args.years * frame_idx / max(1, args.frames - 1):,.0f}")

    comm.Barrier()
    runtime = time.perf_counter() - t0

    if rank == 0:
        gif_path = out_dir / args.gif_name
        images = read_frames_same_shape(frame_paths)
        imageio.mimsave(gif_path, images, fps=args.fps)
        write_csv(out_dir / "visual_scale_results.csv", args, size, int(total_sample), runtime)
        # Save a clear last frame link, too.
        shutil.copyfile(frame_paths[-1], out_dir / "last_frame_visual_scale.png")
        if not args.keep_frames:
            shutil.rmtree(frames_dir)
        print("\n=== Done ===")
        print(f"GIF: {gif_path}")
        print(f"Last frame: {out_dir / 'last_frame_visual_scale.png'}")
        print(f"CSV: {out_dir / 'visual_scale_results.csv'}")
        print(f"Runtime: {runtime:.3f} seconds")
        print("Use this GIF only as a representative scale visualization in the report.")


if __name__ == "__main__":
    main()
