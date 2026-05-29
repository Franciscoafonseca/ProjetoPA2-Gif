#!/usr/bin/env python3
"""
galaxy_mpi_ultra_capacity.py

Parallel galaxy / N-body simulation using mpi4py, with improved instrumentation, Leapfrog integration, energy diagnostics, CSV runtime export, and showcase-quality GIF output.

MPI methods used intentionally:
- comm.bcast(): broadcast simulation configuration from rank 0.
- comm.scatter(): distribute the initial particles/chunks from rank 0.
- comm.allgather(): share local particle counts / load-balance metadata.
- comm.isend() and comm.irecv(): ring-based exchange of particle blocks during force computation.
- comm.reduce(): verify total particles on rank 0.
- comm.allreduce(): compute global diagnostics such as kinetic energy and max radius.
- comm.gather(): gather distributed particles on rank 0 only when a frame must be plotted.
- comm.Barrier(): synchronize processes before/after timing.

Run examples:
  python galaxy_mpi_ultra.py --allow-serial --particles 250 --years 8000 --dt 25 --plot-interval 500
  mpiexec -n 4 python galaxy_mpi_ultra.py --particles 400 --years 8000 --dt 25 --plot-interval 1000
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Force non-interactive plotting, suitable for terminals and MPI runs.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.patheffects as path_effects

try:
    import imageio.v2 as imageio
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing imageio. Install it with: conda install -c conda-forge imageio pillow") from exc


# ---------------------------------------------------------------------------
# Serial fallback so the preview GIF can be generated without mpiexec.
# ---------------------------------------------------------------------------
class _DummyRequest:
    def __init__(self, payload: Any = None) -> None:
        self.payload = payload

    def wait(self) -> Any:
        return self.payload


class _DummyComm:
    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def bcast(self, obj: Any, root: int = 0) -> Any:
        return obj

    def scatter(self, seq: Optional[List[Any]], root: int = 0) -> Any:
        if seq is None:
            raise RuntimeError("Dummy scatter received None on serial rank.")
        return seq[0]

    def gather(self, obj: Any, root: int = 0) -> List[Any]:
        return [obj]

    def allgather(self, obj: Any) -> List[Any]:
        return [obj]

    def allreduce(self, value: Any, op: Any = None) -> Any:
        return value

    def reduce(self, value: Any, op: Any = None, root: int = 0) -> Any:
        return value

    def isend(self, obj: Any, dest: int, tag: int = 0) -> _DummyRequest:
        return _DummyRequest(None)

    def irecv(self, source: int, tag: int = 0) -> _DummyRequest:
        return _DummyRequest(None)

    def Barrier(self) -> None:
        return None


class _DummyMPI:
    SUM = "sum"
    MAX = "max"
    COMM_WORLD = _DummyComm()


def get_mpi(allow_serial: bool):
    try:
        from mpi4py import MPI  # type: ignore
        return MPI, MPI.COMM_WORLD, False
    except Exception as exc:
        if allow_serial:
            return _DummyMPI, _DummyMPI.COMM_WORLD, True
        raise SystemExit(
            "mpi4py is not available. Install it or run with --allow-serial for a non-MPI preview."
        ) from exc


# ---------------------------------------------------------------------------
# Argument parsing and utilities.
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel ultra-instrumented galaxy simulation using mpi4py."
    )
    parser.add_argument("--particles", type=int, default=360, help="Number of stars/particles.")
    parser.add_argument("--years", type=float, default=10000.0, help="Total simulated years.")
    parser.add_argument("--dt", type=float, default=25.0, help="Time step in simulated years.")
    parser.add_argument("--plot-interval", type=float, default=1000.0, help="Years between GIF frames.")
    parser.add_argument("--galaxy-radius", type=float, default=100.0, help="Approximate galaxy radius.")
    parser.add_argument("--thickness", type=float, default=5.0, help="Vertical disk thickness.")
    parser.add_argument("--spiral-arms", type=int, default=3, help="Number of initial spiral arms.")
    parser.add_argument("--arm-twist", type=float, default=3.35, help="Spiral twist strength.")
    parser.add_argument("--halo-fraction", type=float, default=0.08, help="Fraction of stars in a diffuse halo.")
    parser.add_argument("--g-const", type=float, default=2.0e-9, help="Scaled gravitational constant.")
    parser.add_argument("--central-mass", type=float, default=1.6e8, help="Mass of the central black-hole-like object.")
    parser.add_argument("--softening", type=float, default=8.0, help="Softening length to avoid singular forces.")
    parser.add_argument("--rotation-factor", type=float, default=0.84, help="Initial circular velocity multiplier.")
    parser.add_argument("--velocity-noise", type=float, default=0.0035, help="Small random velocity noise for visual texture.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="showcase_output", help="Directory for frames and GIF.")
    parser.add_argument("--gif-name", type=str, default="galaxy_ultra.gif", help="Output GIF filename.")
    parser.add_argument("--fps", type=int, default=10, help="GIF frames per second.")
    parser.add_argument("--trail-length", type=int, default=14, help="Number of previous plotted frames used as trails.")
    parser.add_argument("--background-stars", type=int, default=900, help="Decorative background stars in each plot.")
    parser.add_argument("--pair-block", type=int, default=512, help="Block size for vectorized pair-force computation.")
    parser.add_argument("--dpi", type=int, default=145, help="Frame DPI.")
    parser.add_argument("--fig-width", type=float, default=16.2, help="Frame width in inches.")
    parser.add_argument("--fig-height", type=float, default=8.4, help="Frame height in inches.")
    parser.add_argument("--keep-frames", action="store_true", help="Keep PNG frames after GIF generation.")
    parser.add_argument("--mode", choices=["custom", "report", "beauty", "benchmark"], default="custom",
                        help="Preset mode: report uses 1000-year frames; beauty favors smooth GIFs; benchmark minimizes visual overhead.")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF creation. Useful for clean runtime benchmarks.")
    parser.add_argument("--no-plot", action="store_true", help="Skip all plotting, frame gathering and energy snapshot diagnostics. Required for huge-scale feasibility tests.")
    parser.add_argument("--scale-only", action="store_true", help="Do not allocate particles or run physics. Only print memory/interactions/runtime estimates for very large N.")
    parser.add_argument("--capacity-only", action="store_true", help="Actually allocate local particle arrays on each rank without all-pairs physics or plots. Useful to test how many particles fit in distributed memory.")
    parser.add_argument("--touch-memory", choices=["none", "light", "full"], default="light", help="For --capacity-only: none allocates arrays only, light touches a few elements, full writes every element to force real memory commitment.")
    parser.add_argument("--calibration-rate", type=float, default=6.68e7, help="Estimated pair-interactions per second from local benchmark, used by --scale-only. Default based on the 600-particle/4-rank run.")
    parser.add_argument("--csv-name", type=str, default="runtime_results.csv", help="CSV file written in the output directory.")
    parser.add_argument("--allow-serial", action="store_true", help="Allow preview without mpi4py/mpiexec.")
    return parser.parse_args()


def apply_mode_presets(args: argparse.Namespace, rank: int = 0) -> None:
    """Apply high-level presets while keeping all CLI parameters editable.

    custom: no changes.
    report: follows the assignment wording more strictly: plot every 1000 years.
    beauty: makes a smoother GIF with more frames and longer trails.
    benchmark: reduces rendering overhead and skips GIF creation by default.
    """
    if args.mode == "report":
        args.plot_interval = 1000.0
        args.fps = min(args.fps, 8)
        args.trail_length = min(args.trail_length, 12)
    elif args.mode == "beauty":
        args.plot_interval = min(args.plot_interval, 300.0)
        args.fps = max(args.fps, 12)
        args.trail_length = max(args.trail_length, 20)
        args.dt = min(args.dt, 10.0)
        args.softening = max(args.softening, 14.0)
        args.rotation_factor = min(args.rotation_factor, 0.86)
        args.central_mass = max(args.central_mass, 2.8e8)
    elif args.mode == "benchmark":
        args.plot_interval = max(args.plot_interval, 1000.0)
        args.fps = min(args.fps, 6)
        args.trail_length = min(args.trail_length, 6)
        args.background_stars = min(args.background_stars, 200)
        args.dpi = min(args.dpi, 100)
        args.no_gif = True


def partition_counts(n: int, size: int) -> List[int]:
    base = n // size
    rem = n % size
    return [base + (1 if r < rem else 0) for r in range(size)]


def split_chunks(ids: np.ndarray, pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, size: int) -> List[Dict[str, np.ndarray]]:
    counts = partition_counts(len(ids), size)
    chunks = []
    start = 0
    for c in counts:
        end = start + c
        chunks.append({
            "ids": ids[start:end].astype(np.int64, copy=True),
            "pos": pos[start:end].astype(np.float64, copy=True),
            "vel": vel[start:end].astype(np.float64, copy=True),
            "mass": mass[start:end].astype(np.float64, copy=True),
        })
        start = end
    return chunks


# ---------------------------------------------------------------------------
# Physics and initial conditions.
# ---------------------------------------------------------------------------
def create_cinematic_galaxy(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    n = args.particles
    ids = np.arange(n, dtype=np.int64)

    # Masses are log-normal: many small stars, a few brighter/heavier ones.
    mass = rng.lognormal(mean=6.15, sigma=0.38, size=n).astype(np.float64)

    # Disk + halo split. Disk stars form visible spiral arms; halo adds depth.
    is_halo = rng.random(n) < np.clip(args.halo_fraction, 0.0, 0.5)
    disk = ~is_halo

    r = np.empty(n, dtype=np.float64)
    theta = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)

    nd = int(np.count_nonzero(disk))
    nh = n - nd

    # Disk radius distribution: dense central bulge + extended disk.
    u = rng.random(nd)
    r_disk = args.galaxy_radius * (0.16 + 0.84 * np.power(u, 0.62))
    arm_index = rng.integers(0, max(args.spiral_arms, 1), size=nd)
    arm_angle = 2.0 * np.pi * arm_index / max(args.spiral_arms, 1)
    theta_disk = arm_angle + args.arm_twist * (r_disk / args.galaxy_radius) + rng.normal(0.0, 0.16, nd)
    z_disk = rng.normal(0.0, args.thickness * (0.35 + 0.65 * r_disk / args.galaxy_radius), nd)

    # Halo: spherical, low-density points around the disk.
    r_halo = args.galaxy_radius * (0.35 + 0.85 * rng.random(nh))
    theta_halo = rng.uniform(0.0, 2.0 * np.pi, nh)
    phi_halo = np.arccos(rng.uniform(-0.65, 0.65, nh))
    z_halo = r_halo * np.cos(phi_halo) * 0.65
    r_halo_xy = r_halo * np.sin(phi_halo)

    r[disk] = r_disk
    theta[disk] = theta_disk
    z[disk] = z_disk
    r[is_halo] = r_halo_xy
    theta[is_halo] = theta_halo
    z[is_halo] = z_halo

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pos = np.column_stack([x, y, z]).astype(np.float64)

    # Initial tangential velocity around z-axis, plus small noise for visual texture.
    radius_3d = np.linalg.norm(pos, axis=1) + args.softening
    total_mass_estimate = args.central_mass + np.sum(mass) * np.minimum(1.0, radius_3d / args.galaxy_radius)
    circular_speed = np.sqrt(args.g_const * total_mass_estimate / radius_3d) * args.rotation_factor

    tangential = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros(n)])
    vel = tangential * circular_speed[:, None]
    vel += rng.normal(0.0, args.velocity_noise, size=(n, 3))
    vel[:, 2] *= 0.25

    # Shift center of mass and momentum to avoid drifting out of view.
    pos -= np.average(pos, axis=0, weights=mass)
    vel -= np.average(vel, axis=0, weights=mass)
    return ids, pos, vel.astype(np.float64), mass


def acceleration_from_block(
    local_pos: np.ndarray,
    block_pos: np.ndarray,
    block_mass: np.ndarray,
    g_const: float,
    softening: float,
    pair_block: int,
) -> np.ndarray:
    """Acceleration on local particles caused by a block of particles."""
    if local_pos.size == 0 or block_pos.size == 0:
        return np.zeros_like(local_pos)
    acc = np.zeros_like(local_pos)
    eps2 = softening * softening
    m = len(block_pos)
    step = max(1, pair_block)
    for start in range(0, m, step):
        end = min(start + step, m)
        bp = block_pos[start:end]
        bm = block_mass[start:end]
        diff = bp[None, :, :] - local_pos[:, None, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff) + eps2
        inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
        acc += g_const * np.einsum("ijk,j,ij->ik", diff, bm, inv_dist3)
    return acc


def compute_acceleration_ring(
    comm: Any,
    rank: int,
    size: int,
    local_pos: np.ndarray,
    local_mass: np.ndarray,
    args: argparse.Namespace,
    timers: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Ring-based force computation.

    Each rank computes acceleration on its own particles. Particle blocks are circulated
    with non-blocking point-to-point communication (isend/irecv). This is useful here:
    it avoids making rank 0 a bottleneck and demonstrates true process-to-process exchange.
    """
    acc = np.zeros_like(local_pos)

    block_pos = np.ascontiguousarray(local_pos)
    block_mass = np.ascontiguousarray(local_mass)
    owner = rank

    for hop in range(size):
        t_phys = time.perf_counter()
        acc += acceleration_from_block(
            local_pos, block_pos, block_mass, args.g_const, args.softening, args.pair_block
        )
        if timers is not None:
            timers["physics"] += time.perf_counter() - t_phys
        if size > 1:
            nxt = (rank + 1) % size
            prv = (rank - 1) % size
            payload = (block_pos, block_mass, owner)
            t_comm = time.perf_counter()
            send_req = comm.isend(payload, dest=nxt, tag=9000 + hop)
            recv_req = comm.irecv(source=prv, tag=9000 + hop)
            block_pos, block_mass, owner = recv_req.wait()
            send_req.wait()
            if timers is not None:
                timers["communication"] += time.perf_counter() - t_comm

    # Central black-hole-like object at origin: improves orbital structure and readability.
    if args.central_mass > 0 and local_pos.size:
        diff = -local_pos
        dist2 = np.einsum("ij,ij->i", diff, diff) + args.softening * args.softening
        inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
        t_phys = time.perf_counter()
        acc += args.g_const * args.central_mass * diff * inv_dist3[:, None]
        if timers is not None:
            timers["physics"] += time.perf_counter() - t_phys

    return acc


def total_energy_from_snapshot(snapshot: Dict[str, np.ndarray], args: argparse.Namespace) -> Tuple[float, float, float, float]:
    """Compute kinetic, potential, total energy and maximum radius on rank 0.

    This diagnostic is intentionally computed only when a frame is gathered. It is not
    in the inner loop, so it gives a useful physical check without dominating the MPI
    simulation. The pair potential uses i < j to avoid double counting.
    """
    pos = snapshot["pos"]
    vel = snapshot["vel"]
    mass = snapshot["mass"]
    if len(pos) == 0:
        return 0.0, 0.0, 0.0, 0.0
    kinetic = 0.5 * float(np.sum(mass * np.einsum("ij,ij->i", vel, vel)))
    eps2 = args.softening * args.softening
    potential = 0.0
    n = len(pos)
    for i in range(n - 1):
        diff = pos[i + 1:] - pos[i]
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff) + eps2)
        potential -= float(args.g_const * mass[i] * np.sum(mass[i + 1:] / dist))
    if args.central_mass > 0:
        r = np.sqrt(np.einsum("ij,ij->i", pos, pos) + eps2)
        potential -= float(args.g_const * args.central_mass * np.sum(mass / r))
    total = kinetic + potential
    max_radius = float(np.max(np.linalg.norm(pos, axis=1)))
    return kinetic, potential, total, max_radius


def local_diagnostics(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray) -> Tuple[float, float, float]:
    if len(pos) == 0:
        return 0.0, 0.0, 0.0
    kinetic = 0.5 * float(np.sum(mass * np.einsum("ij,ij->i", vel, vel)))
    max_radius = float(np.max(np.linalg.norm(pos, axis=1)))
    avg_speed_weighted = float(np.average(np.linalg.norm(vel, axis=1), weights=mass))
    return kinetic, max_radius, avg_speed_weighted


# ---------------------------------------------------------------------------
# Plotting and GIF generation.
# ---------------------------------------------------------------------------
def collect_snapshot(chunks: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    ids = np.concatenate([c["ids"] for c in chunks])
    pos = np.concatenate([c["pos"] for c in chunks])
    vel = np.concatenate([c["vel"] for c in chunks])
    mass = np.concatenate([c["mass"] for c in chunks])
    order = np.argsort(ids)
    return {
        "ids": ids[order],
        "pos": pos[order],
        "vel": vel[order],
        "mass": mass[order],
    }


def make_background(seed: int, count: int, extent: float):
    """Create a reusable decorative star field plus a soft nebula texture."""
    rng = np.random.default_rng(seed + 100_000)
    bx = rng.uniform(-extent, extent, count)
    by = rng.uniform(-extent, extent, count)
    bz = rng.uniform(-0.35 * extent, 0.35 * extent, count)
    # Most background stars are tiny; a few become bright anchor points.
    bs = rng.lognormal(mean=-0.25, sigma=0.55, size=count)
    bs = np.clip(bs, 0.18, 3.6)

    grid_size = 320
    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    xx, yy = np.meshgrid(x, y)
    nebula = np.zeros((grid_size, grid_size), dtype=np.float64)
    for _ in range(9):
        cx = rng.uniform(-0.80 * extent, 0.80 * extent)
        cy = rng.uniform(-0.80 * extent, 0.80 * extent)
        sx = rng.uniform(0.15 * extent, 0.45 * extent)
        sy = rng.uniform(0.12 * extent, 0.38 * extent)
        amp = rng.uniform(0.25, 0.90)
        nebula += amp * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2.0)
    nebula += 0.05 * rng.random((grid_size, grid_size))
    nebula /= max(float(nebula.max()), 1e-12)
    return {"stars": (bx, by, bz, bs), "nebula": nebula, "extent": extent}


def render_frame(
    snapshot: Dict[str, np.ndarray],
    history: deque,
    frame_path: Path,
    year: float,
    frame_index: int,
    args: argparse.Namespace,
    mpi_info: Dict[str, Any],
    diagnostics: Dict[str, float],
    background,
) -> None:
    """Render one high-legibility frame for the animated GIF."""
    pos = snapshot["pos"]
    vel = snapshot["vel"]
    mass = snapshot["mass"]
    speed = np.linalg.norm(vel, axis=1)
    radius = np.linalg.norm(pos, axis=1)

    extent = args.galaxy_radius * 1.42
    bg_extent = background["extent"]
    bx, by, bz, bs = background["stars"]
    nebula = background["nebula"]

    # Custom maps: a subtle nebula and a bright star palette.
    nebula_cmap = LinearSegmentedColormap.from_list(
        "deep_space_nebula",
        ["#03030a", "#071129", "#14165a", "#35136b", "#79287d", "#ff9f1c"],
    )
    star_cmap = LinearSegmentedColormap.from_list(
        "stellar_heat",
        ["#5b7cff", "#35d6ff", "#f4fbff", "#ffd56a", "#ff7a2f", "#ff3b8f"],
    )

    fig = plt.figure(figsize=(args.fig_width, args.fig_height), dpi=args.dpi, facecolor="#02030a")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.04, 1.0], wspace=0.055)
    ax2d = fig.add_subplot(grid[0, 0])
    ax3d = fig.add_subplot(grid[0, 1], projection="3d")

    for ax in [ax2d, ax3d]:
        ax.set_facecolor("#02030a")

    # Background nebula and stars. This makes motion easier to perceive frame-to-frame.
    ax2d.imshow(
        nebula,
        extent=[-bg_extent, bg_extent, -bg_extent, bg_extent],
        origin="lower",
        cmap=nebula_cmap,
        alpha=0.32,
        interpolation="bilinear",
        zorder=0,
    )
    ax2d.scatter(bx, by, s=bs * 2.2, c="#ffffff", alpha=0.26, linewidths=0, zorder=1)
    ax2d.scatter(bx[::7], by[::7], s=bs[::7] * 8.0, c="#9fc5ff", alpha=0.08, linewidths=0, zorder=1)
    ax3d.scatter(bx, by, bz, s=bs * 0.75, c="#ffffff", alpha=0.13, linewidths=0)

    # Density glow behind the disk so the spiral arms are visible even with many points.
    if len(pos) > 0:
        ax2d.hexbin(
            pos[:, 0],
            pos[:, 1],
            gridsize=72,
            extent=(-extent, extent, -extent, extent),
            cmap="magma",
            bins="log",
            mincnt=1,
            alpha=0.22,
            linewidths=0.0,
            zorder=2,
        )

    # Trails in 2D: oldest are faint, newest are bright. This is the main readability boost.
    if len(history) > 1 and args.trail_length > 0:
        hist = list(history)
        max_segments = min(len(hist) - 1, args.trail_length)
        for h in range(max_segments):
            prev = hist[-max_segments + h]["pos"][:, :2]
            curr = hist[-max_segments + h + 1]["pos"][:, :2]
            segments = np.stack([prev, curr], axis=1)
            age = (h + 1) / max_segments
            alpha = 0.025 + 0.18 * age
            width = 0.20 + 0.62 * age
            lc = LineCollection(segments, colors=(0.45, 0.80, 1.0, alpha), linewidths=width, zorder=3)
            ax2d.add_collection(lc)

        # 3D history as translucent layers; less noisy than thousands of 3D lines.
        for h, old in enumerate(hist[:-1]):
            age = (h + 1) / max(1, len(hist))
            alpha = 0.010 + 0.040 * age
            old_pos = old["pos"]
            ax3d.scatter(old_pos[:, 0], old_pos[:, 1], old_pos[:, 2], s=1.0 + 1.8 * age, c="#6fb4ff", alpha=alpha, linewidths=0)

    # Colour and size encoding.
    inner = np.percentile(radius, 4) if len(radius) else 0.0
    outer = np.percentile(radius, 97) if len(radius) else 1.0
    norm = Normalize(vmin=inner, vmax=outer)
    color_values = norm(radius)
    mass_norm = (mass - np.min(mass)) / (np.ptp(mass) + 1e-12)
    speed_norm = speed / (np.percentile(speed, 96) + 1e-12) if len(speed) else speed
    sizes = np.clip(7.0 + 28.0 * mass_norm + 10.0 * speed_norm, 5.0, 48.0)

    # Two-layer stars: glow layer + crisp point layer.
    ax2d.scatter(pos[:, 0], pos[:, 1], s=sizes * 4.8, c=color_values, cmap=star_cmap, alpha=0.080, linewidths=0, zorder=4)
    ax2d.scatter(pos[:, 0], pos[:, 1], s=sizes * 1.7, c=color_values, cmap=star_cmap, alpha=0.18, linewidths=0, zorder=5)
    ax2d.scatter(pos[:, 0], pos[:, 1], s=sizes, c=color_values, cmap=star_cmap, alpha=0.96, linewidths=0.05, edgecolors="#fff7d8", zorder=6)

    # Bright core / central mass.
    ax2d.scatter([0], [0], s=1450, c="#ff8a00", alpha=0.055, linewidths=0, zorder=7)
    ax2d.scatter([0], [0], s=560, c="#ffd36a", alpha=0.16, linewidths=0, zorder=8)
    ax2d.scatter([0], [0], s=135, c="#fff7c2", alpha=1.0, edgecolors="#ff9f1c", linewidths=1.1, zorder=9)

    # 3D plot with disk projection to make depth clearer.
    floor_z = -extent * 0.38
    ax3d.scatter(pos[:, 0], pos[:, 1], np.full(len(pos), floor_z), s=sizes * 0.18, c=color_values, cmap=star_cmap, alpha=0.075, linewidths=0)
    ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=sizes * 0.82, c=color_values, cmap=star_cmap, alpha=0.90, linewidths=0.0, depthshade=True)
    ax3d.scatter([0], [0], [0], s=180, c="#fff7c2", alpha=1.0, edgecolors="#ff9f1c", linewidths=0.9)
    ax3d.scatter([0], [0], [0], s=900, c="#ff8a00", alpha=0.045, linewidths=0)

    # Axes style.
    ax2d.set_xlim(-extent, extent)
    ax2d.set_ylim(-extent, extent)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_title("2D disk: spiral arms, glow density and trails", color="white", fontsize=14, pad=11)
    ax2d.set_xlabel("x", color="#cfd8ff")
    ax2d.set_ylabel("y", color="#cfd8ff")
    ax2d.tick_params(colors="#8fa0d6", labelsize=8)
    ax2d.grid(color="#27395f", alpha=0.24, linewidth=0.6)

    ax3d.set_xlim(-extent, extent)
    ax3d.set_ylim(-extent, extent)
    ax3d.set_zlim(-extent * 0.40, extent * 0.40)
    ax3d.set_title("3D perspective: disk thickness and halo", color="white", fontsize=14, pad=11)
    ax3d.set_xlabel("x", color="#cfd8ff", labelpad=-2)
    ax3d.set_ylabel("y", color="#cfd8ff", labelpad=-2)
    ax3d.set_zlabel("z", color="#cfd8ff", labelpad=-2)
    ax3d.tick_params(colors="#8fa0d6", labelsize=7)
    ax3d.view_init(elev=25 + 4.5 * math.sin(frame_index / 5.0), azim=38 + frame_index * 3.6)
    ax3d.xaxis.pane.set_facecolor((0.02, 0.02, 0.07, 0.20))
    ax3d.yaxis.pane.set_facecolor((0.02, 0.02, 0.07, 0.20))
    ax3d.zaxis.pane.set_facecolor((0.02, 0.02, 0.07, 0.20))
    ax3d.grid(True)

    # On-frame HUD. White with stroke keeps it readable on all backgrounds.
    title = f"Parallel Galaxy Simulation — ultra edition"
    st = fig.suptitle(title, color="#f8fbff", fontsize=19, y=0.988, fontweight="bold")
    st.set_path_effects([path_effects.withStroke(linewidth=3, foreground="#02030a")])

    hud = (
        f"year {year:,.0f}   •   N={len(pos)}   •   MPI ranks={mpi_info['size']}   •   frame {frame_index}\n"
        f"Ek={diagnostics['kinetic']:.3e}   Ep={diagnostics.get('potential', 0.0):.3e}   "
        f"drift={diagnostics.get('energy_drift', 0.0):+.2f}%   Rmax={diagnostics['max_radius']:.1f}   dt={args.dt:g}"
    )
    fig.text(
        0.022,
        0.035,
        hud,
        color="#e8efff",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.50", facecolor="#05091c", edgecolor="#37508a", alpha=0.72),
    )
    fig.text(
        0.655,
        0.035,
        "MPI: bcast • scatter • allgather • isend/irecv ring • gather • reduce • allreduce • Barrier",
        color="#b7c8ff",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#05091c", edgecolor="#283a68", alpha=0.58),
    )

    # Small label near the core.
    core_label = ax2d.text(6, 7, "central mass", color="#ffe2a3", fontsize=9, zorder=10)
    core_label.set_path_effects([path_effects.withStroke(linewidth=2.2, foreground="#02030a")])

    frame_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(frame_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.11)
    plt.close(fig)

def create_gif(frame_paths: List[Path], gif_path: Path, fps: int) -> None:
    images = [imageio.imread(str(p)) for p in frame_paths]
    imageio.mimsave(str(gif_path), images, duration=1.0 / max(1, fps), loop=0)



# ---------------------------------------------------------------------------
# Huge-scale feasibility / dry-run estimates.
# ---------------------------------------------------------------------------
def human_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} EB"


def human_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.2f} minutes"
    hours = minutes / 60.0
    if hours < 24:
        return f"{hours:.2f} hours"
    days = hours / 24.0
    if days < 365:
        return f"{days:.2f} days"
    years = days / 365.25
    return f"{years:.2f} years"


def run_scale_only(args: argparse.Namespace, rank: int, size: int, MPI: Any, comm: Any) -> None:
    """Safe mode for 1e8/1e9 discussions: no allocation and no N^2 loop."""
    steps = int(math.ceil(args.years / args.dt))
    force_evals = max(0, 2 * steps)  # kick-drift-kick computes acceleration twice per step
    n = int(args.particles)
    counts = partition_counts(n, size)
    local_max = max(counts) if counts else 0

    # Core arrays: ids int64 + pos float64[3] + vel float64[3] + mass float64.
    bytes_per_particle_core = 8 + 3 * 8 + 3 * 8 + 8
    local_mem_max = local_max * bytes_per_particle_core
    global_mem_core = n * bytes_per_particle_core
    # Rank 0 in the current implementation creates full arrays and split chunks, so peak is higher.
    rank0_peak_conservative = global_mem_core * 3.0

    interactions_per_force_eval = float(n) * float(n)
    total_interactions = interactions_per_force_eval * float(force_evals)
    est_seconds = total_interactions / max(float(args.calibration_rate), 1.0)

    total_particles_checked = comm.reduce(local_max if rank == 0 else 0, op=MPI.SUM, root=0)
    # Intentional simple collective use in scale-only: synchronize and gather rank counts.
    gathered_counts = comm.allgather(counts[rank] if rank < len(counts) else 0)
    comm.Barrier()

    if rank == 0:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / args.csv_name
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "particles", "ranks", "years", "dt", "steps", "force_evals",
                "total_interactions", "calibration_rate_interactions_per_sec",
                "estimated_runtime_seconds", "estimated_runtime_human",
                "core_memory_all_particles", "max_local_memory_per_rank",
                "rank0_peak_memory_conservative", "counts"
            ])
            writer.writerow([
                n, size, args.years, args.dt, steps, force_evals,
                f"{total_interactions:.6e}", f"{args.calibration_rate:.6e}",
                f"{est_seconds:.6e}", human_seconds(est_seconds),
                human_bytes(global_mem_core), human_bytes(local_mem_max),
                human_bytes(rank0_peak_conservative), gathered_counts
            ])

        print("=== Parallel Galaxy Simulation: HPC scale/capacity estimate ===")
        print(f"Particles requested: {n:,} | ranks: {size} | counts per rank: {gathered_counts}")
        print(f"Years: {args.years:g} | dt: {args.dt:g} | steps: {steps} | force evaluations: {force_evals}")
        print("No particle arrays were allocated and no plots were generated.")
        print("This mode is intended for the report discussion of 10^8 / 10^9 feasibility.")
        print(f"All-pairs interactions per force evaluation: {interactions_per_force_eval:.3e}")
        print(f"Total pair interactions for this run: {total_interactions:.3e}")
        print(f"Estimated runtime from calibration: {human_seconds(est_seconds)}")
        print(f"Core memory for all particles: {human_bytes(global_mem_core)}")
        print(f"Max local core memory per rank after scatter: {human_bytes(local_mem_max)}")
        print(f"Conservative rank-0 peak during current initialization: {human_bytes(rank0_peak_conservative)}")
        print(f"CSV: {csv_path}")
        print("Conclusion: a direct O(N^2) run at this scale is not feasible on a normal laptop/desktop.")




def run_capacity_only(args: argparse.Namespace, rank: int, size: int, MPI: Any, comm: Any) -> None:
    """Distributed memory-capacity test: allocate only this rank's local arrays.

    This is intentionally not an all-pairs simulation. It answers a different
    question for the report: can the machine represent N particles in a
    distributed layout without rank 0 holding the whole galaxy at once?
    """
    n = int(args.particles)
    counts = partition_counts(n, size)
    local_n = counts[rank] if rank < len(counts) else 0
    start_id = sum(counts[:rank])
    bytes_per_particle_core = 8 + 3 * 8 + 3 * 8 + 8
    local_bytes = local_n * bytes_per_particle_core
    global_bytes = n * bytes_per_particle_core

    comm.Barrier()
    t0 = MPI.Wtime()
    ok = True
    err_msg = ""
    checksum = 0.0
    try:
        ids = np.arange(start_id, start_id + local_n, dtype=np.int64)
        pos = np.empty((local_n, 3), dtype=np.float64)
        vel = np.empty((local_n, 3), dtype=np.float64)
        mass = np.empty(local_n, dtype=np.float64)

        if args.touch_memory == "full":
            # Full writes force the OS to commit the pages. This is the most honest
            # memory test, but it can be slow and may fail if RAM is insufficient.
            pos.fill(0.0)
            vel.fill(0.0)
            mass.fill(1.0)
        elif args.touch_memory == "light":
            # Light touch avoids leaving arrays entirely virtual while keeping the
            # test safer for laptops. It is not as strict as full.
            if local_n > 0:
                idx = np.linspace(0, local_n - 1, num=min(1024, local_n), dtype=np.int64)
                pos[idx, 0] = 0.0
                pos[idx, 1] = 0.0
                pos[idx, 2] = 0.0
                vel[idx, 0] = 0.0
                vel[idx, 1] = 0.0
                vel[idx, 2] = 0.0
                mass[idx] = 1.0
        # A tiny checksum prevents the allocation from being optimized away and
        # gives rank-local evidence of successful allocation.
        if local_n > 0:
            checksum = float(ids[0] + ids[-1]) + float(local_n)
    except MemoryError as exc:
        ok = False
        err_msg = f"MemoryError: {exc}"
    except Exception as exc:
        ok = False
        err_msg = f"{type(exc).__name__}: {exc}"

    alloc_time = MPI.Wtime() - t0
    ok_all = comm.allreduce(1 if ok else 0, op=MPI.SUM)
    total_allocated = comm.reduce(local_n if ok else 0, op=MPI.SUM, root=0)
    max_alloc_time = comm.allreduce(alloc_time, op=MPI.MAX)
    max_local_bytes = comm.allreduce(local_bytes, op=MPI.MAX)
    checksum_sum = comm.allreduce(checksum, op=MPI.SUM)
    gathered_counts = comm.allgather(local_n)
    gathered_status = comm.gather({"rank": rank, "ok": ok, "error": err_msg, "local_n": local_n, "local_memory": human_bytes(local_bytes), "alloc_time": alloc_time}, root=0)
    comm.Barrier()

    if rank == 0:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / args.csv_name
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "particles", "ranks", "capacity_success", "touch_memory", "total_allocated_particles",
                "core_memory_all_particles", "max_local_core_memory_per_rank", "max_allocation_time_seconds",
                "counts", "checksum", "rank_status"
            ])
            writer.writerow([
                n, size, bool(ok_all == size), args.touch_memory, total_allocated,
                human_bytes(global_bytes), human_bytes(max_local_bytes), f"{max_alloc_time:.6f}",
                gathered_counts, f"{checksum_sum:.6e}", gathered_status
            ])

        print("=== Parallel Galaxy Simulation: distributed capacity test ===")
        print(f"Particles requested: {n:,} | ranks: {size} | counts per rank: {gathered_counts}")
        print(f"Touch mode: {args.touch_memory}")
        print("No all-pairs force calculation, no plots, no GIF.")
        print("This test only checks whether a distributed representation can be allocated across ranks.")
        print(f"Capacity success on all ranks: {ok_all == size}")
        print(f"Total allocated particles: {int(total_allocated or 0):,}")
        print(f"Core memory for all particles: {human_bytes(global_bytes)}")
        print(f"Max local core memory per rank: {human_bytes(max_local_bytes)}")
        print(f"Max allocation/touch time across ranks: {max_alloc_time:.4f} seconds")
        print(f"Checksum: {checksum_sum:.6e}")
        if ok_all != size:
            print("At least one rank failed:")
            for st in gathered_status:
                if not st.get("ok"):
                    print(f"  rank {st.get('rank')}: {st.get('error')}")
        print(f"CSV: {csv_path}")
        print("Conclusion: this is a memory-capacity test, not a feasible direct O(N^2) simulation.")

# ---------------------------------------------------------------------------
# Main simulation.
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    MPI, comm, serial_fallback = get_mpi(args.allow_serial)
    rank = comm.Get_rank()
    size = comm.Get_size()

    apply_mode_presets(args, rank)

    if args.particles <= 0:
        raise SystemExit("--particles must be positive")
    if args.dt <= 0 or args.years <= 0 or args.plot_interval <= 0:
        raise SystemExit("--years, --dt and --plot-interval must be positive")

    # Broadcast the full configuration so all ranks are explicitly synchronized.
    config = vars(args) if rank == 0 else None
    config = comm.bcast(config, root=0)
    for key, value in config.items():
        setattr(args, key, value)

    if args.scale_only:
        run_scale_only(args, rank, size, MPI, comm)
        return

    if args.capacity_only:
        run_capacity_only(args, rank, size, MPI, comm)
        return

    if rank == 0:
        ids, pos, vel, mass = create_cinematic_galaxy(args)
        chunks = split_chunks(ids, pos, vel, mass, size)
    else:
        chunks = None

    # Scatter initial particle ownership across MPI ranks.
    local = comm.scatter(chunks, root=0)
    local_ids = np.ascontiguousarray(local["ids"], dtype=np.int64)
    local_pos = np.ascontiguousarray(local["pos"], dtype=np.float64)
    local_vel = np.ascontiguousarray(local["vel"], dtype=np.float64)
    local_mass = np.ascontiguousarray(local["mass"], dtype=np.float64)

    local_count = int(len(local_ids))
    counts = comm.allgather(local_count)
    total_particles_checked = comm.reduce(local_count, op=MPI.SUM, root=0)

    if rank == 0:
        out_dir = Path(args.output_dir)
        frame_dir = out_dir / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_dir.mkdir(parents=True, exist_ok=True)
        background = make_background(args.seed, args.background_stars, args.galaxy_radius * 1.35)
        history: deque = deque(maxlen=max(1, args.trail_length + 1))
        frame_paths: List[Path] = []
    else:
        background = None
        history = None
        frame_paths = []

    if rank == 0:
        print("=== Parallel Galaxy Simulation: ultra edition ===")
        print(f"Particles: {args.particles} | ranks: {size} | counts per rank: {counts}")
        print(f"reduce() check: total particles across ranks = {total_particles_checked}")
        print("MPI methods used: bcast, scatter, allgather, isend, irecv, gather, reduce, allreduce, Barrier")
        print("Integrator: Leapfrog / velocity-Verlet style kick-drift-kick")
        print("Diagnostics: kinetic + potential + total energy, energy drift, separated runtime timers, CSV export")
        if serial_fallback:
            print("Running in serial fallback mode because mpi4py was not imported.")

    steps = int(math.ceil(args.years / args.dt))
    plot_every_steps = max(1, int(round(args.plot_interval / args.dt)))
    actual_plot_interval = plot_every_steps * args.dt
    if rank == 0 and abs(actual_plot_interval - args.plot_interval) > 1e-9:
        print(
            f"Warning: requested plot interval {args.plot_interval:g}, "
            f"using {actual_plot_interval:g} because dt={args.dt:g}."
        )

    timers: Dict[str, float] = {
        "physics": 0.0,
        "communication": 0.0,
        "diagnostics": 0.0,
        "plotting": 0.0,
        "gif": 0.0,
    }
    initial_total_energy: Optional[float] = None
    final_energy_drift = 0.0
    final_kinetic = final_potential = final_total_energy = final_rmax = 0.0

    comm.Barrier()
    total_start = time.perf_counter()
    sim_start = time.perf_counter()

    # Initial frame + frames every plot interval.
    for step in range(steps + 1):
        current_year = min(step * args.dt, args.years)

        if (not args.no_plot) and (step % plot_every_steps == 0 or step == steps):
            t_comm = time.perf_counter()
            local_diag = local_diagnostics(local_pos, local_vel, local_mass)
            global_kinetic = comm.allreduce(local_diag[0], op=MPI.SUM)
            global_max_radius = comm.allreduce(local_diag[1], op=MPI.MAX)
            # Weighted average speed: sum(mass*speed)/sum(mass), computed with allreduce.
            local_speed_weighted_sum = float(np.sum(local_mass * np.linalg.norm(local_vel, axis=1))) if local_count else 0.0
            local_mass_sum = float(np.sum(local_mass)) if local_count else 0.0
            global_speed_weighted_sum = comm.allreduce(local_speed_weighted_sum, op=MPI.SUM)
            global_mass_sum = comm.allreduce(local_mass_sum, op=MPI.SUM)
            global_avg_speed = global_speed_weighted_sum / max(global_mass_sum, 1e-12)

            snapshot_local = {
                "ids": local_ids.copy(),
                "pos": local_pos.copy(),
                "vel": local_vel.copy(),
                "mass": local_mass.copy(),
            }
            gathered = comm.gather(snapshot_local, root=0)
            timers["communication"] += time.perf_counter() - t_comm
            if rank == 0:
                snapshot = collect_snapshot(gathered)

                t_diag = time.perf_counter()
                kinetic_root, potential_root, total_energy, rmax_root = total_energy_from_snapshot(snapshot, args)
                timers["diagnostics"] += time.perf_counter() - t_diag
                if initial_total_energy is None:
                    initial_total_energy = total_energy if abs(total_energy) > 1e-30 else 1e-30
                final_energy_drift = 100.0 * (total_energy - initial_total_energy) / max(abs(initial_total_energy), 1e-30)
                final_kinetic, final_potential, final_total_energy, final_rmax = kinetic_root, potential_root, total_energy, rmax_root

                history.append(snapshot)
                frame_path = Path(args.output_dir) / "frames" / f"frame_{len(frame_paths):04d}.png"
                t_plot = time.perf_counter()
                render_frame(
                    snapshot=snapshot,
                    history=history,
                    frame_path=frame_path,
                    year=current_year,
                    frame_index=len(frame_paths),
                    args=args,
                    mpi_info={"size": size, "counts": counts},
                    diagnostics={
                        "kinetic": float(kinetic_root),
                        "potential": float(potential_root),
                        "total_energy": float(total_energy),
                        "energy_drift": float(final_energy_drift),
                        "max_radius": float(rmax_root),
                        "avg_speed": float(global_avg_speed),
                    },
                    background=background,
                )
                timers["plotting"] += time.perf_counter() - t_plot
                frame_paths.append(frame_path)
                print(
                    f"Frame {len(frame_paths):03d} | year={current_year:,.0f} | "
                    f"Ek={kinetic_root:.3e} | Ep={potential_root:.3e} | "
                    f"E={total_energy:.3e} | drift={final_energy_drift:+.2f}% | Rmax={rmax_root:.1f}"
                )

        if step == steps:
            break

        # Symplectic kick-drift-kick integration: more stable than plain Euler for orbits.
        acc = compute_acceleration_ring(comm, rank, size, local_pos, local_mass, args, timers)
        local_vel += 0.5 * acc * args.dt
        local_pos += local_vel * args.dt
        acc_new = compute_acceleration_ring(comm, rank, size, local_pos, local_mass, args, timers)
        local_vel += 0.5 * acc_new * args.dt

        # Recentre around global centre-of-mass to keep the visualization stable.
        local_m = float(np.sum(local_mass))
        local_com_num = np.sum(local_pos * local_mass[:, None], axis=0) if local_count else np.zeros(3)
        t_comm = time.perf_counter()
        global_com_num = np.array([comm.allreduce(float(local_com_num[i]), op=MPI.SUM) for i in range(3)], dtype=np.float64)
        global_mass = comm.allreduce(local_m, op=MPI.SUM)
        timers["communication"] += time.perf_counter() - t_comm
        global_com = global_com_num / max(global_mass, 1e-12)
        local_pos -= global_com

    comm.Barrier()
    sim_elapsed = time.perf_counter() - sim_start
    sim_elapsed_max = comm.allreduce(sim_elapsed, op=MPI.MAX)

    gif_path = None
    last_frame = None
    if rank == 0:
        gif_path = Path(args.output_dir) / args.gif_name
        last_frame = Path(args.output_dir) / "last_frame.png"
        if (not args.no_gif) and frame_paths:
            t_gif = time.perf_counter()
            create_gif(frame_paths, gif_path, args.fps)
            timers["gif"] += time.perf_counter() - t_gif
        if frame_paths:
            import shutil
            shutil.copyfile(frame_paths[-1], last_frame)
        if not args.keep_frames:
            for p in frame_paths:
                try:
                    p.unlink()
                except OSError:
                    pass
            try:
                (Path(args.output_dir) / "frames").rmdir()
            except OSError:
                pass

    comm.Barrier()
    total_elapsed = time.perf_counter() - total_start
    total_elapsed_max = comm.allreduce(total_elapsed, op=MPI.MAX)
    max_physics = comm.allreduce(timers["physics"], op=MPI.MAX)
    max_comm = comm.allreduce(timers["communication"], op=MPI.MAX)
    max_diag = comm.allreduce(timers["diagnostics"], op=MPI.MAX)
    max_plot = comm.allreduce(timers["plotting"], op=MPI.MAX)
    max_gif = comm.allreduce(timers["gif"], op=MPI.MAX)

    if rank == 0:
        csv_path = Path(args.output_dir) / args.csv_name
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "particles", "ranks", "years", "dt", "requested_plot_interval", "actual_plot_interval",
                "frames", "mode", "simulation_runtime", "total_runtime", "physics_time", "mpi_time",
                "diagnostics_time", "plotting_time", "gif_time", "final_kinetic", "final_potential",
                "final_total_energy", "final_energy_drift_percent", "final_rmax"
            ])
            writer.writerow([
                args.particles, size, args.years, args.dt, args.plot_interval, actual_plot_interval,
                len(frame_paths), args.mode, f"{sim_elapsed_max:.6f}", f"{total_elapsed_max:.6f}",
                f"{max_physics:.6f}", f"{max_comm:.6f}", f"{max_diag:.6f}", f"{max_plot:.6f}",
                f"{max_gif:.6f}", f"{final_kinetic:.8e}", f"{final_potential:.8e}",
                f"{final_total_energy:.8e}", f"{final_energy_drift:.6f}", f"{final_rmax:.6f}"
            ])

        print("\n=== Done ===")
        print(f"Simulation runtime (max across ranks): {sim_elapsed_max:.4f} seconds")
        print(f"Total runtime incl. plotting/GIF/wait: {total_elapsed_max:.4f} seconds")
        print(f"Physics computation time: {max_physics:.4f} seconds")
        print(f"MPI communication/synchronization time: {max_comm:.4f} seconds")
        print(f"Diagnostics time: {max_diag:.4f} seconds")
        print(f"Plotting time: {max_plot:.4f} seconds")
        print(f"GIF creation time: {max_gif:.4f} seconds")
        print(f"Final energy drift: {final_energy_drift:+.3f}%")
        if args.no_plot:
            print("Plots/frames: skipped because --no-plot was used")
        if args.no_gif or not frame_paths:
            print("GIF: skipped because --no-gif or --no-plot was used")
        else:
            print(f"GIF: {gif_path}")
        print(f"Last frame: {last_frame if frame_paths else None}")
        print(f"CSV: {csv_path}")
        print("Use the simulation runtime for speedup, and total runtime for end-to-end cost.")


if __name__ == "__main__":
    main()
