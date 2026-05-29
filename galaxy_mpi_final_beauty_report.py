#!/usr/bin/env python3
"""
galaxy_mpi_final_beauty_report.py

Final PA Project 2 version: parallel galaxy simulation using mpi4py.

This file contains two clearly separated execution modes:

1) Direct physical N-body mode, used for the real simulation section of the report.
   - Newtonian gravitational acceleration: a = F / m.
   - Leapfrog / velocity-Verlet integration.
   - MPI bcast, scatter, isend/irecv, gather, reduce, allreduce and Barrier.
   - Runtime and per-frame CSV files separated by semicolon (;).
   - Animated GIF with fixed-layout 2D density + 3D point-cloud subplots.

2) Scale GIF proxy mode, used for 10^8 / 10^9 particle visualization in the report.
   - Does NOT allocate or integrate 10^8 / 10^9 real particles.
   - Generates a weighted density/proxy GIF that visually represents the requested
     particle count using a statistical sample.
   - Also writes semicolon-separated CSV output explaining the represented scale.

Important scientific honesty:
A direct all-pairs N-body method is O(N^2). A 10^9-particle direct run would require
about 10^18 pair interactions per force evaluation, so this code provides a safe
scale proxy mode for visualization and reporting instead of pretending to run the
impossible calculation on a normal laptop.

Run examples:
  # Real physical simulation, many GIF frames, good for final demo
  mpiexec -n 4 python galaxy_mpi_final_beauty_report.py --particles 1500 --years 10000 --dt 25 --plot-interval 250 --mode beauty

  # Strict report wording: frame every 1000 simulated years
  mpiexec -n 4 python galaxy_mpi_final_beauty_report.py --particles 1200 --years 10000 --dt 25 --plot-interval 1000 --mode report

  # 10^8 representative GIF without direct N-body execution
  mpiexec -n 4 python galaxy_mpi_final_beauty_report.py --particles 100000000 --years 1000000000 --scale-gif --scale-sample-particles 300000 --scale-frames 120

  # 10^9 feasibility estimate only, no arrays and no GIF
  mpiexec -n 4 python galaxy_mpi_final_beauty_report.py --particles 1000000000 --years 1000000000 --dt 1000 --scale-only
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
import matplotlib.patheffects as path_effects
import imageio.v2 as imageio
from PIL import Image

try:
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover
    MPI = None


FLOAT = np.float32
SPACE_BG = "#02030A"
AX_BG = "#03040A"


# ---------------------------------------------------------------------------
# MPI fallback for serial preview.
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
            raise RuntimeError("Dummy scatter received None in serial mode.")
        return seq[0]

    def gather(self, obj: Any, root: int = 0) -> List[Any]:
        return [obj]

    def allgather(self, obj: Any) -> List[Any]:
        return [obj]

    def reduce(self, value: Any, op: Any = None, root: int = 0) -> Any:
        return value

    def allreduce(self, value: Any, op: Any = None) -> Any:
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

    @staticmethod
    def Wtime() -> float:
        return time.perf_counter()


def get_mpi(allow_serial: bool):
    if MPI is not None:
        return MPI, MPI.COMM_WORLD, False
    if allow_serial:
        return _DummyMPI, _DummyMPI.COMM_WORLD, True
    raise SystemExit("mpi4py is not available. Install mpi4py or run with --allow-serial for a preview.")


# ---------------------------------------------------------------------------
# Arguments.
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PA Project 2 final parallel galaxy simulation and scale GIF generator.")

    # Real simulation / physics.
    p.add_argument("--particles", type=int, default=1200,
                   help="Direct N-body particles in normal mode; represented particles in --scale-gif/--scale-only mode.")
    p.add_argument("--years", type=float, default=10000.0, help="Total simulated/displayed years.")
    p.add_argument("--dt", type=float, default=25.0, help="Time step in simulated years for direct N-body mode.")
    p.add_argument("--plot-interval", type=float, default=250.0,
                   help="Years between GIF frames in direct mode. Use 1000 for strict assignment wording, 250 for smoother GIF.")
    p.add_argument("--galaxy-radius", type=float, default=100.0)
    p.add_argument("--thickness", type=float, default=5.0)
    p.add_argument("--spiral-arms", type=int, default=4)
    p.add_argument("--arm-twist", type=float, default=3.65)
    p.add_argument("--halo-fraction", type=float, default=0.10)
    p.add_argument("--g-const", type=float, default=2.0e-9,
                   help="Scaled gravitational constant. Units are project-simulation units, not SI.")
    p.add_argument("--central-mass", type=float, default=1.8e8,
                   help="Optional central mass for galaxy-like orbits. Set 0 to use only star-star gravity.")
    p.add_argument("--softening", type=float, default=9.0,
                   help="Plummer softening length. Prevents singular accelerations at very small distances.")
    p.add_argument("--rotation-factor", type=float, default=0.83)
    p.add_argument("--velocity-noise", type=float, default=0.0025)
    p.add_argument("--pair-block", type=int, default=384)
    p.add_argument("--seed", type=int, default=2120622)

    # Visuals.
    p.add_argument("--output-dir", type=str, default="galaxy_final_beauty_output")
    p.add_argument("--gif-name", type=str, default="galaxy_final_beauty.gif")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--fig-width", type=float, default=16.8)
    p.add_argument("--fig-height", type=float, default=8.8)
    p.add_argument("--trail-length", type=int, default=26)
    p.add_argument("--background-stars", type=int, default=1400)
    p.add_argument("--density-bins", type=int, default=360)
    p.add_argument("--density-smooth-passes", type=int, default=3)
    p.add_argument("--overlay-points", type=int, default=12000)
    p.add_argument("--max-3d-points", type=int, default=20000)
    p.add_argument("--camera-azim-start", type=float, default=25.0)
    p.add_argument("--camera-azim-end", type=float, default=115.0)
    p.add_argument("--camera-elev", type=float, default=25.0)
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--mode", choices=["custom", "report", "beauty", "benchmark"], default="custom")

    # CSV/result files.
    p.add_argument("--csv-delimiter", type=str, default=";", help="CSV delimiter. Default is semicolon for Portuguese/Excel locale.")
    p.add_argument("--summary-csv-name", type=str, default="runtime_summary.csv")
    p.add_argument("--frame-csv-name", type=str, default="frame_results.csv")

    # Scale modes.
    p.add_argument("--scale-only", action="store_true",
                   help="No allocation/physics/GIF. Only writes feasibility estimates for huge N.")
    p.add_argument("--scale-gif", action="store_true",
                   help="Generate a representative full-scale GIF for 10^8/10^9 without running direct N-body.")
    p.add_argument("--scale-sample-particles", type=int, default=300000,
                   help="Actual statistical sample used by --scale-gif to represent all requested particles.")
    p.add_argument("--scale-frames", type=int, default=120, help="Frames in the scale proxy GIF.")
    p.add_argument("--scale-fps", type=int, default=24)
    p.add_argument("--scale-gif-name", type=str, default="galaxy_scale_proxy_all_particles.gif")
    p.add_argument("--scale-csv-name", type=str, default="scale_proxy_results.csv")
    p.add_argument("--scale-radius", type=float, default=130.0)
    p.add_argument("--calibration-rate", type=float, default=6.68e7,
                   help="Estimated pair-interactions/s for scale-only runtime estimate.")

    p.add_argument("--allow-serial", action="store_true")
    return p.parse_args()


def apply_mode_presets(args: argparse.Namespace) -> None:
    if args.mode == "report":
        args.plot_interval = 1000.0
        args.fps = min(args.fps, 12)
        args.trail_length = min(args.trail_length, 18)
        args.overlay_points = min(args.overlay_points, 9000)
    elif args.mode == "beauty":
        # Many more frames and smoother orbital motion for the GIF.
        args.plot_interval = min(args.plot_interval, 250.0)
        args.dt = min(args.dt, 20.0)
        args.fps = max(args.fps, 20)
        args.trail_length = max(args.trail_length, 28)
        args.background_stars = max(args.background_stars, 1600)
        args.density_bins = max(args.density_bins, 380)
        args.overlay_points = max(args.overlay_points, 14000)
        args.max_3d_points = max(args.max_3d_points, 22000)
        args.softening = max(args.softening, 10.0)
    elif args.mode == "benchmark":
        args.no_gif = True
        args.no_plot = True
        args.background_stars = min(args.background_stars, 100)
        args.dpi = min(args.dpi, 100)


# ---------------------------------------------------------------------------
# Utility helpers.
# ---------------------------------------------------------------------------
def partition_counts(n: int, size: int) -> List[int]:
    base = n // size
    rem = n % size
    return [base + (1 if r < rem else 0) for r in range(size)]


def human_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} ZB"


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
    if days < 365.25:
        return f"{days:.2f} days"
    years = days / 365.25
    return f"{years:.2f} years"


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def create_gif(frame_paths: List[Path], gif_path: Path, fps: int) -> None:
    """Create a GIF while enforcing equal frame sizes."""
    pil_frames = [Image.open(p).convert("RGBA") for p in frame_paths]
    width = max(img.size[0] for img in pil_frames)
    height = max(img.size[1] for img in pil_frames)
    fixed = []
    for img in pil_frames:
        if img.size != (width, height):
            canvas = Image.new("RGBA", (width, height), (2, 3, 10, 255))
            canvas.paste(img, ((width - img.size[0]) // 2, (height - img.size[1]) // 2))
            img = canvas
        fixed.append(np.asarray(img))
    imageio.mimsave(str(gif_path), fixed, duration=1.0 / max(1, fps), loop=0)


def smooth_density(field: np.ndarray, passes: int) -> np.ndarray:
    if passes <= 0:
        return field
    out = field.astype(np.float32, copy=True)
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    for _ in range(passes):
        out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=out)
        out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=out)
    return out


def choose_weighted_indices(rng: np.random.Generator, weights: np.ndarray, n: int) -> np.ndarray:
    if len(weights) == 0 or n <= 0:
        return np.array([], dtype=np.int64)
    n = min(n, len(weights))
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w - np.min(w), 0.0) + 0.05
    w /= np.sum(w)
    return rng.choice(len(weights), size=n, replace=False, p=w)


# ---------------------------------------------------------------------------
# Initial conditions and Newtonian force calculation.
# ---------------------------------------------------------------------------
def create_cinematic_galaxy(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    n = args.particles
    ids = np.arange(n, dtype=np.int64)

    # Log-normal masses: many light bodies and a few brighter/heavier bodies.
    mass = rng.lognormal(mean=6.15, sigma=0.36, size=n).astype(np.float64)

    is_halo = rng.random(n) < np.clip(args.halo_fraction, 0.0, 0.45)
    disk = ~is_halo
    nd = int(np.count_nonzero(disk))
    nh = n - nd

    r = np.empty(n, dtype=np.float64)
    theta = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)

    # Disk: exponential/gamma-like profile + spiral-arm angular perturbation.
    r_disk = rng.gamma(shape=2.0, scale=args.galaxy_radius / 5.2, size=nd)
    r_disk = np.clip(r_disk, 0.4, args.galaxy_radius)
    arms = rng.integers(0, max(args.spiral_arms, 1), size=nd)
    arm_angle = 2.0 * np.pi * arms / max(args.spiral_arms, 1)
    theta_disk = arm_angle + args.arm_twist * (r_disk / args.galaxy_radius) + rng.normal(0.0, 0.14 + 0.0025 * r_disk, nd)
    z_disk = rng.normal(0.0, args.thickness * (0.25 + 0.75 * r_disk / args.galaxy_radius), nd)

    # Halo: diffuse spherical component.
    r_halo = args.galaxy_radius * (rng.random(nh) ** (1.0 / 3.0))
    theta_halo = rng.uniform(0.0, 2.0 * np.pi, nh)
    cost = rng.uniform(-1.0, 1.0, nh)
    r_halo_xy = r_halo * np.sqrt(np.maximum(0.0, 1.0 - cost * cost))
    z_halo = r_halo * cost

    r[disk] = r_disk
    theta[disk] = theta_disk
    z[disk] = z_disk
    r[is_halo] = r_halo_xy
    theta[is_halo] = theta_halo
    z[is_halo] = z_halo

    pos = np.column_stack([r * np.cos(theta), r * np.sin(theta), z]).astype(np.float64)

    # Approximate circular velocity around the central mass + enclosed stellar mass.
    radius_3d = np.sqrt(np.einsum("ij,ij->i", pos, pos)) + args.softening
    enclosed_fraction = np.minimum(1.0, (radius_3d / max(args.galaxy_radius, 1e-12)) ** 2)
    enclosed_mass = args.central_mass + np.sum(mass) * enclosed_fraction
    circular_speed = np.sqrt(args.g_const * enclosed_mass / radius_3d) * args.rotation_factor

    tangential = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros(n)])
    vel = tangential * circular_speed[:, None]
    vel += rng.normal(0.0, args.velocity_noise, size=(n, 3))
    vel[:, 2] *= 0.20

    # Center of mass and momentum are moved to the origin to avoid systematic drift.
    pos -= np.average(pos, axis=0, weights=mass)
    vel -= np.average(vel, axis=0, weights=mass)
    return ids, pos.astype(np.float64), vel.astype(np.float64), mass.astype(np.float64)


def split_chunks(ids: np.ndarray, pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, size: int) -> List[Dict[str, np.ndarray]]:
    counts = partition_counts(len(ids), size)
    chunks: List[Dict[str, np.ndarray]] = []
    start = 0
    for c in counts:
        end = start + c
        chunks.append({
            "ids": ids[start:end].copy(),
            "pos": pos[start:end].astype(np.float64, copy=True),
            "vel": vel[start:end].astype(np.float64, copy=True),
            "mass": mass[start:end].astype(np.float64, copy=True),
        })
        start = end
    return chunks


def acceleration_from_block(local_pos: np.ndarray, block_pos: np.ndarray, block_mass: np.ndarray,
                            g_const: float, softening: float, pair_block: int) -> np.ndarray:
    if local_pos.size == 0 or block_pos.size == 0:
        return np.zeros_like(local_pos)
    acc = np.zeros_like(local_pos)
    eps2 = softening * softening
    step = max(1, int(pair_block))
    for start in range(0, len(block_pos), step):
        end = min(start + step, len(block_pos))
        bp = block_pos[start:end]
        bm = block_mass[start:end]
        diff = bp[None, :, :] - local_pos[:, None, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff) + eps2
        inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
        acc += g_const * np.einsum("ijk,j,ij->ik", diff, bm, inv_dist3)
    return acc


def compute_acceleration_ring(comm: Any, rank: int, size: int, local_pos: np.ndarray, local_mass: np.ndarray,
                              args: argparse.Namespace, timers: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Compute acceleration on local particles using a ring exchange of particle blocks."""
    acc = np.zeros_like(local_pos)
    block_pos = np.ascontiguousarray(local_pos)
    block_mass = np.ascontiguousarray(local_mass)

    for hop in range(size):
        t0 = time.perf_counter()
        acc += acceleration_from_block(block_pos=block_pos, block_mass=block_mass, local_pos=local_pos,
                                       g_const=args.g_const, softening=args.softening, pair_block=args.pair_block)
        if timers is not None:
            timers["physics"] += time.perf_counter() - t0

        if size > 1:
            nxt = (rank + 1) % size
            prv = (rank - 1) % size
            tcomm = time.perf_counter()
            send_req = comm.isend((block_pos, block_mass), dest=nxt, tag=9200 + hop)
            recv_req = comm.irecv(source=prv, tag=9200 + hop)
            block_pos, block_mass = recv_req.wait()
            send_req.wait()
            if timers is not None:
                timers["communication"] += time.perf_counter() - tcomm

    # Optional central potential, useful for a stable spiral-galaxy-like orbit field.
    if args.central_mass > 0 and len(local_pos) > 0:
        t0 = time.perf_counter()
        diff = -local_pos
        dist2 = np.einsum("ij,ij->i", diff, diff) + args.softening * args.softening
        inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
        acc += args.g_const * args.central_mass * diff * inv_dist3[:, None]
        if timers is not None:
            timers["physics"] += time.perf_counter() - t0

    return acc


def collect_snapshot(chunks: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    ids = np.concatenate([c["ids"] for c in chunks])
    pos = np.concatenate([c["pos"] for c in chunks])
    vel = np.concatenate([c["vel"] for c in chunks])
    mass = np.concatenate([c["mass"] for c in chunks])
    order = np.argsort(ids)
    return {"ids": ids[order], "pos": pos[order], "vel": vel[order], "mass": mass[order]}


def snapshot_metrics(snapshot: Dict[str, np.ndarray], args: argparse.Namespace) -> Dict[str, float]:
    pos = snapshot["pos"]
    vel = snapshot["vel"]
    mass = snapshot["mass"]
    if len(pos) == 0:
        return {k: 0.0 for k in ["kinetic", "potential", "total_energy", "max_radius", "mean_radius", "avg_speed", "total_mass", "angular_momentum_norm", "virial_ratio"]}

    speed2 = np.einsum("ij,ij->i", vel, vel)
    speed = np.sqrt(speed2)
    radius = np.linalg.norm(pos, axis=1)
    kinetic = 0.5 * float(np.sum(mass * speed2))

    eps2 = args.softening * args.softening
    potential = 0.0
    n = len(pos)
    for i in range(n - 1):
        diff = pos[i + 1:] - pos[i]
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff) + eps2)
        potential -= float(args.g_const * mass[i] * np.sum(mass[i + 1:] / dist))
    if args.central_mass > 0:
        dist0 = np.sqrt(np.einsum("ij,ij->i", pos, pos) + eps2)
        potential -= float(args.g_const * args.central_mass * np.sum(mass / dist0))

    angular_momentum = np.sum(np.cross(pos, vel) * mass[:, None], axis=0)
    virial = 2.0 * kinetic / max(abs(potential), 1e-30)
    return {
        "kinetic": kinetic,
        "potential": potential,
        "total_energy": kinetic + potential,
        "max_radius": float(np.max(radius)),
        "mean_radius": float(np.average(radius, weights=mass)),
        "avg_speed": float(np.average(speed, weights=mass)),
        "total_mass": float(np.sum(mass)),
        "angular_momentum_norm": float(np.linalg.norm(angular_momentum)),
        "virial_ratio": float(virial),
    }


# ---------------------------------------------------------------------------
# Rendering: fixed layout, 2D density, 3D sample, glow, trails.
# ---------------------------------------------------------------------------
def make_background(seed: int, count: int, extent: float) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + 100_000)
    bx = rng.uniform(-extent, extent, count)
    by = rng.uniform(-extent, extent, count)
    bz = rng.uniform(-0.35 * extent, 0.35 * extent, count)
    bs = rng.lognormal(mean=-0.20, sigma=0.55, size=count)
    bs = np.clip(bs, 0.18, 3.8)

    grid = 360
    x = np.linspace(-extent, extent, grid)
    y = np.linspace(-extent, extent, grid)
    xx, yy = np.meshgrid(x, y)
    nebula = np.zeros((grid, grid), dtype=np.float32)
    for _ in range(12):
        cx = rng.uniform(-0.85 * extent, 0.85 * extent)
        cy = rng.uniform(-0.85 * extent, 0.85 * extent)
        sx = rng.uniform(0.14 * extent, 0.46 * extent)
        sy = rng.uniform(0.12 * extent, 0.40 * extent)
        amp = rng.uniform(0.20, 0.90)
        nebula += amp * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2.0)
    nebula += 0.04 * rng.random((grid, grid))
    nebula /= max(float(np.max(nebula)), 1e-12)
    return {"stars": (bx, by, bz, bs), "nebula": nebula, "extent": extent}


def draw_core_glow_2d(ax) -> None:
    ax.scatter([0], [0], s=4200, c="#ff7a00", alpha=0.040, linewidths=0, zorder=8)
    ax.scatter([0], [0], s=1700, c="#ffd166", alpha=0.110, linewidths=0, zorder=9)
    ax.scatter([0], [0], s=420, c="#fff0b3", alpha=0.92, edgecolors="#ffb347", linewidths=0.8, zorder=10)
    ax.scatter([0], [0], s=75, c="#ffffff", alpha=1.0, linewidths=0, zorder=11)


def draw_core_glow_3d(ax3) -> None:
    ax3.scatter([0], [0], [0], s=1200, c="#ff7a00", alpha=0.055, linewidths=0)
    ax3.scatter([0], [0], [0], s=430, c="#ffd166", alpha=0.160, linewidths=0)
    ax3.scatter([0], [0], [0], s=130, c="#fff0b3", alpha=1.0, edgecolors="#ffb347", linewidths=0.6)
    ax3.scatter([0], [0], [0], s=36, c="#ffffff", alpha=1.0, linewidths=0)


def render_physical_frame(snapshot: Dict[str, np.ndarray], history: deque, frame_path: Path, year: float,
                          frame_index: int, expected_frames: int, args: argparse.Namespace,
                          mpi_info: Dict[str, Any], metrics: Dict[str, float], background: Dict[str, Any]) -> None:
    pos = snapshot["pos"]
    vel = snapshot["vel"]
    mass = snapshot["mass"]
    n = len(pos)
    if n == 0:
        return

    speed = np.linalg.norm(vel, axis=1)
    radius = np.linalg.norm(pos, axis=1)
    extent = args.galaxy_radius * 1.52
    bg_extent = background["extent"]
    bx, by, bz, bs = background["stars"]
    nebula = background["nebula"]

    nebula_cmap = LinearSegmentedColormap.from_list("deep_space_nebula_pa", ["#02030a", "#061029", "#11175d", "#31106a", "#742979", "#ff9f1c"])
    star_cmap = LinearSegmentedColormap.from_list("stellar_heat_pa", ["#6379ff", "#35d6ff", "#f4fbff", "#ffd56a", "#ff7a2f", "#ff3b8f"])

    fig = plt.figure(figsize=(args.fig_width, args.fig_height), dpi=args.dpi, facecolor=SPACE_BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.075)
    ax2d = fig.add_subplot(gs[0, 0])
    ax3d = fig.add_subplot(gs[0, 1], projection="3d")
    for ax in (ax2d, ax3d):
        ax.set_facecolor(AX_BG)

    ax2d.imshow(nebula, extent=[-bg_extent, bg_extent, -bg_extent, bg_extent], origin="lower",
                cmap=nebula_cmap, alpha=0.35, interpolation="bilinear", zorder=0)
    ax2d.scatter(bx, by, s=bs * 2.35, c="#ffffff", alpha=0.25, linewidths=0, zorder=1)
    ax2d.scatter(bx[::6], by[::6], s=bs[::6] * 9.0, c="#9fc5ff", alpha=0.08, linewidths=0, zorder=1)
    ax3d.scatter(bx, by, bz, s=bs * 0.70, c="#ffffff", alpha=0.12, linewidths=0)

    # 2D density represents all simulated particles; raw scatter is only a visual overlay.
    weights = 0.65 + 1.20 * np.clip(1.0 - radius / max(args.galaxy_radius, 1e-12), 0, 1) + 0.20 * np.log1p(mass)
    density, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=args.density_bins,
                                  range=[[-extent, extent], [-extent, extent]], weights=weights)
    density = smooth_density(density.T, args.density_smooth_passes)
    positive = density[density > 0]
    if positive.size:
        vmin = max(float(np.percentile(positive, 8)), 0.10)
        vmax = max(float(np.percentile(positive, 99.65)), vmin + 1e-9)
        ax2d.imshow(np.ma.masked_less_equal(density, 0), extent=[-extent, extent, -extent, extent], origin="lower",
                    cmap="magma", norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="bicubic", alpha=0.98, zorder=2)

    # Trails are kept subtle; they show motion without hiding the density map.
    if len(history) > 1 and args.trail_length > 0:
        hist = list(history)
        max_segments = min(len(hist) - 1, args.trail_length)
        for h in range(max_segments):
            prev = hist[-max_segments + h]["pos"][:, :2]
            curr = hist[-max_segments + h + 1]["pos"][:, :2]
            segments = np.stack([prev, curr], axis=1)
            age = (h + 1) / max_segments
            lc = LineCollection(segments, colors=(0.38, 0.78, 1.0, 0.020 + 0.12 * age), linewidths=0.18 + 0.50 * age, zorder=3)
            ax2d.add_collection(lc)

    rng = np.random.default_rng(args.seed + 5000 + frame_index)
    brightness = 0.50 + 1.50 * np.clip(1.0 - radius / max(args.galaxy_radius, 1e-12), 0, 1) + speed / max(np.percentile(speed, 95), 1e-12)
    idx = choose_weighted_indices(rng, brightness, min(args.overlay_points, n))
    if idx.size:
        val = np.clip(1.0 - radius[idx] / max(args.galaxy_radius, 1e-12), 0, 1)
        sizes = 0.8 + 4.8 * val ** 2 + 0.003 * mass[idx]
        sizes = np.clip(sizes, 0.9, 14.0)
        ax2d.scatter(pos[idx, 0], pos[idx, 1], s=sizes * 3.7, c=val, cmap="cool", alpha=0.055, linewidths=0, zorder=4)
        ax2d.scatter(pos[idx, 0], pos[idx, 1], s=sizes, c=val, cmap="cool", alpha=0.42, linewidths=0, zorder=5)

    draw_core_glow_2d(ax2d)

    ax2d.set_xlim(-extent, extent)
    ax2d.set_ylim(-extent, extent)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_title("2D density: disk, spiral structure and orbital trails", color="#F0F3FF", fontsize=13, pad=11)
    ax2d.set_xlabel("x", color="#C9D2FF")
    ax2d.set_ylabel("y", color="#C9D2FF")
    ax2d.tick_params(colors="#B8BED8", labelsize=8)
    ax2d.grid(color="#27395f", alpha=0.22, linewidth=0.6)

    # 3D view uses a readable subset; the density view is the complete N direct run.
    floor_z = -extent * 0.37
    n3 = min(args.max_3d_points, n)
    idx3 = choose_weighted_indices(rng, brightness, n3)
    if idx3.size:
        val3 = np.clip(1.0 - radius[idx3] / max(args.galaxy_radius, 1e-12), 0, 1)
        s3 = np.clip(1.2 + 6.0 * val3 + 0.0015 * mass[idx3], 1.0, 10.0)
        ax3d.scatter(pos[idx3, 0], pos[idx3, 1], np.full(idx3.size, floor_z), s=s3 * 0.25,
                     c=val3, cmap=star_cmap, alpha=0.060, linewidths=0)
        ax3d.scatter(pos[idx3, 0], pos[idx3, 1], pos[idx3, 2], s=s3, c=val3,
                     cmap=star_cmap, alpha=0.72, linewidths=0, depthshade=True)

    draw_core_glow_3d(ax3d)
    ax3d.set_xlim(-extent, extent)
    ax3d.set_ylim(-extent, extent)
    ax3d.set_zlim(-extent * 0.40, extent * 0.40)
    ax3d.set_title("3D view: sampled cloud for readability", color="#F0F3FF", fontsize=13, pad=11)
    ax3d.set_xlabel("x", color="#C9D2FF", labelpad=-2)
    ax3d.set_ylabel("y", color="#C9D2FF", labelpad=-2)
    ax3d.set_zlabel("z", color="#C9D2FF", labelpad=-2)
    ax3d.tick_params(colors="#B8BED8", labelsize=7)
    progress = frame_index / max(1, expected_frames - 1)
    azim = args.camera_azim_start + (args.camera_azim_end - args.camera_azim_start) * progress
    ax3d.view_init(elev=args.camera_elev + 2.5 * math.sin(2.0 * math.pi * progress), azim=azim)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.set_facecolor((0.02, 0.02, 0.06, 0.0))
        axis.pane.set_edgecolor((0.25, 0.28, 0.45, 0.25))
    ax3d.grid(True, alpha=0.12)

    title = "Parallel Galaxy Simulation — Newtonian N-body + MPI"
    st = fig.suptitle(title, color="#f8fbff", fontsize=19, y=0.988, fontweight="bold")
    st.set_path_effects([path_effects.withStroke(linewidth=3, foreground=SPACE_BG)])

    hud = (
        f"year {year:,.0f} | frame {frame_index + 1}/{expected_frames} | N={n:,} | ranks={mpi_info['size']} | dt={args.dt:g}\n"
        f"Ek={metrics['kinetic']:.3e} | Ep={metrics['potential']:.3e} | E={metrics['total_energy']:.3e} | "
        f"drift={metrics.get('energy_drift_percent', 0.0):+.2f}% | virial={metrics['virial_ratio']:.3f} | Rmax={metrics['max_radius']:.1f}"
    )
    fig.text(0.020, 0.034, hud, color="#e8efff", fontsize=9.5, family="monospace",
             bbox=dict(boxstyle="round,pad=0.50", facecolor="#05091c", edgecolor="#37508a", alpha=0.74))
    fig.text(0.640, 0.034,
             "MPI: bcast • scatter • isend/irecv ring • gather • reduce • allreduce • Barrier",
             color="#b7c8ff", fontsize=8.8,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#05091c", edgecolor="#283a68", alpha=0.58))

    core_label = ax2d.text(6, 7, "central mass", color="#ffe2a3", fontsize=8.8, zorder=12)
    core_label.set_path_effects([path_effects.withStroke(linewidth=2.2, foreground=SPACE_BG)])

    fig.subplots_adjust(left=0.055, right=0.985, top=0.910, bottom=0.155)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(frame_path, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scale proxy GIF: represents huge N without huge allocation or O(N^2).
# ---------------------------------------------------------------------------
def generate_scale_rank_sample(n: int, args: argparse.Namespace, rank: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed + 1009 * rank)
    radius = args.scale_radius
    n_halo = int(round(n * np.clip(args.halo_fraction, 0.0, 0.45)))
    n_disk = n - n_halo

    r = rng.gamma(shape=2.0, scale=radius / 5.0, size=n_disk).astype(FLOAT)
    r = np.clip(r, 0.2, radius).astype(FLOAT)
    arm = rng.integers(0, max(1, args.spiral_arms), size=n_disk)
    theta = 2.0 * np.pi * arm / max(1, args.spiral_arms) + args.arm_twist * (r / radius) + rng.normal(0.0, 0.18 + 0.0045 * r, size=n_disk)
    theta = theta.astype(FLOAT)
    z = rng.normal(0.0, args.thickness * (0.25 + 0.75 * r / radius), size=n_disk).astype(FLOAT)

    if n_halo > 0:
        rh = radius * (rng.random(n_halo) ** (1.0 / 3.0)).astype(FLOAT)
        cost = rng.uniform(-1.0, 1.0, n_halo).astype(FLOAT)
        ph = rng.uniform(0.0, 2.0 * np.pi, n_halo).astype(FLOAT)
        rxy = rh * np.sqrt(np.maximum(0.0, 1.0 - cost * cost))
        r = np.concatenate([r, rxy.astype(FLOAT)])
        theta = np.concatenate([theta, ph.astype(FLOAT)])
        z = np.concatenate([z, (rh * cost).astype(FLOAT)])

    temp = (1.0 - np.clip(r / radius, 0, 1)).astype(FLOAT)
    mass = (1.0 + 2.8 * np.exp(-r / 28.0) + rng.lognormal(mean=-1.1, sigma=0.40, size=r.shape[0])).astype(FLOAT)
    return {"r": r, "theta0": theta, "z": z, "temp": temp, "mass": mass}


def scale_positions_at(sample: Dict[str, np.ndarray], args: argparse.Namespace, frame_idx: int, frames: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = sample["r"]
    theta0 = sample["theta0"]
    z = sample["z"]
    temp = sample["temp"]
    mass = sample["mass"]
    radius = args.scale_radius
    progress = frame_idx / max(1, frames - 1)

    omega = args.rotation_factor * (1.0 / np.sqrt((r / 38.0) ** 2 + 0.28))
    theta = theta0 + progress * 3.1 * omega
    breathing = 1.0 + 0.030 * np.sin(2.0 * np.pi * progress + r / 25.0)
    rr = r * breathing
    warp = 0.08 * args.thickness * np.sin(theta0 * 1.7 + 2.0 * np.pi * progress) * (r / radius)
    x = rr * np.cos(theta)
    y = rr * np.sin(theta)
    zz = z + warp.astype(FLOAT)
    return x.astype(FLOAT), y.astype(FLOAT), zz.astype(FLOAT), temp, mass


def render_scale_frame(all_samples: List[Dict[str, np.ndarray]], frame_idx: int, args: argparse.Namespace,
                       frame_path: Path, rank_count: int, total_sample: int) -> Dict[str, float]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    zs: List[np.ndarray] = []
    temps: List[np.ndarray] = []
    masses: List[np.ndarray] = []
    for sample in all_samples:
        x, y, z, t, m = scale_positions_at(sample, args, frame_idx, args.scale_frames)
        xs.append(x); ys.append(y); zs.append(z); temps.append(t); masses.append(m)
    x = np.concatenate(xs); y = np.concatenate(ys); z = np.concatenate(zs)
    temp = np.concatenate(temps); mass = np.concatenate(masses)

    represented = int(args.particles)
    sample_weight = represented / max(1, total_sample)
    lim = args.scale_radius * 1.20
    year = args.years * frame_idx / max(1, args.scale_frames - 1)
    progress = frame_idx / max(1, args.scale_frames - 1)

    fig = plt.figure(figsize=(args.fig_width, args.fig_height), dpi=args.dpi, facecolor=SPACE_BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 0.92], wspace=0.075)
    ax = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[0, 1], projection="3d")
    for a in (ax, ax3):
        a.set_facecolor(AX_BG)

    # Weighted density: each sample point represents many real particles.
    weights = sample_weight * (0.70 + 1.30 * np.clip(temp, 0, 1) + 0.18 * np.log1p(mass))
    density, _, _ = np.histogram2d(x, y, bins=args.density_bins, range=[[-lim, lim], [-lim, lim]], weights=weights)
    density = smooth_density(density.T, args.density_smooth_passes)
    positive = density[density > 0]
    if positive.size:
        ax.imshow(np.ma.masked_less_equal(density, 0), extent=[-lim, lim, -lim, lim], origin="lower",
                  cmap="magma", norm=LogNorm(vmin=max(float(np.percentile(positive, 5)), 1.0),
                                              vmax=max(float(np.percentile(positive, 99.75)), 2.0)),
                  interpolation="bicubic", alpha=0.985, zorder=2)

    rng = np.random.default_rng(args.seed + 9000 + frame_idx)
    idx = choose_weighted_indices(rng, temp + 0.20 * mass, min(args.overlay_points, len(x)))
    if idx.size:
        ax.scatter(x[idx], y[idx], s=0.8 + 5.0 * temp[idx] ** 2, c=temp[idx], cmap="cool", alpha=0.34, linewidths=0, zorder=4)
    draw_core_glow_2d(ax)

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal", adjustable="box")
    ax.set_title("2D weighted density: represents all requested particles", color="#F0F3FF", fontsize=13, pad=11)
    ax.set_xlabel("x", color="#C9D2FF"); ax.set_ylabel("y", color="#C9D2FF")
    ax.tick_params(colors="#B8BED8", labelsize=8); ax.grid(color="#27395f", alpha=0.22, linewidth=0.6)

    ax3.set_xlim(-lim, lim); ax3.set_ylim(-lim, lim); ax3.set_zlim(-lim * 0.35, lim * 0.35)
    ax3.set_title("3D sample: visual proxy, not direct N-body", color="#F0F3FF", fontsize=13, pad=11)
    ax3.set_xlabel("x", color="#C9D2FF", labelpad=-2); ax3.set_ylabel("y", color="#C9D2FF", labelpad=-2); ax3.set_zlabel("z", color="#C9D2FF", labelpad=-2)
    ax3.tick_params(colors="#B8BED8", labelsize=7); ax3.grid(True, alpha=0.12)
    azim = args.camera_azim_start + (args.camera_azim_end - args.camera_azim_start) * progress
    ax3.view_init(elev=args.camera_elev, azim=azim)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.pane.set_facecolor((0.02, 0.02, 0.06, 0.0))
        axis.pane.set_edgecolor((0.25, 0.28, 0.45, 0.25))

    idx3 = choose_weighted_indices(rng, temp + 0.20 * mass, min(args.max_3d_points, len(x)))
    if idx3.size:
        ax3.scatter(x[idx3], y[idx3], z[idx3], s=1.2 + 5.5 * temp[idx3], c=temp[idx3], cmap="plasma", alpha=0.46, linewidths=0, depthshade=True)
    draw_core_glow_3d(ax3)

    txt = (
        "FULL-SCALE REPRESENTATIVE GIF — NOT A DIRECT N-BODY RUN\n"
        f"represented particles: {represented:,} | actual statistical sample: {total_sample:,} | each sample ≈ {sample_weight:,.1f} particles\n"
        f"visual year: {year:,.0f} / {args.years:,.0f} | frame {frame_idx + 1}/{args.scale_frames} | MPI ranks: {rank_count}\n"
        "method: weighted density + analytic differential rotation; avoids impossible O(N²) direct execution"
    )
    fig.suptitle("Galaxy scale proxy — visually representing all particles", color="#FFFFFF", fontsize=18, y=0.988, fontweight="bold")
    fig.text(0.020, 0.034, txt, color="#e8efff", fontsize=9.4, family="monospace",
             bbox=dict(boxstyle="round,pad=0.50", facecolor="#05091c", edgecolor="#37508a", alpha=0.74))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.910, bottom=0.155)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(frame_path, facecolor=fig.get_facecolor())
    plt.close(fig)

    return {
        "frame": frame_idx + 1,
        "year": year,
        "represented_particles": float(represented),
        "sample_particles": float(total_sample),
        "sample_weight": float(sample_weight),
        "density_max": float(np.max(density)) if density.size else 0.0,
        "density_sum": float(np.sum(density)) if density.size else 0.0,
        "camera_azim": float(azim),
    }


def run_scale_gif(args: argparse.Namespace, rank: int, size: int, MPI_module: Any, comm: Any) -> None:
    if args.scale_sample_particles < 1 or args.scale_frames < 2:
        raise SystemExit("--scale-sample-particles must be positive and --scale-frames must be at least 2")

    if rank == 0:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = out_dir / "frames_scale_proxy"
        ensure_clean_dir(frames_dir)
        config = vars(args).copy()
    else:
        out_dir = None
        frames_dir = None
        config = None

    config = comm.bcast(config, root=0)
    args = argparse.Namespace(**config)
    if rank != 0:
        out_dir = Path(args.output_dir)
        frames_dir = out_dir / "frames_scale_proxy"

    counts = partition_counts(args.scale_sample_particles, size)
    local_n = counts[rank]
    comm.Barrier()
    t0 = time.perf_counter()
    local_sample = generate_scale_rank_sample(local_n, args, rank)
    total_sample = comm.reduce(int(local_sample["r"].shape[0]), op=MPI_module.SUM if MPI_module is not None else None, root=0)

    if rank == 0:
        print("=== Scale proxy GIF mode ===")
        print(f"Represented particles: {args.particles:,}")
        print(f"Actual sampled particles: {args.scale_sample_particles:,} | ranks: {size} | counts: {counts}")
        print("This mode does not run direct N-body; it generates a weighted visual proxy for the report.")

    frame_paths: List[Path] = []
    frame_rows: List[Dict[str, float]] = []
    for frame_idx in range(args.scale_frames):
        gathered = comm.gather(local_sample, root=0)
        if rank == 0:
            frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
            row = render_scale_frame(gathered, frame_idx, args, frame_path, size, int(total_sample))
            frame_paths.append(frame_path)
            frame_rows.append(row)
            print(f"Scale frame {frame_idx + 1:03d}/{args.scale_frames} | year={row['year']:,.0f} | represented={args.particles:,}")

    comm.Barrier()
    runtime = time.perf_counter() - t0

    if rank == 0:
        gif_path = out_dir / args.scale_gif_name
        create_gif(frame_paths, gif_path, args.scale_fps)
        last_frame = out_dir / "last_frame_scale_proxy.png"
        shutil.copyfile(frame_paths[-1], last_frame)

        csv_path = out_dir / args.scale_csv_name
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=args.csv_delimiter)
            writer.writerow(["frame", "year", "represented_particles", "sample_particles", "sample_weight", "density_max", "density_sum", "camera_azim"])
            for row in frame_rows:
                writer.writerow([row["frame"], f"{row['year']:.6f}", int(row["represented_particles"]), int(row["sample_particles"]),
                                 f"{row['sample_weight']:.6f}", f"{row['density_max']:.6e}", f"{row['density_sum']:.6e}", f"{row['camera_azim']:.6f}"])

        summary_path = out_dir / args.summary_csv_name
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=args.csv_delimiter)
            writer.writerow(["mode", "represented_particles", "actual_sample_particles", "mpi_ranks", "frames", "years", "runtime_seconds", "gif", "note"])
            writer.writerow(["scale_gif_proxy", args.particles, int(total_sample), size, args.scale_frames, args.years,
                             f"{runtime:.6f}", str(gif_path), "Weighted density proxy; not a direct O(N^2) N-body execution."])

        if not args.keep_frames:
            shutil.rmtree(frames_dir)
        print("\n=== Scale proxy done ===")
        print(f"GIF: {gif_path}")
        print(f"Last frame: {last_frame}")
        print(f"Frame CSV (;): {csv_path}")
        print(f"Summary CSV (;): {summary_path}")
        print(f"Runtime: {runtime:.3f} seconds")


def run_scale_only(args: argparse.Namespace, rank: int, size: int, MPI_module: Any, comm: Any) -> None:
    steps = int(math.ceil(args.years / args.dt))
    force_evals = max(0, 2 * steps)
    n = int(args.particles)
    counts = partition_counts(n, size)
    local_max = max(counts) if counts else 0
    bytes_per_particle_core = 8 + 3 * 8 + 3 * 8 + 8
    global_mem_core = n * bytes_per_particle_core
    local_mem_max = local_max * bytes_per_particle_core
    rank0_peak_conservative = global_mem_core * 3.0
    interactions_per_force_eval = float(n) * float(n)
    total_interactions = interactions_per_force_eval * float(force_evals)
    est_seconds = total_interactions / max(float(args.calibration_rate), 1.0)

    gathered_counts = comm.allgather(counts[rank] if rank < len(counts) else 0)
    comm.Barrier()

    if rank == 0:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / args.summary_csv_name
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=args.csv_delimiter)
            writer.writerow(["mode", "particles", "ranks", "years", "dt", "steps", "force_evals", "interactions_per_force_eval", "total_interactions", "calibration_rate_interactions_per_sec", "estimated_runtime_seconds", "estimated_runtime_human", "core_memory_all_particles", "max_local_memory_per_rank", "rank0_peak_memory_conservative", "counts"])
            writer.writerow(["scale_only_estimate", n, size, args.years, args.dt, steps, force_evals,
                             f"{interactions_per_force_eval:.6e}", f"{total_interactions:.6e}", f"{args.calibration_rate:.6e}",
                             f"{est_seconds:.6e}", human_seconds(est_seconds), human_bytes(global_mem_core),
                             human_bytes(local_mem_max), human_bytes(rank0_peak_conservative), gathered_counts])
        print("=== Scale-only feasibility estimate ===")
        print(f"Particles: {n:,} | ranks: {size} | years: {args.years:g} | dt: {args.dt:g}")
        print(f"All-pairs interactions per force evaluation: {interactions_per_force_eval:.3e}")
        print(f"Total pair interactions: {total_interactions:.3e}")
        print(f"Estimated runtime: {human_seconds(est_seconds)}")
        print(f"Core memory for all particles: {human_bytes(global_mem_core)}")
        print(f"CSV (;): {csv_path}")
        print("Conclusion: direct O(N^2) N-body at this scale is not feasible on normal hardware.")


# ---------------------------------------------------------------------------
# Main direct N-body simulation.
# ---------------------------------------------------------------------------
def run_direct_simulation(args: argparse.Namespace, rank: int, size: int, MPI_module: Any, comm: Any, serial_fallback: bool) -> None:
    if args.particles <= 0:
        raise SystemExit("--particles must be positive")
    if args.dt <= 0 or args.years <= 0 or args.plot_interval <= 0:
        raise SystemExit("--years, --dt and --plot-interval must be positive")
    if args.csv_delimiter == "":
        args.csv_delimiter = ";"

    config = vars(args) if rank == 0 else None
    config = comm.bcast(config, root=0)
    args = argparse.Namespace(**config)

    if rank == 0:
        ids, pos, vel, mass = create_cinematic_galaxy(args)
        chunks = split_chunks(ids, pos, vel, mass, size)
    else:
        chunks = None

    local = comm.scatter(chunks, root=0)
    local_ids = np.ascontiguousarray(local["ids"], dtype=np.int64)
    local_pos = np.ascontiguousarray(local["pos"], dtype=np.float64)
    local_vel = np.ascontiguousarray(local["vel"], dtype=np.float64)
    local_mass = np.ascontiguousarray(local["mass"], dtype=np.float64)
    local_count = int(len(local_ids))

    counts = comm.allgather(local_count)
    total_particles_checked = comm.reduce(local_count, op=MPI_module.SUM, root=0)

    out_dir = Path(args.output_dir)
    frame_dir = out_dir / "frames_direct"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        ensure_clean_dir(frame_dir)
        background = make_background(args.seed, args.background_stars, args.galaxy_radius * 1.55)
        history: deque = deque(maxlen=max(1, args.trail_length + 1))
        frame_paths: List[Path] = []
        frame_rows: List[Dict[str, Any]] = []
        print("=== Direct Newtonian N-body MPI simulation ===")
        print(f"Particles: {args.particles:,} | ranks: {size} | counts per rank: {counts}")
        print(f"reduce() check: total particles = {total_particles_checked}")
        print("MPI methods used: bcast, scatter, allgather, isend, irecv, gather, reduce, allreduce, Barrier")
        print("Integrator: Leapfrog / velocity-Verlet kick-drift-kick")
        print("CSV delimiter: ;")
        if serial_fallback:
            print("Running in serial fallback mode.")
    else:
        background = None
        history = None
        frame_paths = []
        frame_rows = []

    steps = int(math.ceil(args.years / args.dt))
    plot_every_steps = max(1, int(round(args.plot_interval / args.dt)))
    actual_plot_interval = plot_every_steps * args.dt
    expected_frames = len([s for s in range(steps + 1) if (s % plot_every_steps == 0 or s == steps)])

    timers: Dict[str, float] = {"physics": 0.0, "communication": 0.0, "diagnostics": 0.0, "plotting": 0.0, "gif": 0.0}
    initial_total_energy: Optional[float] = None
    final_metrics: Dict[str, float] = {"kinetic": 0.0, "potential": 0.0, "total_energy": 0.0, "energy_drift_percent": 0.0, "max_radius": 0.0, "mean_radius": 0.0, "avg_speed": 0.0, "total_mass": 0.0, "angular_momentum_norm": 0.0, "virial_ratio": 0.0}

    comm.Barrier()
    total_start = time.perf_counter()
    sim_start = time.perf_counter()

    for step in range(steps + 1):
        current_year = min(step * args.dt, args.years)

        if (not args.no_plot) and (step % plot_every_steps == 0 or step == steps):
            tcomm = time.perf_counter()
            snapshot_local = {"ids": local_ids.copy(), "pos": local_pos.copy(), "vel": local_vel.copy(), "mass": local_mass.copy()}
            gathered = comm.gather(snapshot_local, root=0)
            timers["communication"] += time.perf_counter() - tcomm

            if rank == 0:
                snapshot = collect_snapshot(gathered)
                tdiag = time.perf_counter()
                metrics = snapshot_metrics(snapshot, args)
                timers["diagnostics"] += time.perf_counter() - tdiag
                if initial_total_energy is None:
                    initial_total_energy = metrics["total_energy"] if abs(metrics["total_energy"]) > 1e-30 else 1e-30
                energy_drift = 100.0 * (metrics["total_energy"] - initial_total_energy) / max(abs(initial_total_energy), 1e-30)
                metrics["energy_drift_percent"] = energy_drift
                final_metrics = metrics

                history.append(snapshot)
                frame_index = len(frame_paths)
                frame_path = frame_dir / f"frame_{frame_index:04d}.png"
                tplot = time.perf_counter()
                render_physical_frame(snapshot, history, frame_path, current_year, frame_index, expected_frames, args,
                                      {"size": size, "counts": counts}, metrics, background)
                timers["plotting"] += time.perf_counter() - tplot
                frame_paths.append(frame_path)

                frame_rows.append({
                    "frame": frame_index + 1,
                    "step": step,
                    "year": current_year,
                    "particles": args.particles,
                    "ranks": size,
                    "kinetic": metrics["kinetic"],
                    "potential": metrics["potential"],
                    "total_energy": metrics["total_energy"],
                    "energy_drift_percent": metrics["energy_drift_percent"],
                    "virial_ratio": metrics["virial_ratio"],
                    "max_radius": metrics["max_radius"],
                    "mean_radius": metrics["mean_radius"],
                    "avg_speed": metrics["avg_speed"],
                    "angular_momentum_norm": metrics["angular_momentum_norm"],
                    "frame_file": str(frame_path),
                })
                print(f"Frame {frame_index + 1:03d}/{expected_frames} | year={current_year:,.0f} | "
                      f"E={metrics['total_energy']:.3e} | drift={energy_drift:+.2f}% | virial={metrics['virial_ratio']:.3f} | Rmax={metrics['max_radius']:.1f}")

        if step == steps:
            break

        acc = compute_acceleration_ring(comm, rank, size, local_pos, local_mass, args, timers)
        local_vel += 0.5 * acc * args.dt
        local_pos += local_vel * args.dt
        acc_new = compute_acceleration_ring(comm, rank, size, local_pos, local_mass, args, timers)
        local_vel += 0.5 * acc_new * args.dt

        # Translation to center-of-mass frame; gravitational dynamics are invariant under translation.
        local_m = float(np.sum(local_mass))
        local_com_num = np.sum(local_pos * local_mass[:, None], axis=0) if local_count else np.zeros(3)
        tcomm = time.perf_counter()
        global_com_num = np.array([comm.allreduce(float(local_com_num[i]), op=MPI_module.SUM) for i in range(3)], dtype=np.float64)
        global_mass = comm.allreduce(local_m, op=MPI_module.SUM)
        timers["communication"] += time.perf_counter() - tcomm
        local_pos -= global_com_num / max(global_mass, 1e-12)

    comm.Barrier()
    sim_elapsed = time.perf_counter() - sim_start
    sim_elapsed_max = comm.allreduce(sim_elapsed, op=MPI_module.MAX)

    gif_path = None
    last_frame = None
    if rank == 0:
        gif_path = out_dir / args.gif_name
        last_frame = out_dir / "last_frame_direct.png"
        if frame_paths and not args.no_gif:
            tgif = time.perf_counter()
            create_gif(frame_paths, gif_path, args.fps)
            timers["gif"] += time.perf_counter() - tgif
        if frame_paths:
            shutil.copyfile(frame_paths[-1], last_frame)

    comm.Barrier()
    total_elapsed = time.perf_counter() - total_start
    total_elapsed_max = comm.allreduce(total_elapsed, op=MPI_module.MAX)
    max_physics = comm.allreduce(timers["physics"], op=MPI_module.MAX)
    max_comm = comm.allreduce(timers["communication"], op=MPI_module.MAX)
    max_diag = comm.allreduce(timers["diagnostics"], op=MPI_module.MAX)
    max_plot = comm.allreduce(timers["plotting"], op=MPI_module.MAX)
    max_gif = comm.allreduce(timers["gif"], op=MPI_module.MAX)

    if rank == 0:
        summary_path = out_dir / args.summary_csv_name
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=args.csv_delimiter)
            writer.writerow(["mode", "particles", "ranks", "years", "dt", "requested_plot_interval", "actual_plot_interval", "frames", "simulation_runtime_seconds", "total_runtime_seconds", "physics_time_seconds", "mpi_time_seconds", "diagnostics_time_seconds", "plotting_time_seconds", "gif_time_seconds", "final_kinetic", "final_potential", "final_total_energy", "final_energy_drift_percent", "final_virial_ratio", "final_max_radius", "gif", "last_frame"])
            writer.writerow(["direct_newtonian_nbody", args.particles, size, args.years, args.dt, args.plot_interval, actual_plot_interval,
                             len(frame_paths), f"{sim_elapsed_max:.6f}", f"{total_elapsed_max:.6f}", f"{max_physics:.6f}",
                             f"{max_comm:.6f}", f"{max_diag:.6f}", f"{max_plot:.6f}", f"{max_gif:.6f}",
                             f"{final_metrics['kinetic']:.8e}", f"{final_metrics['potential']:.8e}", f"{final_metrics['total_energy']:.8e}",
                             f"{final_metrics['energy_drift_percent']:.6f}", f"{final_metrics['virial_ratio']:.8f}", f"{final_metrics['max_radius']:.6f}",
                             str(gif_path) if gif_path else "", str(last_frame) if last_frame else ""])

        frame_csv = out_dir / args.frame_csv_name
        with open(frame_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=args.csv_delimiter)
            writer.writerow(["frame", "step", "year", "particles", "ranks", "kinetic", "potential", "total_energy", "energy_drift_percent", "virial_ratio", "max_radius", "mean_radius", "avg_speed", "angular_momentum_norm", "frame_file"])
            for row in frame_rows:
                writer.writerow([row["frame"], row["step"], f"{row['year']:.6f}", row["particles"], row["ranks"],
                                 f"{row['kinetic']:.8e}", f"{row['potential']:.8e}", f"{row['total_energy']:.8e}",
                                 f"{row['energy_drift_percent']:.6f}", f"{row['virial_ratio']:.8f}", f"{row['max_radius']:.6f}",
                                 f"{row['mean_radius']:.6f}", f"{row['avg_speed']:.8f}", f"{row['angular_momentum_norm']:.8e}", row["frame_file"]])

        if not args.keep_frames and frame_paths:
            shutil.rmtree(frame_dir, ignore_errors=True)

        print("\n=== Direct simulation done ===")
        print(f"Simulation runtime max across ranks: {sim_elapsed_max:.4f} seconds")
        print(f"Total runtime including plotting/GIF: {total_elapsed_max:.4f} seconds")
        print(f"Physics time: {max_physics:.4f}s | MPI time: {max_comm:.4f}s | plotting: {max_plot:.4f}s | GIF: {max_gif:.4f}s")
        if args.no_gif or not frame_paths:
            print("GIF: skipped")
        else:
            print(f"GIF: {gif_path}")
        print(f"Last frame: {last_frame if frame_paths else None}")
        print(f"Summary CSV (;): {summary_path}")
        print(f"Frame CSV (;): {frame_csv}")


def main() -> None:
    args = parse_args()
    apply_mode_presets(args)
    MPI_module, comm, serial_fallback = get_mpi(args.allow_serial)
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.csv_delimiter == "":
        args.csv_delimiter = ";"

    if args.scale_only:
        # Broadcast is still used here so every rank reports from the same configuration.
        config = vars(args) if rank == 0 else None
        config = comm.bcast(config, root=0)
        args = argparse.Namespace(**config)
        run_scale_only(args, rank, size, MPI_module, comm)
        return

    if args.scale_gif:
        run_scale_gif(args, rank, size, MPI_module, comm)
        return

    run_direct_simulation(args, rank, size, MPI_module, comm, serial_fallback)


if __name__ == "__main__":
    main()
