#!/usr/bin/env python3
"""
galaxy_mpi_cinematic.py

Parallel galaxy / N-body simulation using mpi4py, with a cinematic GIF output.

MPI methods used intentionally:
- comm.bcast(): broadcast simulation configuration from rank 0.
- comm.scatter(): distribute the initial particles/chunks from rank 0.
- comm.allgather(): share local particle counts / load-balance metadata.
- comm.isend() and comm.irecv(): ring-based exchange of particle blocks during force computation.
- comm.allreduce(): compute global diagnostics such as kinetic energy and max radius.
- comm.gather(): gather distributed particles on rank 0 only when a frame must be plotted.
- comm.Barrier(): synchronize processes before/after timing.

Run examples:
  python galaxy_mpi_cinematic.py --allow-serial --particles 250 --years 8000 --dt 25 --plot-interval 500
  mpiexec -n 4 python galaxy_mpi_cinematic.py --particles 400 --years 8000 --dt 25 --plot-interval 1000
"""

from __future__ import annotations

import argparse
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
from matplotlib.colors import Normalize

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
        description="Parallel cinematic galaxy simulation using mpi4py."
    )
    parser.add_argument("--particles", type=int, default=300, help="Number of stars/particles.")
    parser.add_argument("--years", type=float, default=10000.0, help="Total simulated years.")
    parser.add_argument("--dt", type=float, default=25.0, help="Time step in simulated years.")
    parser.add_argument("--plot-interval", type=float, default=1000.0, help="Years between GIF frames.")
    parser.add_argument("--galaxy-radius", type=float, default=100.0, help="Approximate galaxy radius.")
    parser.add_argument("--thickness", type=float, default=5.0, help="Vertical disk thickness.")
    parser.add_argument("--spiral-arms", type=int, default=4, help="Number of initial spiral arms.")
    parser.add_argument("--arm-twist", type=float, default=2.7, help="Spiral twist strength.")
    parser.add_argument("--halo-fraction", type=float, default=0.10, help="Fraction of stars in a diffuse halo.")
    parser.add_argument("--g-const", type=float, default=2.0e-9, help="Scaled gravitational constant.")
    parser.add_argument("--central-mass", type=float, default=1.2e8, help="Mass of the central black-hole-like object.")
    parser.add_argument("--softening", type=float, default=8.0, help="Softening length to avoid singular forces.")
    parser.add_argument("--rotation-factor", type=float, default=0.78, help="Initial circular velocity multiplier.")
    parser.add_argument("--velocity-noise", type=float, default=0.0035, help="Small random velocity noise for visual texture.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="cinematic_output", help="Directory for frames and GIF.")
    parser.add_argument("--gif-name", type=str, default="galaxy_cinematic.gif", help="Output GIF filename.")
    parser.add_argument("--fps", type=int, default=8, help="GIF frames per second.")
    parser.add_argument("--trail-length", type=int, default=10, help="Number of previous plotted frames used as trails.")
    parser.add_argument("--background-stars", type=int, default=500, help="Decorative background stars in each plot.")
    parser.add_argument("--pair-block", type=int, default=512, help="Block size for vectorized pair-force computation.")
    parser.add_argument("--dpi", type=int, default=150, help="Frame DPI.")
    parser.add_argument("--fig-width", type=float, default=15.5, help="Frame width in inches.")
    parser.add_argument("--fig-height", type=float, default=7.8, help="Frame height in inches.")
    parser.add_argument("--keep-frames", action="store_true", help="Keep PNG frames after GIF generation.")
    parser.add_argument("--allow-serial", action="store_true", help="Allow preview without mpi4py/mpiexec.")
    return parser.parse_args()


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
        acc += acceleration_from_block(
            local_pos, block_pos, block_mass, args.g_const, args.softening, args.pair_block
        )
        if size > 1:
            nxt = (rank + 1) % size
            prv = (rank - 1) % size
            payload = (block_pos, block_mass, owner)
            send_req = comm.isend(payload, dest=nxt, tag=9000 + hop)
            recv_req = comm.irecv(source=prv, tag=9000 + hop)
            block_pos, block_mass, owner = recv_req.wait()
            send_req.wait()

    # Central black-hole-like object at origin: improves orbital structure and readability.
    if args.central_mass > 0 and local_pos.size:
        diff = -local_pos
        dist2 = np.einsum("ij,ij->i", diff, diff) + args.softening * args.softening
        inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
        acc += args.g_const * args.central_mass * diff * inv_dist3[:, None]

    return acc


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


def make_background(seed: int, count: int, extent: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 100_000)
    bx = rng.uniform(-extent, extent, count)
    by = rng.uniform(-extent, extent, count)
    bz = rng.uniform(-0.35 * extent, 0.35 * extent, count)
    bs = rng.uniform(0.3, 1.6, count)
    return bx, by, bz, bs


def render_frame(
    snapshot: Dict[str, np.ndarray],
    history: deque,
    frame_path: Path,
    year: float,
    frame_index: int,
    args: argparse.Namespace,
    mpi_info: Dict[str, Any],
    diagnostics: Dict[str, float],
    background: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    pos = snapshot["pos"]
    vel = snapshot["vel"]
    mass = snapshot["mass"]
    speed = np.linalg.norm(vel, axis=1)
    radius = np.linalg.norm(pos, axis=1)

    extent = args.galaxy_radius * 1.35
    bx, by, bz, bs = background

    fig = plt.figure(figsize=(args.fig_width, args.fig_height), dpi=args.dpi, facecolor="#03030a")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.03, 1.0], wspace=0.06)
    ax2d = fig.add_subplot(grid[0, 0])
    ax3d = fig.add_subplot(grid[0, 1], projection="3d")

    for ax in [ax2d, ax3d]:
        ax.set_facecolor("#03030a")

    # Background star field.
    ax2d.scatter(bx, by, s=bs, c="#ffffff", alpha=0.18, linewidths=0)
    ax3d.scatter(bx, by, bz, s=bs * 0.8, c="#ffffff", alpha=0.10, linewidths=0)

    # Trails in the 2D view: line collection is much faster and clearer than many plot() calls.
    if len(history) > 1 and args.trail_length > 0:
        hist = list(history)
        max_segments = min(len(hist) - 1, args.trail_length)
        for h in range(max_segments):
            prev = hist[-max_segments + h]["pos"][:, :2]
            curr = hist[-max_segments + h + 1]["pos"][:, :2]
            segments = np.stack([prev, curr], axis=1)
            alpha = 0.045 + 0.12 * (h + 1) / max_segments
            lc = LineCollection(segments, colors=(0.45, 0.75, 1.0, alpha), linewidths=0.42)
            ax2d.add_collection(lc)

        # A soft 3D trail cloud, less cluttered than line trails.
        for h, old in enumerate(hist[:-1]):
            alpha = 0.012 + 0.035 * (h + 1) / max(1, len(hist))
            old_pos = old["pos"]
            ax3d.scatter(old_pos[:, 0], old_pos[:, 1], old_pos[:, 2], s=1.2, c="#77aaff", alpha=alpha, linewidths=0)

    # Star colour: inner/core stars yellow-white; outer stars blue/purple. Size follows mass and speed.
    norm = Normalize(vmin=np.percentile(radius, 3), vmax=np.percentile(radius, 96))
    color_values = norm(radius)
    sizes = 5.5 + 22.0 * (mass - np.min(mass)) / (np.ptp(mass) + 1e-12)
    sizes += 6.0 * speed / (np.percentile(speed, 95) + 1e-12)
    sizes = np.clip(sizes, 4.0, 38.0)

    ax2d.scatter(
        pos[:, 0], pos[:, 1],
        s=sizes,
        c=color_values,
        cmap="plasma",
        alpha=0.90,
        linewidths=0.0,
    )
    ax2d.scatter([0], [0], s=210, c="#fff4b0", alpha=0.95, edgecolors="#ff9f1c", linewidths=0.8)
    ax2d.scatter([0], [0], s=820, c="#ffb000", alpha=0.055, linewidths=0)

    ax3d.scatter(
        pos[:, 0], pos[:, 1], pos[:, 2],
        s=sizes * 0.72,
        c=color_values,
        cmap="plasma",
        alpha=0.86,
        linewidths=0.0,
        depthshade=True,
    )
    ax3d.scatter([0], [0], [0], s=150, c="#fff4b0", alpha=0.98, edgecolors="#ff9f1c", linewidths=0.7)

    # Styling.
    ax2d.set_xlim(-extent, extent)
    ax2d.set_ylim(-extent, extent)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_title("2D galactic disk + orbital trails", color="white", fontsize=14, pad=10)
    ax2d.set_xlabel("x", color="#cfd8ff")
    ax2d.set_ylabel("y", color="#cfd8ff")
    ax2d.tick_params(colors="#8fa0d6", labelsize=8)
    ax2d.grid(color="#23304f", alpha=0.23, linewidth=0.6)

    ax3d.set_xlim(-extent, extent)
    ax3d.set_ylim(-extent, extent)
    ax3d.set_zlim(-extent * 0.40, extent * 0.40)
    ax3d.set_title("3D perspective", color="white", fontsize=14, pad=10)
    ax3d.set_xlabel("x", color="#cfd8ff", labelpad=-2)
    ax3d.set_ylabel("y", color="#cfd8ff", labelpad=-2)
    ax3d.set_zlabel("z", color="#cfd8ff", labelpad=-2)
    ax3d.tick_params(colors="#8fa0d6", labelsize=7)
    ax3d.view_init(elev=24 + 5 * math.sin(frame_index / 5), azim=35 + frame_index * 3.0)
    ax3d.xaxis.pane.set_facecolor((0.02, 0.02, 0.06, 0.18))
    ax3d.yaxis.pane.set_facecolor((0.02, 0.02, 0.06, 0.18))
    ax3d.zaxis.pane.set_facecolor((0.02, 0.02, 0.06, 0.18))
    ax3d.grid(True)

    title = (
        f"Parallel Galaxy Simulation — year {year:,.0f} | "
        f"N={len(pos)} | ranks={mpi_info['size']} | frame={frame_index}"
    )
    fig.suptitle(title, color="#f8fbff", fontsize=18, y=0.985, fontweight="bold")

    diag = (
        f"MPI: bcast + scatter + allgather + isend/irecv ring + gather + allreduce + Barrier\n"
        f"E_k={diagnostics['kinetic']:.3e} | max radius={diagnostics['max_radius']:.1f} | "
        f"avg speed={diagnostics['avg_speed']:.3f} | dt={args.dt:g}"
    )
    fig.text(0.015, 0.018, diag, color="#b8c7ff", fontsize=9, family="monospace")
    fig.text(0.73, 0.018, "central mass shown as bright core", color="#ffd98a", fontsize=9)

    frame_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(frame_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def create_gif(frame_paths: List[Path], gif_path: Path, fps: int) -> None:
    images = [imageio.imread(str(p)) for p in frame_paths]
    imageio.mimsave(str(gif_path), images, duration=1.0 / max(1, fps), loop=0)


# ---------------------------------------------------------------------------
# Main simulation.
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    MPI, comm, serial_fallback = get_mpi(args.allow_serial)
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.particles <= 0:
        raise SystemExit("--particles must be positive")
    if args.dt <= 0 or args.years <= 0 or args.plot_interval <= 0:
        raise SystemExit("--years, --dt and --plot-interval must be positive")

    # Broadcast the full configuration so all ranks are explicitly synchronized.
    config = vars(args) if rank == 0 else None
    config = comm.bcast(config, root=0)
    for key, value in config.items():
        setattr(args, key, value)

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
        print("=== Parallel Galaxy Simulation: cinematic edition ===")
        print(f"Particles: {args.particles} | ranks: {size} | counts per rank: {counts}")
        print("MPI methods used: bcast, scatter, allgather, isend, irecv, gather, allreduce, Barrier")
        if serial_fallback:
            print("Running in serial fallback mode because mpi4py was not imported.")

    steps = int(math.ceil(args.years / args.dt))
    plot_every_steps = max(1, int(round(args.plot_interval / args.dt)))

    comm.Barrier()
    t0 = time.perf_counter()

    # Initial frame + frames every plot interval.
    for step in range(steps + 1):
        current_year = min(step * args.dt, args.years)

        if step % plot_every_steps == 0 or step == steps:
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
            if rank == 0:
                snapshot = collect_snapshot(gathered)
                history.append(snapshot)
                frame_path = Path(args.output_dir) / "frames" / f"frame_{len(frame_paths):04d}.png"
                render_frame(
                    snapshot=snapshot,
                    history=history,
                    frame_path=frame_path,
                    year=current_year,
                    frame_index=len(frame_paths),
                    args=args,
                    mpi_info={"size": size, "counts": counts},
                    diagnostics={
                        "kinetic": float(global_kinetic),
                        "max_radius": float(global_max_radius),
                        "avg_speed": float(global_avg_speed),
                    },
                    background=background,
                )
                frame_paths.append(frame_path)
                print(f"Frame {len(frame_paths):03d} | year={current_year:,.0f} | Ek={global_kinetic:.3e} | Rmax={global_max_radius:.1f}")

        if step == steps:
            break

        # Symplectic kick-drift-kick integration: more stable than plain Euler for orbits.
        acc = compute_acceleration_ring(comm, rank, size, local_pos, local_mass, args)
        local_vel += 0.5 * acc * args.dt
        local_pos += local_vel * args.dt
        acc_new = compute_acceleration_ring(comm, rank, size, local_pos, local_mass, args)
        local_vel += 0.5 * acc_new * args.dt

        # Recentre around global centre-of-mass to keep the visualization stable.
        local_m = float(np.sum(local_mass))
        local_com_num = np.sum(local_pos * local_mass[:, None], axis=0) if local_count else np.zeros(3)
        global_com_num = np.array([comm.allreduce(float(local_com_num[i]), op=MPI.SUM) for i in range(3)], dtype=np.float64)
        global_mass = comm.allreduce(local_m, op=MPI.SUM)
        global_com = global_com_num / max(global_mass, 1e-12)
        local_pos -= global_com

    comm.Barrier()
    elapsed = time.perf_counter() - t0
    elapsed_max = comm.allreduce(elapsed, op=MPI.MAX)

    if rank == 0:
        gif_path = Path(args.output_dir) / args.gif_name
        create_gif(frame_paths, gif_path, args.fps)
        last_frame = Path(args.output_dir) / "last_frame.png"
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

        print("\n=== Done ===")
        print(f"Total runtime (max across ranks): {elapsed_max:.4f} seconds")
        print(f"GIF: {gif_path}")
        print(f"Last frame: {last_frame}")
        print("Use the runtime value in the report table.")


if __name__ == "__main__":
    main()
