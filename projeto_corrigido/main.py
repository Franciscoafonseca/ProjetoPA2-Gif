#!/usr/bin/env python3
"""
main.py

This is the entry point that joins all grading-oriented modules:
- physical_correctness.py
- mpi_parallelization.py
- runtime_measurement.py
- gif_visualization.py

Recommended execution:

1) Beautiful physical N-body GIF with many frames:
   mpiexec -n 4 python main.py --particles 1500 --years 10000 --dt 25 --plot-interval 250 --mode beauty

2) More literal assignment mode, plot every 1000 simulated years:
   mpiexec -n 4 python main.py --particles 1200 --years 10000 --dt 25 --plot-interval 1000 --mode report

3) 10^8 representative GIF without direct N-body execution:
   mpiexec -n 4 python main.py --particles 100000000 --years 1000000000 --scale-gif --scale-sample-particles 300000 --scale-frames 120

4) 10^9 representative GIF without direct N-body execution:
   mpiexec -n 4 python main.py --particles 1000000000 --years 1000000000 --scale-gif --scale-sample-particles 300000 --scale-frames 120
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from physical_correctness import (
    create_initial_conditions,
    local_basic_diagnostics,
    snapshot_energy_diagnostics,
)
from mpi_parallelization import (
    allreduce_basic_diagnostics,
    bcast_config,
    compute_acceleration_ring,
    gather_snapshot,
    get_mpi,
    partition_counts,
    scatter_particles,
    split_particle_chunks,
)
from runtime_measurement import (
    TimerBook,
    estimate_direct_nbody_cost,
    write_csv_semicolon,
    write_runtime_summary,
)
from gif_visualization import (
    create_gif,
    generate_scale_rank_sample,
    make_background,
    render_direct_frame,
    render_scale_frame,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Modular MPI galaxy simulation for PA Project 2.")

    # Core physical simulation.
    p.add_argument("--particles", type=int, default=1500, help="Number of particles for direct N-body mode, or represented particles for --scale-gif.")
    p.add_argument("--years", type=float, default=10000.0, help="Total simulated/displayed years.")
    p.add_argument("--dt", type=float, default=25.0, help="Time step in simulated years for direct N-body mode.")
    p.add_argument("--plot-interval", type=float, default=250.0, help="Years between frames. Use 1000 for the strict assignment wording.")
    p.add_argument("--galaxy-radius", type=float, default=120.0)
    p.add_argument("--thickness", type=float, default=7.0)
    p.add_argument("--spiral-arms", type=int, default=4)
    p.add_argument("--arm-twist", type=float, default=4.8)
    p.add_argument("--halo-fraction", type=float, default=0.10)
    p.add_argument("--g-const", type=float, default=2.0e-9)
    p.add_argument("--central-mass", type=float, default=1.8e8)
    p.add_argument("--softening", type=float, default=10.0)
    p.add_argument("--rotation-factor", type=float, default=0.82)
    p.add_argument("--velocity-noise", type=float, default=0.0030)
    p.add_argument("--pair-block", type=int, default=512)
    p.add_argument("--energy-sample-limit", type=int, default=5000, help="Exact potential energy up to this many particles; sampled estimate above it.")

    # Output and visual settings.
    p.add_argument("--output-dir", type=str, default="galaxy_modular_output")
    p.add_argument("--gif-name", type=str, default="galaxy_physical.gif")
    p.add_argument("--fps", type=int, default=14)
    p.add_argument("--dpi", type=int, default=125)
    p.add_argument("--fig-width", type=float, default=15.5)
    p.add_argument("--fig-height", type=float, default=7.8)
    p.add_argument("--density-bins", type=int, default=300)
    p.add_argument("--max-3d-points", type=int, default=15000)
    p.add_argument("--trail-length", type=int, default=18)
    p.add_argument("--background-stars", type=int, default=900)
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--no-plot", action="store_true", help="Run physics and CSV only; no frames/GIF.")

    # Presets.
    p.add_argument("--mode", choices=["custom", "report", "beauty", "benchmark"], default="beauty")
    p.add_argument("--seed", type=int, default=2120622)
    p.add_argument("--allow-serial", action="store_true", help="Allow running without mpi4py, useful for preview/debug.")

    # Scale GIF / feasibility mode.
    p.add_argument("--scale-gif", action="store_true", help="Generate representative weighted-density GIF for huge N without direct N-body execution.")
    p.add_argument("--scale-sample-particles", type=int, default=300000, help="Actual sampled points used to represent all particles in --scale-gif.")
    p.add_argument("--scale-frames", type=int, default=120)
    p.add_argument("--scale-gif-name", type=str, default="galaxy_scale_proxy_all_particles.gif")
    p.add_argument("--scale-overlay-points", type=int, default=14000)
    p.add_argument("--scale-only", action="store_true", help="Only write feasibility estimates; no physical simulation and no GIF.")
    p.add_argument("--calibration-rate", type=float, default=6.68e7, help="Estimated pair interactions per second for runtime extrapolation.")

    return p.parse_args()


def apply_mode_presets(args: argparse.Namespace) -> None:
    """High-level presets. CLI values can still be overridden by editing command arguments."""
    if args.mode == "report":
        args.plot_interval = 1000.0
        args.fps = min(args.fps, 10)
        args.trail_length = min(args.trail_length, 12)
        args.density_bins = min(args.density_bins, 260)
    elif args.mode == "beauty":
        # More frames and smoother GIF than the strict assignment mode.
        args.plot_interval = min(args.plot_interval, 250.0)
        args.fps = max(args.fps, 14)
        args.trail_length = max(args.trail_length, 18)
        args.density_bins = max(args.density_bins, 300)
        args.max_3d_points = max(args.max_3d_points, 15000)
        args.softening = max(args.softening, 10.0)
    elif args.mode == "benchmark":
        args.no_gif = True
        args.no_plot = True
        args.plot_interval = max(args.plot_interval, 1000.0)
        args.background_stars = min(args.background_stars, 150)
        args.dpi = min(args.dpi, 95)


def validate_args(args: argparse.Namespace) -> None:
    if args.particles <= 0:
        raise SystemExit("--particles must be positive")
    if args.years <= 0:
        raise SystemExit("--years must be positive")
    if args.dt <= 0:
        raise SystemExit("--dt must be positive")
    if args.plot_interval <= 0:
        raise SystemExit("--plot-interval must be positive")
    if args.scale_sample_particles <= 0:
        raise SystemExit("--scale-sample-particles must be positive")
    if args.scale_frames < 2:
        raise SystemExit("--scale-frames must be at least 2")


def run_scale_estimate_only(args: argparse.Namespace, rank: int, size: int, out_dir: Path) -> None:
    estimate = estimate_direct_nbody_cost(args.particles, args.years, args.dt, size, args.calibration_rate)
    if rank == 0:
        write_csv_semicolon(out_dir / "scale_estimate.csv", [estimate])
        print("=== Scale-only estimate ===")
        print(f"Particles: {args.particles:,}")
        print(f"Total interactions: {estimate['total_interactions']:.3e}")
        print(f"Estimated direct runtime: {estimate['estimated_runtime_human']}")
        print(f"CSV: {out_dir / 'scale_estimate.csv'}")


def run_scale_gif(args: argparse.Namespace, MPI: Any, comm: Any, rank: int, size: int, out_dir: Path) -> None:
    """Generate the 10^8 / 10^9 representative GIF without direct N-body execution."""
    import time

    frames_dir = out_dir / "frames_scale_proxy"
    if rank == 0:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

    counts = partition_counts(int(args.scale_sample_particles), size)
    local_n = counts[rank]

    comm.Barrier()
    start = time.perf_counter()
    local_sample = generate_scale_rank_sample(local_n, args, rank)
    local_count = int(local_sample["r"].shape[0])
    total_sample = comm.reduce(local_count, op=MPI.SUM, root=0)

    estimate = estimate_direct_nbody_cost(args.particles, args.years, args.dt, size, args.calibration_rate)
    frame_paths: List[Path] = []
    frame_records: List[Dict[str, Any]] = []

    if rank == 0:
        print("=== Scale GIF mode ===")
        print(f"Represented particles: {args.particles:,}")
        print(f"Visual sample: {args.scale_sample_particles:,} | ranks: {size} | sample counts: {counts}")
        print("This creates a weighted density GIF. It does not run direct O(N^2) N-body physics.")

    for frame_idx in range(int(args.scale_frames)):
        gathered = comm.gather(local_sample, root=0)
        if rank == 0:
            frame_path = frames_dir / f"scale_frame_{frame_idx:04d}.png"
            rec = render_scale_frame(
                gathered,
                frame_path,
                frame_idx,
                int(args.scale_frames),
                args,
                size,
                int(args.particles),
            )
            frame_records.append(rec)
            frame_paths.append(frame_path)
            print(f"Scale frame {frame_idx + 1:03d}/{args.scale_frames}")

    comm.Barrier()
    runtime_without_gif = time.perf_counter() - start

    if rank == 0:
        gif_start = time.perf_counter()
        gif_path = out_dir / args.scale_gif_name
        create_gif(frame_paths, gif_path, args.fps)
        gif_time = time.perf_counter() - gif_start
        total_runtime = runtime_without_gif + gif_time

        summary = {
            "mode": "scale_gif_weighted_density_proxy",
            "represented_particles": args.particles,
            "visual_sample_particles": total_sample,
            "mpi_ranks": size,
            "frames": args.scale_frames,
            "fps": args.fps,
            "gif_path": str(gif_path),
            "runtime_without_gif_seconds": f"{runtime_without_gif:.6f}",
            "gif_creation_seconds": f"{gif_time:.6f}",
            "total_runtime_seconds": f"{total_runtime:.6f}",
            "important_note": "GIF represents all particles by weighted density; no direct O(N^2) N-body execution was performed.",
        }
        summary.update({f"estimate_{k}": v for k, v in estimate.items()})
        write_csv_semicolon(out_dir / "runtime_summary.csv", [summary])
        write_csv_semicolon(out_dir / "frame_results.csv", frame_records)
        print(f"GIF: {gif_path}")
        print(f"CSV summary: {out_dir / 'runtime_summary.csv'}")
        print(f"CSV frames: {out_dir / 'frame_results.csv'}")
        if not args.keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)


def run_direct_physics(args: argparse.Namespace, MPI: Any, comm: Any, rank: int, size: int, out_dir: Path, serial_fallback: bool) -> None:
    """Run the real Newtonian direct N-body simulation."""
    timers = TimerBook()

    if rank == 0:
        frames_dir = out_dir / "frames_direct_physics"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        ids, pos, vel, mass = create_initial_conditions(args)
        chunks = split_particle_chunks(ids, pos, vel, mass, size)
    else:
        frames_dir = None
        chunks = None

    local = scatter_particles(comm, chunks)
    counts = comm.allgather(int(len(local["ids"])))
    total_checked = comm.reduce(int(len(local["ids"])), op=MPI.SUM, root=0)

    if rank == 0:
        print("=== Direct physical N-body mode ===")
        print(f"Particles: {args.particles:,} | MPI ranks: {size} | counts per rank: {counts}")
        print(f"reduce() particle check: {total_checked}")
        print("MPI methods: bcast, scatter, allgather, isend/irecv, gather, reduce, allreduce, Barrier")
        print("Integrator: Leapfrog / velocity-Verlet")
        if serial_fallback:
            print("Serial fallback active: mpi4py was not used.")

    steps = int(math.ceil(float(args.years) / float(args.dt)))
    plot_every_steps = max(1, int(round(float(args.plot_interval) / float(args.dt))))
    frame_count_estimate = (steps // plot_every_steps) + 1
    if steps % plot_every_steps != 0:
        frame_count_estimate += 1

    history: deque = deque(maxlen=max(1, int(args.trail_length) + 1)) if rank == 0 else deque(maxlen=1)
    background = make_background(int(args.seed), int(args.background_stars), float(args.galaxy_radius) * 1.45) if rank == 0 and not args.no_plot else None
    frame_paths: List[Path] = []
    frame_records: List[Dict[str, Any]] = []
    initial_energy = None
    frame_number = 0

    comm.Barrier()
    total_start = time.perf_counter()

    # Initial acceleration.
    acc = compute_acceleration_ring(comm, rank, size, local["pos"], local["mass"], args, timers.values)

    for step in range(steps + 1):
        current_year = min(step * float(args.dt), float(args.years))
        should_plot = (step % plot_every_steps == 0) or (step == steps)

        if should_plot:
            t_diag = time.perf_counter()
            local_diag = local_basic_diagnostics(local["pos"], local["vel"], local["mass"])
            reduced_diag = allreduce_basic_diagnostics(comm, MPI, local_diag)
            snapshot = gather_snapshot(comm, local, root=0)
            timers.add("diagnostics", time.perf_counter() - t_diag)

            if rank == 0 and snapshot is not None:
                t_energy = time.perf_counter()
                energy_diag = snapshot_energy_diagnostics(snapshot, args)
                timers.add("diagnostics", time.perf_counter() - t_energy)
                if initial_energy is None:
                    initial_energy = energy_diag["total_energy"]
                if initial_energy is not None and abs(initial_energy) > 1e-30:
                    energy_diag["energy_drift_percent"] = 100.0 * (energy_diag["total_energy"] - initial_energy) / abs(initial_energy)
                else:
                    energy_diag["energy_drift_percent"] = 0.0
                energy_diag.update(reduced_diag)

                history.append({"pos": snapshot["pos"].copy()})

                if not args.no_plot:
                    t_plot = time.perf_counter()
                    frame_path = frames_dir / f"frame_{frame_number:04d}.png"
                    render_direct_frame(
                        snapshot,
                        history,
                        frame_path,
                        current_year,
                        frame_number,
                        frame_count_estimate,
                        args,
                        {"size": size},
                        energy_diag,
                        background,
                    )
                    timers.add("plotting", time.perf_counter() - t_plot)
                    frame_paths.append(frame_path)

                rec = {
                    "frame": frame_number,
                    "step": step,
                    "year": f"{current_year:.6f}",
                    "particles": args.particles,
                    "mpi_ranks": size,
                    "kinetic": f"{energy_diag['kinetic']:.12e}",
                    "potential": f"{energy_diag['potential']:.12e}",
                    "total_energy": f"{energy_diag['total_energy']:.12e}",
                    "energy_drift_percent": f"{energy_diag['energy_drift_percent']:.8f}",
                    "virial_ratio": f"{energy_diag['virial_ratio']:.8f}",
                    "max_radius": f"{energy_diag['max_radius']:.8f}",
                    "mean_radius": f"{energy_diag['mean_radius']:.8f}",
                    "mean_speed": f"{energy_diag['mean_speed']:.12e}",
                    "angular_momentum_norm": f"{energy_diag['angular_momentum_norm']:.12e}",
                    "potential_is_estimated": int(energy_diag.get("potential_is_estimated", 0.0)),
                }
                frame_records.append(rec)
                print(f"Frame {frame_number + 1:03d} | year={current_year:,.0f} | E drift={energy_diag['energy_drift_percent']:+.3f}%")
                frame_number += 1

        if step == steps:
            break

        # Leapfrog / velocity-Verlet update.
        half_dt = 0.5 * float(args.dt)
        local["vel"] += half_dt * acc
        local["pos"] += float(args.dt) * local["vel"]
        new_acc = compute_acceleration_ring(comm, rank, size, local["pos"], local["mass"], args, timers.values)
        local["vel"] += half_dt * new_acc
        acc = new_acc

    comm.Barrier()
    total_runtime = time.perf_counter() - total_start
    timers.values["total"] = total_runtime

    if rank == 0:
        gif_path = out_dir / args.gif_name
        gif_time = 0.0
        if frame_paths and not args.no_gif and not args.no_plot:
            t_gif = time.perf_counter()
            create_gif(frame_paths, gif_path, int(args.fps))
            gif_time = time.perf_counter() - t_gif
            timers.add("gif", gif_time)
            timers.values["total"] += gif_time

        summary_extra = {
            "mpi_ranks": size,
            "particle_counts_per_rank": counts,
            "total_particles_reduce_check": total_checked,
            "steps": steps,
            "frames": frame_number,
            "actual_plot_interval_years": plot_every_steps * float(args.dt),
            "csv_delimiter": ";",
            "gif_path": str(gif_path) if frame_paths and not args.no_gif and not args.no_plot else "not_generated",
            "physics_model": "direct Newtonian all-pairs N-body with softening and central mass",
            "integrator": "Leapfrog / velocity-Verlet",
            "mpi_methods": "bcast;scatter;allgather;isend;irecv;gather;reduce;allreduce;Barrier",
        }
        write_runtime_summary(out_dir / "runtime_summary.csv", args, summary_extra, timers.as_dict())
        write_csv_semicolon(out_dir / "frame_results.csv", frame_records)
        print(f"Runtime CSV: {out_dir / 'runtime_summary.csv'}")
        print(f"Frame CSV: {out_dir / 'frame_results.csv'}")
        if frame_paths and not args.no_gif and not args.no_plot:
            print(f"GIF: {gif_path}")
        if not args.keep_frames and not args.no_plot:
            shutil.rmtree(frames_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    apply_mode_presets(args)
    validate_args(args)

    MPI, comm, serial_fallback = get_mpi(args.allow_serial)
    rank = comm.Get_rank()
    size = comm.Get_size()

    # bcast: all ranks receive exactly the same configuration.
    args = bcast_config(comm, rank, args)

    out_dir = Path(args.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.scale_only:
        run_scale_estimate_only(args, rank, size, out_dir)
        return

    if args.scale_gif:
        run_scale_gif(args, MPI, comm, rank, size, out_dir)
        return

    run_direct_physics(args, MPI, comm, rank, size, out_dir, serial_fallback)


if __name__ == "__main__":
    main()
