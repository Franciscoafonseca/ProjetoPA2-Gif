#!/usr/bin/env python3
"""
main.py

Entrada principal do projeto PA2 — Simulação Paralela de Galáxia.

Execuções recomendadas dentro de:
C:\\Projetos\\Pa\\ProjetoPA2-Gif\\projeto_corrigido>

1) GIF bonito, suave e com mais frames:
   mpiexec -n 4 python main.py --mode beauty --particles 2200 --years 12000 --dt 20 --plot-interval 200

2) Modo fiel ao enunciado, frame a cada 1000 anos:
   mpiexec -n 4 python main.py --mode report --particles 1800 --years 10000 --dt 25 --plot-interval 1000

3) Benchmark sem GIF, para medições de runtime:
   mpiexec -n 4 python main.py --mode benchmark --particles 2500 --years 5000 --dt 25

4) Proxy visual para escala 10^9 sem executar O(N^2):
   mpiexec -n 4 python main.py --scale-gif --particles 1000000000 --years 1000000000 --scale-sample-particles 250000 --scale-frames 140
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

from gif_visualization import (
    create_gif,
    generate_scale_rank_sample,
    make_background,
    render_direct_frame,
    render_scale_frame,
)
from mpi_parallelization import (
    allreduce_basic_diagnostics,
    bcast_config,
    compute_acceleration_ring,
    create_local_particles,
    get_mpi,
    partition_counts,
    gather_snapshot,
)
from physical_correctness import local_basic_diagnostics, snapshot_energy_diagnostics
from runtime_measurement import (
    TimerBook,
    estimate_direct_nbody_cost,
    write_csv_semicolon,
    write_runtime_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel galaxy simulation with mpi4py, Newtonian gravity and GIF output.")

    # Simulação física.
    p.add_argument("--particles", type=int, default=2200, help="Número de partículas no modo físico direto.")
    p.add_argument("--years", type=float, default=12000.0, help="Anos simulados.")
    p.add_argument("--dt", type=float, default=20.0, help="Passo temporal em anos simulados.")
    p.add_argument("--plot-interval", type=float, default=200.0, help="Intervalo entre frames. Usa 1000 para modo enunciado.")
    p.add_argument("--g-const", type=float, default=2.15e-9)
    p.add_argument("--central-mass", type=float, default=2.25e8)
    p.add_argument("--disk-mass-proxy", type=float, default=6.0e7)
    p.add_argument("--softening", type=float, default=9.0)
    p.add_argument("--pair-block", type=int, default=384, help="Bloco vetorizado do cálculo de forças.")
    p.add_argument("--energy-sample-limit", type=int, default=2500)

    # Estrutura galáctica.
    p.add_argument("--galaxy-radius", type=float, default=125.0)
    p.add_argument("--thickness", type=float, default=6.5)
    p.add_argument("--spiral-arms", type=int, default=4)
    p.add_argument("--arm-twist", type=float, default=4.9)
    p.add_argument("--arm-width", type=float, default=0.115)
    p.add_argument("--bar-fraction", type=float, default=0.18)
    p.add_argument("--bulge-fraction", type=float, default=0.085)
    p.add_argument("--halo-fraction", type=float, default=0.085)
    p.add_argument("--halo-radius-factor", type=float, default=1.30)
    p.add_argument("--rotation-factor", type=float, default=0.88)
    p.add_argument("--mass-log-mean", type=float, default=6.12)
    p.add_argument("--mass-log-sigma", type=float, default=0.38)
    p.add_argument("--velocity-noise", type=float, default=0.0016)
    p.add_argument("--radial-velocity-noise", type=float, default=0.0022)
    p.add_argument("--vertical-velocity-noise", type=float, default=0.0012)
    p.add_argument("--bulge-velocity-noise", type=float, default=0.006)
    p.add_argument("--halo-velocity-noise", type=float, default=0.004)

    # Visualização.
    p.add_argument("--output-dir", type=str, default="galaxy_modular_output")
    p.add_argument("--gif-name", type=str, default="galaxy_physical_smooth.gif")
    p.add_argument("--fps", type=int, default=18)
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--fig-width", type=float, default=15.8)
    p.add_argument("--fig-height", type=float, default=7.9)
    p.add_argument("--density-bins", type=int, default=340)
    p.add_argument("--max-2d-points", type=int, default=48000)
    p.add_argument("--max-3d-points", type=int, default=18000)
    p.add_argument("--trail-length", type=int, default=24)
    p.add_argument("--trail-particles", type=int, default=220)
    p.add_argument("--background-stars", type=int, default=1300)
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--no-plot", action="store_true")

    # Presets e execução.
    p.add_argument("--mode", choices=["custom", "beauty", "report", "benchmark"], default="beauty")
    p.add_argument("--seed", type=int, default=2120622)
    p.add_argument("--allow-serial", action="store_true", help="Permite testar sem MPI real. Não usar na entrega final.")

    # Proxy de escala para 10^8/10^9.
    p.add_argument("--scale-gif", action="store_true")
    p.add_argument("--scale-only", action="store_true")
    p.add_argument("--scale-sample-particles", type=int, default=250000)
    p.add_argument("--scale-frames", type=int, default=140)
    p.add_argument("--scale-gif-name", type=str, default="galaxy_scale_proxy_1e9.gif")
    p.add_argument("--calibration-rate", type=float, default=6.5e7, help="Interações par/segundo por rank para estimativa O(N^2).")

    return p.parse_args()


def apply_mode_presets(args: argparse.Namespace) -> None:
    """Presets para não ter de escrever muitos parâmetros nos testes."""
    if args.mode == "beauty":
        # Mais frames e visual mais suave que o enunciado estrito.
        args.plot_interval = min(float(args.plot_interval), 200.0)
        args.dt = min(float(args.dt), 20.0)
        args.fps = max(int(args.fps), 18)
        args.trail_length = max(int(args.trail_length), 24)
        args.trail_particles = max(int(args.trail_particles), 220)
        args.density_bins = max(int(args.density_bins), 340)
        args.max_3d_points = max(int(args.max_3d_points), 18000)
        args.arm_width = min(float(args.arm_width), 0.12)
        args.softening = max(float(args.softening), 9.0)
    elif args.mode == "report":
        # Fiel à frase do enunciado: plot de 1000 em 1000 anos.
        args.plot_interval = 1000.0
        args.fps = min(int(args.fps), 12)
        args.trail_length = min(int(args.trail_length), 14)
        args.density_bins = min(int(args.density_bins), 280)
    elif args.mode == "benchmark":
        args.no_plot = True
        args.no_gif = True
        args.plot_interval = max(float(args.plot_interval), 1000.0)
        args.density_bins = min(int(args.density_bins), 160)
        args.background_stars = min(int(args.background_stars), 150)


def validate_args(args: argparse.Namespace) -> None:
    checks = [
        (args.particles > 0, "--particles tem de ser positivo"),
        (args.years > 0, "--years tem de ser positivo"),
        (args.dt > 0, "--dt tem de ser positivo"),
        (args.plot_interval > 0, "--plot-interval tem de ser positivo"),
        (args.spiral_arms >= 1, "--spiral-arms tem de ser >= 1"),
        (args.scale_sample_particles > 0, "--scale-sample-particles tem de ser positivo"),
        (args.scale_frames >= 2, "--scale-frames tem de ser >= 2"),
    ]
    for ok, msg in checks:
        if not ok:
            raise SystemExit(msg)


def run_scale_estimate_only(args: argparse.Namespace, rank: int, size: int, out_dir: Path) -> None:
    estimate = estimate_direct_nbody_cost(args.particles, args.years, args.dt, size, args.calibration_rate)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv_semicolon(out_dir / "scale_estimate.csv", [estimate])
        print("=== Estimativa O(N^2) para escala real ===")
        print(f"Partículas: {args.particles:,}")
        print(f"Passos: {estimate['steps']}")
        print(f"Interações totais: {estimate['total_interactions']}")
        print(f"Runtime estimado: {estimate['estimated_runtime_human']}")
        print(f"CSV: {out_dir / 'scale_estimate.csv'}")


def run_scale_gif(args: argparse.Namespace, MPI: Any, comm: Any, rank: int, size: int, out_dir: Path) -> None:
    """GIF proxy para N enorme sem executar N-body direto."""
    frames_dir = out_dir / "frames_scale_proxy"
    if rank == 0:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

    counts = partition_counts(int(args.scale_sample_particles), int(size))
    local_n = counts[int(rank)]
    comm.Barrier()
    start = time.perf_counter()

    local_sample = generate_scale_rank_sample(local_n, args, rank)
    local_count = int(local_sample["r"].shape[0])
    total_sample = comm.reduce(local_count, op=MPI.SUM, root=0)
    estimate = estimate_direct_nbody_cost(args.particles, args.years, args.dt, size, args.calibration_rate)

    if rank == 0:
        print("=== Modo scale-gif / proxy visual ===")
        print(f"Partículas representadas: {args.particles:,}")
        print(f"Amostra desenhada: {int(total_sample):,} | ranks: {size} | frames: {args.scale_frames}")
        print("Nota: este modo NÃO executa N-body O(N^2); é para visualização e discussão de escala.")

    frame_paths: List[Path] = []
    frame_records: List[Dict[str, Any]] = []
    for frame_idx in range(int(args.scale_frames)):
        gathered = comm.gather(local_sample, root=0)
        if rank == 0:
            frame_path = frames_dir / f"scale_frame_{frame_idx:04d}.png"
            rec = render_scale_frame(gathered, frame_path, frame_idx, int(args.scale_frames), args, size, int(args.particles))
            frame_paths.append(frame_path)
            frame_records.append(rec)
            print(f"Scale frame {frame_idx + 1:03d}/{args.scale_frames}")
        comm.Barrier()

    runtime_without_gif = time.perf_counter() - start
    if rank == 0:
        gif_start = time.perf_counter()
        gif_path = out_dir / args.scale_gif_name
        create_gif(frame_paths, gif_path, int(args.fps))
        gif_time = time.perf_counter() - gif_start
        total_runtime = runtime_without_gif + gif_time

        summary = {
            "mode": "scale_gif_weighted_density_proxy",
            "represented_particles": int(args.particles),
            "visual_sample_particles": int(total_sample),
            "mpi_ranks": int(size),
            "frames": int(args.scale_frames),
            "fps": int(args.fps),
            "gif_path": str(gif_path),
            "runtime_without_gif_seconds": f"{runtime_without_gif:.6f}",
            "gif_creation_seconds": f"{gif_time:.6f}",
            "total_runtime_seconds": f"{total_runtime:.6f}",
            "important_note": "Proxy visual ponderada; não é simulação direta O(N^2).",
        }
        summary.update({f"estimate_{k}": v for k, v in estimate.items()})
        write_csv_semicolon(out_dir / "runtime_summary.csv", [summary])
        write_csv_semicolon(out_dir / "frame_results.csv", frame_records)
        print(f"GIF: {gif_path}")
        print(f"CSV runtime: {out_dir / 'runtime_summary.csv'}")
        if not args.keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)


def run_direct_physics(
    args: argparse.Namespace,
    MPI: Any,
    comm: Any,
    rank: int,
    size: int,
    out_dir: Path,
    serial_fallback: bool,
) -> None:
    """Executa a simulação N-body direta com MPI."""
    timers = TimerBook()

    frames_dir = out_dir / "frames_direct_physics"
    if rank == 0:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

    # scatter() dos IDs + geração paralela local das partículas.
    t_init = time.perf_counter()
    local = create_local_particles(comm, rank, size, args)
    timers.add("initial_conditions", time.perf_counter() - t_init)

    counts = comm.allgather(int(len(local["ids"])))
    total_checked = comm.reduce(int(len(local["ids"])), op=MPI.SUM, root=0)

    if rank == 0:
        print("=== Simulação física direta N-body ===")
        print(f"Partículas: {args.particles:,} | ranks MPI: {size} | distribuição: {counts}")
        print(f"Validação reduce(): {total_checked:,} partículas")
        print("MPI: bcast, scatter, allgather, isend/irecv, gather, reduce, allreduce, Barrier")
        print(f"Frames previstos: ~{math.floor(args.years / args.plot_interval) + 1}")
        if serial_fallback:
            print("AVISO: fallback serial ativo; não usar isto na entrega final.")

    steps = int(math.ceil(float(args.years) / float(args.dt)))
    plot_every_steps = max(1, int(round(float(args.plot_interval) / float(args.dt))))
    frame_count_estimate = int(math.floor(steps / plot_every_steps)) + 1
    if steps % plot_every_steps != 0:
        frame_count_estimate += 1

    history: deque = deque(maxlen=max(1, int(args.trail_length) + 1))
    background = make_background(int(args.seed), int(args.background_stars), float(args.galaxy_radius) * 1.45) if rank == 0 else None

    comm.Barrier()
    total_start = time.perf_counter()

    acc = compute_acceleration_ring(comm, rank, size, local["pos"], local["mass"], args, timers.values)
    initial_energy: float | None = None
    frame_paths: List[Path] = []
    frame_records: List[Dict[str, Any]] = []
    frame_number = 0

    for step in range(steps + 1):
        current_year = min(float(step) * float(args.dt), float(args.years))
        should_plot = (step % plot_every_steps == 0) or (step == steps)

        if should_plot:
            t_diag = time.perf_counter()
            local_diag = local_basic_diagnostics(local["pos"], local["vel"], local["mass"])
            reduced_diag = allreduce_basic_diagnostics(comm, MPI, local_diag)
            snapshot = gather_snapshot(comm, local, root=0)
            timers.add("diagnostics_and_gather", time.perf_counter() - t_diag)

            if rank == 0 and snapshot is not None:
                energy_start = time.perf_counter()
                energy_diag = snapshot_energy_diagnostics(snapshot, args)
                timers.add("energy_diagnostics", time.perf_counter() - energy_start)

                if initial_energy is None:
                    initial_energy = float(energy_diag["total_energy"])
                if initial_energy is not None and abs(initial_energy) > 1e-30:
                    drift = 100.0 * (float(energy_diag["total_energy"]) - initial_energy) / abs(initial_energy)
                else:
                    drift = 0.0
                energy_diag["energy_drift_percent"] = float(drift)
                energy_diag.update(reduced_diag)

                history.append({"pos": snapshot["pos"].copy()})

                if not args.no_plot:
                    t_plot = time.perf_counter()
                    frame_path = frames_dir / f"frame_{frame_number:04d}.png"
                    render_direct_frame(
                        snapshot,
                        list(history),
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
                    "frame": int(frame_number),
                    "step": int(step),
                    "year": f"{current_year:.6f}",
                    "particles": int(args.particles),
                    "mpi_ranks": int(size),
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
                print(
                    f"Frame {frame_number + 1:03d}/{frame_count_estimate} | "
                    f"ano={current_year:,.0f} | drift energia={energy_diag['energy_drift_percent']:+.3f}%"
                )
                frame_number += 1

        if step == steps:
            break

        # Integrador Leapfrog / velocity-Verlet:
        # v(t+dt/2), x(t+dt), a(t+dt), v(t+dt)
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
        if frame_paths and not args.no_gif and not args.no_plot:
            t_gif = time.perf_counter()
            create_gif(frame_paths, gif_path, int(args.fps))
            gif_seconds = time.perf_counter() - t_gif
            timers.add("gif_creation", gif_seconds)
            timers.values["total"] += gif_seconds
        else:
            gif_path = Path("not_generated")

        summary_extra = {
            "mpi_ranks": int(size),
            "particle_counts_per_rank": str(counts),
            "total_particles_reduce_check": int(total_checked),
            "steps": int(steps),
            "frames": int(frame_number),
            "actual_plot_interval_years": float(plot_every_steps) * float(args.dt),
            "gif_path": str(out_dir / args.gif_name) if frame_paths and not args.no_gif and not args.no_plot else "not_generated",
            "physics_model": "Newtonian all-pairs N-body with gravitational softening and central mass",
            "integrator": "Leapfrog / velocity-Verlet",
            "mpi_methods": "bcast; scatter; allgather; isend; irecv; gather; reduce; allreduce; Barrier",
            "csv_delimiter": ";",
        }
        write_runtime_summary(out_dir / "runtime_summary.csv", args, summary_extra, timers.as_dict())
        write_csv_semicolon(out_dir / "frame_results.csv", frame_records)

        print(f"Runtime CSV: {out_dir / 'runtime_summary.csv'}")
        print(f"Frame CSV: {out_dir / 'frame_results.csv'}")
        if frame_paths and not args.no_gif and not args.no_plot:
            print(f"GIF: {out_dir / args.gif_name}")
        if not args.keep_frames and not args.no_plot:
            shutil.rmtree(frames_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    apply_mode_presets(args)
    validate_args(args)

    MPI, comm, serial_fallback = get_mpi(args.allow_serial)
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())

    # bcast() garante que todos os processos usam exatamente os mesmos parâmetros.
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
