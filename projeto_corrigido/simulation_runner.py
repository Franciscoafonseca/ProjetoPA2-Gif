#!/usr/bin/env python3
"""
simulation_runner.py

Lógica de execução do projeto PA2 — Simulação Paralela de Galáxia.

Este ficheiro contém:
- run_scale_estimate_only(): estimativa O(N^2) para escalas grandes;
- run_scale_gif(): GIF proxy para 10^8 / 10^9 partículas;
- run_direct_physics(): simulação física direta N-body com MPI.

O main.py fica apenas responsável por argumentos, validação e inicialização.
"""

from __future__ import annotations

import math
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from gif_visualization import (
    create_gif,
    generate_scale_rank_sample,
    make_background,
    render_direct_frame,
    render_scale_frame,
)
from mpi_parallelization import (
    aggregate_timer_values,
    allreduce_basic_diagnostics,
    center_local_particles,
    compute_acceleration_ring,
    create_local_particles,
    gather_snapshot,
    partition_counts,
)
from physical_correctness import (
    local_basic_diagnostics,
    snapshot_energy_diagnostics,
)
from runtime_measurement import (
    TimerBook,
    estimate_direct_nbody_cost,
    write_csv_semicolon,
    write_runtime_summary,
)


def run_scale_estimate_only(args: Any, rank: int, size: int, out_dir: Path) -> None:
    """
    Estima o custo do método direto O(N^2) para escalas grandes.

    Este modo não executa a simulação física. Serve para justificar no relatório
    porque 10^8 ou 10^9 partículas não são viáveis com all-pairs direto.
    """

    estimate = estimate_direct_nbody_cost(
        int(args.particles),
        float(args.years),
        float(args.dt),
        int(size),
        float(args.calibration_rate),
    )

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv_semicolon(out_dir / "scale_estimate.csv", [estimate])

        print("=== Estimativa O(N^2) para escala real ===")
        print(f"Partículas: {int(args.particles):,}")
        print(f"Passos: {estimate['steps']}")
        print(f"Interações totais: {estimate['total_interactions']}")
        print(f"Runtime estimado: {estimate['estimated_runtime_human']}")
        print(f"CSV: {out_dir / 'scale_estimate.csv'}")


def run_scale_gif(args: Any, MPI: Any, comm: Any, rank: int, size: int, out_dir: Path) -> None:
    """
    Gera um GIF proxy para N muito grande sem executar N-body direto.

    Este modo é apenas visual/explicativo. Ele desenha uma amostra distribuída
    pelos ranks e guarda também a estimativa do custo O(N^2).
    """

    frames_dir = out_dir / "frames_scale_proxy"

    if rank == 0:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

    counts = partition_counts(int(args.scale_sample_particles), int(size))
    local_n = int(counts[int(rank)])

    comm.Barrier()
    start = time.perf_counter()

    local_sample = generate_scale_rank_sample(local_n, args, int(rank))
    local_count = int(local_sample["r"].shape[0])
    total_sample = comm.reduce(local_count, op=MPI.SUM, root=0)

    estimate = estimate_direct_nbody_cost(
        int(args.particles),
        float(args.years),
        float(args.dt),
        int(size),
        float(args.calibration_rate),
    )

    if rank == 0:
        print("=== Modo scale-gif / proxy visual ===")
        print(f"Partículas representadas: {int(args.particles):,}")
        print(f"Amostra desenhada: {int(total_sample):,} | ranks: {int(size)} | frames: {int(args.scale_frames)}")
        print("Nota: este modo NÃO executa N-body O(N^2); é para visualização e discussão de escala.")

    frame_paths: List[Path] = []
    frame_records: List[Dict[str, Any]] = []

    for frame_idx in range(int(args.scale_frames)):
        gathered = comm.gather(local_sample, root=0)

        if rank == 0:
            frame_path = frames_dir / f"scale_frame_{frame_idx:04d}.png"

            record = render_scale_frame(
                gathered,
                frame_path,
                int(frame_idx),
                int(args.scale_frames),
                args,
                int(size),
                int(args.particles),
            )

            frame_paths.append(frame_path)
            frame_records.append(record)

            print(f"Scale frame {frame_idx + 1:03d}/{int(args.scale_frames)}")

        # Mantém os ranks alinhados durante a geração dos frames proxy.
        comm.Barrier()

    runtime_without_gif = time.perf_counter() - start

    if rank == 0:
        gif_start = time.perf_counter()
        gif_path = out_dir / str(args.scale_gif_name)

        create_gif(frame_paths, gif_path, int(args.fps))

        gif_time = time.perf_counter() - gif_start
        total_runtime = runtime_without_gif + gif_time

        summary: Dict[str, Any] = {
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
            "important_note": "Proxy visual ponderado; não é simulação direta O(N^2).",
        }

        summary.update({f"estimate_{key}": value for key, value in estimate.items()})

        write_csv_semicolon(out_dir / "runtime_summary.csv", [summary])
        write_csv_semicolon(out_dir / "frame_results.csv", frame_records)

        print(f"GIF: {gif_path}")
        print(f"CSV runtime: {out_dir / 'runtime_summary.csv'}")

        if not bool(args.keep_frames):
            shutil.rmtree(frames_dir, ignore_errors=True)


def run_direct_physics(
    args: Any,
    MPI: Any,
    comm: Any,
    rank: int,
    size: int,
    out_dir: Path,
    serial_fallback: bool,
) -> None:
    """
    Executa a simulação física direta N-body com MPI.

    Estratégia:
    - scatter() dos IDs das partículas;
    - geração local das condições iniciais;
    - cálculo das acelerações com troca em anel isend()/irecv();
    - atualização Leapfrog / velocity-Verlet;
    - gather() periódico dos snapshots para plotting no rank 0;
    - escrita dos CSVs de runtime e frames.
    """

    timers = TimerBook()

    frames_dir = out_dir / "frames_direct_physics"
    if rank == 0:
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

    # 1. scatter() dos IDs + geração paralela local das partículas.
    t_init = time.perf_counter()
    local = create_local_particles(comm, int(rank), int(size), args)
    timers.add("initial_conditions", time.perf_counter() - t_init)

    # 2. Correção inicial do centro de massa.
    # Importante porque as condições iniciais são geradas por amostragem estatística em paralelo.
    initial_centering_info: Dict[str, float] = {}
    if bool(args.center_initial_conditions):
        t_center = time.perf_counter()
        initial_centering_info = center_local_particles(
            comm,
            MPI,
            local,
            center_velocity=True,
        )
        timers.add("recentering", time.perf_counter() - t_center)

    # 3. Diagnóstico da distribuição.
    counts = comm.allgather(int(len(local["ids"])))
    total_checked = comm.reduce(int(len(local["ids"])), op=MPI.SUM, root=0)

    if rank == 0:
        print("=== Simulação física direta N-body ===")
        print(f"Partículas: {int(args.particles):,} | ranks MPI: {int(size)} | distribuição: {counts}")
        print(f"Validação reduce(): {int(total_checked):,} partículas")

        if initial_centering_info:
            print(
                "Centro de massa inicial corrigido: "
                f"COM=({initial_centering_info['com_x']:.4e}, "
                f"{initial_centering_info['com_y']:.4e}, "
                f"{initial_centering_info['com_z']:.4e}), "
                f"|V_COM|={initial_centering_info['com_speed']:.4e}"
            )

        print("MPI: bcast, scatter, allgather, isend/irecv, gather, reduce, allreduce, Barrier")

        if serial_fallback:
            print("AVISO: fallback serial ativo; não usar isto na entrega final.")

    steps = int(math.ceil(float(args.years) / float(args.dt)))
    plot_every_steps = max(1, int(round(float(args.plot_interval) / float(args.dt))))

    frame_count_estimate = int(math.floor(steps / plot_every_steps)) + 1
    if steps % plot_every_steps != 0:
        frame_count_estimate += 1

    if rank == 0:
        print(f"Passos: {steps}")
        print(f"Plot a cada {plot_every_steps} passos")
        print(f"Frames previstos: ~{frame_count_estimate}")

    history = deque(maxlen=max(1, int(args.trail_length) + 1))

    if rank == 0:
        background = make_background(
            int(args.seed),
            int(args.background_stars),
            float(args.galaxy_radius) * 1.45,
        )
    else:
        background = None

    comm.Barrier()
    total_start = time.perf_counter()

    # Aceleração inicial.
    acc = compute_acceleration_ring(
        comm,
        int(rank),
        int(size),
        local["pos"],
        local["mass"],
        args,
        timers.values,
    )

    initial_energy: Optional[float] = None
    frame_paths: List[Path] = []
    frame_records: List[Dict[str, Any]] = []
    frame_number = 0

    for step in range(steps + 1):
        current_year = min(float(step) * float(args.dt), float(args.years))
        should_plot = (step % plot_every_steps == 0) or (step == steps)

        if should_plot:
            t_diag = time.perf_counter()

            local_diag = local_basic_diagnostics(
                local["pos"],
                local["vel"],
                local["mass"],
            )

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
                    drift = (
                        100.0
                        * (float(energy_diag["total_energy"]) - initial_energy)
                        / abs(initial_energy)
                    )
                else:
                    drift = 0.0

                energy_diag["energy_drift_percent"] = float(drift)
                energy_diag.update(reduced_diag)

                history.append({"pos": snapshot["pos"].copy()})

                if not bool(args.no_plot):
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
                        {"size": int(size)},
                        energy_diag,
                        background,
                    )

                    timers.add("plotting", time.perf_counter() - t_plot)
                    frame_paths.append(frame_path)

                frame_record = {
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

                frame_records.append(frame_record)

                print(
                    f"Frame {frame_number + 1:03d}/{frame_count_estimate} | "
                    f"ano={current_year:,.0f} | "
                    f"drift energia={energy_diag['energy_drift_percent']:+.3f}%"
                )

                frame_number += 1

        if step == steps:
            break

        # Integrador Leapfrog / velocity-Verlet:
        # v(t + dt/2), x(t + dt), a(t + dt), v(t + dt).
        half_dt = 0.5 * float(args.dt)

        local["vel"] += half_dt * acc
        local["pos"] += float(args.dt) * local["vel"]

        new_acc = compute_acceleration_ring(
            comm,
            int(rank),
            int(size),
            local["pos"],
            local["mass"],
            args,
            timers.values,
        )

        local["vel"] += half_dt * new_acc
        acc = new_acc

        # Opcional para testes longos/visuais. Mantém a galáxia centrada,
        # mas por defeito fica desligado para evitar correção artificial.
        if int(args.recenter_every) > 0 and ((step + 1) % int(args.recenter_every) == 0):
            t_center = time.perf_counter()
            center_local_particles(comm, MPI, local, center_velocity=True)
            timers.add("recentering", time.perf_counter() - t_center)

    comm.Barrier()
    total_runtime = time.perf_counter() - total_start
    timers.values["total"] = total_runtime

    gif_path = Path("not_generated")
    if rank == 0:
        if frame_paths and not bool(args.no_gif) and not bool(args.no_plot):
            gif_start = time.perf_counter()
            gif_path = out_dir / str(args.gif_name)

            create_gif(frame_paths, gif_path, int(args.fps))

            gif_seconds = time.perf_counter() - gif_start
            timers.add("gif_creation", gif_seconds)
            timers.values["total"] += gif_seconds
        else:
            gif_path = Path("not_generated")

    # Todos os ranks participam. O tempo paralelo relevante é o máximo por fase.
    timer_summary = aggregate_timer_values(comm, MPI, timers.values)

    if rank == 0:
        summary_extra: Dict[str, Any] = {
            "mpi_ranks": int(size),
            "particle_counts_per_rank": str(counts),
            "total_particles_reduce_check": int(total_checked),
            "steps": int(steps),
            "frames": int(frame_number),
            "actual_plot_interval_years": float(plot_every_steps) * float(args.dt),
            "center_initial_conditions": bool(args.center_initial_conditions),
            "initial_center_of_mass_before_correction": (
                str(initial_centering_info) if initial_centering_info else "not_applied"
            ),
            "recenter_every_steps": int(args.recenter_every),
            "gif_path": str(gif_path) if frame_paths and not bool(args.no_gif) and not bool(args.no_plot) else "not_generated",
            "physics_model": "Newtonian all-pairs N-body with gravitational softening and central mass",
            "integrator": "Leapfrog / velocity-Verlet",
            "mpi_methods": "bcast; scatter; allgather; isend; irecv; gather; reduce; allreduce; Barrier",
            "csv_delimiter": ";",
        }

        write_runtime_summary(
            out_dir / "runtime_summary.csv",
            args,
            summary_extra,
            timer_summary,
        )

        write_csv_semicolon(out_dir / "frame_results.csv", frame_records)

        print(f"Runtime CSV: {out_dir / 'runtime_summary.csv'}")
        print(f"Frame CSV: {out_dir / 'frame_results.csv'}")

        if frame_paths and not bool(args.no_gif) and not bool(args.no_plot):
            print(f"GIF: {gif_path}")

        if not bool(args.keep_frames) and not bool(args.no_plot):
            shutil.rmtree(frames_dir, ignore_errors=True)
