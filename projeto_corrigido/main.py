#!/usr/bin/env python3
"""
main.py

Entrada principal do projeto PA2 — Simulação Paralela de Galáxia.

Este ficheiro fica apenas responsável por:
- ler argumentos;
- aplicar presets;
- validar parâmetros;
- inicializar MPI;
- distribuir a configuração por bcast();
- chamar o modo de execução correto em simulation_runner.py.

Execuções recomendadas dentro de:
C:\Projetos\Pa\ProjetoPA2-Gif\projeto_corrigido>

1) GIF bonito, suave e com mais frames:
   mpiexec -n 4 python main.py --mode beauty --particles 2200 --years 12000 --dt 20 --plot-interval 200

2) Modo fiel ao enunciado, frame a cada 1000 anos:
   mpiexec -n 4 python main.py --mode report --particles 1800 --years 10000 --dt 25

3) Benchmark sem GIF, para medições de runtime:
   mpiexec -n 4 python main.py --mode benchmark --particles 2500 --years 5000 --dt 25

4) Proxy visual para escala 10^9 sem executar O(N^2):
   mpiexec -n 4 python main.py --scale-gif --particles 1000000000 --years 1000000000 --scale-sample-particles 250000 --scale-frames 140

5) Estimativa O(N^2) para escala grande:
   mpiexec -n 4 python main.py --scale-only --particles 1000000000 --years 1000000000 --dt 1000
"""

from __future__ import annotations

# Importante para testes grandes no Windows/MS-MPI:
# deve ser definido antes de qualquer import que possa carregar mpi4py.
# Ajuda a evitar "Message truncated" em irecv() com objetos Python maiores.
import os
os.environ.setdefault("MPI4PY_RC_IRECV_BUFSZ", "268435456")

import argparse
from pathlib import Path

from mpi_parallelization import bcast_config, get_mpi
from simulation_runner import (
    run_direct_physics,
    run_scale_estimate_only,
    run_scale_gif,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel galaxy simulation with mpi4py, Newtonian gravity and GIF output."
    )

    # Simulação física.
    parser.add_argument("--particles", type=int, default=2200, help="Número de partículas no modo físico direto.")
    parser.add_argument("--years", type=float, default=12000.0, help="Anos simulados.")
    parser.add_argument("--dt", type=float, default=20.0, help="Passo temporal em anos simulados.")
    parser.add_argument(
        "--plot-interval",
        type=float,
        default=200.0,
        help="Intervalo entre frames em anos simulados. No modo report é forçado para 1000.",
    )
    parser.add_argument("--g-const", type=float, default=2.15e-9)
    parser.add_argument("--central-mass", type=float, default=2.25e8)
    parser.add_argument("--disk-mass-proxy", type=float, default=6.0e7)
    parser.add_argument("--softening", type=float, default=9.0)
    parser.add_argument("--pair-block", type=int, default=384, help="Bloco vetorizado do cálculo de forças.")
    parser.add_argument("--energy-sample-limit", type=int, default=2500)

    parser.add_argument(
        "--no-center-initial-conditions",
        dest="center_initial_conditions",
        action="store_false",
        help="Desativa a correção inicial do centro de massa.",
    )
    parser.set_defaults(center_initial_conditions=True)

    parser.add_argument(
        "--recenter-every",
        type=int,
        default=0,
        help="Opcional: recentra a galáxia a cada N passos. 0 mantém apenas a correção inicial.",
    )

    # Estrutura galáctica.
    parser.add_argument("--galaxy-radius", type=float, default=125.0)
    parser.add_argument("--thickness", type=float, default=6.5)
    parser.add_argument("--spiral-arms", type=int, default=4)
    parser.add_argument("--arm-twist", type=float, default=4.9)
    parser.add_argument("--arm-width", type=float, default=0.115)
    parser.add_argument("--bar-fraction", type=float, default=0.18)
    parser.add_argument("--bulge-fraction", type=float, default=0.085)
    parser.add_argument("--halo-fraction", type=float, default=0.085)
    parser.add_argument("--halo-radius-factor", type=float, default=1.30)
    parser.add_argument("--rotation-factor", type=float, default=0.88)
    parser.add_argument("--mass-log-mean", type=float, default=6.12)
    parser.add_argument("--mass-log-sigma", type=float, default=0.38)
    parser.add_argument("--velocity-noise", type=float, default=0.0016)
    parser.add_argument("--radial-velocity-noise", type=float, default=0.0022)
    parser.add_argument("--vertical-velocity-noise", type=float, default=0.0012)
    parser.add_argument("--bulge-velocity-noise", type=float, default=0.006)
    parser.add_argument("--halo-velocity-noise", type=float, default=0.004)

    # Visualização.
    parser.add_argument("--output-dir", type=str, default="galaxy_modular_output")
    parser.add_argument("--gif-name", type=str, default="galaxy_physical_smooth.gif")
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--fig-width", type=float, default=15.8)
    parser.add_argument("--fig-height", type=float, default=7.9)
    parser.add_argument("--density-bins", type=int, default=340)
    parser.add_argument("--max-2d-points", type=int, default=48000)
    parser.add_argument("--max-3d-points", type=int, default=18000)
    parser.add_argument("--trail-length", type=int, default=24)
    parser.add_argument("--trail-particles", type=int, default=220)
    parser.add_argument("--background-stars", type=int, default=1300)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-plot", action="store_true")

    # Presets e execução.
    parser.add_argument(
        "--mode",
        choices=["custom", "beauty", "report", "benchmark"],
        default="beauty",
        help="Preset de execução.",
    )
    parser.add_argument("--seed", type=int, default=2120622)
    parser.add_argument("--allow-serial", action="store_true", help="Permite testar sem MPI real. Não usar na entrega final.")

    # Proxy/estimativa de escala para 10^8/10^9.
    parser.add_argument("--scale-gif", action="store_true")
    parser.add_argument("--scale-only", action="store_true")
    parser.add_argument("--scale-sample-particles", type=int, default=250000)
    parser.add_argument("--scale-frames", type=int, default=140)
    parser.add_argument("--scale-gif-name", type=str, default="galaxy_scale_proxy_1e9.gif")
    parser.add_argument(
        "--calibration-rate",
        type=float,
        default=6.5e7,
        help="Interações par/segundo por rank para estimativa O(N^2).",
    )

    return parser.parse_args()


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
        # Benchmark sem GIF/plot para medir a física e a comunicação.
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
        (args.recenter_every >= 0, "--recenter-every tem de ser >= 0"),
        (args.pair_block > 0, "--pair-block tem de ser positivo"),
        (args.fps > 0, "--fps tem de ser positivo"),
    ]

    for ok, message in checks:
        if not ok:
            raise SystemExit(message)


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

    run_direct_physics(
        args=args,
        MPI=MPI,
        comm=comm,
        rank=rank,
        size=size,
        out_dir=out_dir,
        serial_fallback=serial_fallback,
    )


if __name__ == "__main__":
    main()
