#!/usr/bin/env python3
"""
Corre uma matriz de benchmarks MPI para comparar PCs diferentes.
Executar a partir da pasta projeto_corrigido, onde está main.py.
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def write_manifest(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Run PA galaxy MPI benchmark matrix.")
    p.add_argument("--particles", default="500,1000,1500,2000", help="Lista CSV de partículas.")
    p.add_argument("--ranks", default="1,2,4", help="Lista CSV de processos MPI.")
    p.add_argument("--repeats", type=int, default=2, help="Repetições por teste. Usa 2 ou 3 para relatório final.")
    p.add_argument("--years", type=float, default=1000.0)
    p.add_argument("--dt", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=2120622)
    p.add_argument("--mpiexec", default="mpiexec")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--results-root", default="results")
    p.add_argument("--with-plot-test", action="store_true", help="Também mede overhead de plotting, sem gerar GIF.")
    p.add_argument("--plot-particles", type=int, default=1000)
    p.add_argument("--plot-rank", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    particles = parse_int_list(args.particles)
    ranks = parse_int_list(args.ranks)
    machine = socket.gethostname()
    results_root = Path(args.results_root) / machine
    results_root.mkdir(parents=True, exist_ok=True)

    # Evita oversubscription interna de BLAS/OpenMP dentro de cada processo MPI.
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    manifest: list[dict] = []
    total_jobs = len(particles) * len(ranks) * int(args.repeats)
    job_idx = 0

    def run_job(test_kind: str, n: int, rank_count: int, rep: int, mode: str, extra_flags: list[str]) -> None:
        nonlocal job_idx
        job_idx += 1
        test_id = f"{test_kind}_N{n}_P{rank_count}_R{rep}"
        out_dir = results_root / test_id
        cmd = [
            args.mpiexec,
            "-n",
            str(rank_count),
            args.python,
            "main.py",
            "--mode",
            mode,
            "--particles",
            str(n),
            "--years",
            str(args.years),
            "--dt",
            str(args.dt),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(out_dir),
        ] + extra_flags
        print("\n" + "=" * 80)
        print(f"[{job_idx}/{total_jobs}] {test_id}")
        print(" ".join(cmd))
        t0 = time.perf_counter()
        status = "dry_run"
        returncode = 0
        if not args.dry_run:
            proc = subprocess.run(cmd, env=env)
            returncode = int(proc.returncode)
            status = "ok" if returncode == 0 else "failed"
        wall = time.perf_counter() - t0
        manifest.append(
            {
                "machine": machine,
                "test_id": test_id,
                "test_kind": test_kind,
                "particles": n,
                "mpi_ranks": rank_count,
                "repeat": rep,
                "years": args.years,
                "dt": args.dt,
                "mode": mode,
                "status": status,
                "returncode": returncode,
                "external_wall_seconds": f"{wall:.6f}",
                "output_dir": str(out_dir),
                "python": platform.python_version(),
                "os_cpu_count": os.cpu_count(),
            }
        )
        write_manifest(results_root / "experiment_manifest.csv", manifest)
        if returncode != 0:
            raise SystemExit(f"Teste falhou: {test_id}")

    for n in particles:
        for rank_count in ranks:
            for rep in range(1, int(args.repeats) + 1):
                run_job("benchmark", n, rank_count, rep, "benchmark", [])

    if args.with_plot_test:
        # Mede gathering + plotting, mas sem GIF, para comparar contra o benchmark puro.
        for rep in range(1, int(args.repeats) + 1):
            run_job(
                "plot_no_gif",
                int(args.plot_particles),
                int(args.plot_rank),
                rep,
                "custom",
                ["--plot-interval", "1000", "--no-gif"],
            )

    print("\nConcluído.")
    print(f"Manifest: {results_root / 'experiment_manifest.csv'}")
    print("Depois corre: python merge_results.py")


if __name__ == "__main__":
    main()
