#!/usr/bin/env python3
"""Agrega runtime_summary.csv de todos os PCs e calcula speedup/eficiência."""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_semicolon_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=";"))


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default


def to_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(str(x).replace(",", ".")))
    except Exception:
        return default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
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


def infer_machine(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        i = parts.index("results")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def infer_test_id(path: Path) -> str:
    # .../results/MACHINE/test_id/runtime_summary.csv
    try:
        return path.parent.name
    except Exception:
        return "unknown"


def main() -> None:
    root = Path("results")
    rows: list[dict[str, Any]] = []
    for path in root.rglob("runtime_summary.csv"):
        for r in read_semicolon_csv(path):
            r = dict(r)
            r["machine"] = infer_machine(path)
            r["test_id"] = infer_test_id(path)
            rows.append(r)

    if not rows:
        raise SystemExit("Não encontrei results/**/runtime_summary.csv")

    write_csv(root / "combined_runtime_results.csv", rows)

    # Agrega repetições por máquina, N, P, anos, dt, modo.
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (
            r.get("machine", "unknown"),
            to_int(r.get("particles")),
            to_int(r.get("mpi_ranks")),
            to_float(r.get("years")),
            to_float(r.get("dt")),
            r.get("mode", ""),
        )
        groups[key].append(r)

    agg_rows: list[dict[str, Any]] = []
    for (machine, n, p, years, dt, mode), vals in sorted(groups.items()):
        total_times = [to_float(v.get("total_runtime_seconds")) for v in vals]
        physics_times = [to_float(v.get("timer_physics_seconds")) for v in vals]
        comm_times = [to_float(v.get("timer_communication_seconds")) for v in vals]
        gather_times = [to_float(v.get("timer_diagnostics_and_gather_seconds")) for v in vals]
        plot_times = [to_float(v.get("timer_plotting_seconds")) for v in vals]
        gif_times = [to_float(v.get("timer_gif_creation_seconds")) for v in vals]
        steps = to_int(vals[0].get("steps"))
        frames = to_int(vals[0].get("frames"))
        mean_total = statistics.mean(total_times)
        agg_rows.append(
            {
                "machine": machine,
                "particles": n,
                "mpi_ranks": p,
                "years": years,
                "dt": dt,
                "mode": mode,
                "repeats": len(vals),
                "steps": steps,
                "frames": frames,
                "mean_total_seconds": f"{mean_total:.6f}",
                "min_total_seconds": f"{min(total_times):.6f}",
                "max_total_seconds": f"{max(total_times):.6f}",
                "mean_physics_seconds": f"{statistics.mean(physics_times):.6f}",
                "mean_communication_seconds": f"{statistics.mean(comm_times):.6f}",
                "mean_gather_seconds": f"{statistics.mean(gather_times):.6f}",
                "mean_plotting_seconds": f"{statistics.mean(plot_times):.6f}",
                "mean_gif_seconds": f"{statistics.mean(gif_times):.6f}",
                "physics_percent_total": f"{100 * statistics.mean(physics_times) / mean_total if mean_total else 0:.2f}",
                "communication_percent_total": f"{100 * statistics.mean(comm_times) / mean_total if mean_total else 0:.2f}",
                "plotting_percent_total": f"{100 * statistics.mean(plot_times) / mean_total if mean_total else 0:.2f}",
            }
        )

    # Speedup por máquina e N, usando P=1 como baseline do mesmo modo.
    baseline: dict[tuple, float] = {}
    for r in agg_rows:
        if to_int(r["mpi_ranks"]) == 1:
            baseline[(r["machine"], r["particles"], r["years"], r["dt"], r["mode"])] = to_float(r["mean_total_seconds"])

    for r in agg_rows:
        key = (r["machine"], r["particles"], r["years"], r["dt"], r["mode"])
        base = baseline.get(key, 0.0)
        p = to_int(r["mpi_ranks"])
        total = to_float(r["mean_total_seconds"])
        speedup = base / total if base and total else 0.0
        efficiency = speedup / p if p else 0.0
        r["speedup_vs_1_rank_same_machine"] = f"{speedup:.4f}"
        r["efficiency_vs_1_rank_same_machine_percent"] = f"{100 * efficiency:.2f}"

    write_csv(root / "aggregated_scaling_results.csv", agg_rows)

    # Markdown para colar no relatório.
    md_lines = [
        "# Tabelas para relatório",
        "",
        "## Resultados agregados por máquina, partículas e processos",
        "",
        "| Máquina | N | Processos | Tempo médio (s) | Speedup | Eficiência | Física % | Comunicação % | Plotting % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in agg_rows:
        if r.get("mode") != "benchmark":
            continue
        md_lines.append(
            f"| {r['machine']} | {r['particles']} | {r['mpi_ranks']} | {r['mean_total_seconds']} | "
            f"{r['speedup_vs_1_rank_same_machine']} | {r['efficiency_vs_1_rank_same_machine_percent']}% | "
            f"{r['physics_percent_total']}% | {r['communication_percent_total']}% | {r['plotting_percent_total']}% |"
        )
    (root / "report_ready_tables.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Escrevi: {root / 'combined_runtime_results.csv'}")
    print(f"Escrevi: {root / 'aggregated_scaling_results.csv'}")
    print(f"Escrevi: {root / 'report_ready_tables.md'}")


if __name__ == "__main__":
    main()
