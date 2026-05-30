#!/usr/bin/env python3
"""runtime_measurement.py — medição de runtime e ficheiros CSV."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class TimerBook:
    values: Dict[str, float] = field(default_factory=dict)

    def add(self, name: str, seconds: float) -> None:
        self.values[name] = self.values.get(name, 0.0) + float(seconds)

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.values.items()}


def _human_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.2f} min"
    hours = minutes / 60.0
    if hours < 24:
        return f"{hours:.2f} h"
    days = hours / 24.0
    if days < 365:
        return f"{days:.2f} dias"
    return f"{days / 365.25:.2e} anos"


def write_csv_semicolon(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def write_runtime_summary(path: Path, args: Any, extra: Dict[str, Any], timers: Dict[str, float]) -> None:
    total = float(timers.get("total", 0.0))
    row: Dict[str, Any] = {
        "particles": int(args.particles),
        "years": float(args.years),
        "dt": float(args.dt),
        "plot_interval": float(args.plot_interval),
        "mode": str(args.mode),
        "total_runtime_seconds": f"{total:.6f}",
        "total_runtime_human": _human_seconds(total),
    }
    for k, v in extra.items():
        row[k] = v
    for k, v in timers.items():
        row[f"timer_{k}_seconds"] = f"{float(v):.6f}"
    write_csv_semicolon(path, [row])


def estimate_direct_nbody_cost(
    particles: int,
    years: float,
    dt: float,
    ranks: int,
    pair_rate_per_second: float,
) -> Dict[str, Any]:
    """Estimativa simples para explicar por que O(N^2) não escala para 10^9."""
    particles = int(particles)
    steps = max(1, int(round(float(years) / float(dt))))
    ranks = max(1, int(ranks))
    interactions_per_step = particles * max(0, particles - 1)
    total_interactions = interactions_per_step * steps
    effective_rate = max(1.0, float(pair_rate_per_second) * ranks)
    runtime = total_interactions / effective_rate
    return {
        "particles": particles,
        "steps": steps,
        "interactions_per_step": f"{interactions_per_step:.6e}",
        "total_interactions": f"{total_interactions:.6e}",
        "assumed_pair_rate_per_rank_per_second": f"{float(pair_rate_per_second):.6e}",
        "ranks": ranks,
        "estimated_runtime_seconds": f"{runtime:.6e}",
        "estimated_runtime_human": _human_seconds(runtime),
        "note": "Estimativa para método direto O(N^2); para escala real usar Barnes-Hut/FMM/GPU/árvore distribuída.",
    }
