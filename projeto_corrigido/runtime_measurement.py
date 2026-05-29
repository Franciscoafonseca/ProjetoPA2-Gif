#!/usr/bin/env python3
"""
runtime_measurement.py

Criterion covered: Runtime measurement.

This module writes semicolon-separated CSV files and produces feasibility estimates
for huge target scales such as 10^8 and 10^9 particles.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


class TimerBook:
    """Simple named timer accumulator."""

    def __init__(self) -> None:
        self.values: Dict[str, float] = {
            "physics": 0.0,
            "communication": 0.0,
            "diagnostics": 0.0,
            "plotting": 0.0,
            "gif": 0.0,
            "total": 0.0,
        }

    def add(self, name: str, seconds: float) -> None:
        self.values[name] = self.values.get(name, 0.0) + float(seconds)

    def now(self) -> float:
        return time.perf_counter()

    def as_dict(self) -> Dict[str, float]:
        return dict(self.values)


def human_seconds(seconds: float) -> str:
    seconds = float(seconds)
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


def human_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} ZB"


def estimate_direct_nbody_cost(particles: int, years: float, dt: float, ranks: int, calibration_rate: float) -> Dict[str, Any]:
    """Estimate cost of direct all-pairs N-body without executing it."""
    n = int(particles)
    steps = int(__import__("math").ceil(float(years) / float(dt))) if dt > 0 else 0
    force_evals = max(0, 2 * steps)  # kick-drift-kick uses two acceleration stages per full step.
    interactions_per_force_eval = float(n) * float(n)
    total_interactions = interactions_per_force_eval * float(force_evals)
    estimated_seconds = total_interactions / max(float(calibration_rate), 1.0)

    bytes_per_particle = 8 + 3 * 8 + 3 * 8 + 8  # id + pos + vel + mass
    global_memory = n * bytes_per_particle
    max_local_memory = ((n + int(ranks) - 1) // max(1, int(ranks))) * bytes_per_particle
    rank0_peak = global_memory * 3.0

    return {
        "particles": n,
        "years": float(years),
        "dt": float(dt),
        "steps": steps,
        "force_evals": force_evals,
        "interactions_per_force_eval": interactions_per_force_eval,
        "total_interactions": total_interactions,
        "calibration_rate_interactions_per_second": float(calibration_rate),
        "estimated_runtime_seconds": estimated_seconds,
        "estimated_runtime_human": human_seconds(estimated_seconds),
        "bytes_per_particle_core": bytes_per_particle,
        "global_memory_core_human": human_bytes(global_memory),
        "max_local_memory_core_human": human_bytes(max_local_memory),
        "rank0_peak_memory_conservative_human": human_bytes(rank0_peak),
    }


def write_csv_semicolon(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dictionaries as a semicolon-separated CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_runtime_summary(path: Path, args, rows: Dict[str, Any], timers: Dict[str, float]) -> None:
    """Write one semicolon-separated runtime summary CSV."""
    out: Dict[str, Any] = {
        "particles": getattr(args, "particles", ""),
        "years": getattr(args, "years", ""),
        "dt": getattr(args, "dt", ""),
        "plot_interval": getattr(args, "plot_interval", ""),
        "mode": getattr(args, "mode", ""),
        "gif_name": getattr(args, "gif_name", ""),
    }
    out.update(rows)
    for key, value in timers.items():
        out[f"time_{key}_seconds"] = f"{float(value):.6f}"
    write_csv_semicolon(path, [out])
