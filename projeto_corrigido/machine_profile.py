#!/usr/bin/env python3
"""Regista perfil da máquina para comparar resultados entre colegas."""
from __future__ import annotations

import csv
import os
import platform
import socket
import subprocess
from pathlib import Path
from datetime import datetime


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=4)
        return " ".join(out.strip().split())
    except Exception:
        return ""


def windows_cpu_name() -> str:
    # Funciona em Windows sem dependências externas; em Linux/Mac simplesmente fica vazio.
    val = _run(["wmic", "cpu", "get", "name", "/value"])
    if "Name=" in val:
        return val.split("Name=", 1)[1].strip()
    return ""


def windows_ram_gb() -> str:
    val = _run(["wmic", "computersystem", "get", "TotalPhysicalMemory", "/value"])
    if "TotalPhysicalMemory=" in val:
        try:
            b = int(val.split("TotalPhysicalMemory=", 1)[1].strip())
            return f"{b / (1024 ** 3):.2f}"
        except Exception:
            pass
    return ""


def main() -> None:
    out_dir = Path("results") / socket.gethostname()
    out_dir.mkdir(parents=True, exist_ok=True)
    cpu_name = windows_cpu_name() or platform.processor() or platform.machine()
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "machine": socket.gethostname(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "python": platform.python_version(),
        "architecture": platform.architecture()[0],
        "cpu_name": cpu_name,
        "logical_cores_os_cpu_count": os.cpu_count() or "unknown",
        "ram_gb_windows_wmic": windows_ram_gb(),
        "notes": "Preencher manualmente no relatório: CPU exato, RAM, se estava ligado à corrente, plano de energia.",
    }
    path = out_dir / "machine_profile.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter=";")
        writer.writeheader()
        writer.writerow(row)
    print(f"Perfil gravado em: {path}")
    for k, v in row.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
