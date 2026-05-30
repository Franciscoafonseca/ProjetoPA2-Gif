#!/usr/bin/env python3
"""
gif_visualization.py

Visualização da galáxia:
- frame com dois subplots obrigatórios: 2D e 3D;
- estética escura, suave, com densidade, estrelas e pequenos rastos;
- GIF com mais frames e melhor fluidez quando usado o modo beauty.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

Array = np.ndarray


def make_background(seed: int, n: int, limit: float) -> Dict[str, Array]:
    """Campo de estrelas de fundo fixo para tornar o GIF mais bonito."""
    rng = np.random.default_rng(int(seed) + 44)
    return {
        "x": rng.uniform(-limit, limit, int(n)),
        "y": rng.uniform(-limit, limit, int(n)),
        "s": rng.uniform(0.04, 0.55, int(n)),
        "a": rng.uniform(0.06, 0.30, int(n)),
    }


def _set_space_axes(ax: Any, limit: float, title: str) -> None:
    ax.set_facecolor("#030713")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_title(title, color="#eaf6ff", fontsize=12, pad=10, weight="bold")
    ax.tick_params(colors="#9bb7d4", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#18304a")
        spine.set_alpha(0.8)
    ax.set_xlabel("x", color="#a8c7e6", fontsize=9)
    ax.set_ylabel("y", color="#a8c7e6", fontsize=9)


def _sample_indices(n: int, max_points: int, seed: int) -> Array:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed) + n + max_points)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def _safe_norm(values: Array) -> Array:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    lo, hi = np.percentile(values, [2, 98])
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def render_direct_frame(
    snapshot: Dict[str, Array],
    history: Sequence[Dict[str, Array]],
    frame_path: Path,
    current_year: float,
    frame_number: int,
    frame_count_estimate: int,
    args: Any,
    mpi_info: Dict[str, Any],
    diagnostics: Dict[str, float],
    background: Dict[str, Array] | None = None,
) -> Dict[str, Any]:
    """Cria um frame PNG da simulação física direta."""
    pos = np.asarray(snapshot["pos"], dtype=np.float64)
    mass = np.asarray(snapshot["mass"], dtype=np.float64)
    n = int(pos.shape[0])
    limit = float(args.galaxy_radius) * 1.28

    frame_path = Path(frame_path)
    frame_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(float(args.fig_width), float(args.fig_height)), dpi=int(args.dpi), facecolor="#030713")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.08)
    ax2d = fig.add_subplot(grid[0, 0])
    ax3d = fig.add_subplot(grid[0, 1], projection="3d")

    _set_space_axes(ax2d, limit, "Vista 2D — densidade e braços espirais")
    ax2d.set_aspect("equal", adjustable="box")

    if background is not None:
        ax2d.scatter(background["x"], background["y"], s=background["s"], c="#b8d8ff", alpha=background["a"], linewidths=0)

    if n:
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]
        r = np.sqrt(x * x + y * y + z * z)

        # Brilho de densidade: ajuda a perceber a estrutura global.
        bins = int(args.density_bins)
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[-limit, limit], [-limit, limit]], weights=np.sqrt(mass))
        if np.max(hist) > 0:
            ax2d.imshow(
                hist.T,
                origin="lower",
                extent=[-limit, limit, -limit, limit],
                cmap="magma",
                norm=PowerNorm(gamma=0.45),
                alpha=0.82,
                interpolation="bicubic",
            )

        # Rastos curtos de algumas estrelas para suavidade visual.
        max_trails = int(getattr(args, "trail_particles", 180))
        if len(history) > 2 and max_trails > 0:
            rng = np.random.default_rng(int(args.seed) + 77)
            trail_idx = _sample_indices(n, min(max_trails, n), int(args.seed) + 90)
            for idx in trail_idx:
                pts = np.array([h["pos"][idx, :2] for h in history if h["pos"].shape[0] > idx])
                if pts.shape[0] > 1:
                    ax2d.plot(pts[:, 0], pts[:, 1], color="#7fc7ff", alpha=0.08, linewidth=0.55)

        # Estrelas por cima da densidade.
        max_points_2d = int(getattr(args, "max_2d_points", 45000))
        idx2 = _sample_indices(n, max_points_2d, int(args.seed) + frame_number)
        c2 = _safe_norm(r[idx2])
        s2 = np.clip(0.25 + 3.5 * _safe_norm(np.sqrt(mass[idx2])), 0.25, 4.2)
        ax2d.scatter(
            x[idx2],
            y[idx2],
            s=s2,
            c=c2,
            cmap="cool",
            alpha=0.72,
            linewidths=0,
        )
        # Núcleo brilhante.
        core = r < np.percentile(r, 6)
        if np.any(core):
            ax2d.scatter(x[core], y[core], s=2.4, c="#fff2b2", alpha=0.55, linewidths=0)

        # Vista 3D com subamostra.
        ax3d.set_facecolor("#030713")
        idx3 = _sample_indices(n, int(args.max_3d_points), int(args.seed) + 5 + frame_number)
        c3 = _safe_norm(r[idx3])
        s3 = np.clip(0.6 + 5.0 * _safe_norm(np.sqrt(mass[idx3])), 0.6, 6.0)
        ax3d.scatter(x[idx3], y[idx3], z[idx3], s=s3, c=c3, cmap="plasma", alpha=0.70, linewidths=0, depthshade=True)
        ax3d.set_xlim(-limit, limit)
        ax3d.set_ylim(-limit, limit)
        zlim = max(float(args.thickness) * 8.0, limit * 0.24)
        ax3d.set_zlim(-zlim, zlim)
        ax3d.view_init(elev=26, azim=38 + 0.55 * frame_number)
    else:
        ax3d.text2D(0.3, 0.5, "Sem partículas", transform=ax3d.transAxes, color="white")

    ax3d.set_title("Vista 3D — disco, espessura e halo", color="#eaf6ff", fontsize=12, pad=10, weight="bold")
    ax3d.tick_params(colors="#9bb7d4", labelsize=8)
    ax3d.set_xlabel("x", color="#a8c7e6", fontsize=9)
    ax3d.set_ylabel("y", color="#a8c7e6", fontsize=9)
    ax3d.set_zlabel("z", color="#a8c7e6", fontsize=9)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.set_facecolor((0.01, 0.03, 0.07, 0.92))
        pane.set_edgecolor("#18304a")
    ax3d.grid(True, alpha=0.16)

    progress = 100.0 * (frame_number + 1) / max(1, frame_count_estimate)
    text = (
        f"Ano simulado: {current_year:,.0f}\n"
        f"Frame: {frame_number + 1}/{frame_count_estimate}  ({progress:.0f}%)\n"
        f"Partículas: {n:,} | ranks MPI: {mpi_info.get('size', '?')}\n"
        f"Virial 2K/|U|: {diagnostics.get('virial_ratio', 0.0):.3f}\n"
        f"Drift energia: {diagnostics.get('energy_drift_percent', 0.0):+.3f}%"
    )
    fig.text(0.018, 0.965, text, color="#d8ecff", fontsize=9, va="top", family="monospace")
    fig.suptitle("Simulação Paralela de Galáxia — Newton + MPI", color="#ffffff", fontsize=15, weight="bold", y=0.995)

    fig.savefig(frame_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    return {
        "frame": int(frame_number),
        "year": float(current_year),
        "path": str(frame_path),
        "particles": int(n),
        "max_radius": float(diagnostics.get("max_radius", 0.0)),
        "energy_drift_percent": float(diagnostics.get("energy_drift_percent", 0.0)),
    }


def create_gif(frame_paths: Iterable[Path], gif_path: Path, fps: int = 16) -> None:
    """Combina PNGs num GIF animado."""
    frame_paths = [Path(p) for p in frame_paths]
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    if not frame_paths:
        raise ValueError("Não há frames para criar o GIF.")

    duration = 1.0 / max(1, int(fps))
    try:
        import imageio.v2 as imageio

        images = [imageio.imread(str(p)) for p in frame_paths]
        imageio.mimsave(str(gif_path), images, duration=duration, loop=0)
    except Exception:
        from PIL import Image

        images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 * duration),
            loop=0,
            optimize=True,
        )


def generate_scale_rank_sample(local_n: int, args: Any, rank: int) -> Dict[str, Array]:
    """
    Amostra visual para representar N muito grande sem executar O(N^2).

    Útil para discutir a escala 10^8/10^9 no relatório: o GIF é uma proxy visual
    ponderada, não uma simulação direta.
    """
    rng = np.random.default_rng(int(args.seed) + 5000 + int(rank))
    local_n = int(local_n)
    radius = float(args.galaxy_radius)
    arms = max(1, int(args.spiral_arms))

    r = rng.gamma(shape=2.1, scale=radius / 6.0, size=local_n)
    r = np.clip(r, 0.2, radius)
    arm = rng.integers(0, arms, size=local_n)
    theta = 2.0 * np.pi * arm / arms + float(args.arm_twist) * np.log1p(r / (radius * 0.08))
    theta += rng.normal(0.0, float(args.arm_width) * (0.6 + r / radius), size=local_n)
    z = rng.normal(0.0, float(args.thickness) * (0.2 + 0.8 * r / radius), size=local_n)
    mass = rng.lognormal(float(args.mass_log_mean), float(args.mass_log_sigma), size=local_n)
    return {"r": r.astype(np.float64), "theta": theta.astype(np.float64), "z": z.astype(np.float64), "mass": mass.astype(np.float64)}


def render_scale_frame(
    gathered: List[Dict[str, Array]],
    frame_path: Path,
    frame_idx: int,
    total_frames: int,
    args: Any,
    size: int,
    represented_particles: int,
) -> Dict[str, Any]:
    """Frame visual proxy para a escala 10^8/10^9."""
    r = np.concatenate([g["r"] for g in gathered])
    theta0 = np.concatenate([g["theta"] for g in gathered])
    z = np.concatenate([g["z"] for g in gathered])
    mass = np.concatenate([g["mass"] for g in gathered])

    t = frame_idx / max(1, total_frames - 1)
    # Rotação diferencial: núcleo roda mais depressa que periferia.
    omega = 2.2 * np.exp(-r / (float(args.galaxy_radius) * 0.75)) + 0.16
    theta = theta0 + 2.0 * np.pi * t * omega
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    fake_snapshot = {
        "ids": np.arange(r.size, dtype=np.int64),
        "pos": np.column_stack([x, y, z]),
        "vel": np.zeros((r.size, 3), dtype=np.float64),
        "mass": mass,
    }
    diag = {
        "virial_ratio": 0.0,
        "energy_drift_percent": 0.0,
        "max_radius": float(np.max(r)) if r.size else 0.0,
    }
    bg = make_background(int(args.seed), int(args.background_stars), float(args.galaxy_radius) * 1.45)
    rec = render_direct_frame(
        fake_snapshot,
        [],
        frame_path,
        t * float(args.years),
        frame_idx,
        total_frames,
        args,
        {"size": size},
        diag,
        bg,
    )
    rec["represented_particles"] = int(represented_particles)
    rec["visual_sample_particles"] = int(r.size)
    rec["mode"] = "scale_proxy"
    return rec
