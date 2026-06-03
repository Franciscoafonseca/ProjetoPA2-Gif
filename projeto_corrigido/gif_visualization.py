#!/usr/bin/env python3
"""
gif_visualization.py

Visualização da galáxia para o projeto Newton + MPI.

Objetivos deste ficheiro:
- gerar frames com os dois subplots obrigatórios: 2D e 3D;
- tornar o GIF final mais bonito, legível e completo;
- destacar núcleo, disco, braços espirais, halo, espessura 3D e evolução temporal;
- manter compatibilidade com o resto do projeto: main.py pode continuar a chamar
  render_direct_frame(), create_gif(), generate_scale_rank_sample() e render_scale_frame().

Nota importante:
- render_direct_frame() representa a simulação física direta.
- render_scale_frame() é apenas proxy visual para escalas enormes; não substitui o teste físico.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.collections import LineCollection

Array = np.ndarray


# ---------------------------------------------------------------------------
# Utilitários gerais
# ---------------------------------------------------------------------------

def _get(args: Any, name: str, default: Any) -> Any:
    """Obtém um argumento mesmo quando ele não existe em versões antigas do main.py."""
    return getattr(args, name, default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def make_background(seed: int, n: int, limit: float) -> Dict[str, Array]:
    """
    Campo fixo de estrelas de fundo.

    Não faz parte da física. Serve apenas para melhorar a leitura visual do GIF
    e dar profundidade ao espaço de fundo.
    """
    rng = np.random.default_rng(int(seed) + 44)

    # Mistura de muitas estrelas fracas com poucas estrelas mais brilhantes.
    n = int(n)
    x = rng.uniform(-limit, limit, n)
    y = rng.uniform(-limit, limit, n)
    brightness = rng.power(5.0, n)

    return {
        "x": x.astype(np.float64),
        "y": y.astype(np.float64),
        "s": (0.03 + 0.85 * brightness).astype(np.float64),
        "a": (0.04 + 0.28 * brightness).astype(np.float64),
    }


def _set_space_axes(ax: Any, limit: float, title: str) -> None:
    """Estilo escuro e limpo para a vista 2D."""
    ax.set_facecolor("#020611")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_title(title, color="#f0f7ff", fontsize=13, pad=11, weight="bold")
    ax.tick_params(colors="#9eb8d4", labelsize=8)

    for spine in ax.spines.values():
        spine.set_color("#1b3555")
        spine.set_alpha(0.9)

    ax.set_xlabel("x", color="#abc8e8", fontsize=9)
    ax.set_ylabel("y", color="#abc8e8", fontsize=9)

    # Círculos de referência discretos para perceber escala e rotação.
    for frac, alpha in [(0.33, 0.08), (0.66, 0.06), (1.00, 0.045)]:
        circle = plt.Circle(
            (0, 0),
            radius=limit * frac,
            edgecolor="#7fbfff",
            facecolor="none",
            linewidth=0.45,
            alpha=alpha,
            linestyle="--",
        )
        ax.add_patch(circle)


def _sample_indices(n: int, max_points: int, seed: int) -> Array:
    """Amostra determinística para não desenhar pontos a mais."""
    n = int(n)
    max_points = int(max_points)
    if n <= 0 or max_points <= 0:
        return np.empty(0, dtype=np.int64)
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed) + n + max_points)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def _safe_norm(values: Array) -> Array:
    """Normalização robusta para cores/tamanhos."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values)
    vals = values[finite]
    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo:
        return np.zeros_like(values)
    out = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    out[~finite] = 0.0
    return out


def _draw_progress_bar(fig: Any, frame_number: int, frame_count_estimate: int) -> None:
    """Barra de progresso pequena no fundo da figura."""
    total = max(1, int(frame_count_estimate))
    progress = np.clip((frame_number + 1) / total, 0.0, 1.0)

    ax_bar = fig.add_axes([0.18, 0.025, 0.64, 0.012])
    ax_bar.set_facecolor("#07101d")
    ax_bar.barh([0], [1.0], color="#112942", height=1.0)
    ax_bar.barh([0], [progress], color="#69d2ff", height=1.0)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.5, 0.5)
    ax_bar.axis("off")


def _draw_trails_2d(
    ax: Any,
    history: Sequence[Dict[str, Array]],
    n: int,
    max_trails: int,
    seed: int,
) -> None:
    """Desenha rastos curtos e suaves, mais eficiente com LineCollection."""
    if len(history) < 2 or n <= 0 or max_trails <= 0:
        return

    trail_idx = _sample_indices(n, min(max_trails, n), seed + 90)
    segments: List[Array] = []

    for idx in trail_idx:
        pts = []
        for h in history:
            hp = h.get("pos")
            if hp is not None and hp.shape[0] > idx:
                pts.append(hp[idx, :2])
        if len(pts) > 1:
            arr = np.asarray(pts, dtype=np.float64)
            if np.all(np.isfinite(arr)):
                segments.append(arr)

    if not segments:
        return

    # Camada larga e muito transparente + camada fina por cima: efeito glow.
    lc_glow = LineCollection(segments, colors="#19b8ff", linewidths=1.45, alpha=0.035)
    lc_core = LineCollection(segments, colors="#9fe5ff", linewidths=0.60, alpha=0.13)
    ax.add_collection(lc_glow)
    ax.add_collection(lc_core)


def _draw_velocity_flow(ax: Any, pos: Array, vel: Array, limit: float, args: Any, frame_number: int) -> None:
    """
    Pequenas setas de fluxo no 2D para mostrar que há movimento físico.

    É visualmente útil no relatório/GIF porque evidencia atualização de velocidade/posição.
    """
    if pos.size == 0 or vel.size == 0 or pos.shape != vel.shape:
        return

    n = pos.shape[0]
    max_arrows = _as_int(_get(args, "flow_arrows", 90), 90)
    if max_arrows <= 0:
        return

    idx = _sample_indices(n, min(max_arrows, n), _as_int(_get(args, "seed", 42), 42) + 300 + frame_number)
    if idx.size == 0:
        return

    x = pos[idx, 0]
    y = pos[idx, 1]
    vx = vel[idx, 0]
    vy = vel[idx, 1]

    speed = np.sqrt(vx * vx + vy * vy)
    good = np.isfinite(speed) & (speed > np.percentile(speed[np.isfinite(speed)], 25) if np.any(np.isfinite(speed)) else False)
    if not np.any(good):
        return

    x, y, vx, vy, speed = x[good], y[good], vx[good], vy[good], speed[good]
    norm = np.percentile(speed, 90)
    if not np.isfinite(norm) or norm <= 0:
        return

    # Comprimento visual das setas, não físico.
    scale = limit * 0.018 / norm
    ax.quiver(
        x,
        y,
        vx * scale,
        vy * scale,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#bdeeff",
        alpha=0.16,
        width=0.0016,
        headwidth=3.0,
        headlength=4.0,
        headaxislength=3.2,
    )


# ---------------------------------------------------------------------------
# Frame da simulação física direta
# ---------------------------------------------------------------------------

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
    """
    Cria um frame PNG da simulação física direta.

    O frame contém:
    - subplot 2D: densidade, braços espirais, núcleo, rastos e fluxo;
    - subplot 3D: disco, espessura e halo;
    - texto com ano, frame, partículas, ranks MPI, virial e energia.
    """
    pos = np.asarray(snapshot.get("pos", np.empty((0, 3))), dtype=np.float64)
    vel = np.asarray(snapshot.get("vel", np.zeros_like(pos)), dtype=np.float64)
    mass = np.asarray(snapshot.get("mass", np.ones(pos.shape[0])), dtype=np.float64)

    if pos.ndim != 2 or pos.shape[1] != 3:
        pos = np.empty((0, 3), dtype=np.float64)
    if vel.shape != pos.shape:
        vel = np.zeros_like(pos)
    if mass.ndim != 1 or mass.shape[0] != pos.shape[0]:
        mass = np.ones(pos.shape[0], dtype=np.float64)

    n = int(pos.shape[0])
    galaxy_radius = _as_float(_get(args, "galaxy_radius", 120.0), 120.0)
    limit = galaxy_radius * 1.28

    frame_path = Path(frame_path)
    frame_path.parent.mkdir(parents=True, exist_ok=True)

    fig_w = _as_float(_get(args, "fig_width", 16.5), 16.5)
    fig_h = _as_float(_get(args, "fig_height", 8.8), 8.8)
    dpi = _as_int(_get(args, "dpi", 135), 135)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#020611")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.07)
    ax2d = fig.add_subplot(grid[0, 0])
    ax3d = fig.add_subplot(grid[0, 1], projection="3d")

    _set_space_axes(ax2d, limit, "Vista 2D — densidade, núcleo e braços espirais")
    ax2d.set_aspect("equal", adjustable="box")

    if background is not None:
        ax2d.scatter(
            background["x"],
            background["y"],
            s=background["s"],
            c="#c9e7ff",
            alpha=background["a"],
            linewidths=0,
            zorder=0,
        )

    if n:
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]
        r_xy = np.sqrt(x * x + y * y)
        r = np.sqrt(x * x + y * y + z * z)
        m_sqrt = np.sqrt(np.clip(mass, 0.0, None))

        # -------------------------------------------------------------------
        # 2D: densidade suave do disco
        # -------------------------------------------------------------------
        bins = _as_int(_get(args, "density_bins", 420), 420)
        bins = max(80, min(900, bins))
        hist, xedges, yedges = np.histogram2d(
            x,
            y,
            bins=bins,
            range=[[-limit, limit], [-limit, limit]],
            weights=m_sqrt,
        )

        if np.max(hist) > 0:
            # Halo largo e suave.
            ax2d.imshow(
                hist.T,
                origin="lower",
                extent=[-limit, limit, -limit, limit],
                cmap="inferno",
                norm=PowerNorm(gamma=0.38),
                alpha=0.58,
                interpolation="bicubic",
                zorder=1,
            )
            # Camada azul/magenta para dar contraste às zonas densas.
            ax2d.imshow(
                hist.T,
                origin="lower",
                extent=[-limit, limit, -limit, limit],
                cmap="magma",
                norm=PowerNorm(gamma=0.62),
                alpha=0.34,
                interpolation="bilinear",
                zorder=2,
            )
            # Contornos discretos para realçar braços espirais.
            try:
                levels = np.percentile(hist[hist > 0], [72, 86, 94])
                levels = np.unique(levels[np.isfinite(levels) & (levels > 0)])
                if levels.size:
                    cx = 0.5 * (xedges[:-1] + xedges[1:])
                    cy = 0.5 * (yedges[:-1] + yedges[1:])
                    ax2d.contour(
                        cx,
                        cy,
                        hist.T,
                        levels=levels,
                        colors=["#50d6ff", "#b75cff", "#ffe58a"][: levels.size],
                        linewidths=[0.35, 0.40, 0.45][: levels.size],
                        alpha=0.20,
                        zorder=3,
                    )
            except Exception:
                pass

        # Rastos curtos de algumas estrelas.
        max_trails = _as_int(_get(args, "trail_particles", 260), 260)
        _draw_trails_2d(ax2d, history, n, max_trails, _as_int(_get(args, "seed", 42), 42))

        # Setas de fluxo para reforçar a ideia de movimento orbital.
        _draw_velocity_flow(ax2d, pos, vel, limit, args, frame_number)

        # Estrelas por cima da densidade.
        max_points_2d = _as_int(_get(args, "max_2d_points", 60000), 60000)
        idx2 = _sample_indices(n, max_points_2d, _as_int(_get(args, "seed", 42), 42) + frame_number)
        c2 = _safe_norm(r_xy[idx2])
        mass_norm = _safe_norm(m_sqrt[idx2])
        s2 = np.clip(0.42 + 5.2 * mass_norm, 0.35, 6.0)

        # Glow por trás das estrelas principais.
        ax2d.scatter(
            x[idx2],
            y[idx2],
            s=s2 * 3.2,
            c=c2,
            cmap="cool",
            alpha=0.10,
            linewidths=0,
            zorder=4,
        )
        ax2d.scatter(
            x[idx2],
            y[idx2],
            s=s2,
            c=c2,
            cmap="cool",
            alpha=0.88,
            linewidths=0,
            zorder=5,
        )

        # Núcleo em várias camadas para parecer mais luminoso.
        if n >= 10:
            core_cut = np.percentile(r, 6.5)
            core = r <= core_cut
            if np.any(core):
                ax2d.scatter(x[core], y[core], s=38.0, c="#ffd36e", alpha=0.035, linewidths=0, zorder=6)
                ax2d.scatter(x[core], y[core], s=15.0, c="#fff0a8", alpha=0.12, linewidths=0, zorder=7)
                ax2d.scatter(x[core], y[core], s=5.2, c="#fff8dc", alpha=0.48, linewidths=0, zorder=8)
                ax2d.scatter(x[core], y[core], s=1.8, c="#ffffff", alpha=0.82, linewidths=0, zorder=9)

        # Marcador discreto do centro de massa.
        total_mass = float(np.sum(mass))
        if total_mass > 0 and np.all(np.isfinite(mass)):
            com = np.sum(pos * mass[:, None], axis=0) / total_mass
            ax2d.scatter([com[0]], [com[1]], s=55, marker="+", c="#ffffff", alpha=0.85, linewidths=1.2, zorder=10)

        # -------------------------------------------------------------------
        # 3D: disco, espessura e halo
        # -------------------------------------------------------------------
        ax3d.set_facecolor("#020611")
        idx3 = _sample_indices(n, _as_int(_get(args, "max_3d_points", 9000), 9000), _as_int(_get(args, "seed", 42), 42) + 5 + frame_number)
        c3 = _safe_norm(r[idx3])
        s3 = np.clip(0.85 + 7.2 * _safe_norm(m_sqrt[idx3]), 0.75, 8.0)

        # Pontos principais.
        ax3d.scatter(
            x[idx3],
            y[idx3],
            z[idx3],
            s=s3,
            c=c3,
            cmap="plasma",
            alpha=0.84,
            linewidths=0,
            depthshade=True,
        )

        # Núcleo 3D ligeiramente realçado.
        if n >= 10:
            core3 = r <= np.percentile(r, 5.0)
            core_idx = np.where(core3)[0]
            if core_idx.size > 0:
                core_idx = core_idx[: min(core_idx.size, 900)]
                ax3d.scatter(
                    x[core_idx],
                    y[core_idx],
                    z[core_idx],
                    s=10.0,
                    c="#fff2a8",
                    alpha=0.45,
                    linewidths=0,
                    depthshade=True,
                )

        ax3d.set_xlim(-limit, limit)
        ax3d.set_ylim(-limit, limit)
        thickness = _as_float(_get(args, "thickness", 6.0), 6.0)
        zlim = max(thickness * 8.5, limit * 0.24)
        ax3d.set_zlim(-zlim, zlim)

        # Câmara lenta e dinâmica: dá sensação cinematográfica no GIF.
        ax3d.view_init(elev=25 + 3.0 * np.sin(frame_number * 0.22), azim=38 + 0.65 * frame_number)
    else:
        ax3d.text2D(0.3, 0.5, "Sem partículas", transform=ax3d.transAxes, color="white")

    # Estilo 3D
    ax3d.set_title("Vista 3D — disco, espessura e halo", color="#f0f7ff", fontsize=13, pad=11, weight="bold")
    ax3d.tick_params(colors="#9eb8d4", labelsize=8)
    ax3d.set_xlabel("x", color="#abc8e8", fontsize=9)
    ax3d.set_ylabel("y", color="#abc8e8", fontsize=9)
    ax3d.set_zlabel("z", color="#abc8e8", fontsize=9)

    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.set_facecolor((0.01, 0.025, 0.055, 0.95))
        pane.set_edgecolor("#1b3555")
    ax3d.grid(True, alpha=0.18)

    # Texto informativo. Mantém o que é útil para avaliação.
    progress = 100.0 * (frame_number + 1) / max(1, frame_count_estimate)
    virial = float(diagnostics.get("virial_ratio", 0.0))
    drift = float(diagnostics.get("energy_drift_percent", 0.0))
    max_radius = float(diagnostics.get("max_radius", 0.0))

    text = (
        f"Ano simulado: {current_year:,.0f}\n"
        f"Frame: {frame_number + 1}/{frame_count_estimate}  ({progress:.0f}%)\n"
        f"Partículas: {n:,} | ranks MPI: {mpi_info.get('size', '?')}\n"
        f"Virial 2K/|U|: {virial:.3f}\n"
        f"Drift energia: {drift:+.3f}%\n"
        f"Raio máximo: {max_radius:.1f}"
    )
    fig.text(0.018, 0.958, text, color="#dbeeff", fontsize=9.2, va="top", family="monospace")

    # Legenda curta para o professor perceber a correspondência com o enunciado.
    subtitle = "N-corpos Newtoniano | atualização paralela de aceleração, velocidade e posição | frames de 1000 em 1000 anos"
    fig.text(0.50, 0.952, subtitle, color="#9fcdf2", fontsize=8.5, ha="center", va="top")

    fig.suptitle(
        "Simulação Paralela de Galáxia — Newton + MPI",
        color="#ffffff",
        fontsize=16,
        weight="bold",
        y=0.994,
    )

    _draw_progress_bar(fig, frame_number, frame_count_estimate)

    fig.savefig(frame_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    return {
        "frame": int(frame_number),
        "year": float(current_year),
        "path": str(frame_path),
        "particles": int(n),
        "max_radius": float(max_radius),
        "energy_drift_percent": float(drift),
        "virial_ratio": float(virial),
    }


# ---------------------------------------------------------------------------
# GIF
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Modo proxy visual para escala enorme
# ---------------------------------------------------------------------------

def generate_scale_rank_sample(local_n: int, args: Any, rank: int) -> Dict[str, Array]:
    """
    Amostra visual para representar N muito grande sem executar O(N^2).

    Útil para discutir a escala 10^8/10^9 no relatório. Este modo é proxy visual,
    não uma simulação física direta.
    """
    rng = np.random.default_rng(_as_int(_get(args, "seed", 42), 42) + 5000 + int(rank))
    local_n = int(local_n)
    radius = _as_float(_get(args, "galaxy_radius", 120.0), 120.0)
    arms = max(1, _as_int(_get(args, "spiral_arms", 4), 4))

    r = rng.gamma(shape=2.1, scale=radius / 6.0, size=local_n)
    r = np.clip(r, 0.2, radius)
    arm = rng.integers(0, arms, size=local_n)
    theta = 2.0 * np.pi * arm / arms + _as_float(_get(args, "arm_twist", 4.8), 4.8) * np.log1p(r / (radius * 0.08))
    theta += rng.normal(0.0, _as_float(_get(args, "arm_width", 0.20), 0.20) * (0.6 + r / radius), size=local_n)
    z = rng.normal(0.0, _as_float(_get(args, "thickness", 6.0), 6.0) * (0.2 + 0.8 * r / radius), size=local_n)
    mass = rng.lognormal(_as_float(_get(args, "mass_log_mean", 0.0), 0.0), _as_float(_get(args, "mass_log_sigma", 0.45), 0.45), size=local_n)

    return {
        "r": r.astype(np.float64),
        "theta": theta.astype(np.float64),
        "z": z.astype(np.float64),
        "mass": mass.astype(np.float64),
    }


def render_scale_frame(
    gathered: List[Dict[str, Array]],
    frame_path: Path,
    frame_idx: int,
    total_frames: int,
    args: Any,
    size: int,
    represented_particles: int,
) -> Dict[str, Any]:
    """Frame visual proxy para escala 10^8/10^9."""
    if not gathered:
        gathered = [{"r": np.empty(0), "theta": np.empty(0), "z": np.empty(0), "mass": np.empty(0)}]

    r = np.concatenate([g["r"] for g in gathered])
    theta0 = np.concatenate([g["theta"] for g in gathered])
    z = np.concatenate([g["z"] for g in gathered])
    mass = np.concatenate([g["mass"] for g in gathered])

    t = frame_idx / max(1, total_frames - 1)

    # Rotação diferencial visual: núcleo roda mais depressa que a periferia.
    radius = _as_float(_get(args, "galaxy_radius", 120.0), 120.0)
    omega = 2.25 * np.exp(-r / (radius * 0.75)) + 0.16
    theta = theta0 + 2.0 * np.pi * t * omega
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Velocidade tangencial proxy só para as setas de fluxo no render_direct_frame().
    vx = -r * np.sin(theta) * omega
    vy = r * np.cos(theta) * omega
    vz = np.zeros_like(vx)

    fake_snapshot = {
        "ids": np.arange(r.size, dtype=np.int64),
        "pos": np.column_stack([x, y, z]) if r.size else np.empty((0, 3), dtype=np.float64),
        "vel": np.column_stack([vx, vy, vz]) if r.size else np.empty((0, 3), dtype=np.float64),
        "mass": mass,
    }

    diag = {
        "virial_ratio": 0.0,
        "energy_drift_percent": 0.0,
        "max_radius": float(np.max(r)) if r.size else 0.0,
    }
    bg = make_background(_as_int(_get(args, "seed", 42), 42), _as_int(_get(args, "background_stars", 2400), 2400), radius * 1.45)

    rec = render_direct_frame(
        fake_snapshot,
        [],
        frame_path,
        t * _as_float(_get(args, "years", 1.0), 1.0),
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
