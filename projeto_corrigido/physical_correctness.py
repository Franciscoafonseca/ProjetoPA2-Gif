#!/usr/bin/env python3
"""
physical_correctness.py

Modelo físico da simulação de galáxia para PA Project 2.

O objetivo deste ficheiro é manter a física separada do MPI:
- geração de condições iniciais com disco espiral, bojo central e halo;
- massa, posição e velocidade inicial de cada estrela;
- aceleração gravitacional newtoniana com softening;
- diagnósticos físicos para o relatório.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

Array = np.ndarray


def _rng_for_rank(seed: int, rank: int) -> np.random.Generator:
    """Gerador independente por rank, reprodutível para testes."""
    return np.random.default_rng(int(seed) + 1009 * int(rank) + 9176)


def generate_local_initial_conditions(
    ids: Array,
    args,
    rank: int = 0,
) -> Tuple[Array, Array, Array, Array]:
    """
    Gera apenas as partículas locais deste rank.

    Isto torna a geração inicial paralela: o rank 0 só distribui os IDs por scatter(),
    e cada processo cria o seu bloco usando distribuições estatísticas.

    Estrutura da galáxia:
    - disco fino com braços espirais logarítmicos;
    - bojo central mais denso;
    - halo 3D difuso;
    - velocidades tangenciais aproximadas para órbitas quase circulares.
    """
    ids = np.asarray(ids, dtype=np.int64)
    n = int(ids.size)
    rng = _rng_for_rank(int(args.seed), rank)

    if n == 0:
        return ids, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    radius = float(args.galaxy_radius)
    thickness = float(args.thickness)
    arms = max(1, int(args.spiral_arms))

    # Tipos de estrela/partícula. As probabilidades criam disco dominante,
    # bojo visível e halo pouco denso.
    u = rng.random(n)
    is_bulge = u < float(args.bulge_fraction)
    is_halo = (u >= float(args.bulge_fraction)) & (u < float(args.bulge_fraction) + float(args.halo_fraction))
    is_disk = ~(is_bulge | is_halo)

    pos = np.zeros((n, 3), dtype=np.float64)
    vel = np.zeros((n, 3), dtype=np.float64)

    # Massas log-normais: muitas estrelas leves e algumas mais pesadas.
    mass = rng.lognormal(mean=float(args.mass_log_mean), sigma=float(args.mass_log_sigma), size=n)
    mass = mass.astype(np.float64)

    # -----------------------------
    # Disco espiral
    # -----------------------------
    nd = int(np.count_nonzero(is_disk))
    if nd:
        # Gamma cria centro denso e cauda externa, mais parecido com disco galáctico.
        r = rng.gamma(shape=2.25, scale=radius / 6.2, size=nd)
        r = np.clip(r, 0.45, radius)

        arm = rng.integers(0, arms, size=nd)
        arm_angle = (2.0 * np.pi * arm) / arms

        # Espiral logarítmica/diferencial. Menos ruído no centro, mais difuso fora.
        arm_width = float(args.arm_width)
        twist = float(args.arm_twist)
        theta = arm_angle + twist * np.log1p(r / max(radius * 0.08, 1e-9))
        theta += rng.normal(0.0, arm_width * (0.55 + 0.85 * r / radius), size=nd)

        # Barra central suave, ajuda a imagem parecer uma galáxia estruturada.
        bar_mask = r < radius * float(args.bar_fraction)
        if np.any(bar_mask):
            theta[bar_mask] = rng.normal(0.0, 0.28, size=int(np.count_nonzero(bar_mask)))
            theta[bar_mask] += rng.choice([0.0, np.pi], size=int(np.count_nonzero(bar_mask)))

        z = rng.normal(0.0, thickness * (0.18 + 0.75 * r / radius), size=nd)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pos[is_disk, 0] = x
        pos[is_disk, 1] = y
        pos[is_disk, 2] = z

        # Velocidade orbital aproximada: v = sqrt(G*M(<r)/r).
        enclosed = float(args.central_mass) + float(args.disk_mass_proxy) * (r / radius) ** 1.55
        v_circ = np.sqrt(float(args.g_const) * enclosed / np.maximum(r, float(args.softening)))
        v_circ *= float(args.rotation_factor)

        tangential = np.column_stack((-np.sin(theta), np.cos(theta), np.zeros(nd)))
        radial = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(nd)))
        vel[is_disk] = tangential * v_circ[:, None]
        vel[is_disk] += radial * rng.normal(0.0, float(args.radial_velocity_noise), size=(nd, 1))
        vel[is_disk, 2] += rng.normal(0.0, float(args.vertical_velocity_noise), size=nd)

    # -----------------------------
    # Bojo central
    # -----------------------------
    nb = int(np.count_nonzero(is_bulge))
    if nb:
        sigma = radius * 0.105
        bulge = rng.normal(0.0, sigma, size=(nb, 3))
        bulge[:, 2] *= 0.62
        pos[is_bulge] = bulge
        # O bojo não roda como o disco; é mais disperso.
        speed = rng.normal(0.0, float(args.bulge_velocity_noise), size=(nb, 3))
        vel[is_bulge] = speed
        mass[is_bulge] *= 1.18

    # -----------------------------
    # Halo esférico difuso
    # -----------------------------
    nh = int(np.count_nonzero(is_halo))
    if nh:
        rr = radius * float(args.halo_radius_factor) * (rng.random(nh) ** (1.0 / 3.0))
        cos_phi = rng.uniform(-1.0, 1.0, size=nh)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=nh)
        sin_phi = np.sqrt(np.maximum(0.0, 1.0 - cos_phi * cos_phi))
        pos[is_halo, 0] = rr * sin_phi * np.cos(phi)
        pos[is_halo, 1] = rr * sin_phi * np.sin(phi)
        pos[is_halo, 2] = rr * cos_phi * 0.72
        vel[is_halo] = rng.normal(0.0, float(args.halo_velocity_noise), size=(nh, 3))
        mass[is_halo] *= 0.72

    # Pequenas perturbações suavizam a evolução sem destruir os braços.
    vel += rng.normal(0.0, float(args.velocity_noise), size=(n, 3))

    return ids, pos.astype(np.float64), vel.astype(np.float64), mass.astype(np.float64)


def acceleration_from_block(
    local_pos: Array,
    block_pos: Array,
    block_mass: Array,
    g_const: float,
    softening: float,
    pair_block: int,
) -> Array:
    """
    Aceleração gravitacional causada por um bloco de partículas.

    Para uma estrela i e uma estrela j:
        a_i += G * m_j * (r_j - r_i) / (|r_j-r_i|^2 + eps^2)^(3/2)

    A divisão em blocos evita criar uma matriz NxN demasiado grande em memória.
    """
    local_pos = np.asarray(local_pos, dtype=np.float64)
    block_pos = np.asarray(block_pos, dtype=np.float64)
    block_mass = np.asarray(block_mass, dtype=np.float64)

    acc = np.zeros_like(local_pos, dtype=np.float64)
    if local_pos.size == 0 or block_pos.size == 0:
        return acc

    chunk = max(32, int(pair_block))
    eps2 = float(softening) ** 2
    g = float(g_const)

    for start in range(0, local_pos.shape[0], chunk):
        end = min(start + chunk, local_pos.shape[0])
        diff = block_pos[None, :, :] - local_pos[start:end, None, :]
        dist2 = np.sum(diff * diff, axis=2) + eps2
        inv_dist3 = dist2 ** -1.5
        # Se diff == 0, a contribuição é zero. O softening evita singularidades.
        weighted = diff * (block_mass[None, :] * inv_dist3)[:, :, None]
        acc[start:end] += g * np.sum(weighted, axis=1)

    return acc


def central_mass_acceleration(local_pos: Array, args) -> Array:
    """Aceleração causada por uma massa central, útil para estabilidade orbital."""
    pos = np.asarray(local_pos, dtype=np.float64)
    if pos.size == 0:
        return np.zeros_like(pos)
    eps2 = float(args.softening) ** 2
    r2 = np.sum(pos * pos, axis=1) + eps2
    inv_r3 = r2 ** -1.5
    return -float(args.g_const) * float(args.central_mass) * pos * inv_r3[:, None]


def local_basic_diagnostics(pos: Array, vel: Array, mass: Array) -> Dict[str, Array | float]:
    """Diagnósticos locais combinados depois por allreduce()."""
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)

    if mass.size == 0:
        return {
            "kinetic": 0.0,
            "mass": 0.0,
            "speed_mass_sum": 0.0,
            "radius_mass_sum": 0.0,
            "max_radius": 0.0,
            "angular_momentum": np.zeros(3, dtype=np.float64),
        }

    speed2 = np.sum(vel * vel, axis=1)
    speed = np.sqrt(speed2)
    radius = np.linalg.norm(pos, axis=1)

    return {
        "kinetic": float(0.5 * np.sum(mass * speed2)),
        "mass": float(np.sum(mass)),
        "speed_mass_sum": float(np.sum(mass * speed)),
        "radius_mass_sum": float(np.sum(mass * radius)),
        "max_radius": float(np.max(radius)),
        "angular_momentum": np.sum(np.cross(pos, vel) * mass[:, None], axis=0),
    }


def snapshot_energy_diagnostics(snapshot: Dict[str, Array], args) -> Dict[str, float]:
    """
    Energia e métricas para o relatório.

    Para muitos pontos, a energia potencial é estimada por amostragem para não
    bloquear a geração do GIF.
    """
    pos = np.asarray(snapshot["pos"], dtype=np.float64)
    vel = np.asarray(snapshot["vel"], dtype=np.float64)
    mass = np.asarray(snapshot["mass"], dtype=np.float64)
    n = int(pos.shape[0])

    if n == 0:
        return {
            "kinetic": 0.0,
            "potential": 0.0,
            "total_energy": 0.0,
            "virial_ratio": 0.0,
            "max_radius": 0.0,
            "mean_radius": 0.0,
            "mean_speed": 0.0,
            "angular_momentum_norm": 0.0,
            "potential_is_estimated": 0.0,
        }

    speed2 = np.sum(vel * vel, axis=1)
    speed = np.sqrt(speed2)
    radius = np.linalg.norm(pos, axis=1)
    kinetic = float(0.5 * np.sum(mass * speed2))

    limit = int(getattr(args, "energy_sample_limit", 2500))
    estimated = 0.0
    if n > limit:
        rng = np.random.default_rng(int(args.seed) + 991)
        idx = rng.choice(n, size=limit, replace=False)
        sample_pos = pos[idx]
        sample_mass = mass[idx]
        scale = (n / limit) ** 2
        estimated = 1.0
    else:
        sample_pos = pos
        sample_mass = mass
        scale = 1.0

    potential = 0.0
    eps2 = float(args.softening) ** 2
    g = float(args.g_const)
    m = sample_pos.shape[0]
    for i in range(m):
        diff = sample_pos[i + 1 :] - sample_pos[i]
        if diff.size == 0:
            continue
        dist = np.sqrt(np.sum(diff * diff, axis=1) + eps2)
        potential -= float(np.sum(g * sample_mass[i] * sample_mass[i + 1 :] / dist))
    potential *= scale

    # Potencial da massa central.
    dist_c = np.sqrt(np.sum(pos * pos, axis=1) + eps2)
    potential -= float(np.sum(g * float(args.central_mass) * mass / dist_c))

    total = kinetic + potential
    virial = float(2.0 * kinetic / abs(potential)) if abs(potential) > 1e-30 else 0.0
    angular = np.sum(np.cross(pos, vel) * mass[:, None], axis=0)

    return {
        "kinetic": kinetic,
        "potential": float(potential),
        "total_energy": float(total),
        "virial_ratio": virial,
        "max_radius": float(np.max(radius)),
        "mean_radius": float(np.average(radius, weights=mass)),
        "mean_speed": float(np.average(speed, weights=mass)),
        "angular_momentum_norm": float(np.linalg.norm(angular)),
        "potential_is_estimated": estimated,
    }
