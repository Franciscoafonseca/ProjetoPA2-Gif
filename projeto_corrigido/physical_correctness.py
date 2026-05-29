#!/usr/bin/env python3
"""
physical_correctness.py

Criterion covered: Physical correctness.

This module contains the physics of the galaxy simulation:
- Newtonian gravity: F = G m1 m2 / r^2 and a = F / m.
- Gravitational softening to avoid numerical singularities at very small distances.
- Physically meaningful initial conditions: disk + halo, masses, tangential orbital velocities.
- Leapfrog / velocity-Verlet integration support.
- Energy and orbital diagnostics for the report.

The MPI communication is intentionally NOT here. This file is about the model and the
mathematics only, so it is easier to explain in the report under "Physical correctness".
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

Array = np.ndarray


def create_initial_conditions(args) -> Tuple[Array, Array, Array, Array]:
    """Create a galaxy on rank 0 before MPI scatter.

    Each star has:
    - id: stable particle identifier;
    - pos: 3D position;
    - vel: 3D velocity;
    - mass: scalar mass.

    Distribution choices used for the report:
    - disk radii follow a gamma-like distribution for a dense core and extended disk;
    - halo particles are sampled from a diffuse spherical distribution;
    - stellar masses follow a log-normal distribution;
    - initial velocities are tangential and approximately circular around the centre.
    """
    rng = np.random.default_rng(int(args.seed))
    n = int(args.particles)
    ids = np.arange(n, dtype=np.int64)

    # Log-normal masses: many lighter stars and a few heavier bodies.
    mass = rng.lognormal(mean=6.15, sigma=0.38, size=n).astype(np.float64)

    halo_fraction = float(np.clip(args.halo_fraction, 0.0, 0.45))
    is_halo = rng.random(n) < halo_fraction
    is_disk = ~is_halo
    n_disk = int(np.count_nonzero(is_disk))
    n_halo = n - n_disk

    r = np.empty(n, dtype=np.float64)
    theta = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)

    # Disk: spiral arms with a concentrated core and extended tails.
    r_disk = rng.gamma(shape=2.0, scale=float(args.galaxy_radius) / 5.2, size=n_disk)
    r_disk = np.clip(r_disk, 0.5, float(args.galaxy_radius))
    arm = rng.integers(0, max(1, int(args.spiral_arms)), size=n_disk)
    base_theta = 2.0 * np.pi * arm / max(1, int(args.spiral_arms))
    theta_disk = base_theta + float(args.arm_twist) * (r_disk / float(args.galaxy_radius))
    theta_disk += rng.normal(0.0, 0.14 + 0.005 * r_disk, size=n_disk)
    z_disk = rng.normal(
        0.0,
        float(args.thickness) * (0.22 + 0.78 * r_disk / float(args.galaxy_radius)),
        size=n_disk,
    )

    # Halo: less dense 3D population around the disk.
    if n_halo > 0:
        r_halo = float(args.galaxy_radius) * (rng.random(n_halo) ** (1.0 / 3.0))
        cos_phi = rng.uniform(-1.0, 1.0, size=n_halo)
        theta_halo = rng.uniform(0.0, 2.0 * np.pi, size=n_halo)
        r_halo_xy = r_halo * np.sqrt(np.maximum(0.0, 1.0 - cos_phi * cos_phi))
        z_halo = r_halo * cos_phi * 0.70
    else:
        r_halo_xy = np.empty(0, dtype=np.float64)
        theta_halo = np.empty(0, dtype=np.float64)
        z_halo = np.empty(0, dtype=np.float64)

    r[is_disk] = r_disk
    theta[is_disk] = theta_disk
    z[is_disk] = z_disk
    r[is_halo] = r_halo_xy
    theta[is_halo] = theta_halo
    z[is_halo] = z_halo

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pos = np.column_stack([x, y, z]).astype(np.float64)

    # Approximate circular velocities around the central mass.
    radius_3d = np.linalg.norm(pos, axis=1) + float(args.softening)
    enclosed_fraction = np.clip(radius_3d / max(float(args.galaxy_radius), 1e-12), 0.0, 1.0)
    enclosed_mass = float(args.central_mass) + np.sum(mass) * enclosed_fraction
    circular_speed = np.sqrt(float(args.g_const) * enclosed_mass / radius_3d) * float(args.rotation_factor)

    tangential = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros(n, dtype=np.float64)])
    vel = tangential * circular_speed[:, None]
    vel += rng.normal(0.0, float(args.velocity_noise), size=(n, 3))
    vel[:, 2] *= 0.20

    # Remove centre-of-mass drift.
    pos -= np.average(pos, axis=0, weights=mass)
    vel -= np.average(vel, axis=0, weights=mass)
    return ids, pos.astype(np.float64), vel.astype(np.float64), mass.astype(np.float64)


def acceleration_from_block(
    local_pos: Array,
    block_pos: Array,
    block_mass: Array,
    g_const: float,
    softening: float,
    pair_block: int,
) -> Array:
    """Acceleration on local particles caused by a block of other particles.

    Newtonian acceleration from a particle j on i:

        a_i += G * m_j * (r_j - r_i) / (|r_j-r_i|^2 + eps^2)^(3/2)

    The particle's own block may include itself. For self-interaction the difference
    vector is zero, therefore the contribution is zero even with softening.
    """
    if local_pos.size == 0 or block_pos.size == 0:
        return np.zeros_like(local_pos)

    acc = np.zeros_like(local_pos, dtype=np.float64)
    eps2 = float(softening) * float(softening)
    step = max(1, int(pair_block))

    for start in range(0, len(block_pos), step):
        end = min(start + step, len(block_pos))
        bp = block_pos[start:end]
        bm = block_mass[start:end]
        diff = bp[None, :, :] - local_pos[:, None, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff) + eps2
        inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
        acc += float(g_const) * np.einsum("ijk,j,ij->ik", diff, bm, inv_dist3)

    return acc


def central_mass_acceleration(local_pos: Array, args) -> Array:
    """Acceleration caused by a massive object at the origin.

    This improves orbital structure and is physically plausible for a galaxy proxy,
    while still keeping the N-body star-star interaction in the code.
    """
    if local_pos.size == 0 or float(args.central_mass) <= 0.0:
        return np.zeros_like(local_pos)
    diff = -local_pos
    eps2 = float(args.softening) * float(args.softening)
    dist2 = np.einsum("ij,ij->i", diff, diff) + eps2
    inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))
    return float(args.g_const) * float(args.central_mass) * diff * inv_dist3[:, None]


def leapfrog_drift_kick(local_pos: Array, local_vel: Array, acc_old: Array, acc_new: Array, dt: float) -> Tuple[Array, Array]:
    """Complete a velocity-Verlet / Leapfrog update.

    This function is provided for clarity. In the main loop the same steps are kept
    explicit because the new acceleration is computed through MPI communication:

        v(t+dt/2) = v(t) + 0.5 dt a(t)
        x(t+dt)   = x(t) + dt v(t+dt/2)
        v(t+dt)   = v(t+dt/2) + 0.5 dt a(t+dt)
    """
    half = 0.5 * float(dt)
    v_half = local_vel + half * acc_old
    new_pos = local_pos + float(dt) * v_half
    new_vel = v_half + half * acc_new
    return new_pos, new_vel


def local_basic_diagnostics(pos: Array, vel: Array, mass: Array) -> Dict[str, Array | float]:
    """Rank-local diagnostics that can be combined with MPI allreduce."""
    if len(pos) == 0:
        return {
            "kinetic": 0.0,
            "mass": 0.0,
            "speed_mass_sum": 0.0,
            "radius_mass_sum": 0.0,
            "max_radius": 0.0,
            "angular_momentum": np.zeros(3, dtype=np.float64),
        }

    speed2 = np.einsum("ij,ij->i", vel, vel)
    speed = np.sqrt(speed2)
    radius = np.linalg.norm(pos, axis=1)
    kinetic = 0.5 * float(np.sum(mass * speed2))
    angular_momentum = np.sum(np.cross(pos, mass[:, None] * vel), axis=0)

    return {
        "kinetic": kinetic,
        "mass": float(np.sum(mass)),
        "speed_mass_sum": float(np.sum(mass * speed)),
        "radius_mass_sum": float(np.sum(mass * radius)),
        "max_radius": float(np.max(radius)),
        "angular_momentum": angular_momentum.astype(np.float64),
    }


def snapshot_energy_diagnostics(snapshot: Dict[str, Array], args) -> Dict[str, float]:
    """Compute global physical diagnostics on rank 0 from a gathered snapshot.

    For normal report-scale runs this is exact. For very large debug runs the pair
    potential can be sampled by setting --energy-sample-limit to a lower value.
    """
    pos = snapshot["pos"]
    vel = snapshot["vel"]
    mass = snapshot["mass"]
    n = len(pos)
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

    speed2 = np.einsum("ij,ij->i", vel, vel)
    speed = np.sqrt(speed2)
    radius = np.linalg.norm(pos, axis=1)
    kinetic = 0.5 * float(np.sum(mass * speed2))

    sample_limit = int(getattr(args, "energy_sample_limit", 5000))
    potential_is_estimated = 0.0
    if n > sample_limit > 1:
        rng = np.random.default_rng(int(args.seed) + 98765)
        idx = np.sort(rng.choice(n, size=sample_limit, replace=False))
        p_pos = pos[idx]
        p_mass = mass[idx]
        scale = (float(n) * float(n - 1)) / (float(sample_limit) * float(sample_limit - 1))
        potential_is_estimated = 1.0
    else:
        p_pos = pos
        p_mass = mass
        scale = 1.0

    eps2 = float(args.softening) * float(args.softening)
    potential_pairs = 0.0
    m = len(p_pos)
    for i in range(m - 1):
        diff = p_pos[i + 1 :] - p_pos[i]
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff) + eps2)
        potential_pairs -= float(args.g_const) * float(p_mass[i]) * float(np.sum(p_mass[i + 1 :] / dist))
    potential_pairs *= scale

    # Central mass potential is exact for all particles.
    potential_central = 0.0
    if float(args.central_mass) > 0.0:
        r = np.sqrt(np.einsum("ij,ij->i", pos, pos) + eps2)
        potential_central = -float(args.g_const) * float(args.central_mass) * float(np.sum(mass / r))

    potential = potential_pairs + potential_central
    total_energy = kinetic + potential
    virial_ratio = (2.0 * kinetic / abs(potential)) if abs(potential) > 1e-30 else 0.0
    angular_momentum = np.sum(np.cross(pos, mass[:, None] * vel), axis=0)

    return {
        "kinetic": kinetic,
        "potential": potential,
        "total_energy": total_energy,
        "virial_ratio": virial_ratio,
        "max_radius": float(np.max(radius)),
        "mean_radius": float(np.average(radius, weights=mass)),
        "mean_speed": float(np.average(speed, weights=mass)),
        "angular_momentum_norm": float(np.linalg.norm(angular_momentum)),
        "potential_is_estimated": potential_is_estimated,
    }
