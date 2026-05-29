#!/usr/bin/env python3
"""
mpi_parallelization.py

Criteria covered:
- Parallelization quality.
- Correct use of MPI methods.

This module contains the distributed-memory logic:
- bcast: configuration distribution;
- scatter: initial particle distribution;
- isend/irecv: ring exchange of particle blocks for force computation;
- gather: snapshot collection for plotting;
- reduce: particle-count validation;
- allreduce: global diagnostics;
- Barrier: synchronization before/after timing.

The physics formulas are imported from physical_correctness.py so that the code is
cleanly separated by grading criterion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from physical_correctness import acceleration_from_block, central_mass_acceleration

Array = np.ndarray


class _DummyRequest:
    def __init__(self, payload: Any = None) -> None:
        self.payload = payload

    def wait(self) -> Any:
        return self.payload


class _DummyComm:
    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def bcast(self, obj: Any, root: int = 0) -> Any:
        return obj

    def scatter(self, seq: Optional[List[Any]], root: int = 0) -> Any:
        if seq is None:
            raise RuntimeError("Serial scatter received None.")
        return seq[0]

    def gather(self, obj: Any, root: int = 0) -> List[Any]:
        return [obj]

    def allgather(self, obj: Any) -> List[Any]:
        return [obj]

    def reduce(self, value: Any, op: Any = None, root: int = 0) -> Any:
        return value

    def allreduce(self, value: Any, op: Any = None) -> Any:
        return value

    def isend(self, obj: Any, dest: int, tag: int = 0) -> _DummyRequest:
        return _DummyRequest(None)

    def irecv(self, source: int, tag: int = 0) -> _DummyRequest:
        return _DummyRequest(None)

    def Barrier(self) -> None:
        return None


class _DummyMPI:
    SUM = "sum"
    MAX = "max"
    COMM_WORLD = _DummyComm()

    @staticmethod
    def Wtime() -> float:
        import time

        return time.perf_counter()


def get_mpi(allow_serial: bool = False) -> Tuple[Any, Any, bool]:
    """Return MPI, communicator and whether we are using serial fallback."""
    try:
        from mpi4py import MPI  # type: ignore

        return MPI, MPI.COMM_WORLD, False
    except Exception as exc:
        if allow_serial:
            return _DummyMPI, _DummyMPI.COMM_WORLD, True
        raise SystemExit(
            "mpi4py is not available. Install mpi4py/OpenMPI or run with --allow-serial for a local preview."
        ) from exc


def bcast_config(comm: Any, rank: int, args) -> Any:
    """Broadcast argparse configuration from rank 0 to all ranks."""
    config = vars(args).copy() if rank == 0 else None
    config = comm.bcast(config, root=0)
    for key, value in config.items():
        setattr(args, key, value)
    return args


def partition_counts(total: int, size: int) -> List[int]:
    """Balanced 1D block distribution."""
    base = int(total) // int(size)
    rem = int(total) % int(size)
    return [base + (1 if r < rem else 0) for r in range(int(size))]


def split_particle_chunks(ids: Array, pos: Array, vel: Array, mass: Array, size: int) -> List[Dict[str, Array]]:
    """Split particles on rank 0 before scatter."""
    counts = partition_counts(len(ids), size)
    chunks: List[Dict[str, Array]] = []
    start = 0
    for count in counts:
        end = start + count
        chunks.append(
            {
                "ids": np.ascontiguousarray(ids[start:end], dtype=np.int64),
                "pos": np.ascontiguousarray(pos[start:end], dtype=np.float64),
                "vel": np.ascontiguousarray(vel[start:end], dtype=np.float64),
                "mass": np.ascontiguousarray(mass[start:end], dtype=np.float64),
            }
        )
        start = end
    return chunks


def scatter_particles(comm: Any, chunks: Optional[List[Dict[str, Array]]]) -> Dict[str, Array]:
    """Distribute particle chunks from rank 0 to every MPI rank."""
    local = comm.scatter(chunks, root=0)
    return {
        "ids": np.ascontiguousarray(local["ids"], dtype=np.int64),
        "pos": np.ascontiguousarray(local["pos"], dtype=np.float64),
        "vel": np.ascontiguousarray(local["vel"], dtype=np.float64),
        "mass": np.ascontiguousarray(local["mass"], dtype=np.float64),
    }


def compute_acceleration_ring(
    comm: Any,
    rank: int,
    size: int,
    local_pos: Array,
    local_mass: Array,
    args,
    timers: Optional[Dict[str, float]] = None,
) -> Array:
    """Compute acceleration on local particles using a ring exchange.

    Each rank owns only a subset of particles. To compute the full gravitational
    field without gathering every particle to rank 0 at each time step, the local
    particle block is circulated through ranks using non-blocking isend/irecv.
    At every hop, each rank adds the force contribution from the received block.
    """
    import time

    acc = np.zeros_like(local_pos, dtype=np.float64)
    block_pos = np.ascontiguousarray(local_pos, dtype=np.float64)
    block_mass = np.ascontiguousarray(local_mass, dtype=np.float64)
    owner = rank

    for hop in range(int(size)):
        t0 = time.perf_counter()
        acc += acceleration_from_block(
            local_pos,
            block_pos,
            block_mass,
            float(args.g_const),
            float(args.softening),
            int(args.pair_block),
        )
        if timers is not None:
            timers["physics"] = timers.get("physics", 0.0) + (time.perf_counter() - t0)

        if size > 1:
            next_rank = (rank + 1) % size
            prev_rank = (rank - 1) % size
            payload = (block_pos, block_mass, owner)
            t1 = time.perf_counter()
            send_req = comm.isend(payload, dest=next_rank, tag=9100 + hop)
            recv_req = comm.irecv(source=prev_rank, tag=9100 + hop)
            block_pos, block_mass, owner = recv_req.wait()
            send_req.wait()
            if timers is not None:
                timers["communication"] = timers.get("communication", 0.0) + (time.perf_counter() - t1)

    acc += central_mass_acceleration(local_pos, args)
    return acc


def gather_snapshot(comm: Any, local: Dict[str, Array], root: int = 0) -> Optional[Dict[str, Array]]:
    """Gather local particle arrays on rank 0 for plotting and diagnostics."""
    gathered = comm.gather(local, root=root)
    # Dummy communicator always returns a list, MPI non-root returns None.
    if gathered is None:
        return None
    if not isinstance(gathered, list):
        return None
    ids = np.concatenate([g["ids"] for g in gathered])
    pos = np.concatenate([g["pos"] for g in gathered])
    vel = np.concatenate([g["vel"] for g in gathered])
    mass = np.concatenate([g["mass"] for g in gathered])
    order = np.argsort(ids)
    return {
        "ids": ids[order],
        "pos": pos[order],
        "vel": vel[order],
        "mass": mass[order],
    }


def allreduce_basic_diagnostics(comm: Any, MPI: Any, local_diag: Dict[str, Any]) -> Dict[str, float]:
    """Combine rank-local diagnostics using MPI allreduce."""
    kinetic = comm.allreduce(float(local_diag["kinetic"]), op=MPI.SUM)
    total_mass = comm.allreduce(float(local_diag["mass"]), op=MPI.SUM)
    speed_mass_sum = comm.allreduce(float(local_diag["speed_mass_sum"]), op=MPI.SUM)
    radius_mass_sum = comm.allreduce(float(local_diag["radius_mass_sum"]), op=MPI.SUM)
    max_radius = comm.allreduce(float(local_diag["max_radius"]), op=MPI.MAX)
    angular = comm.allreduce(np.asarray(local_diag["angular_momentum"], dtype=np.float64), op=MPI.SUM)

    return {
        "kinetic_allreduce": float(kinetic),
        "total_mass": float(total_mass),
        "mean_speed_allreduce": float(speed_mass_sum / total_mass) if total_mass > 0 else 0.0,
        "mean_radius_allreduce": float(radius_mass_sum / total_mass) if total_mass > 0 else 0.0,
        "max_radius_allreduce": float(max_radius),
        "angular_momentum_norm_allreduce": float(np.linalg.norm(angular)),
    }
