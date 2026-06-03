#!/usr/bin/env python3
"""
mpi_parallelization.py

Camada MPI da simulação.

Métodos usados de forma justificada:
- bcast(): todos os ranks recebem a mesma configuração;
- scatter(): rank 0 distribui IDs das partículas;
- allgather(): todos ficam a saber a carga de trabalho por rank;
- isend()/irecv(): troca em anel dos blocos para calcular forças;
- gather(): rank 0 recolhe snapshots para plotting;
- reduce(): valida total de partículas;
- allreduce(): combina diagnósticos físicos globais;
- Barrier(): sincronização antes/depois das fases temporizadas.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from physical_correctness import (
    acceleration_from_block,
    central_mass_acceleration,
    generate_local_initial_conditions,
)

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
            raise RuntimeError("Serial scatter recebeu None.")
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
    """Obtém MPI real ou fallback serial apenas se --allow-serial for usado."""
    try:
        from mpi4py import MPI  # type: ignore

        return MPI, MPI.COMM_WORLD, False
    except Exception as exc:
        if allow_serial:
            return _DummyMPI, _DummyMPI.COMM_WORLD, True
        raise SystemExit(
            "mpi4py não está disponível. Ativa o ambiente py38 e instala mpi4py/openmpi, "
            "ou corre com --allow-serial apenas para pré-visualização."
        ) from exc


def bcast_config(comm: Any, rank: int, args: Any) -> Any:
    """Distribui os parâmetros da simulação a todos os processos."""
    config = vars(args).copy() if rank == 0 else None
    config = comm.bcast(config, root=0)
    for key, value in config.items():
        setattr(args, key, value)
    return args


def partition_counts(total: int, size: int) -> List[int]:
    """Distribuição 1D equilibrada por blocos."""
    total = int(total)
    size = max(1, int(size))
    base = total // size
    rem = total % size
    return [base + (1 if r < rem else 0) for r in range(size)]


def split_id_chunks(total: int, size: int) -> List[Array]:
    """Rank 0 cria apenas IDs; cada rank gera os seus dados localmente."""
    counts = partition_counts(int(total), int(size))
    chunks: List[Array] = []
    start = 0
    for count in counts:
        stop = start + count
        chunks.append(np.arange(start, stop, dtype=np.int64))
        start = stop
    return chunks


def scatter_ids(comm: Any, rank: int, size: int, total_particles: int) -> Array:
    """Distribui IDs das partículas com scatter()."""
    chunks = split_id_chunks(total_particles, size) if rank == 0 else None
    local_ids = comm.scatter(chunks, root=0)
    return np.ascontiguousarray(local_ids, dtype=np.int64)


def create_local_particles(comm: Any, rank: int, size: int, args: Any) -> Dict[str, Array]:
    """Scatter de IDs + geração paralela das condições iniciais em cada rank."""
    local_ids = scatter_ids(comm, rank, size, int(args.particles))
    ids, pos, vel, mass = generate_local_initial_conditions(local_ids, args, rank=rank)
    return {
        "ids": np.ascontiguousarray(ids, dtype=np.int64),
        "pos": np.ascontiguousarray(pos, dtype=np.float64),
        "vel": np.ascontiguousarray(vel, dtype=np.float64),
        "mass": np.ascontiguousarray(mass, dtype=np.float64),
    }


def compute_acceleration_ring(
    comm: Any,
    rank: int,
    size: int,
    local_pos: Array,
    local_mass: Array,
    args: Any,
    timers: Optional[Dict[str, float]] = None,
) -> Array:
    """
    Calcula aceleração nas partículas locais com troca em anel.

    Cada rank começa com o seu bloco. Em cada hop calcula forças desse bloco,
    envia-o ao próximo rank com isend() e recebe o bloco anterior com irecv().
    Assim todos os ranks acabam por considerar todas as partículas sem juntar tudo
    no rank 0 durante a física.
    """
    import time

    local_pos = np.ascontiguousarray(local_pos, dtype=np.float64)
    local_mass = np.ascontiguousarray(local_mass, dtype=np.float64)
    acc = np.zeros_like(local_pos, dtype=np.float64)

    block_pos = local_pos.copy()
    block_mass = local_mass.copy()
    owner = int(rank)

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

        if int(size) > 1 and hop < int(size) - 1:
            next_rank = (int(rank) + 1) % int(size)
            prev_rank = (int(rank) - 1) % int(size)
            payload = (block_pos, block_mass, owner)

            t1 = time.perf_counter()
            send_req = comm.isend(payload, dest=next_rank, tag=8100 + hop)
            recv_req = comm.irecv(source=prev_rank, tag=8100 + hop)
            block_pos, block_mass, owner = recv_req.wait()
            send_req.wait()
            block_pos = np.ascontiguousarray(block_pos, dtype=np.float64)
            block_mass = np.ascontiguousarray(block_mass, dtype=np.float64)
            if timers is not None:
                timers["communication"] = timers.get("communication", 0.0) + (time.perf_counter() - t1)

    acc += central_mass_acceleration(local_pos, args)
    return acc


def gather_snapshot(comm: Any, local: Dict[str, Array], root: int = 0) -> Optional[Dict[str, Array]]:
    """Recolhe posições/velocidades/massas no rank 0 para criar frames."""
    packed = {
        "ids": np.ascontiguousarray(local["ids"], dtype=np.int64),
        "pos": np.ascontiguousarray(local["pos"], dtype=np.float64),
        "vel": np.ascontiguousarray(local["vel"], dtype=np.float64),
        "mass": np.ascontiguousarray(local["mass"], dtype=np.float64),
    }
    gathered = comm.gather(packed, root=root)
    if gathered is None or not isinstance(gathered, list):
        return None

    ids = np.concatenate([g["ids"] for g in gathered])
    pos = np.concatenate([g["pos"] for g in gathered])
    vel = np.concatenate([g["vel"] for g in gathered])
    mass = np.concatenate([g["mass"] for g in gathered])
    order = np.argsort(ids)
    return {"ids": ids[order], "pos": pos[order], "vel": vel[order], "mass": mass[order]}


def allreduce_basic_diagnostics(comm: Any, MPI: Any, local_diag: Dict[str, Any]) -> Dict[str, float]:
    """Combina diagnósticos locais usando allreduce()."""
    kinetic = comm.allreduce(float(local_diag["kinetic"]), op=MPI.SUM)
    total_mass = comm.allreduce(float(local_diag["mass"]), op=MPI.SUM)
    speed_mass_sum = comm.allreduce(float(local_diag["speed_mass_sum"]), op=MPI.SUM)
    radius_mass_sum = comm.allreduce(float(local_diag["radius_mass_sum"]), op=MPI.SUM)
    max_radius = comm.allreduce(float(local_diag["max_radius"]), op=MPI.MAX)
    angular = comm.allreduce(np.asarray(local_diag["angular_momentum"], dtype=np.float64), op=MPI.SUM)

    if total_mass <= 0:
        mean_speed = 0.0
        mean_radius = 0.0
    else:
        mean_speed = float(speed_mass_sum / total_mass)
        mean_radius = float(radius_mass_sum / total_mass)

    return {
        "kinetic_allreduce": float(kinetic),
        "total_mass": float(total_mass),
        "mean_speed_allreduce": mean_speed,
        "mean_radius_allreduce": mean_radius,
        "max_radius_allreduce": float(max_radius),
        "angular_momentum_norm_allreduce": float(np.linalg.norm(angular)),
    }

def center_local_particles(
    comm: Any,
    MPI: Any,
    local: Dict[str, Array],
    center_velocity: bool = True,
) -> Dict[str, float]:
    """
    Corrige o centro de massa global da simulação.

    Esta operação usa allreduce() e é útil sobretudo no arranque: como cada rank
    gera partículas por amostragem aleatória, a galáxia pode começar ligeiramente
    deslocada do referencial da massa central. Recentrar as condições iniciais
    melhora a estabilidade visual/física sem retirar a paralelização.

    Se center_velocity=True, remove também a velocidade média ponderada pela massa,
    reduzindo deriva global artificial.
    """
    mass = np.asarray(local["mass"], dtype=np.float64)
    pos = np.asarray(local["pos"], dtype=np.float64)
    vel = np.asarray(local["vel"], dtype=np.float64)

    local_mass = float(np.sum(mass))
    if local_mass > 0.0:
        local_pos_moment = np.sum(pos * mass[:, None], axis=0)
        local_vel_moment = np.sum(vel * mass[:, None], axis=0)
    else:
        local_pos_moment = np.zeros(3, dtype=np.float64)
        local_vel_moment = np.zeros(3, dtype=np.float64)

    total_mass = float(comm.allreduce(local_mass, op=MPI.SUM))
    global_pos_moment = comm.allreduce(np.asarray(local_pos_moment, dtype=np.float64), op=MPI.SUM)
    global_vel_moment = comm.allreduce(np.asarray(local_vel_moment, dtype=np.float64), op=MPI.SUM)

    if total_mass <= 0.0:
        return {
            "total_mass": 0.0,
            "com_x": 0.0,
            "com_y": 0.0,
            "com_z": 0.0,
            "com_speed": 0.0,
        }

    com = np.asarray(global_pos_moment, dtype=np.float64) / total_mass
    cov = np.asarray(global_vel_moment, dtype=np.float64) / total_mass

    local["pos"] = np.ascontiguousarray(pos - com[None, :], dtype=np.float64)
    if center_velocity:
        local["vel"] = np.ascontiguousarray(vel - cov[None, :], dtype=np.float64)

    return {
        "total_mass": float(total_mass),
        "com_x": float(com[0]),
        "com_y": float(com[1]),
        "com_z": float(com[2]),
        "com_speed": float(np.linalg.norm(cov)),
    }


def aggregate_timer_values(comm: Any, MPI: Any, timer_values: Dict[str, float]) -> Dict[str, float]:
    """
    Agrega tempos por allreduce(MAX) para reportar o custo real paralelo.

    Em MPI, o tempo de uma fase deve ser interpretado como o máximo entre ranks,
    porque todos esperam pelo processo mais lento nas sincronizações.
    """
    keys = [
        "initial_conditions",
        "recentering",
        "physics",
        "communication",
        "diagnostics_and_gather",
        "energy_diagnostics",
        "plotting",
        "gif_creation",
        "total",
    ]
    out: Dict[str, float] = {}
    for key in keys:
        value = float(timer_values.get(key, 0.0))
        out[key] = float(comm.allreduce(value, op=MPI.MAX))
    return out

