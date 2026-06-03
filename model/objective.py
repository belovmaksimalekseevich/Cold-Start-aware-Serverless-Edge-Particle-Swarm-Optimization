"""
Оценка назначения и целевая функция F.

evaluate() — ЧИСТАЯ: по назначению (какая задача на какой узел) считает
задержку каждой задачи, флаг холодного старта и загрузку узлов.

objective() — ЧИСТАЯ: сворачивает результат evaluate() в скаляр F, который
минимизирует PSO:
    F = α·D̄ + β·ΔL + γ·E + δ_cs·CSR
где D̄ — средняя задержка (мс), ΔL — дисбаланс загрузки узлов,
E — энергия (усл.), CSR — доля холодных стартов.

Замечание о масштабе: ΔL и CSR ∈ [0,1] масштабируются ×100, чтобы быть
сопоставимыми по величине со средней задержкой в мс (десятки). Это
модельный выбор весов; абсолютная шкала не влияет на argmin при фикс. весах.
"""
import numpy as np
from .delay import offload_delay, local_delay


def evaluate(assignment, tasks, F_edge, F_loc, warm_sets, T_cold, tau, base_load=None):
    """
    Оптимизационный СУРРОГАТ (по-слотовый, по-задачный cold) — то, что минимизирует PSO.
    assignment[i] = j (узел) либо -1 (локальное исполнение).
    warm_sets = {j: set(тёплых типов)} — срез WarmPool.snapshot().
    base_load[M] — стартовая загрузка узлов (ρ-эквивалент текущего бэклога), опционально.
    Возвращает (delays[c], cold[c], load[M]) — задержки/флаги по задачам, ρ по узлам.
    """
    M = len(F_edge)
    base = np.zeros(M) if base_load is None else np.asarray(base_load, dtype=float)
    by_node = {j: [] for j in range(M)}
    local = []
    for i, j in enumerate(assignment):
        if j < 0:
            local.append(i)
        else:
            by_node[j].append(i)

    n = len(tasks)
    delays = np.zeros(n)
    cold = np.zeros(n)
    load = base.copy()

    for j, idxs in by_node.items():
        if not idxs:
            continue
        cycles = sum(tasks[i].C for i in idxs)
        rho = base[j] + cycles / (F_edge[j] * tau)
        load[j] = rho
        mean_service = float(np.mean([tasks[i].C for i in idxs])) / F_edge[j]
        wj = warm_sets[j]
        for i in idxs:
            is_cold = tasks[i].g not in wj
            delays[i] = offload_delay(tasks[i], F_edge[j], rho, mean_service,
                                      is_cold, T_cold)
            cold[i] = 1.0 if is_cold else 0.0

    for i in local:
        delays[i] = local_delay(tasks[i], F_loc[tasks[i].uid])
        cold[i] = 0.0  # локально нет FaaS-контейнера, холодного старта нет

    return delays, cold, load


def objective(delays, cold, load, cfg) -> float:
    """Скалярная целевая функция F (меньше = лучше)."""
    if len(delays) == 0:
        return 0.0
    mean_delay_ms = float(np.mean(delays)) * 1e3
    active = load[load > 0]
    imbalance = float(active.max() - active.min()) if len(active) > 1 else 0.0
    energy = float(np.sum(load)) * cfg.energy_coef
    csr = float(np.mean(cold))
    return (cfg.alpha * mean_delay_ms
            + cfg.beta * imbalance * 1e2
            + cfg.gamma * energy
            + cfg.delta_cs * csr * 1e2)
