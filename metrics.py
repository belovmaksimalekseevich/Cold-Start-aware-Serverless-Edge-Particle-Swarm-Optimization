"""
Агрегированные метрики по целому прогону симуляции.

Принимают накопленные за все тайм-слоты массивы и считают то, что идёт в статью:
  mean        — средняя задержка, мс
  p95, p99    — хвостовая задержка (то, что важно для URLLC), мс
  csr         — доля холодных стартов
  completion  — доля задач, уложившихся в дедлайн
  load_cv     — коэффициент вариации загрузки узлов (мера дисбаланса)
  gini        — индекс Джини загрузки (альтернативная мера неравномерности)
"""
import numpy as np


def gini(x) -> float:
    """Индекс Джини массива неотрицательных значений (0 = равномерно)."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def summarize(delays_ms, cold_flags, ok_flags, node_loads) -> dict:
    """
    delays_ms  — задержки всех задач, мс
    cold_flags — 0/1 холодный старт по задачам
    ok_flags   — 0/1 уложилась ли в дедлайн
    node_loads — загрузки узлов (по всем слотам или усреднённые)
    """
    d = np.asarray(delays_ms, dtype=float)
    loads = np.asarray(node_loads, dtype=float)
    mean_load = loads.mean() if loads.size else 0.0
    return {
        'mean': float(d.mean()) if d.size else 0.0,
        'p95': float(np.percentile(d, 95)) if d.size else 0.0,
        'p99': float(np.percentile(d, 99)) if d.size else 0.0,
        'csr': float(np.mean(cold_flags)) if len(cold_flags) else 0.0,
        'completion': float(np.mean(ok_flags)) if len(ok_flags) else 0.0,
        'load_cv': float(loads.std() / mean_load) if mean_load > 0 else 0.0,
        'gini': gini(loads),
    }
