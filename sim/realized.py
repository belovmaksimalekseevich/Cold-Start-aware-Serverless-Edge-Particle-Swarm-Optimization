"""
evaluate_realized — РЕАЛИСТИЧНАЯ оценка назначения для ОТЧЁТНЫХ метрик.

В отличие от model.objective.evaluate (оптимизационный суррогат, который
минимизирует PSO: по-слотовый, по-задачный cold, быстрый), здесь используется:
  - реальная очередь с межслотовым бэклогом (NodeQueues, слабость #2);
  - within-slot учёт холодного старта (realized_cold, слабость #4).
Именно эти числа идут в результаты статьи. Разделение «суррогат ≠ метрика» —
осознанное (как в методологии измеряемых DRL-работ).

Задержка задачи на узле = передача + (ожидание_в_очереди + обслуживание) + холод.
Локальная задача = C / F_loc (в очередь узла не идёт, холода нет).
"""
import numpy as np
from model.cold import realized_cold


def evaluate_realized(assignment, tasks, queues, F_loc, warm_sets, T_cold):
    """
    queues: NodeQueues (мутируется — продвигает бэклог).
    Возвращает (delays[c] сек, cold[c], offered_load[M] ρ-эквивалент).
    """
    n = len(tasks)
    cold = realized_cold(assignment, tasks, warm_sets)
    wait_service, offered_load = queues.serve(assignment, tasks)

    delays = np.zeros(n)
    for i, (t, j) in enumerate(zip(tasks, assignment)):
        if j < 0:
            delays[i] = t.C / F_loc[t.uid]
        else:
            t_trans = t.D * 8.0 / t.R
            delays[i] = t_trans + wait_service[i] + (T_cold if cold[i] else 0.0)
    return delays, cold, offered_load
