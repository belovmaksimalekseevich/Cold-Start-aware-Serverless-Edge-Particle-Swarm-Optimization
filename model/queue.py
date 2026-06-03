"""
NodeQueues — реальная дискретная очередь на каждом узле с переносом бэклога
между тайм-слотами. Это ЧЕСТНАЯ замена псевдо-M/M/1 (закрывает слабость #2).

Идея: на узле j копится невыполненная работа (бэклог B[j], в циклах). За слот
длительностью τ сервер успевает обработать F[j]·τ циклов. Задачи, назначенные
на узел в этом слоте, встают в очередь ЗА уже накопленным бэклогом (FIFO).
Перегруженный узел накапливает бэклог из слота в слот → задержка растёт реально,
а не по стационарной формуле, применённой к снимку одного слота.

Изолированный конечный автомат → тестируется отдельно (test_queue.py).
"""
import numpy as np


class NodeQueues:
    def __init__(self, M, F_edge, tau):
        self.M = M
        self.F = np.asarray(F_edge, dtype=float)   # производительность узлов, цикл/с
        self.tau = tau                             # длительность слота, с
        self.B = np.zeros(M)                       # бэклог по узлам, циклов

    def backlog_load(self):
        """ρ-эквивалент текущего бэклога: B/(F·τ). Подаётся методу как стартовая загрузка."""
        return self.B / (self.F * self.tau)

    def serve(self, assignment, tasks):
        """
        Обслужить один тайм-слот.
        assignment[i] = j (узел) либо -1 (локально, в очередь узла не идёт).
        Возвращает (wait_service[n], offered_load[M]):
          wait_service[i] — ожидание+обслуживание задачи i на её узле (с), 0 для локальных;
          offered_load[j] — реализованная загрузка узла (ρ-эквивалент) с учётом бэклога.
        Обновляет бэклог B. FIFO в порядке списка задач.
        """
        n = len(tasks)
        wait_service = np.zeros(n)
        cum = self.B.copy()            # работа впереди по каждому узлу (старт = бэклог)
        added = np.zeros(self.M)       # новая работа, добавленная в этом слоте
        for i, j in enumerate(assignment):
            if j < 0:
                continue
            Ci = tasks[i].C
            wait_service[i] = (cum[j] + Ci) / self.F[j]   # ждём всё впереди + своё обслуживание
            cum[j] += Ci
            added[j] += Ci
        offered_load = (self.B + added) / (self.F * self.tau)
        # сервер слил F·τ циклов за слот; остаток переносится
        self.B = np.maximum(0.0, self.B + added - self.F * self.tau)
        return wait_service, offered_load
