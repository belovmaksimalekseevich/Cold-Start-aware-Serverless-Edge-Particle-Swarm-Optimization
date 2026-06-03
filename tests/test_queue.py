"""Тесты реальной очереди: обслуживание, перенос бэклога, дренаж."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from model.queue import NodeQueues
from model.task import Task

EPS = 1e-9


def _task(C):
    return Task(uid=0, g=0, D=0.0, C=C, R=1e9, T_max=1.0)


def test_single_task_no_backlog():
    # узел F=10, tau=1 -> ёмкость 10 циклов/слот. Задача C=5 -> wait+service = 5/10 = 0.5 c
    q = NodeQueues(M=1, F_edge=[10.0], tau=1.0)
    ws, load = q.serve([0], [_task(5.0)])
    assert abs(ws[0] - 0.5) < EPS
    assert abs(load[0] - 0.5) < EPS
    assert q.B[0] == 0.0                 # 5 < 10, всё обслужено


def test_overload_builds_backlog():
    # C=20 при ёмкости 10 -> wait+service=20/10=2 c; бэклог после = 20-10 = 10
    q = NodeQueues(M=1, F_edge=[10.0], tau=1.0)
    ws, load = q.serve([0], [_task(20.0)])
    assert abs(ws[0] - 2.0) < EPS
    assert abs(q.B[0] - 10.0) < EPS
    assert abs(load[0] - 2.0) < EPS      # ρ-эквивалент = 20/10 = 2.0


def test_backlog_drains_when_idle():
    # стартуем с бэклогом, пустой слот -> бэклог уменьшается на F*tau
    q = NodeQueues(M=1, F_edge=[10.0], tau=1.0)
    q.B[0] = 25.0
    ws, load = q.serve([], [])
    assert abs(q.B[0] - 15.0) < EPS      # 25 - 10 = 15


def test_fifo_behind_backlog():
    # бэклог 10, приходит задача C=5 -> ждёт бэклог + своё = (10+5)/10 = 1.5 c
    q = NodeQueues(M=1, F_edge=[10.0], tau=1.0)
    q.B[0] = 10.0
    ws, _ = q.serve([0], [_task(5.0)])
    assert abs(ws[0] - 1.5) < EPS


def test_two_tasks_fifo_order():
    # две задачи на узел: вторая ждёт первую. C1=3, C2=4, F=10
    q = NodeQueues(M=1, F_edge=[10.0], tau=1.0)
    ws, _ = q.serve([0, 0], [_task(3.0), _task(4.0)])
    assert abs(ws[0] - 3.0 / 10) < EPS           # 0.3
    assert abs(ws[1] - (3.0 + 4.0) / 10) < EPS   # 0.7


def test_local_ignored():
    q = NodeQueues(M=2, F_edge=[10.0, 10.0], tau=1.0)
    ws, load = q.serve([-1, 1], [_task(5.0), _task(5.0)])
    assert ws[0] == 0.0                  # локальная — на узел не идёт
    assert ws[1] > 0.0
    assert q.B[0] == 0.0
