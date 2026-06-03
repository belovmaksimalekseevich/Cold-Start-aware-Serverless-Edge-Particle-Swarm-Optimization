"""Юнит-тесты evaluate() и objective()."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import Config
from model.task import Task
from model.objective import evaluate, objective


def _tasks():
    # 2 задачи: одинаковый тип функции (g=0), разные узлы по назначению
    return [Task(uid=0, g=0, D=50e3, C=5e7, R=50e6, T_max=0.1),
            Task(uid=1, g=0, D=50e3, C=5e7, R=50e6, T_max=0.1)]


def test_evaluate_shapes_and_load():
    F_edge = np.array([5e9, 5e9])
    F_loc = np.array([0.3e9, 0.3e9])
    warm = {0: set(), 1: set()}
    delays, cold, load = evaluate([0, 1], _tasks(), F_edge, F_loc, warm,
                                  T_cold=0.05, tau=0.1)
    assert delays.shape == (2,) and cold.shape == (2,) and load.shape == (2,)
    assert np.all(delays > 0)
    assert load[0] > 0 and load[1] > 0    # обе задачи выгружены


def test_cold_flag_and_penalty():
    F_edge = np.array([5e9, 5e9]); F_loc = np.array([0.3e9, 0.3e9])
    # узел 0 тёплый для g=0, узел 1 холодный
    warm = {0: {0}, 1: set()}
    d_warm, c_warm, _ = evaluate([0], [_tasks()[0]], F_edge, F_loc, warm, 0.05, 0.1)
    d_cold, c_cold, _ = evaluate([1], [_tasks()[0]], F_edge, F_loc, warm, 0.05, 0.1)
    assert c_warm[0] == 0.0 and c_cold[0] == 1.0
    # холодная задержка больше тёплой ровно на T_cold (узлы идентичны)
    assert abs((d_cold[0] - d_warm[0]) - 0.05) < 1e-9


def test_local_no_cold():
    F_edge = np.array([5e9, 5e9]); F_loc = np.array([0.3e9, 0.3e9])
    warm = {0: set(), 1: set()}
    delays, cold, load = evaluate([-1], [_tasks()[0]], F_edge, F_loc, warm, 0.05, 0.1)
    assert cold[0] == 0.0          # локально холодного старта нет
    assert np.all(load == 0)       # узлы не загружены


def test_objective_increases_with_cold_when_weighted():
    cfg = Config(delta_cs=0.3)
    F_edge = np.array([5e9, 5e9]); F_loc = np.array([0.3e9, 0.3e9])
    tasks = _tasks()
    # вариант A: обе на тёплый узел 0
    warmA = {0: {0}, 1: set()}
    dA, cA, lA = evaluate([0, 0], tasks, F_edge, F_loc, warmA, 0.05, 0.1)
    # вариант B: обе на холодный узел 1
    warmB = {0: {0}, 1: set()}
    dB, cB, lB = evaluate([1, 1], tasks, F_edge, F_loc, warmB, 0.05, 0.1)
    FA = objective(dA, cA, lA, cfg)
    FB = objective(dB, cB, lB, cfg)
    assert FB > FA                 # холодные старты ухудшают F


def test_objective_empty():
    cfg = Config()
    assert objective(np.array([]), np.array([]), np.zeros(2), cfg) == 0.0
