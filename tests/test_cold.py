"""Тесты within-slot учёта холодного старта."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.cold import realized_cold
from model.task import Task


def _t(g):
    return Task(uid=0, g=g, D=1.0, C=1.0, R=1e9, T_max=1.0)


def test_first_cold_rest_warm_same_node_type():
    # две задачи типа 0 на холодный узел 0: первая холодная, вторая уже тёплая
    cold = realized_cold([0, 0], [_t(0), _t(0)], {0: set()})
    assert cold[0] == 1.0
    assert cold[1] == 0.0


def test_warm_node_no_cold():
    # узел 0 уже тёплый для типа 0
    cold = realized_cold([0], [_t(0)], {0: {0}})
    assert cold[0] == 0.0


def test_same_type_different_nodes_both_cold():
    # тип 0 на узлы 0 и 1, оба холодные -> обе оплачивают cold (разные контейнеры)
    cold = realized_cold([0, 1], [_t(0), _t(0)], {0: set(), 1: set()})
    assert cold[0] == 1.0 and cold[1] == 1.0


def test_different_types_same_node_both_cold():
    # типы 0 и 1 на узел 0, оба холодные -> обе cold (разные функции)
    cold = realized_cold([0, 0], [_t(0), _t(1)], {0: set()})
    assert cold[0] == 1.0 and cold[1] == 1.0


def test_local_never_cold():
    cold = realized_cold([-1], [_t(0)], {0: set()})
    assert cold[0] == 0.0
