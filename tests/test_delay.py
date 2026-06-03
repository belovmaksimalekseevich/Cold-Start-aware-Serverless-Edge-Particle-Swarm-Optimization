"""Юнит-тесты чистых формул задержки — числа посчитаны руками."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.delay import (t_transmission, t_computation, t_queue, t_cold,
                         offload_delay, local_delay)
from model.task import Task

EPS = 1e-9


def test_transmission():
    # 500 КБ = 500e3 байт = 4e6 бит; канал 200 Мбит/с -> 0.02 с
    assert abs(t_transmission(500e3, 200e6) - 0.02) < EPS


def test_computation():
    # 1e9 циклов на 5 ГГц -> 0.2 с
    assert abs(t_computation(1e9, 5e9) - 0.2) < EPS


def test_queue_basic():
    # ρ=0.5, среднее обслуживание 0.01 с -> Wq = 0.5/0.5 * 0.01 = 0.01 с
    assert abs(t_queue(0.5, 0.01) - 0.01) < EPS
    # ρ=0 -> очереди нет
    assert t_queue(0.0, 0.01) == 0.0
    # ρ растёт -> Wq растёт монотонно
    assert t_queue(0.8, 0.01) > t_queue(0.5, 0.01)


def test_queue_clamp():
    # ρ>=1 не должно давать inf/NaN (clamp до 0.999)
    v = t_queue(1.5, 0.01)
    assert v > 0 and v < float('inf')


def test_cold():
    assert t_cold(True, 0.05) == 0.05
    assert t_cold(False, 0.05) == 0.0


def test_offload_sum():
    task = Task(uid=0, g=0, D=500e3, C=1e9, R=200e6, T_max=0.05)
    # ρ=0.5, mean_service=0.01, тёплая
    warm = offload_delay(task, 5e9, 0.5, 0.01, is_cold=False, T_cold=0.05)
    cold = offload_delay(task, 5e9, 0.5, 0.01, is_cold=True,  T_cold=0.05)
    # ожидаем: 0.02 (trans) + 0.2 (comp) + 0.01 (queue) = 0.23 тёплая
    assert abs(warm - 0.23) < EPS
    # холодная = тёплая + штраф
    assert abs(cold - (warm + 0.05)) < EPS


def test_local():
    task = Task(uid=0, g=0, D=500e3, C=1e9, R=200e6, T_max=0.05)
    # локально на 0.3 ГГц -> 1e9/0.3e9 ≈ 3.333 с
    assert abs(local_delay(task, 0.3e9) - (1e9 / 0.3e9)) < EPS
