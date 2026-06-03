"""Юнит-тесты тёплого пула: освежение, истечение keep-alive, LRU-вытеснение."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.warm_pool import WarmPool


def test_empty_is_cold():
    wp = WarmPool(M=2, keep_alive=3, w_max=4)
    assert not wp.is_warm(0, 5)


def test_use_makes_warm():
    wp = WarmPool(M=2, keep_alive=3, w_max=4)
    wp.update({0: {1}})            # на узле 0 выполнили функцию типа 1
    assert wp.is_warm(0, 1)
    assert not wp.is_warm(1, 1)    # на узле 1 — нет


def test_keep_alive_expiry():
    wp = WarmPool(M=1, keep_alive=3, w_max=4)
    wp.update({0: {1}})            # возраст 1 -> 0
    # бездействуем; возраст растёт 1,2,3 — ещё тёплая (<= keep_alive)
    for _ in range(3):
        wp.update({0: set()})
    assert wp.is_warm(0, 1)
    # ещё один слот -> возраст 4 > 3 -> выселена
    wp.update({0: set()})
    assert not wp.is_warm(0, 1)


def test_refresh_resets_age():
    wp = WarmPool(M=1, keep_alive=2, w_max=4)
    wp.update({0: {1}})
    wp.update({0: set()})          # возраст 1
    wp.update({0: {1}})            # снова вызвали -> возраст 0
    wp.update({0: set()})          # 1
    wp.update({0: set()})          # 2 — всё ещё тёплая
    assert wp.is_warm(0, 1)


def test_lru_eviction():
    # w_max=2: держим максимум 2 тёплых типа, вытесняем самый старый
    wp = WarmPool(M=1, keep_alive=10, w_max=2)
    wp.update({0: {1}})            # типы: {1:0}
    wp.update({0: {2}})            # {1:1, 2:0}
    wp.update({0: {3}})            # пришёл 3 -> {1:2,2:1,3:0} -> переполнение -> выселить старейший (1)
    assert not wp.is_warm(0, 1)    # 1 вытеснен
    assert wp.is_warm(0, 2)
    assert wp.is_warm(0, 3)


def test_snapshot():
    wp = WarmPool(M=2, keep_alive=3, w_max=4)
    wp.update({0: {1, 2}, 1: {5}})
    snap = wp.snapshot()
    assert snap[0] == {1, 2}
    assert snap[1] == {5}
