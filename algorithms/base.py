"""
Общий интерфейс для всех методов выгрузки + контекст тайм-слота.

Любой метод реализует decide(tasks, ctx) -> assignment, где assignment[i] —
индекс узла для задачи i (или -1 = локально). Единый интерфейс делает методы
взаимозаменяемыми: симулятор и тесты работают с любым одинаково.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class SlotContext:
    """Состояние сети на текущем тайм-слоте, доступное методу при принятии решения."""
    cfg: object                 # Config
    F_edge: np.ndarray          # производительность узлов (M,)
    F_loc: np.ndarray           # производительность устройств (N,)
    warm_sets: dict             # {j: set(тёплых типов)} — срез WarmPool
    prev_load: np.ndarray       # загрузка узлов с прошлого слота (M,) — инерция
    rng: object = None          # np.random.Generator — для воспроизводимости
    warm_seed: object = None    # прошлое g_best (для тёплого старта роя), либо None


class OffloadingMethod(ABC):
    """Базовый класс метода выгрузки."""
    name = "base"

    @abstractmethod
    def decide(self, tasks, ctx: SlotContext):
        """Вернуть assignment: список длины len(tasks), значения в [-1, M)."""
        ...

    def reset(self):
        """Сбросить внутреннее состояние (например, тёплый старт) между прогонами."""
        pass
