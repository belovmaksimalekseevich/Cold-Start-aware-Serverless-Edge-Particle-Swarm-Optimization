"""
Task — одна задача (запрос на вычисление) от устройства.

Поля повторяют обозначения из §3.2 диплома + тип функции g (новое, для cold-start):
  g     — тип FaaS-функции (какую функцию надо запустить); от него зависит «тепло»
  D     — объём передаваемых данных, байт
  C     — вычислительная сложность, циклов
  R     — скорость беспроводного канала до узла, бит/с
  T_max — допустимая задержка (дедлайн), с
  uid   — id устройства-источника (нужно для локального исполнения)
"""
from dataclasses import dataclass


@dataclass
class Task:
    uid: int
    g: int
    D: float
    C: float
    R: float
    T_max: float


def sample_task(rng, cfg, gtype: int, uid: int) -> Task:
    """Сгенерировать случайную задачу заданного типа функции для устройства uid."""
    return Task(
        uid=uid,
        g=gtype,
        D=rng.uniform(cfg.D_min, cfg.D_max),
        C=rng.uniform(cfg.C_min, cfg.C_max),
        R=rng.uniform(cfg.R_min, cfg.R_max),
        T_max=rng.uniform(cfg.Tmax_min, cfg.Tmax_max),
    )
