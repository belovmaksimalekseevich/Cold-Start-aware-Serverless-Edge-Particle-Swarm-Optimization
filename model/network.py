"""
Network — статичные параметры сети + генератор прибытий задач.

Хранит производительность узлов (F_edge) и устройств (F_loc), а также
популярность типов функций по Zipf (несколько функций популярны, остальные редки —
реалистично для serverless). Каждый тайм-слот выдаёт пуассоновский поток задач.
"""
import numpy as np
from .task import sample_task


class Network:
    def __init__(self, cfg, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed if seed is None else seed)
        self.F_edge = self.rng.uniform(cfg.F_edge_min, cfg.F_edge_max, cfg.M)
        self.F_loc = self.rng.uniform(cfg.F_loc_min, cfg.F_loc_max, cfg.N)
        ranks = np.arange(1, cfg.G + 1)
        p = 1.0 / ranks ** cfg.zipf_a
        self.ftype_probs = p / p.sum()

    def sample_arrivals(self, lam: float) -> list:
        """Пуассон(lam) задач за слот; тип функции по Zipf-популярности."""
        k = int(self.rng.poisson(lam))
        if k == 0:
            return []
        gtypes = self.rng.choice(self.cfg.G, size=k, p=self.ftype_probs)
        uids = self.rng.integers(0, self.cfg.N, size=k)
        return [sample_task(self.rng, self.cfg, int(g), int(u))
                for g, u in zip(gtypes, uids)]
