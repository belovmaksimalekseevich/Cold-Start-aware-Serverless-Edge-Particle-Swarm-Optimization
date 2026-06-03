"""
OffloadEnv — MDP-среда выгрузки задач для обучения DRL-бейзлайна.

Контракт как у CitiverseSimEnv: reset() -> state, step(action) -> (s, r, done, info).
Эпизод = E_SLOTS тайм-слотов; задачи слота размещаются ПО ОДНОЙ (sequential).
На каждом шаге агент видит текущую задачу + состояние узлов и выбирает узел
(0..M-1) или локальное исполнение (action = M).

Награда = − реализованная задержка размещённой задачи (мс) / scale. Задержка
считается ТОЙ ЖЕ арифметикой, что и симулятор (очередь с бэклогом + within-slot
cold), поэтому DRL обучается на честной модели и сравним с CS-SE-PSO напрямую.

build_state() вынесен наружу — его же использует DRLMethod при инференсе,
чтобы признаки строились идентично обучению.
"""
import numpy as np
from model.network import Network

REWARD_SCALE = 100.0   # масштаб награды (мс -> ~единицы)


def state_dim(cfg):
    return 2 * cfg.M + 3


def action_dim(cfg):
    return cfg.M + 1       # M узлов + локально


def build_state(task, node_load, warm_for_type, progress, cfg):
    """
    node_load: (M,) текущая ρ-загрузка узлов (бэклог + накопленное в слоте).
    warm_for_type: (M,) 0/1 — тёплая ли функция задачи на узле.
    progress: доля размещённых задач в слоте [0,1].
    """
    return np.concatenate([
        np.clip(node_load, 0.0, 2.0) / 2.0,                       # M
        np.asarray(warm_for_type, dtype=float),                   # M
        [min(task.C / cfg.C_max, 1.0),
         min(task.T_max / cfg.Tmax_max, 1.0),
         progress],
    ]).astype(np.float32)


class OffloadEnv:
    def __init__(self, cfg, seed=0, n_slots=50, lam_range=(8, 20)):
        self.cfg = cfg
        self.n_slots = n_slots
        self.lam_range = lam_range
        self.rng = np.random.default_rng(seed)
        self._seed_seq = int(seed)

    # ---------- внутреннее ----------
    def _start_slot(self):
        self.slot_tasks = self.net.sample_arrivals(self.lam)
        # пропускаем пустые слоты
        while not self.slot_tasks and self.slot_idx < self.n_slots:
            self._commit_empty_slot()
            self.slot_tasks = self.net.sample_arrivals(self.lam)
        self.ptr = 0
        self.cum = self.B.copy()            # работа впереди по узлам (старт = бэклог), циклов
        self.added = np.zeros(self.cfg.M)   # новая работа за слот
        self.started = set()                # (j,g), стартовавшие в слоте
        self.warm0 = self.wp.snapshot()     # тёплые пулы на начало слота
        self.used = {}                      # фактически выполненные (узел -> типы)

    def _commit_empty_slot(self):
        self.B = np.maximum(0.0, self.B - self.net.F_edge * self.cfg.tau)
        self.wp.update({})
        self.slot_idx += 1

    def _commit_slot(self):
        self.B = np.maximum(0.0, self.B + self.added - self.net.F_edge * self.cfg.tau)
        self.wp.update(self.used)
        self.slot_idx += 1

    def _node_load_vec(self):
        """ρ-эквивалент: (бэклог + накопленное в слоте) / (F·τ)."""
        return self.cum / (self.net.F_edge * self.cfg.tau)

    def _warm_for_type(self, g):
        return np.array([1.0 if g in self.warm0[j] else 0.0
                         for j in range(self.cfg.M)])

    def _current_state(self):
        t = self.slot_tasks[self.ptr]
        progress = self.ptr / max(len(self.slot_tasks), 1)
        return build_state(t, self._node_load_vec(), self._warm_for_type(t.g),
                           progress, self.cfg)

    # ---------- API ----------
    def reset(self):
        # новый сетевой инстанс на эпизод (разные узлы/каналы) + случайная нагрузка
        self._seed_seq += 1
        self.net = Network(self.cfg, seed=self._seed_seq)
        from model.warm_pool import WarmPool
        self.wp = WarmPool(self.cfg.M, self.cfg.keep_alive, self.cfg.w_max)
        self.B = np.zeros(self.cfg.M)
        self.lam = float(self.rng.integers(self.lam_range[0], self.lam_range[1] + 1))
        self.slot_idx = 0
        self._start_slot()
        return self._current_state()

    def step(self, action):
        cfg = self.cfg
        t = self.slot_tasks[self.ptr]
        M = cfg.M

        if action >= M:                      # локальное исполнение
            delay = t.C / self.net.F_loc[t.uid]
        else:
            j = int(action)
            wait_service = (self.cum[j] + t.C) / self.net.F_edge[j]
            is_cold = (t.g not in self.warm0[j]) and ((j, t.g) not in self.started)
            t_trans = t.D * 8.0 / t.R
            delay = t_trans + wait_service + (cfg.T_cold if is_cold else 0.0)
            # обновить состояние слота
            self.cum[j] += t.C
            self.added[j] += t.C
            self.started.add((j, t.g))
            self.used.setdefault(j, set()).add(t.g)

        reward = -delay * 1e3 / REWARD_SCALE
        self.ptr += 1

        # конец слота?
        if self.ptr >= len(self.slot_tasks):
            self._commit_slot()
            if self.slot_idx >= self.n_slots:
                return self._current_state_safe(), reward, True, {}
            self._start_slot()
            if self.slot_idx >= self.n_slots:   # хвостовые пустые слоты исчерпали эпизод
                return self._current_state_safe(), reward, True, {}

        return self._current_state(), reward, False, {}

    def _current_state_safe(self):
        # терминальное состояние: нулевой вектор корректной размерности
        if self.ptr < len(self.slot_tasks) and self.slot_idx < self.n_slots:
            return self._current_state()
        return np.zeros(state_dim(self.cfg), dtype=np.float32)
