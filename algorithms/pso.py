"""
Бинарный PSO для задачи выгрузки — ядро SE-PSO и CS-SE-PSO (ВЕКТОРИЗОВАН).

Частица — расписание выгрузки на тайм-слот: x ∈ {0,1}^(K×n×M), строка i — one-hot
выбор узла (нулевая строка = локально). Скорость v проецируется в вероятность
сигмоидой (Kennedy&Eberhart 1997), затем сэмплируется бинарно.

Вся «горячая» часть (оценка целевой функции по всему рою и one-hot-репэйр)
векторизована через NumPy — без вложенных Python-циклов по частицам. Это ускоряет
прогон в десятки раз и делает кампанию экспериментов реальной.

SE-PSO     = warm_affinity=False, delta_cs=0  — метод диплома (cold-naive).
CS-SE-PSO  = warm_affinity=True,  delta_cs>0  — предложенный:
  (1) ~40% частиц засеваются на тёплые узлы; (2) one-hot-репэйр предпочитает
  тёплый узел (бонус в score); (3) тёплый старт роя (g_best прошлого слота).
"""
import numpy as np
from .base import OffloadingMethod

VMAX = 4.0
WARM_BONUS = 1.0   # добавка к score за тёплый узел в репэйре (warm-affinity)


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -VMAX, VMAX)))


def _schedule(cfg, q):
    if not cfg.adaptive:
        return cfg.omega, cfg.c1, cfg.c2
    frac = q / max(cfg.Q - 1, 1)
    omega = cfg.omega_start + (cfg.omega_end - cfg.omega_start) * min(frac * 2, 1.0)
    c1 = 2.0 - 1.0 * frac      # 2.0 -> 1.0
    c2 = 1.0 + 1.0 * frac      # 1.0 -> 2.0
    return omega, c1, c2


class _SlotConst:
    """Предвычисленные на слот константы для векторной оценки целевой функции."""
    def __init__(self, tasks, ctx, cfg):
        n, M = len(tasks), cfg.M
        self.n, self.M, self.cfg = n, M, cfg
        self.C = np.array([t.C for t in tasks])                  # (n,)
        self.t_trans = np.array([t.D * 8.0 / t.R for t in tasks])  # (n,)
        self.F_edge = np.asarray(ctx.F_edge)                     # (M,)
        self.local_delay = np.array([t.C / ctx.F_loc[t.uid] for t in tasks])  # (n,)
        # стартовая загрузка узлов (ρ-эквивалент текущего бэклога) — backlog-awareness
        self.base_load = (np.zeros(M) if ctx.prev_load is None
                          else np.asarray(ctx.prev_load, dtype=float))
        # warm_mask[i,j] = функция задачи i тёплая на узле j
        self.warm_mask = np.zeros((n, M))
        for i, t in enumerate(tasks):
            for j in range(M):
                if t.g in ctx.warm_sets[j]:
                    self.warm_mask[i, j] = 1.0


def _batch_objective(x, sc):
    """
    Векторная целевая функция по всему рою.
    x: (K,n,M) one-hot. Возвращает F (K,) и (для g_best) словарь реализаций.
    """
    cfg, F_edge, C = sc.cfg, sc.F_edge, sc.C
    K, n, M = x.shape
    tau = cfg.tau

    sumC = np.einsum('knm,n->km', x, C)              # (K,M) циклов на узел
    cnt = x.sum(axis=1)                              # (K,M) задач на узел
    load = sc.base_load[None, :] + sumC / (F_edge[None, :] * tau)   # (K,M) ρ + бэклог
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_service = np.where(cnt > 0, (sumC / np.maximum(cnt, 1)) / F_edge[None, :], 0.0)
    rho = np.clip(load, 0.0, 0.999)
    queue = np.where(rho > 0, rho / (1.0 - rho) * mean_service, 0.0)  # (K,M)

    A = np.argmax(x, axis=2)                         # (K,n) выбранный узел
    is_local = (x.sum(axis=2) == 0)                  # (K,n) нулевая строка
    Acl = np.clip(A, 0, M - 1)

    F_sel = F_edge[Acl]                              # (K,n)
    comp = C[None, :] / F_sel                         # (K,n)
    queue_sel = np.take_along_axis(queue, Acl, axis=1)               # (K,n)
    warm_sel = np.take_along_axis(sc.warm_mask[None, :, :].repeat(K, 0),
                                  Acl[:, :, None], axis=2)[:, :, 0]   # (K,n)
    cold = (1.0 - warm_sel)                           # (K,n) холодная?=1
    delay = sc.t_trans[None, :] + comp + queue_sel + cfg.T_cold * cold
    # локальные задачи
    delay = np.where(is_local, sc.local_delay[None, :], delay)
    cold = np.where(is_local, 0.0, cold)

    mean_delay_ms = delay.mean(axis=1) * 1e3          # (K,)
    # дисбаланс: max-min загрузки среди активных узлов
    load_max = load.max(axis=1)
    load_min_active = np.where(cnt > 0, load, np.inf).min(axis=1)
    n_active = (cnt > 0).sum(axis=1)
    imbalance = np.where(n_active > 1, load_max - load_min_active, 0.0)
    energy = load.sum(axis=1) * cfg.energy_coef
    csr = cold.mean(axis=1)

    F = (cfg.alpha * mean_delay_ms + cfg.beta * imbalance * 1e2
         + cfg.gamma * energy + cfg.delta_cs * csr * 1e2)
    return F


def _onehot_from_sample(sampled, prob, warm_mask, warm_affinity):
    """
    Из сэмплированной бинарной матрицы (K,n,M) сделать one-hot (K,n,M) + assignment.
    Нулевая строка -> локально (one-hot нулевой, A=-1). Векторизовано.
    """
    K, n, M = sampled.shape
    score = sampled * prob
    if warm_affinity:
        score = score + sampled * (WARM_BONUS * warm_mask[None, :, :])
    rowsum = sampled.sum(axis=2)                      # (K,n)
    A = np.argmax(score, axis=2)                      # (K,n)
    # one-hot
    x = np.zeros((K, n, M))
    kk, ii = np.meshgrid(np.arange(K), np.arange(n), indexing='ij')
    x[kk, ii, A] = 1.0
    # нулевые строки -> локально
    local = (rowsum == 0)
    x[local] = 0.0
    A = np.where(local, -1, A)
    return x, A


class BinaryPSO(OffloadingMethod):
    def __init__(self, warm_affinity: bool, delta_cs=None, name=None):
        self.warm_affinity = warm_affinity
        self.delta_cs_override = delta_cs
        self.name = name or ('cs-se-pso' if warm_affinity else 'se-pso')
        self._prev_best = None

    def reset(self):
        self._prev_best = None

    def decide(self, tasks, ctx):
        cfg = ctx.cfg
        if self.delta_cs_override is not None:
            cfg = cfg.variant(delta_cs=self.delta_cs_override)
        n, M, K, Q = len(tasks), cfg.M, cfg.K, cfg.Q
        if n == 0:
            return []
        rng = ctx.rng if ctx.rng is not None else np.random.default_rng()
        sc = _SlotConst(tasks, ctx, cfg)

        # --- инициализация роя ---
        sampled = (rng.random((K, n, M)) > 0.7)
        x, A = _onehot_from_sample(sampled, rng.random((K, n, M)),
                                   sc.warm_mask, self.warm_affinity)
        # warm-affinity seeding: ~40% частиц — на тёплые узлы
        if self.warm_affinity:
            n_seed = int(0.4 * K)
            for k in range(n_seed):
                for i, t in enumerate(tasks):
                    warm = np.flatnonzero(sc.warm_mask[i] > 0)
                    j = int(rng.choice(warm)) if warm.size else int(rng.integers(M))
                    x[k, i] = 0.0; x[k, i, j] = 1.0
        # тёплый старт: частица 0 = прошлое g_best
        if cfg.warm_start and self._prev_best is not None:
            x[0] = 0.0
            for i, j in enumerate(self._prev_best):
                if 0 <= j < M and i < n:
                    x[0, i, j] = 1.0

        v = np.zeros_like(x)
        f_p = _batch_objective(x, sc)
        p_best = x.copy()
        gi = int(np.argmin(f_p))
        g_best = x[gi].copy(); f_g = float(f_p[gi])

        stagn = 0
        for q in range(Q):
            omega, c1, c2 = _schedule(cfg, q)
            r1 = rng.random((K, n, M)); r2 = rng.random((K, n, M))
            v = omega * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best[None] - x)
            v = np.clip(v, -VMAX, VMAX)
            prob = _sigmoid(v)
            sampled = prob >= rng.random((K, n, M))
            x, A = _onehot_from_sample(sampled, prob, sc.warm_mask, self.warm_affinity)

            f_cur = _batch_objective(x, sc)
            improved = f_cur < f_p
            p_best[improved] = x[improved]
            f_p[improved] = f_cur[improved]
            if f_p.min() < f_g - 1e-12:
                gi = int(np.argmin(f_p)); g_best = p_best[gi].copy(); f_g = float(f_p[gi])
                stagn = 0
            else:
                stagn += 1

            if stagn >= cfg.restart_patience:        # рестарт худших 20%
                worst = np.argsort(f_p)[-max(1, K // 5):]
                rs = (rng.random((len(worst), n, M)) > 0.7)
                xw, _ = _onehot_from_sample(rs, rng.random((len(worst), n, M)),
                                            sc.warm_mask, self.warm_affinity)
                x[worst] = xw; v[worst] = 0.0
                stagn = 0

        # assignment из g_best
        gb_rowsum = g_best.sum(axis=1)
        best_assign = [int(np.argmax(g_best[i])) if gb_rowsum[i] > 0 else -1
                       for i in range(n)]
        self._prev_best = best_assign
        return best_assign


def make_se_pso():
    return BinaryPSO(warm_affinity=False, delta_cs=0.0, name='se-pso')


def make_cs_se_pso(delta_cs=0.3):
    return BinaryPSO(warm_affinity=True, delta_cs=delta_cs, name='cs-se-pso')
