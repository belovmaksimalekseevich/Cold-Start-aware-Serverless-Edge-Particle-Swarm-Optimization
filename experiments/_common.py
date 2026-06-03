"""
Общая инфраструктура экспериментов: параллельный прогон по seeds + кеш на диск.

Принципы:
  - воркеры — top-level функции (для multiprocessing на Windows нужны picklable);
  - методы создаются ПО ИМЕНИ (лямбды не пиклятся) — реестр make_method;
  - сырые результаты по каждому (метод, значение_оси, seed) кешируются в
    results/<exp>.csv → графики строятся БЕЗ перезапуска симуляций; добавить
    seeds можно дозапуском (считаются только недостающие строки).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from config import DEFAULT
from sim.simulator import run_once
from algorithms.baselines import GreedyBalance, Threshold, WarmGreedy
from algorithms.pso import make_se_pso, make_cs_se_pso

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
METRIC_COLS = ['mean', 'p95', 'p99', 'csr', 'completion', 'load_cv', 'gini']


def make_method(name, cfg):
    """Реестр методов по имени (для пиклящихся воркеров)."""
    if name == 'threshold':
        return Threshold()
    if name == 'greedy':
        return GreedyBalance()
    if name == 'warm-greedy':
        return WarmGreedy()
    if name == 'se-pso':
        return make_se_pso()
    if name == 'cs-se-pso':
        return make_cs_se_pso()
    if name.startswith('cs-se-pso:'):                     # cs-se-pso:0.5 -> delta_cs=0.5
        return make_cs_se_pso(float(name.split(':', 1)[1]))
    if name.startswith('pso-w:'):                          # PSO без warm-affinity, δ_cs=value
        from algorithms.pso import BinaryPSO
        return BinaryPSO(warm_affinity=False, delta_cs=float(name.split(':', 1)[1]),
                         name=name)
    raise ValueError(f"Неизвестный метод: {name}")


def _apply_axis(cfg, axis, value, base_lam):
    """Вернуть (cfg, lam) для точки сетки. axis='lam' меняет нагрузку; иначе — поле cfg."""
    if axis == 'lam':
        return cfg, float(value)
    return cfg.variant(**{axis: value}), float(base_lam)


def _worker(job):
    """Один прогон: (method, axis, value, base_lam, n_slots, seed, cfg_kw)."""
    name, axis, value, base_lam, n_slots, seed, cfg_kw = job
    cfg = DEFAULT.variant(**cfg_kw) if cfg_kw else DEFAULT
    cfg, lam = _apply_axis(cfg, axis, value, base_lam)
    method = make_method(name, cfg)
    m = run_once(method, cfg, lam, n_slots, seed)
    return {'method': name, 'value': value, 'seed': seed, 'n_slots': n_slots,
            **{k: m[k] for k in METRIC_COLS}}


def run_grid(exp, methods, axis, values, seeds, n_slots=200,
             base_lam=12, cfg_kw=None, processes=None):
    """
    Прогнать сетку (method × value × seed) параллельно, с кешем results/<exp>.csv.
    Возвращает DataFrame со всеми строками (метрики по каждому seed).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f'{exp}.csv')
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    def is_done(name, value, seed):
        if done.empty or 'n_slots' not in done.columns:
            return False
        return ((done.method == name) & (np.isclose(done.value, value)) &
                (done.seed == seed) & (done.n_slots == n_slots)).any()

    jobs = [(name, axis, v, base_lam, n_slots, s, cfg_kw or {})
            for name in methods for v in values for s in seeds
            if not is_done(name, v, s)]

    if jobs:
        nproc = processes or max(1, cpu_count() - 1)
        print(f"[{exp}] задач к прогону: {len(jobs)} на {nproc} процессах "
              f"(уже в кеше: {len(done)})")
        with Pool(nproc) as pool:
            rows = pool.map(_worker, jobs)
        new = pd.DataFrame(rows)
        done = pd.concat([done, new], ignore_index=True) if not done.empty else new
        done.to_csv(path, index=False)
    else:
        print(f"[{exp}] всё в кеше ({len(done)} строк), прогон не нужен")
    return done


def aggregate(df):
    """Свести по (method, value): среднее + 95% CI по seeds для каждой метрики."""
    out = []
    for (name, value), g in df.groupby(['method', 'value']):
        row = {'method': name, 'value': value, 'n_seeds': len(g)}
        for k in METRIC_COLS:
            vals = g[k].values
            row[k] = float(vals.mean())
            row[k + '_ci'] = (1.96 * float(vals.std(ddof=1)) / np.sqrt(len(vals))
                              if len(vals) > 1 else 0.0)
        out.append(row)
    return pd.DataFrame(out).sort_values(['method', 'value']).reset_index(drop=True)
