"""
Симулятор: прогон метода по N_SLOTS тайм-слотам и сбор метрик.

Каждый тайм-слот:
  1) сеть генерирует прибытия задач (Пуассон + Zipf-типы);
  2) метод принимает решение decide(tasks, ctx) — куда какую задачу;
  3) evaluate() считает РЕАЛИЗОВАННЫЕ задержки/холод/загрузку по этому решению;
  4) тёплый пул обновляется по фактически выполненным (узел, тип);
  5) загрузка переносится в следующий слот с инерцией (плавная динамика).
Накопленные за все слоты величины сворачиваются metrics.summarize().

Один прогон = один seed. run_many() усредняет по нескольким seed с CI.
"""
import numpy as np
from model.network import Network
from model.warm_pool import WarmPool
from model.queue import NodeQueues
from sim.realized import evaluate_realized
from metrics import summarize
from algorithms.base import SlotContext


def run_once(method, cfg, lam, n_slots, seed):
    """
    Один прогон. Метрики РЕАЛИСТИЧНЫЕ: реальная очередь с межслотовым бэклогом
    + within-slot cold. Методам подаётся текущий бэклог как стартовая загрузка
    (backlog-awareness), что и видит реальный MEC-контроллер.
    """
    net = Network(cfg, seed=seed)
    wp = WarmPool(cfg.M, cfg.keep_alive, cfg.w_max)
    queues = NodeQueues(cfg.M, net.F_edge, cfg.tau)
    rng = np.random.default_rng(seed + 10_000)   # отдельный поток для метода
    method.reset()

    all_delays, all_cold, all_ok = [], [], []
    load_acc = np.zeros(cfg.M)
    load_cnt = 0

    for _ in range(n_slots):
        tasks = net.sample_arrivals(lam)
        if not tasks:
            queues.serve([], [])          # узлы продолжают сливать бэклог
            continue

        prev_load = queues.backlog_load()  # текущий бэклог как стартовая загрузка
        ctx = SlotContext(cfg=cfg, F_edge=net.F_edge, F_loc=net.F_loc,
                          warm_sets=wp.snapshot(), prev_load=prev_load, rng=rng)
        assign = method.decide(tasks, ctx)

        delays, cold, load = evaluate_realized(assign, tasks, queues, net.F_loc,
                                               wp.snapshot(), cfg.T_cold)

        # обновить тёплый пул по фактически выполненным (узел, тип)
        used = {}
        for t, j in zip(tasks, assign):
            if j >= 0:
                used.setdefault(j, set()).add(t.g)
        wp.update(used)

        # накопить метрики
        for i, t in enumerate(tasks):
            all_delays.append(delays[i] * 1e3)            # мс
            all_cold.append(cold[i])
            all_ok.append(1.0 if delays[i] <= t.T_max else 0.0)
        active = load[load > 0]
        if active.size:
            load_acc += load
            load_cnt += 1

    mean_loads = (load_acc / load_cnt) if load_cnt else np.zeros(cfg.M)
    return summarize(all_delays, all_cold, all_ok, mean_loads)


def run_many(method_factory, cfg, lam, n_slots=300, seeds=range(8)):
    """
    Усреднить метрики по нескольким seed. method_factory — функция, создающая
    свежий экземпляр метода (чтобы тёплый старт не протекал между прогонами).
    Возвращает {метрика: (mean, ci95)}.
    """
    runs = [run_once(method_factory(), cfg, lam, n_slots, s) for s in seeds]
    keys = runs[0].keys()
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in runs])
        mean = float(vals.mean())
        ci = 1.96 * float(vals.std(ddof=1)) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        out[k] = (mean, ci)
    return out
