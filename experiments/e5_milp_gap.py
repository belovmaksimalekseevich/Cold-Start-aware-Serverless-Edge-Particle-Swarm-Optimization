"""
E5: Optimality gap — CS-SE-PSO vs exact optimum (brute-force same surrogate F).

Методология:
  - Маленький сценарий (M=4, λ=4.0): 5^8=390K > 300K, так что
    пропускаем слоты с k>7 (P≈2%). Для k≤7: 5^7=78125 ≤ 300K.
  - Apples-to-apples: exact_optimum и PSO минимизируют ОДНУ И ТУ ЖЕ
    суррогатную функцию F — честное сравнение без линеаризации.
  - Дополнительно: realized delay gap (cs-se-pso vs optimal assignment
    в одинаковом состоянии очереди, counterfactual).

Параметры прогона (VM, 3 core):
  M=4, G=12, N=50 (устройств), λ=4.0, 100 seeds × 200 слотов
  Ожидаемое время: ~3-5 минут.

Выводит:
  surrogate gap %  : (F_pso - F_opt) / |F_opt| × 100
  realized delay % : (D_pso - D_opt) / D_opt × 100   [ms]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from config import DEFAULT
from model.network import Network
from model.warm_pool import WarmPool
from model.queue import NodeQueues
from sim.realized import evaluate_realized
from model.objective import evaluate as surr_eval, objective as surr_obj
from algorithms.pso import make_cs_se_pso, make_se_pso
from algorithms.milp import exact_optimum
from algorithms.base import SlotContext

# ── конфигурация эксперимента ──────────────────────────────────────────────
# M=4, MAX_K=5: (M+1)^k = 5^5 = 3125 комбо макс.
# λ=3.0: P(k≤5)≈0.916, покрываем ~92% слотов
# 50 seeds × 200 слотов ≈ 2-3 минуты на VM (3 core)
GAP_CFG = DEFAULT.variant(M=4, G=12)
LAM      = 3.0
N_SLOTS  = 200
SEEDS    = list(range(50))
MAX_K    = 5       # 5^5=3125 ≤ 300K; быстро в Python

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'results')


def _run_seed(args):
    """
    Один seed: 200 слотов, per-slot сравнение CS-SE-PSO vs exact_optimum.
    Возвращает dict с накопленными метриками этого seed.
    """
    method_name, seed = args
    cfg   = GAP_CFG
    net   = Network(cfg, seed=seed)
    wp    = WarmPool(cfg.M, cfg.keep_alive, cfg.w_max)
    queues = NodeQueues(cfg.M, net.F_edge, cfg.tau)
    rng   = np.random.default_rng(seed + 10_000)

    if method_name == 'cs-se-pso':
        method = make_cs_se_pso()
    else:
        method = make_se_pso()
    method.reset()

    surr_gaps, delay_gaps = [], []
    n_slots_used = 0
    n_slots_skipped = 0

    for _ in range(N_SLOTS):
        tasks = net.sample_arrivals(LAM)
        if not tasks:
            queues.serve([], [])
            continue

        k = len(tasks)
        if k > MAX_K:
            # слишком много комбо — продвигаем симуляцию методом PSO (без gap)
            warm_snap = wp.snapshot()
            prev_load = queues.backlog_load()
            ctx = SlotContext(cfg=cfg, F_edge=net.F_edge, F_loc=net.F_loc,
                              warm_sets=warm_snap, prev_load=prev_load, rng=rng)
            assign = method.decide(tasks, ctx)
            evaluate_realized(assign, tasks, queues, net.F_loc, warm_snap, cfg.T_cold)
            used = {}
            for t, j in zip(tasks, assign):
                if j >= 0:
                    used.setdefault(j, set()).add(t.g)
            wp.update(used)
            n_slots_skipped += 1
            continue

        warm_snap  = wp.snapshot()
        prev_load  = queues.backlog_load()
        ctx = SlotContext(cfg=cfg, F_edge=net.F_edge, F_loc=net.F_loc,
                          warm_sets=warm_snap, prev_load=prev_load, rng=rng)

        # ── PSO decision ─────────────────────────────────────────────────
        assign_pso = method.decide(tasks, ctx)

        # ── Exact optimum (same surrogate F, apples-to-apples) ───────────
        try:
            F_opt, assign_opt = exact_optimum(tasks, ctx, cfg)
        except ValueError:
            # на всякий случай
            n_slots_skipped += 1
            queues.serve(assign_pso, tasks)
            continue

        # ── Surrogate gap ────────────────────────────────────────────────
        d_p, c_p, l_p = surr_eval(assign_pso, tasks, ctx.F_edge, ctx.F_loc,
                                   ctx.warm_sets, cfg.T_cold, cfg.tau, prev_load)
        F_pso = surr_obj(d_p, c_p, l_p, cfg)
        gap_f = (F_pso - F_opt) / max(abs(F_opt), 1e-9)
        surr_gaps.append(gap_f)

        # ── Realized delay gap (counterfactual, same B snapshot) ─────────
        B_snap = queues.B.copy()
        # PSO realized
        delays_pso, _, _ = evaluate_realized(assign_pso, tasks, queues,
                                             net.F_loc, warm_snap, cfg.T_cold)
        mean_pso = float(np.mean(delays_pso)) * 1e3   # ms

        # Optimal realized (restore B, then run opt assignment)
        queues.B = B_snap.copy()
        delays_opt, _, _ = evaluate_realized(assign_opt, tasks, queues,
                                             net.F_loc, warm_snap, cfg.T_cold)
        mean_opt = float(np.mean(delays_opt)) * 1e3   # ms

        # Restore B to PSO-progressed state (sim follows PSO)
        queues.B = B_snap.copy()
        queues.serve(assign_pso, tasks)

        if mean_opt > 1e-9:
            delay_gaps.append((mean_pso - mean_opt) / mean_opt)

        # ── Обновить warm pool по PSO (что реально произошло) ────────────
        used = {}
        for t, j in zip(tasks, assign_pso):
            if j >= 0:
                used.setdefault(j, set()).add(t.g)
        wp.update(used)
        n_slots_used += 1

    return {
        'method':         method_name,
        'seed':           seed,
        'surr_gap_mean':  float(np.mean(surr_gaps))  if surr_gaps  else float('nan'),
        'delay_gap_mean': float(np.mean(delay_gaps)) if delay_gaps else float('nan'),
        'n_slots_used':   n_slots_used,
        'n_slots_skipped': n_slots_skipped,
    }


def main():
    import time
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'e5_gap.csv')

    jobs = [('cs-se-pso', s) for s in SEEDS] + [('se-pso', s) for s in SEEDS]
    nproc = max(1, cpu_count() - 1)
    print(f"E5: {len(jobs)} jobs, {nproc} процессов, M={GAP_CFG.M}, λ={LAM}, "
          f"{N_SLOTS} slots, {len(SEEDS)} seeds")

    t0 = time.time()
    with Pool(nproc) as pool:
        rows = pool.map(_run_seed, jobs)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    elapsed = time.time() - t0

    print(f"\nГотово за {elapsed:.1f} с\n")
    print(f"{'Метод':<12}  {'Surrogate gap %':>16}  {'Delay gap %':>12}  "
          f"{'Slots used':>10}")
    print('-' * 60)
    for method in ['cs-se-pso', 'se-pso']:
        sub = df[df.method == method]
        sg  = sub.surr_gap_mean.mean()  * 100
        sg_ci = 1.96 * sub.surr_gap_mean.std(ddof=1) / np.sqrt(len(sub)) * 100
        dg  = sub.delay_gap_mean.mean() * 100
        dg_ci = 1.96 * sub.delay_gap_mean.std(ddof=1) / np.sqrt(len(sub)) * 100
        sl  = sub.n_slots_used.mean()
        print(f"{method:<12}  {sg:>+7.2f} ± {sg_ci:.2f}%  {dg:>+7.2f} ± {dg_ci:.2f}%  {sl:.0f}")

    skip_frac = df.n_slots_skipped.mean() / N_SLOTS * 100
    print(f"\nПропущено слотов (k>{MAX_K}): {skip_frac:.1f}%")
    print(f"Результаты: {out_path}")


if __name__ == '__main__':
    main()
