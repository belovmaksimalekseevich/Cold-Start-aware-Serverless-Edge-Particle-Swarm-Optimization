"""Тесты оптимума: точный перебор, MILP-допустимость, gap CS-SE-PSO."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import SMALL
from model.network import Network
from model.warm_pool import WarmPool
from model.objective import evaluate, objective
from algorithms.base import SlotContext
from algorithms.milp import exact_optimum, milp_assign


def _ctx(cfg, seed=0):
    net = Network(cfg, seed=seed)
    wp = WarmPool(cfg.M, cfg.keep_alive, cfg.w_max)
    rng = np.random.default_rng(seed)
    tasks = net.sample_arrivals(lam=4)
    while len(tasks) < 3:
        tasks = net.sample_arrivals(lam=4)
    ctx = SlotContext(cfg=cfg, F_edge=net.F_edge, F_loc=net.F_loc,
                      warm_sets=wp.snapshot(), prev_load=np.zeros(cfg.M), rng=rng)
    return ctx, tasks[:4]   # ограничим N для перебора


def test_exact_is_minimum():
    # точный оптимум должен быть <= F любого случайного назначения
    cfg = SMALL
    ctx, tasks = _ctx(cfg, seed=1)
    F_opt, best = exact_optimum(tasks, ctx, cfg)
    rng = np.random.default_rng(0)
    for _ in range(50):
        assign = [int(rng.integers(-1, cfg.M)) for _ in tasks]
        d, c, l = evaluate(assign, tasks, ctx.F_edge, ctx.F_loc, ctx.warm_sets,
                           cfg.T_cold, cfg.tau)
        F = objective(d, c, l, cfg)
        assert F_opt <= F + 1e-9


def test_exact_assignment_valid():
    cfg = SMALL
    ctx, tasks = _ctx(cfg, seed=2)
    F_opt, best = exact_optimum(tasks, ctx, cfg)
    assert len(best) == len(tasks)
    assert all(-1 <= j < cfg.M for j in best)


def test_milp_feasible_and_valid():
    cfg = SMALL
    ctx, tasks = _ctx(cfg, seed=3)
    assign = milp_assign(tasks, ctx, cfg, load_cap=2.0)
    assert assign is not None
    assert len(assign) == len(tasks)
    assert all(-1 <= j < cfg.M for j in assign)


def test_cs_se_pso_near_optimum():
    # CS-SE-PSO должен быть в разумных % от точного оптимума на малой задаче
    from algorithms.pso import make_cs_se_pso
    cfg = SMALL
    ctx, tasks = _ctx(cfg, seed=4)
    F_opt, _ = exact_optimum(tasks, ctx, cfg)
    m = make_cs_se_pso()
    assign = m.decide(tasks, ctx)
    d, c, l = evaluate(assign, tasks, ctx.F_edge, ctx.F_loc, ctx.warm_sets,
                       cfg.T_cold, cfg.tau)
    F_pso = objective(d, c, l, cfg)
    gap = (F_pso - F_opt) / (abs(F_opt) + 1e-9)
    assert gap < 0.5, f"gap слишком большой: {gap:.2%}"  # мягкий порог для N=4
