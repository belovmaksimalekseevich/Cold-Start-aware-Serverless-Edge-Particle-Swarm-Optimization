"""
Baselines for comparison.

GreedyBalance  - cold-naive: route each task to least-queued node.
Threshold      - offload only if local execution exceeds deadline.
WarmGreedy     - warm-aware greedy: prefer warm nodes, fall back to least-loaded.
"""
import numpy as np
from .base import OffloadingMethod
from model.delay import t_queue, local_delay


class GreedyBalance(OffloadingMethod):
    name = "greedy"

    def decide(self, tasks, ctx):
        cfg = ctx.cfg
        load = ctx.prev_load.copy()
        assign = []
        for t in tasks:
            best_j, best_cost = 0, float("inf")
            for j in range(cfg.M):
                mean_service = t.C / ctx.F_edge[j]
                cost = t_queue(load[j], mean_service)
                if cost < best_cost:
                    best_cost, best_j = cost, j
            assign.append(best_j)
            load[best_j] += t.C / (ctx.F_edge[best_j] * cfg.tau)
        return assign


class Threshold(OffloadingMethod):
    name = "threshold"

    def decide(self, tasks, ctx):
        cfg = ctx.cfg
        load = ctx.prev_load.copy()
        assign = []
        for t in tasks:
            if local_delay(t, ctx.F_loc[t.uid]) <= t.T_max:
                assign.append(-1)
            else:
                j = int(np.argmin(load))
                assign.append(j)
                load[j] += t.C / (ctx.F_edge[j] * cfg.tau)
        return assign


class WarmGreedy(OffloadingMethod):
    """Warm-aware greedy: prefer nodes where the function type is already warm.
    If no warm node exists for the requested function type, falls back to
    least-loaded (identical to GreedyBalance).
    Purpose: isolates the contribution of warm-awareness from PSO optimisation.
    """
    name = "warm_greedy"

    def decide(self, tasks, ctx):
        cfg = ctx.cfg
        load = ctx.prev_load.copy()
        assign = []
        for t in tasks:
            warm_nodes = [j for j in range(cfg.M)
                          if t.g in ctx.warm_sets.get(j, set())]
            if warm_nodes:
                best_j, best_cost = warm_nodes[0], float("inf")
                for j in warm_nodes:
                    mean_service = t.C / ctx.F_edge[j]
                    cost = t_queue(load[j], mean_service)
                    if cost < best_cost:
                        best_cost, best_j = cost, j
            else:
                best_j = int(np.argmin(load))
            assign.append(best_j)
            load[best_j] += t.C / (ctx.F_edge[best_j] * cfg.tau)
        return assign
