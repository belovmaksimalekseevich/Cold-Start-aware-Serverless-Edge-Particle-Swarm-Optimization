"""
Оптимум для оценки близости CS-SE-PSO к идеалу (optimality gap).

Два инструмента:
  exact_optimum()  — ТОЧНЫЙ перебор всех назначений на малых N. Считает ТУ ЖЕ
                     целевую функцию F (model.objective.objective), что минимизирует
                     PSO → честное apples-to-apples сравнение без линеаризации.
                     Это главный аргумент строгости.
  milp_assign()    — MILP (PuLP/CBC) для средних N: минимизирует задержку
                     (передача+вычисление+холод, без очереди) + cold-rate при
                     ограничении на пиковую загрузку. Идеализированная нижняя
                     граница (очередь игнорируется → оптимистична). Масштабируемо.

Optimality gap = (F_метода − F_opt) / F_opt.
"""
import itertools
import numpy as np
import pulp

from model.objective import evaluate, objective


def exact_optimum(tasks, ctx, cfg, max_combos=300_000):
    """
    Точный минимум F перебором всех (M+1)^N назначений (узлы 0..M-1 или -1=локально).
    Возвращает (F_opt, best_assignment). Бросает, если перебор слишком большой.
    """
    n, M = len(tasks), cfg.M
    if n == 0:
        return 0.0, []
    combos = (M + 1) ** n
    if combos > max_combos:
        raise ValueError(f"Перебор слишком большой: {(M+1)}^{n}={combos} > {max_combos}. "
                         f"Используйте меньший N/M или milp_assign().")
    choices = list(range(M)) + [-1]
    base = None if ctx.prev_load is None else np.asarray(ctx.prev_load, float)
    F_opt, best = float('inf'), None
    for combo in itertools.product(choices, repeat=n):
        assign = list(combo)
        d, c, l = evaluate(assign, tasks, ctx.F_edge, ctx.F_loc, ctx.warm_sets,
                           cfg.T_cold, cfg.tau, base_load=base)
        F = objective(d, c, l, cfg)
        if F < F_opt:
            F_opt, best = F, assign
    return F_opt, best


def milp_assign(tasks, ctx, cfg, load_cap=1.2):
    """
    MILP (PuLP/CBC): минимизирует Σ (передача+вычисление+холод) + δ_cs·Σcold,
    при ограничении load_j ≤ load_cap. Очередь игнорируется → оптимистичная
    нижняя граница на задержку. Возвращает assignment (или None при неудаче).
    """
    n, M = len(tasks), cfg.M
    if n == 0:
        return []
    base = np.zeros(M) if ctx.prev_load is None else np.asarray(ctx.prev_load, float)

    prob = pulp.LpProblem("offload", pulp.LpMinimize)
    # x[i,j] = 1 если задача i на узел j; x[i,M] = локально
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
         for i in range(n) for j in range(M + 1)}

    def task_cost(i, j):
        t = tasks[i]
        if j == M:                                   # локально
            return t.C / ctx.F_loc[t.uid]
        t_trans = t.D * 8.0 / t.R
        t_comp = t.C / ctx.F_edge[j]
        cold = cfg.T_cold if t.g not in ctx.warm_sets[j] else 0.0
        return t_trans + t_comp + cold

    # цель: средняя задержка (мс) + δ_cs·CSR (масштаб как в objective)
    delay_term = pulp.lpSum(task_cost(i, j) * x[(i, j)]
                            for i in range(n) for j in range(M + 1)) * 1e3 / n * cfg.alpha
    cold_term = pulp.lpSum((1 if (j < M and tasks[i].g not in ctx.warm_sets[j]) else 0)
                           * x[(i, j)] for i in range(n) for j in range(M + 1))
    prob += delay_term + cfg.delta_cs * 1e2 * cold_term / n

    # каждая задача — ровно одно назначение
    for i in range(n):
        prob += pulp.lpSum(x[(i, j)] for j in range(M + 1)) == 1
    # ограничение пиковой загрузки узла (с учётом бэклога)
    for j in range(M):
        load_j = base[j] + pulp.lpSum(tasks[i].C / (ctx.F_edge[j] * cfg.tau) * x[(i, j)]
                                      for i in range(n))
        prob += load_j <= load_cap

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    assign = []
    for i in range(n):
        j_sel = next((j for j in range(M + 1) if pulp.value(x[(i, j)]) > 0.5), M)
        assign.append(-1 if j_sel == M else j_sel)
    return assign
