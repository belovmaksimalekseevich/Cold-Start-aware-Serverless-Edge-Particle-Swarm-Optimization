"""
Smoke-проверка полного метода: прогон 4 методов на одном режиме.
Не эксперимент для статьи — только убедиться, что всё стыкуется и CS-SE-PSO
ведёт себя осмысленно (ниже cold-rate и хвост, чем у cold-naive).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from config import DEFAULT
from sim.simulator import run_many
from algorithms.baselines import GreedyBalance, Threshold
from algorithms.pso import make_se_pso, make_cs_se_pso

LAM = 12
N_SLOTS = 150
SEEDS = range(5)

methods = {
    'threshold':  lambda: Threshold(),
    'greedy':     lambda: GreedyBalance(),
    'se-pso':     make_se_pso,        # cold-naive оптимизатор (метод диплома)
    'cs-se-pso':  make_cs_se_pso,     # предложенный (cold-aware)
}

print(f"Режим: lam={LAM}, slots={N_SLOTS}, seeds={len(list(SEEDS))}\n")
print(f"{'метод':<12} {'mean,мс':>8} {'p95,мс':>8} {'p99,мс':>8} "
      f"{'CSR':>6} {'compl':>7} {'load_cv':>8}")
print("-" * 62)
t0 = time.time()
for name, factory in methods.items():
    r = run_many(factory, DEFAULT, LAM, n_slots=N_SLOTS, seeds=SEEDS)
    print(f"{name:<12} {r['mean'][0]:8.1f} {r['p95'][0]:8.1f} {r['p99'][0]:8.1f} "
          f"{r['csr'][0]:6.2f} {r['completion'][0]:7.2f} {r['load_cv'][0]:8.2f}")
print(f"\nВремя: {time.time()-t0:.1f} c")
print("\nОжидаем: у cs-se-pso CSR и хвост (p95/p99) заметно НИЖЕ, чем у se-pso/greedy.")
