"""
E1 + E2 — задержка (mean/p95/p99) и cold-start rate vs интенсивность λ.

Один прогон сетки покрывает оба эксперимента (E2 = та же сетка, метрика csr).
Все методы; результаты с 95% CI по seeds; кеш в results/e1.csv.

    python experiments/e1_latency.py            # полный (100 seeds)
    python experiments/e1_latency.py --quick    # быстрый (8 seeds, проверка)
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from experiments._common import run_grid, aggregate

METHODS = ['threshold', 'greedy', 'se-pso', 'cs-se-pso']
LAMBDAS = [5, 8, 12, 16, 20, 25, 30, 40]


def main(quick=False):
    seeds = range(8) if quick else range(100)
    n_slots = 100 if quick else 200
    t0 = time.time()
    df = run_grid('e1', METHODS, axis='lam', values=LAMBDAS,
                  seeds=seeds, n_slots=n_slots)
    agg = aggregate(df)
    print(f"\nГотово за {time.time()-t0:.1f} c. Строк: {len(df)}, seeds/точку: "
          f"{df.groupby(['method','value']).size().min()}")
    # краткая сводка: задержка и CSR при двух уровнях нагрузки
    for lam in (12, 30):
        print(f"\n--- λ={lam} ---")
        sub = agg[agg.value == lam]
        print(f"{'метод':<11} {'mean':>7} {'p95':>7} {'p99':>7} {'CSR':>6}")
        for _, r in sub.iterrows():
            print(f"{r['method']:<11} {r['mean']:7.1f} {r['p95']:7.1f} "
                  f"{r['p99']:7.1f} {r['csr']:6.2f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    main(args.quick)
