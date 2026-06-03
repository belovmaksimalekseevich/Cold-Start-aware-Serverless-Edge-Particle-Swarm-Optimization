"""
E4 — Парето-граница «баланс нагрузки vs cold-start rate».

Один оптимизатор БЕЗ warm-affinity, варьируем только вес δ_cs целевой функции
(от 0 = cold-naive до большого = сильный приоритет тепла). С ростом δ_cs метод
консолидирует одинаковые функции → CSR падает, но дисбаланс нагрузки (load_cv)
растёт. Это изолирует конфликт В ВЕСАХ (β·ΔL vs δ_cs·CSR) — нетривиальная находка:
оптимум зависит от рабочей точки оператора, а не предопределён.

    python experiments/e4_pareto.py [--quick]
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from experiments._common import run_grid, aggregate

DELTAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
METHODS = [f'pso-w:{d}' for d in DELTAS]


def main(quick=False):
    seeds = range(8) if quick else range(100)
    n_slots = 100 if quick else 200
    t0 = time.time()
    df = run_grid('e4', METHODS, axis='lam', values=[12],
                  seeds=seeds, n_slots=n_slots)
    agg = aggregate(df)
    print(f"\nГотово за {time.time()-t0:.1f} c. Парето «дисбаланс vs CSR»:\n")
    print(f"{'δ_cs':>6} | {'load_cv':>8} {'CSR':>6} {'mean,мс':>8} {'p95':>7}")
    for d, m in zip(DELTAS, METHODS):
        r = agg[agg.method == m]
        if len(r):
            r = r.iloc[0]
            print(f"{d:6.2f} | {r['load_cv']:8.3f} {r['csr']:6.3f} "
                  f"{r['mean']:8.1f} {r['p95']:7.1f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    main(args.quick)
