"""
E3 — sweep по стоимости холодного старта T_cold (при фиксированной λ).

Ключевой график: cold-naive методы (threshold, greedy, se-pso) деградируют ~линейно
с ростом T_cold, а cs-se-pso держит почти плоскую задержку; хвост (p95/p99) расходится
сильнее среднего. Демонстрирует «холод дорожает -> выигрыш растёт».

    python experiments/e3_tcold.py [--quick]
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from experiments._common import run_grid, aggregate

METHODS = ['threshold', 'greedy', 'se-pso', 'cs-se-pso']
TCOLD = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]   # секунды


def main(quick=False):
    seeds = range(8) if quick else range(100)
    n_slots = 100 if quick else 200
    t0 = time.time()
    df = run_grid('e3', METHODS, axis='T_cold', values=TCOLD,
                  seeds=seeds, n_slots=n_slots, base_lam=12)
    agg = aggregate(df)
    print(f"\nГотово за {time.time()-t0:.1f} c. seeds/точку: "
          f"{df.groupby(['method','value']).size().min()}")
    print(f"\n{'T_cold':>7} | " + " | ".join(f"{m[:9]:>9}" for m in METHODS) + "   (mean, мс)")
    for tc in TCOLD:
        row = f"  {tc*1e3:5.0f}мс |"
        for m in METHODS:
            v = agg[(agg.method == m) & (abs(agg.value - tc) < 1e-9)]['mean']
            row += f" {v.iloc[0]:9.1f} |" if len(v) else f" {'—':>9} |"
        print(row)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    main(args.quick)
