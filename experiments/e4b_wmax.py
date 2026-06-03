"""E4B — Парето баланс-vs-тепло при ТЕСНОЙ памяти узла W_max (вариант B).
При малом W_max удержание функций тёплыми требует консолидации -> реальный
компромисс: с ростом δ_cs растёт дисбаланс. Сравниваем W_max=1,2 (vs 4 из e4.csv)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._common import run_grid, aggregate

DELTAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
METHODS = [f'pso-w:{d}' for d in DELTAS]

if __name__ == '__main__':
    for wmax in (1, 2):
        df = run_grid(f'e4_wmax{wmax}', METHODS, axis='lam', values=[12],
                      seeds=range(100), n_slots=200, cfg_kw={'w_max': wmax})
        agg = aggregate(df)
        print(f'--- W_max={wmax} ---')
        for d, m in zip(DELTAS, METHODS):
            r = agg[agg.method == m]
            if len(r):
                r = r.iloc[0]
                print(f'delta={d:.2f} load_cv={r["load_cv"]:.3f} '
                      f'csr={r["csr"]:.3f} p95={r["p95"]:.1f}')
    print('E4B_DONE')
