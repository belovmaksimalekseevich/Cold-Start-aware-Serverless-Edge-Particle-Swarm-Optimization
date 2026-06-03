"""
Построение рисунков F3–F6 (English, Springer LNCS-стиль) из results/*.csv.

Читает кешированные результаты, не запускает симуляции. Каждый рисунок — отдельная
функция (регенерируются независимо). Пропускает отсутствующие csv.

    python plotting/make_figures.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments._common import aggregate, RESULTS_DIR

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# единый стиль методов
STYLE = {
    'threshold':  ('Threshold',            'tab:gray',   's', '--'),
    'greedy':     ('Greedy (least-loaded)','tab:orange', '^', '--'),
    'se-pso':     ('SE-PSO (cold-naive)',  'tab:green',  'D', '-'),
    'cs-se-pso':  ('CS-SE-PSO (proposed)', 'tab:blue',   'o', '-'),
}
plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3,
                     'figure.dpi': 150, 'savefig.bbox': 'tight'})


def _load(exp):
    p = os.path.join(RESULTS_DIR, f'{exp}.csv')
    return aggregate(pd.read_csv(p)) if os.path.exists(p) else None


def fig_latency_vs_load():
    """F3: средняя и хвостовая (p95) задержка vs λ."""
    agg = _load('e1')
    if agg is None:
        print('F3: нет e1.csv'); return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for metric, ax, title in [('mean', axes[0], 'Mean latency'),
                              ('p95', axes[1], '95th-percentile (tail) latency')]:
        for m, (label, col, mk, ls) in STYLE.items():
            d = agg[agg.method == m].sort_values('value')
            if d.empty:
                continue
            ax.errorbar(d.value, d[metric], yerr=d[metric + '_ci'],
                        label=label, color=col, marker=mk, linestyle=ls,
                        markersize=5, capsize=3, linewidth=1.8)
        ax.set_xlabel('Task arrival rate $\\lambda$ (tasks/slot)')
        ax.set_ylabel(f'{title} (ms)')
        ax.set_title(title)
    axes[0].legend(fontsize=9)
    fig.savefig(os.path.join(FIG_DIR, 'F3_latency_vs_load.png'))
    plt.close(fig)
    print('F3 сохранён')


def fig_csr_vs_load():
    """F4: cold-start rate vs λ."""
    agg = _load('e1')
    if agg is None:
        print('F4: нет e1.csv'); return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m, (label, col, mk, ls) in STYLE.items():
        d = agg[agg.method == m].sort_values('value')
        if d.empty:
            continue
        ax.errorbar(d.value, d.csr, yerr=d.csr_ci, label=label, color=col,
                    marker=mk, linestyle=ls, markersize=5, capsize=3, linewidth=1.8)
    ax.set_xlabel('Task arrival rate $\\lambda$ (tasks/slot)')
    ax.set_ylabel('Cold-start rate')
    ax.set_title('Cold-start rate vs load')
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(FIG_DIR, 'F4_coldstart_rate.png'))
    plt.close(fig)
    print('F4 сохранён')


def fig_tcold_sweep():
    """F5: задержка vs T_cold (линейная деградация cold-naive vs плоская CS-SE-PSO)."""
    agg = _load('e3')
    if agg is None:
        print('F5: нет e3.csv'); return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for metric, ax, title in [('mean', axes[0], 'Mean latency'),
                              ('p95', axes[1], '95th-percentile latency')]:
        for m, (label, col, mk, ls) in STYLE.items():
            d = agg[agg.method == m].sort_values('value')
            if d.empty:
                continue
            ax.errorbar(d.value * 1e3, d[metric], yerr=d[metric + '_ci'],
                        label=label, color=col, marker=mk, linestyle=ls,
                        markersize=5, capsize=3, linewidth=1.8)
        ax.set_xlabel('Cold-start latency $T_{cold}$ (ms)')
        ax.set_ylabel(f'{title} (ms)')
        ax.set_title(title)
    axes[0].legend(fontsize=9)
    fig.savefig(os.path.join(FIG_DIR, 'F5_tcold_sweep.png'))
    plt.close(fig)
    print('F5 сохранён')


def fig_pareto():
    """F6: Парето-граница «дисбаланс нагрузки vs cold-start rate» по δ_cs."""
    agg = _load('e4')
    if agg is None:
        print('F6: нет e4.csv'); return
    d = agg.copy()
    d['delta'] = d.method.str.replace('pso-w:', '', regex=False).astype(float)
    d = d.sort_values('delta')
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.plot(d.csr, d.load_cv, '-o', color='tab:blue', linewidth=1.8, markersize=6)
    for _, r in d.iterrows():
        ax.annotate(f"$\\delta$={r['delta']:.2f}", (r['csr'], r['load_cv']),
                    textcoords='offset points', xytext=(6, 4), fontsize=8)
    ax.set_xlabel('Cold-start rate')
    ax.set_ylabel('Load imbalance (CV of node load)')
    ax.set_title('Balance–warmth trade-off (vary cold weight $\\delta_{cs}$)')
    fig.savefig(os.path.join(FIG_DIR, 'F6_pareto.png'))
    plt.close(fig)
    print('F6 сохранён')


if __name__ == '__main__':
    fig_latency_vs_load()
    fig_csr_vs_load()
    fig_tcold_sweep()
    fig_pareto()
    print(f'Рисунки в {FIG_DIR}')
