# CS-SE-PSO: Cold-Start-Aware Serverless Edge Particle Swarm Optimization

---

## Problem

Serverless (FaaS) deployments on MEC edge nodes suffer from **cold-start penalties**: routing a task to a node where the function container is not warm adds 50–200 ms of initialization delay. Conventional task-offloading optimizers (greedy, PSO, DRL) ignore container warm state entirely — they minimize transmission and queueing delay, but blindly trigger cold starts that dominate tail latency under high load.

## Proposed Method

**CS-SE-PSO** is a binary Particle Swarm Optimization variant with three cold-start-aware modifications:

1. **Cold-start objective term** — adds `δ_cs × CSR × 100` to the per-slot optimization objective, making cold-start rate an explicit optimization target alongside delay and load balance.
2. **Warm-affinity seeding** — 40% of initial particles are biased toward nodes where the required function container is already warm, accelerating convergence.
3. **Warm-start across slots** — the best assignment from the previous 100 ms slot is injected as a seed particle in the next slot.

The algorithm is **training-free**, re-optimizes every 100 ms slot in **17 ms** on a single CPU core, and requires no offline training or topology-specific warm-up.

---

## Key Results (100 seeds × 200 slots)

| Method | Mean latency | p95 latency | CSR | Training |
|---|---|---|---|---|
| Threshold | 49.0 ms | 94.4 ms | 0.268 | — |
| Greedy | 47.6 ms | 95.6 ms | 0.300 | — |
| SE-PSO (cold-naive) | 44.6 ms | 94.7 ms | 0.212 | — |
| **CS-SE-PSO** | **36.1 ms** | **67.1 ms** | **0.035** | **none** |
| WarmGreedy | 34.2 ms | 64.0 ms | 0.007 | — |

*At λ = 30 tasks/slot, T_cold = 50 ms, M = 8 edge nodes.*

- **−19% mean latency** and **−29% p95 latency** vs. cold-naive SE-PSO at high load
- Cold-start rate reduced from **0.21 → 0.035** (6×)
- CS-SE-PSO beats WarmGreedy on **p95 by 3–4% at moderate loads** (λ = 12–20) through explicit load-balance optimization
- Stays within **2.35% of the combinatorial optimum** (gap analysis vs. exact brute-force)
- Graceful degradation vs. T_cold: only +9.6 ms over 0–200 ms range (vs. +13 ms for SE-PSO, +72 ms for Greedy)

---

## Repository Structure

```
cs-se-pso/
├── algorithms/
│   ├── pso.py            # CS-SE-PSO and SE-PSO (binary PSO, warm-affinity seeding)
│   ├── baselines.py      # Threshold, Greedy, WarmGreedy baselines
│   ├── milp.py           # Exact MILP solver for optimality-gap analysis
│   ├── drl_method.py     # DRL baseline wrapper
│   └── drl/              # DRL agent, model, environment, replay buffer
├── model/
│   ├── warm_pool.py      # LRU warm-pool with keep-alive tracking
│   ├── objective.py      # Surrogate objective F(a)
│   ├── delay.py          # Transmission + queueing + cold-start delay
│   ├── queue.py          # FIFO queue with inter-slot backlog carry-over
│   ├── cold.py           # Cold-start detection and tracking
│   ├── task.py           # Task dataclass (D_i, C_i, g_i, uid_i)
│   └── network.py        # MEC network topology generator
├── sim/
│   ├── simulator.py      # Main simulation loop (200 slots × N seeds)
│   └── realized.py       # Realized evaluator (true FIFO, real cold tracking)
├── experiments/
│   ├── _common.py        # Shared runner, algorithm registry, parallel execution
│   ├── e1_latency.py     # E1: mean/p95/p99 latency + CSR vs. arrival rate λ
│   ├── e3_tcold.py       # E3: sensitivity to cold-start penalty T_cold
│   ├── e4_pareto.py      # E4: Pareto sweep over β and δ_cs
│   ├── e4b_wmax.py       # Sensitivity to warm-pool capacity w_max
│   └── e5_milp_gap.py    # E5: optimality-gap vs. exact MILP
├── plotting/
│   └── make_figures.py   # Matplotlib figure generation (F3–F6)
├── figures/
│   ├── F3_latency_vs_load.png   # Mean and p95 latency vs. λ
│   ├── F4_coldstart_rate.png    # Cold-start rate vs. λ
│   ├── F5_tcold_sweep.png       # Latency vs. T_cold at λ = 12
│   └── F6_pareto.png            # Pareto: load balance vs. cold-start rate
├── results/
│   ├── e1.csv            # Raw E1 results (100 seeds × 9 λ values)
│   ├── e3.csv            # Raw E3 results (T_cold sweep)
│   ├── e4.csv            # Raw E4 results (δ_cs sweep)
│   └── e5_gap.csv        # Raw E5 results (MILP optimality gap)
├── tests/                # Unit tests for all model components
├── config.py             # Default simulation parameters
├── metrics.py            # Metric aggregation utilities
├── smoke_methods.py      # Quick sanity-check runner
├── train_drl.py          # DRL baseline training script
└── requirements.txt      # Python dependencies
```

---

## Simulation Parameters

| Parameter | Value |
|---|---|
| Edge nodes M | 8 |
| User devices N | 50 |
| Function types G | 12 |
| Slot duration τ | 100 ms |
| Arrival rate λ | 4–30 tasks/slot (Poisson) |
| Function type distribution | Zipf(a = 1.3) |
| Cold-start penalty T_cold | 50 ms (default), 0–200 ms sweep |
| Keep-alive | 5 slots |
| Warm-pool capacity w_max | 4 types/node |
| PSO swarm size K | 40 |
| PSO iterations Q | 30 |
| Warm-affinity seed ratio | 40% (16/40 particles) |
| Seeds | 100 |
| Slots per run | 200 |

---

## Running Experiments

```bash
pip install -r requirements.txt

# E1: latency vs. arrival rate (main result)
python -m experiments.e1_latency

# E3: sensitivity to T_cold
python -m experiments.e3_tcold

# E4: delta_cs sweep
python -m experiments.e4_pareto

# E5: optimality gap (requires PuLP/CBC)
python -m experiments.e5_milp_gap

# Reproduce all figures
python -m plotting.make_figures
```

Results are saved to `results/` as CSV. Figures are saved to `figures/`.

---

## Requirements

- Python 3.10+
- NumPy, SciPy, Matplotlib
- PuLP + CBC solver (for E5 optimality gap only)
- PyTorch (for DRL baseline training only)

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this code, please cite:

```
Belov M.A. CS-SE-PSO: Cold-Start-Aware Serverless Edge Particle Swarm Optimization
for Task Offloading in Mobile Edge Computing.
Proceedings of DCCN 2026, Springer LNCS (2026).
```

---

## License

MIT
