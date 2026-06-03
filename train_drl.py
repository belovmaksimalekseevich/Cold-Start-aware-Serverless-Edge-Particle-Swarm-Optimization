"""
Обучение DRL-бейзлайна (Double-Dueling DQN + PER + n-step) на OffloadEnv.

Замеряем ВРЕМЯ ОБУЧЕНИЯ и число шагов — это прямой аргумент в пользу CS-SE-PSO
(training-free): покажем, что DQN достигает сопоставимого качества ценой N минут
обучения, тогда как CS-SE-PSO работает сразу.

    python train_drl.py --episodes 1500 --slots 50
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from config import DEFAULT
from algorithms.drl.offload_env import OffloadEnv, state_dim, action_dim
from algorithms.drl.agent import DQNAgent


def train(cfg, episodes, slots, seed=0, ckpt='results/dqn_offload.pth', update_every=4):
    env = OffloadEnv(cfg, seed=seed, n_slots=slots, lam_range=(8, 20))
    # грубая оценка числа ОБНОВЛЕНИЙ сети для расписаний lr/eps/beta
    est_steps = episodes * slots * 14 // update_every
    agent = DQNAgent(state_dim=state_dim(cfg), action_dim=action_dim(cfg),
                     total_steps=est_steps, checkpoint_path=ckpt)

    t0 = time.time()
    total_steps = 0
    recent = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            a = agent.select_action(s)
            s2, r, done, _ = env.step(a)
            agent.push(s, a, r, s2, done)
            if total_steps % update_every == 0:   # обновляем сеть раз в k шагов (×k скорость)
                agent.update()
            s = s2
            ep_r += r
            total_steps += 1
        agent.record_episode_reward(ep_r)
        recent.append(ep_r)
        if (ep + 1) % 100 == 0:
            mean_r = np.mean(recent[-100:])
            elapsed = time.time() - t0
            print(f"эп {ep+1:5d}/{episodes} | шагов {total_steps:7d} | "
                  f"reward(100) {mean_r:7.2f} | eps {agent.eps:.3f} | "
                  f"{elapsed:6.1f} c")

    train_time = time.time() - t0
    agent.save(ckpt)
    print(f"\n=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")
    print(f"эпизодов: {episodes}, шагов: {total_steps}")
    print(f"время обучения: {train_time:.1f} c ({train_time/60:.1f} мин)")
    print(f"чекпойнт: {ckpt}")
    # сохраним метрики обучения для статьи
    os.makedirs('results', exist_ok=True)
    with open('results/dqn_train_cost.txt', 'w', encoding='utf-8') as f:
        f.write(f"episodes={episodes}\nsteps={total_steps}\n"
                f"train_time_s={train_time:.1f}\ntrain_time_min={train_time/60:.2f}\n")
    return train_time, total_steps


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=1500)
    ap.add_argument('--slots', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--update_every', type=int, default=4)
    ap.add_argument('--ckpt', type=str, default='results/dqn_offload.pth')
    args = ap.parse_args()
    train(DEFAULT, args.episodes, args.slots, args.seed, args.ckpt, args.update_every)
