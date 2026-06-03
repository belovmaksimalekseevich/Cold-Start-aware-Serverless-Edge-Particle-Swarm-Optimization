"""
DRLMethod — обёртка обученной DQN-политики под интерфейс OffloadingMethod.

Грузит чекпойнт, и в decide(tasks, ctx) размещает задачи слота по одной, строя
состояние ИДЕНТИЧНО обучению (build_state из offload_env), беря текущий бэклог из
ctx.prev_load и накапливая внутрислотовую загрузку. Это позволяет сравнивать DRL с
CS-SE-PSO на том же честном симуляторе.
"""
import numpy as np
import torch
from .base import OffloadingMethod
from .drl.offload_env import build_state, state_dim, action_dim
from .drl.model import DuelingDQN


class DRLMethod(OffloadingMethod):
    name = "dqn"

    def __init__(self, cfg, checkpoint_path):
        self.cfg = cfg
        self.sdim = state_dim(cfg)
        self.adim = action_dim(cfg)
        self.net = DuelingDQN(self.sdim, self.adim)
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        self.net.load_state_dict(ckpt['online'])
        self.net.eval()

    @torch.no_grad()
    def decide(self, tasks, ctx):
        cfg = self.cfg
        M = cfg.M
        n = len(tasks)
        if n == 0:
            return []
        F_edge = np.asarray(ctx.F_edge)
        base = (np.zeros(M) if ctx.prev_load is None
                else np.asarray(ctx.prev_load, dtype=float))
        cum = base * (F_edge * cfg.tau)          # бэклог в циклах
        assign = []
        for i, t in enumerate(tasks):
            node_load = cum / (F_edge * cfg.tau)
            warm = np.array([1.0 if t.g in ctx.warm_sets[j] else 0.0 for j in range(M)])
            s = build_state(t, node_load, warm, i / n, cfg)
            q = self.net(torch.FloatTensor(s).unsqueeze(0))
            a = int(q.argmax(dim=-1).item())
            if a >= M:
                assign.append(-1)                 # локально
            else:
                assign.append(a)
                cum[a] += t.C                      # накапливаем внутрислотовую загрузку
        return assign
