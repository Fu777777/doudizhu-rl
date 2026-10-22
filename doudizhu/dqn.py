"""DQN / Double DQN：Q 网络、经验回放、目标网络、epsilon 衰减、动作掩码。"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .cards import NUM_ACTIONS


class QNet(nn.Module):
    """简单 MLP：状态 -> 每个动作的 Q 值。"""

    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, NUM_ACTIONS))

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """经验回放：存 (s, a, r, s', done, s'的合法掩码)，随机采样打破相关性。"""

    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, *t):
        self.buf.append(t)

    def sample(self, n):
        s, a, r, s2, d, m2 = zip(*random.sample(self.buf, n))
        return (torch.tensor(np.array(s)), torch.tensor(a), torch.tensor(r, dtype=torch.float32),
                torch.tensor(np.array(s2)), torch.tensor(d, dtype=torch.float32), torch.tensor(np.array(m2)))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, obs_dim, double=False, lr=1e-3, gamma=0.99, batch=128,
                 buffer=50000, target_update=500, eps_start=1.0, eps_end=0.05, eps_decay_steps=20000):
        self.q = QNet(obs_dim)
        self.target = QNet(obs_dim)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer)
        self.double, self.gamma, self.batch, self.target_update = double, gamma, batch, target_update
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.steps = 0

    @property
    def epsilon(self):
        """线性衰减的探索率。"""
        f = min(1.0, self.steps / self.eps_decay_steps)
        return self.eps_start + f * (self.eps_end - self.eps_start)

    def act(self, obs, mask, greedy=False):
        """epsilon-greedy，且只在合法动作中选择（非法动作 Q 置为 -inf）。"""
        if not greedy and random.random() < self.epsilon:
            return int(random.choice(np.flatnonzero(mask)))
        with torch.no_grad():
            q = self.q(torch.tensor(obs).unsqueeze(0))[0]
        q[~torch.tensor(mask)] = -float("inf")
        return int(q.argmax())

    def learn(self):
        self.steps += 1
        if len(self.buffer) < self.batch:
            return None
        s, a, r, s2, d, m2 = self.buffer.sample(self.batch)
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next_t = self.target(s2)
            if self.double:
                # Double DQN：在线网络选动作，目标网络评估，缓解 Q 值高估
                q_next_o = self.q(s2).masked_fill(~m2, -1e9)
                best = q_next_o.argmax(1, keepdim=True)
                q_next = q_next_t.gather(1, best).squeeze(1)
            else:
                # 普通 DQN：目标网络既选又评（仅在合法动作中取 max）
                q_next = q_next_t.masked_fill(~m2, -1e9).max(1).values
            target = r + self.gamma * (1 - d) * q_next
        loss = nn.functional.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        if self.steps % self.target_update == 0:  # 定期硬同步目标网络
            self.target.load_state_dict(self.q.state_dict())
        return loss.item()

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path))
