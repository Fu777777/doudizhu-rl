"""DQN / Double DQN：Q 网络、经验回放、目标网络、epsilon 衰减、动作掩码。"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .cards import NUM_ACTIONS


class QNet(nn.Module):
    """MLP：状态 -> 每个动作的 Q 值。dueling=True 时拆成 V(s) + A(s,a) - mean(A)。"""

    def __init__(self, obs_dim, hidden=256, dueling=False):
        super().__init__()
        self.dueling = dueling
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.adv = nn.Linear(hidden, NUM_ACTIONS)   # 普通 DQN 时就是 Q 头
        if dueling:
            self.val = nn.Linear(hidden, 1)          # 状态价值 V(s)

    def forward(self, x):
        h = self.body(x)
        a = self.adv(h)
        if not self.dueling:
            return a
        # 减去均值保证 V/A 可辨识（Wang et al. 2016）
        return self.val(h) + a - a.mean(1, keepdim=True)


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
    def __init__(self, obs_dim, double=False, dueling=False, lr=1e-3, gamma=0.99, batch=128,
                 buffer=50000, target_update=500, eps_start=1.0, eps_end=0.05, eps_decay_steps=20000):
        self.q = QNet(obs_dim, dueling=dueling)
        self.target = QNet(obs_dim, dueling=dueling)
        self.expert = None      # 可选：专家数据 (s, a, mask)，用于微调时的行为克隆辅助损失
        self.bc_weight = 0.0
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
        if self.expert is not None and self.bc_weight > 0:
            # DQfD 风格：微调时继续在专家数据上加一项交叉熵，防止预训练知识被自举误差冲掉
            es, ea, em = self.expert
            idx = torch.randint(len(es), (self.batch,))
            logits = self.q(es[idx]).masked_fill(~em[idx], -1e9)
            loss = loss + self.bc_weight * nn.functional.cross_entropy(logits, ea[idx])
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        if self.steps % self.target_update == 0:  # 定期硬同步目标网络
            self.target.load_state_dict(self.q.state_dict())
        return loss.item()

    def save(self, path):
        torch.save({"state": self.q.state_dict(), "dueling": self.q.dueling}, path)

    def load(self, path):
        """兼容旧格式（只存 state_dict）；返回后 target 也同步。"""
        ck = torch.load(path)
        if "state" not in ck:
            ck = {"state": {k.replace("net.0", "body.0").replace("net.2", "body.2").replace("net.4", "adv"): v
                            for k, v in ck.items()}, "dueling": False}
        if ck["dueling"] != self.q.dueling:
            self.q = QNet(self.q.body[0].in_features, dueling=ck["dueling"])
            self.target = QNet(self.q.body[0].in_features, dueling=ck["dueling"])
            self.opt = torch.optim.Adam(self.q.parameters(), lr=self.opt.param_groups[0]["lr"])
        self.q.load_state_dict(ck["state"])
        self.target.load_state_dict(ck["state"])
