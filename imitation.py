"""模仿学习预训练：RuleAgent 当地主自对弈生成专家数据 -> 行为克隆(BC)训练 Q 网络。

用法：python imitation.py --games 3000 --epochs 10 [--dueling]
输出：checkpoints/bc.pt（或 bc_dueling.pt）与 checkpoints/expert.pt（专家数据，供微调辅助损失用）
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from doudizhu.agents import RuleAgent
from doudizhu.dqn import DQNAgent
from doudizhu.env import OBS_DIM, DoudizhuEnv
from evaluate import play


def collect(games, seed):
    """规则地主 vs 规则农民，记录地主每一步的 (状态, 动作, 合法掩码)。"""
    env, expert = DoudizhuEnv([RuleAgent(), RuleAgent()]), RuleAgent()
    S, A, M, wins = [], [], [], 0
    for g in range(games):
        obs, info = env.reset(seed=seed + g)
        done = False
        while not done:
            a = expert.act(env.view(0), info["action_mask"])
            S.append(obs); A.append(a); M.append(info["action_mask"])
            obs, r, done, _, info = env.step(a)
        wins += r > 0
    print(f"专家数据：{games} 局，{len(S)} 个样本，专家胜率 {wins / games:.1%}")
    return torch.tensor(np.array(S)), torch.tensor(A), torch.tensor(np.array(M))


def train_bc(agent, data, epochs, batch=256, lr=1e-3):
    """把 Q 网络输出当 logits，在合法动作上做交叉熵（掩码外置 -1e9）。"""
    S, A, M = data
    opt = torch.optim.Adam(agent.q.parameters(), lr=lr)
    n = len(S)
    for ep in range(epochs):
        perm, tot, correct = torch.randperm(n), 0.0, 0
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            logits = agent.q(S[j]).masked_fill(~M[j], -1e9)
            loss = nn.functional.cross_entropy(logits, A[j])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(j); correct += (logits.argmax(1) == A[j]).sum().item()
        print(f"  BC epoch {ep + 1:2d} | loss {tot / n:.4f} | 训练集动作一致率 {correct / n:.1%}", flush=True)
    agent.target.load_state_dict(agent.q.state_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--dueling", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    t0 = time.time()
    data = collect(args.games, seed=500_000)  # 与训练(0~)、评估(10000~)种子区间都不重叠
    agent = DQNAgent(OBS_DIM, dueling=args.dueling)
    train_bc(agent, data, args.epochs)
    env = DoudizhuEnv([RuleAgent(), RuleAgent()])
    wr = play(env, lambda o, m, e: agent.act(o, m, greedy=True), 200, 50_000)
    print(f"BC 模型评估(对规则农民200局) {wr:.1%} | 用时 {time.time() - t0:.0f}s")
    os.makedirs("checkpoints", exist_ok=True)
    name = "bc_dueling" if args.dueling else "bc"
    agent.save(f"checkpoints/{name}.pt")
    torch.save(data, "checkpoints/expert.pt")
    print("已保存", f"checkpoints/{name}.pt", "checkpoints/expert.pt")


if __name__ == "__main__":
    main()
