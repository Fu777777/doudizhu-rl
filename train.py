"""训练脚本：DQN 或 Double DQN 地主 vs 农民（默认混合随机/规则农民，逐步偏向规则农民）。"""
import argparse
import os
import random
import time

import numpy as np
import torch

from doudizhu.agents import RandomAgent, RuleAgent
from doudizhu.dqn import DQNAgent
from doudizhu.env import OBS_DIM, DoudizhuEnv
from evaluate import play


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dqn", "double"], default="double")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--dueling", action="store_true", help="使用 Dueling 网络结构")
    ap.add_argument("--init", default=None, help="用预训练(BC)权重初始化，如 checkpoints/bc.pt")
    ap.add_argument("--eps-start", type=float, default=1.0, help="BC 初始化时建议调小，如 0.2")
    ap.add_argument("--bc-weight", type=float, default=0.0, help="微调时专家交叉熵辅助损失权重")
    ap.add_argument("--name", default=None, help="输出模型名，默认按算法命名")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    agent = DQNAgent(OBS_DIM, double=args.algo == "double", dueling=args.dueling, eps_start=args.eps_start)
    if args.init:
        agent.load(args.init)  # 行为克隆预训练权重 -> 在线网络与目标网络
        print("从预训练权重初始化:", args.init)
    if args.bc_weight > 0:
        agent.expert, agent.bc_weight = torch.load("checkpoints/expert.pt"), args.bc_weight
    rule_env = DoudizhuEnv([RuleAgent(), RuleAgent()])
    rand_env = DoudizhuEnv([RandomAgent(1), RandomAgent(2)])
    eval_env = DoudizhuEnv([RuleAgent(), RuleAgent()])
    greedy = lambda o, m, e: agent.act(o, m, greedy=True)
    t0, wins, losses = time.time(), 0, []

    for ep in range(1, args.episodes + 1):
        # 课程式对手：前期多打随机农民，后期 80% 打规则农民
        env = rule_env if random.random() < min(0.8, ep / args.episodes + 0.2) else rand_env
        obs, info = env.reset(seed=args.seed * 1_000_000 + ep)
        done = False
        while not done:
            a = agent.act(obs, info["action_mask"])
            obs2, r, done, _, info = env.step(a)
            # 下一状态是“对手出完后轮到我”的状态，同时存它的合法掩码供目标值计算
            agent.buffer.push(obs, a, r, obs2, float(done), info["action_mask"])
            l = agent.learn()
            if l is not None:
                losses.append(l)
            obs = obs2
        wins += r > 0
        if ep % args.eval_every == 0:
            wr = play(eval_env, greedy, 200, 50_000)
            print(f"[{args.algo}] ep {ep:5d} | 训练近{args.eval_every}局胜率 {wins / args.eval_every:.1%} | "
                  f"eps {agent.epsilon:.2f} | loss {np.mean(losses[-1000:]):.4f} | "
                  f"评估(对规则农民200局) {wr:.1%} | {time.time() - t0:.0f}s", flush=True)
            wins = 0

    os.makedirs("checkpoints", exist_ok=True)
    name = args.name or ("double_dqn" if args.algo == "double" else "dqn")
    path = f"checkpoints/{name}.pt"
    agent.save(path)
    print("模型已保存:", path)


if __name__ == "__main__":
    main()
