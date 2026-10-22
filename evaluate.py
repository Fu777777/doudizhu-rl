"""评估脚本：地主（DQN/随机/规则）对战 两个规则农民 或 两个随机农民，统计胜率。"""
import argparse

import numpy as np
import torch

from doudizhu.agents import RandomAgent, RuleAgent
from doudizhu.dqn import DQNAgent
from doudizhu.env import OBS_DIM, DoudizhuEnv


def play(env, policy, games, seed):
    """policy(obs, mask, env) -> 动作；返回地主胜率。"""
    wins = 0
    for g in range(games):
        obs, info = env.reset(seed=seed + g)  # 同一组种子 => 各策略面对相同的牌局
        # 审核修复：随机农民每局按局号重新播种，使结果与“先评估了哪些模型”无关（原实现 rng 状态跨模型延续）
        for i, opp in enumerate(env.opponents):
            if hasattr(opp, "rng"):
                opp.rng = np.random.default_rng((seed + g) * 10 + i)
        done = False
        while not done:
            obs, r, done, _, info = env.step(policy(obs, info["action_mask"], env))
        wins += r > 0
    return wins / games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["checkpoints/dqn.pt", "checkpoints/double_dqn.pt", "checkpoints/dueling_double.pt",
                                                     "checkpoints/bc.pt", "checkpoints/bc_double.pt", "checkpoints/bc_double_aux.pt"])
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=10_000)  # 与训练种子区间不重叠
    args = ap.parse_args()
    torch.set_num_threads(1)

    policies = {"随机地主": lambda o, m, e, a=RandomAgent(0): a.act(None, m),
                "规则地主": lambda o, m, e, a=RuleAgent(): a.act(e.view(0), m)}
    for path in args.models:
        agent = DQNAgent(OBS_DIM)
        agent.load(path)
        policies[path] = lambda o, m, e, a=agent: a.act(o, m, greedy=True)

    for opp_name, opps in [("规则农民", [RuleAgent(), RuleAgent()]), ("随机农民", [RandomAgent(1), RandomAgent(2)])]:
        env = DoudizhuEnv(opps)
        print(f"== 对手: {opp_name}，{args.games} 局 ==")
        for name, pol in policies.items():
            print(f"  {name:28s} 地主胜率 {play(env, pol, args.games, args.seed):.1%}")


if __name__ == "__main__":
    main()
