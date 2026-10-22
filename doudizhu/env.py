"""自定义 gymnasium 环境：智能体扮演地主（玩家 0），两个农民由给定策略控制。"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .cards import ACTIONS, NUM_ACTIONS, NUM_RANKS, PASS, full_deck, legal_mask

# 状态维度：自己手牌15 + 已出牌15 + 上一手牌15 + 三家剩余牌数3 + 上一手是谁出的3 + 两个农民各自已出牌15*2
OBS_DIM = NUM_RANKS * 5 + 6


class DoudizhuEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, opponents, verbose=False):
        """opponents: 两个农民策略对象，需实现 act(game_view, mask) -> 动作下标。"""
        super().__init__()
        self.opponents = opponents
        self.verbose = verbose
        self.observation_space = spaces.Box(0.0, 1.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    # ---------- 游戏核心 ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        deck = full_deck()
        self.np_random.shuffle(deck)
        self.hands = [np.zeros(NUM_RANKS, dtype=np.int8) for _ in range(3)]
        for i, c in enumerate(deck[:51]):
            self.hands[i % 3][c] += 1
        for c in deck[51:]:  # 3 张底牌归地主
            self.hands[0][c] += 1
        self.played = [np.zeros(NUM_RANKS, dtype=np.int8) for _ in range(3)]
        self.last_action = None   # 当前需要压的牌（None=自由出牌）
        self.last_player = -1     # 打出 last_action 的玩家
        self.current = 0          # 地主先出
        return self._obs(0), {"action_mask": self.mask(0)}

    def mask(self, p):
        last = None if self.last_player in (-1, p) else self.last_action
        return legal_mask(self.hands[p], last)

    def _apply(self, p, idx):
        """玩家 p 执行动作 idx，返回该玩家是否出完牌。"""
        a = ACTIONS[idx]
        if self.verbose:
            print(f"  {['地主', '农民1', '农民2'][p]}: {a}")
        if idx != PASS:
            self.hands[p] -= a.counts
            self.played[p] += a.counts
            self.last_action, self.last_player = a, p
        self.current = (p + 1) % 3
        return self.hands[p].sum() == 0

    def _obs(self, p):
        """以玩家 p 的视角编码状态（各计数除以 4 归一化到 [0,1]）。"""
        last = np.zeros(NUM_RANKS) if self.last_action is None or self.last_player == p else self.last_action.counts
        who = np.zeros(3)
        if self.last_player >= 0:
            who[(self.last_player - p) % 3] = 1  # 相对位置：0 自己，1 下家，2 上家
        played_all = sum(self.played)
        left = np.array([self.hands[(p + i) % 3].sum() for i in range(3)]) / 20.0
        return np.concatenate([
            self.hands[p] / 4, played_all / 4, np.asarray(last) / 4, left, who,
            self.played[(p + 1) % 3] / 4, self.played[(p + 2) % 3] / 4,
        ]).astype(np.float32)

    def view(self, p):
        """给规则/随机对手用的“可见信息”。"""
        return {"hand": self.hands[p].copy(), "last_action": self.last_action,
                "last_player": self.last_player, "me": p,
                "left": [int(h.sum()) for h in self.hands]}

    # ---------- gym 接口 ----------
    def step(self, action):
        assert self.mask(0)[action], f"非法动作 {ACTIONS[action]}"
        if self._apply(0, action):
            return self._obs(0), 1.0, True, False, {"action_mask": self.mask(0), "winner": "landlord"}
        for p in (1, 2):  # 两个农民依次出牌
            idx = self.opponents[p - 1].act(self.view(p), self.mask(p))
            if self._apply(p, idx):
                return self._obs(0), -1.0, True, False, {"action_mask": self.mask(0), "winner": "farmer"}
        return self._obs(0), 0.0, False, False, {"action_mask": self.mask(0)}
